"""One inference pass for Mixed/four-illumination AP from a selected FP32 EMA.

This module never constructs a training loader, criterion, optimizer or solver.
Evaluation is explicit; importing it does not access any dataset or checkpoint.
"""
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import yaml
from .efficiency_protocol import ALGORITHM_SUITE


SCHEMA = 'paper_v2_ap_record_v1'
EVALUATION = {
    'precision': 'FP32', 'autocast': False, 'tf32': False,
    'iou_type': 'bbox', 'iou_thresholds': [round(.50 + .05 * i, 2) for i in range(10)],
    'max_dets': [1, 10, 100], 'input_size': [640, 640],
    'num_top_queries': 300, 'backend': 'PyTorch eager eval',
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_digest(mapping):
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        raise ValueError('illumination_map must be a JSON mapping')
    normalized = {str(key): value for key, value in mapping.items()}
    if len(normalized) != len(mapping):
        raise ValueError('illumination_map contains duplicate int/string keys')
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(checkpoint):
    stat = checkpoint.stat()
    return {'path': str(checkpoint), 'size_bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'model_state': 'ema'}


def _strict_run_binding(run_dir, model, seed):
    from tools.paper_v2.run import load_resolved, verify_recipe

    cfg = yaml.safe_load((run_dir / 'config_resolved.yml').read_text(encoding='utf-8'))
    info = json.loads((run_dir / 'run_info.json').read_text(encoding='utf-8'))
    # Parse the completion marker too: a truncated marker is not a completed run.
    json.loads((run_dir / 'completed.json').read_text(encoding='utf-8'))
    if (info.get('experiment') != model or type(info.get('seed')) is not int
            or info['seed'] != seed or type(cfg.get('seed')) is not int or cfg['seed'] != seed):
        raise ValueError('AP model/seed does not match saved run metadata')
    try:
        verify_recipe(cfg)
    except (AssertionError, KeyError) as exc:
        raise ValueError('AP requires the shared Paper v2 120-epoch recipe') from exc
    expected = load_resolved(model, expand_env=False)['innovation']
    actual = cfg['innovation']
    if (actual['instance_mask_sync'] != expected['instance_mask_sync'] or
            any(actual[key]['enabled'] != expected[key]['enabled']
                for key in ('vq_dense_o2o', 'li_copyblend', 'dc_sta'))):
        raise ValueError('AP model name does not match saved algorithm switches')
    if cfg.get('eval_spatial_size') != [640, 640] or cfg['PostProcessor']['num_top_queries'] != 300:
        raise ValueError('AP requires native 640 input and postprocessor top-k=300')
    selection = json.loads((run_dir / 'selection.json').read_text(encoding='utf-8'))
    if (selection.get('model') != 'ema' or selection.get('split') != 'mixed_validation'
            or selection.get('precision') != 'FP32' or selection.get('autocast') is not False
            or selection.get('tf32') is not False):
        raise ValueError('AP requires EMA FP32 Mixed Validation selection with autocast/TF32 disabled')
    epoch = selection.get('epoch')
    if type(epoch) is not int or not 0 <= epoch < 120:
        raise ValueError('Selected epoch must be zero-based within the 120-epoch recipe')
    checkpoint = run_dir / 'best.pth'
    return cfg, checkpoint, selection


def _binding(run_dir, model, seed, split, data_root, illumination_map, run_policy='strict'):
    # Lazy imports avoid a circular dependency with the unified CLI.
    from tools.paper_v2.run import check_assets

    if split not in ('test', 'validation'):
        raise ValueError('split must be test or validation')
    if type(seed) is not int or seed not in (0, 42, 3407):
        raise ValueError('Paper v2 seeds are 0, 42 and 3407')
    run_dir = Path(run_dir).resolve()
    if run_policy == 'strict':
        cfg, checkpoint, selection = _strict_run_binding(run_dir, model, seed)
    elif run_policy == 'existing-checkpoint':
        from tools.paper_v2.checkpoint_binding import bind_existing_checkpoint
        cfg, checkpoint, selection, provenance = bind_existing_checkpoint(run_dir, model, seed)
        cfg['_evaluation_provenance'] = provenance
    else:
        raise ValueError(f'Unknown run policy: {run_policy}')
    epoch = selection['epoch']
    identity = _checkpoint_identity(checkpoint)

    cfg = copy.deepcopy(cfg)
    if split == 'test':
        data_root = data_root or os.environ.get('TOMATO_DATA_ROOT')
        if not data_root:
            raise FileNotFoundError('Test evaluation requires --data-root or TOMATO_DATA_ROOT')
    if data_root is not None:
        root = Path(data_root).resolve()
        suffix = 'test' if split == 'test' else 'val'
        cfg['val_dataloader']['dataset'].update(
            img_folder=str(root / 'images' / suffix),
            ann_file=str(root / 'annotations' / f'instances_{suffix}.json'))
    check_assets(cfg, evaluation=True)
    ds = cfg['val_dataloader']['dataset']
    if ds.get('visible_mask_sidecar') or ds.get('return_masks', False):
        raise ValueError('AP must use the native bbox-only evaluation dataset')
    if ds['category_ids'] != cfg['PostProcessor'].get('category_ids'):
        raise ValueError('Dataset/postprocessor category mappings disagree')
    if cfg['val_dataloader'].get('drop_last', False):
        raise ValueError('AP cannot drop the final evaluation batch')
    annotation = Path(ds['ann_file']).resolve()
    dataset = {'annotation_file': str(annotation), 'annotation_sha256': _sha256(annotation),
               'image_folder': str(Path(ds['img_folder']).resolve()),
               'illumination_map_sha256': _mapping_digest(illumination_map)}
    cfg['DINOv3STAs']['weights_path'] = None
    return cfg, checkpoint, identity, epoch, dataset


def record_path(output_dir, model, seed):
    return Path(output_dir).resolve() / 'raw' / model / f'seed_{seed}' / 'metrics.json'


def reusable_record(run_dir, model, seed, output_dir, split='test', device='cuda:0',
                    data_root=None, illumination_map=None, run_policy='strict'):
    """Return an existing bound record, None if absent; reject stale provenance.

    This checks metadata and file fingerprints without decoding images or running
    inference. Existing-checkpoint mode also inspects checkpoint metadata/tensors.
    A changed checkpoint/dataset needs a new output directory.
    """
    path = record_path(output_dir, model, seed)
    if not path.exists():
        return None
    cfg, _, identity, epoch, dataset = _binding(
        run_dir, model, seed, split, data_root, illumination_map, run_policy)
    record = json.loads(path.read_text(encoding='utf-8'))
    expected = {'schema': SCHEMA, 'model': model, 'seed': seed, 'split': split,
                'selected_epoch_zero_based': epoch, 'checkpoint': identity,
                'dataset': dataset, 'evaluation': EVALUATION}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Existing AP record has different run/data/protocol provenance: {path}')
    recorded_digest = (record.get('run_provenance') or {}).get('inference_config_sha256')
    current_digest = (cfg.get('_evaluation_provenance') or {}).get('inference_config_sha256')
    if recorded_digest and current_digest and recorded_digest != current_digest:
        raise ValueError(f'Existing AP record has different inference configuration provenance: {path}')
    if not (path.parent / 'predictions.json').is_file():
        raise ValueError(f'Existing AP record is missing predictions.json: {path.parent}')
    from tools.paper_v2.ap_summary import validate_record
    validate_record(record, split=split)
    return record


def _create_config(cfg):
    from tools.paper_v2.run import create_config
    return create_config(cfg)


def _infer(model, postprocessor, loader, device):
    from engine.data.dataset.coco_eval import convert_to_xywh
    from engine.solver.det_engine import fp32_evaluation

    coco_gt = loader.dataset.coco
    expected_ids = set(coco_gt.getImgIds())
    seen, predictions = set(), []
    with torch.inference_mode(), fp32_evaluation(model, device, enabled=True):
        for samples, targets in loader:
            if samples.dtype != torch.float32 or list(samples.shape[1:]) != [3, 640, 640]:
                raise ValueError('AP expects the native FP32 [N,3,640,640] evaluation preprocessing')
            ids = [int(target['image_id'].item()) for target in targets]
            if len(ids) != len(set(ids)) or seen.intersection(ids) or not set(ids) <= expected_ids:
                raise ValueError('Evaluation loader contains duplicate or unexpected image IDs')
            samples = samples.to(device)
            sizes = torch.stack([target['orig_size'] for target in targets]).to(device)
            results = postprocessor(model(samples), sizes)
            if not isinstance(results, list) or len(results) != len(ids):
                raise ValueError('AP requires the native non-deploy per-image postprocessor output')
            for image_id, result in zip(ids, results):
                boxes = convert_to_xywh(result['boxes']).detach().cpu()
                scores = result['scores'].detach().cpu()
                labels = result['labels'].detach().cpu()
                if not (len(boxes) == len(scores) == len(labels)):
                    raise ValueError('Postprocessor instance fields have inconsistent lengths')
                if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                    raise ValueError('Non-finite predictions cannot be reported as AP')
                predictions.extend({'image_id': image_id, 'category_id': int(label),
                                    'bbox': box, 'score': score}
                                   for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()))
            seen.update(ids)
    if seen != expected_ids:
        raise ValueError(f'Evaluation loader missed {len(expected_ids - seen)} dataset images')
    return coco_gt, predictions


def _publish(path, record, predictions):
    """Publish both complete JSON files together, refusing existing campaigns."""
    destination = path.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'AP result directory already exists: {destination}')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}_', dir=destination.parent))
    try:
        (temporary / 'predictions.json').write_text(
            json.dumps(predictions, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        (temporary / 'metrics.json').write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if destination.exists():
            raise FileExistsError(f'AP result directory already exists: {destination}')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def evaluate_run(run_dir, model, seed, output_dir, split='test', device='cuda:0',
                 data_root=None, illumination_map=None, run_policy='strict'):
    """Evaluate one best.pth, write raw/<model>/seed_<seed>, return record.

    The caller explicitly invokes this function for Test; no Test access occurs
    in training or validation selection. Existing records are never overwritten.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise ValueError('AP evaluation is single-process; do not use torchrun/DDP')
    path = record_path(output_dir, model, seed)
    from tools.paper_v2.ap_summary import validate_campaign
    validate_campaign(output_dir, split)
    if path.parent.exists():
        raise FileExistsError(f'AP result directory already exists: {path.parent}')
    cfg, checkpoint, identity, epoch, dataset = _binding(
        run_dir, model, seed, split, data_root, illumination_map, run_policy)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if type(state.get('last_epoch')) is not int or state['last_epoch'] != epoch:
        raise ValueError('best.pth epoch differs from the recorded EMA selection')
    ema = state.get('ema', {}).get('module')
    if not isinstance(ema, dict) or not ema:
        raise ValueError('best.pth must contain the selected EMA module state')
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in ema.values() if isinstance(value, torch.Tensor)):
        raise ValueError('Selected checkpoint EMA must retain FP32 tensors')
    obj = _create_config(cfg)
    detector = obj.model
    from tools.paper_v2.checkpoint_binding import load_evaluation_state
    load_evaluation_state(detector, ema, run_policy, cfg)
    print(f'Loaded FP32 EMA from {checkpoint} (epoch {epoch + 1}, strict=True); evaluation only.', flush=True)
    detector = detector.to(device).eval()
    postprocessor = obj.postprocessor.to(device).eval()
    if getattr(postprocessor, 'deploy_mode', False):
        raise ValueError('AP requires the native postprocessor category mapping')
    del ema, state
    coco_gt, predictions = _infer(detector, postprocessor, obj.val_dataloader, device)
    from tools.paper_v2.ap_metrics import evaluate_predictions
    result = evaluate_predictions(coco_gt, predictions, illumination_map=illumination_map)
    if _checkpoint_identity(checkpoint) != identity or _sha256(dataset['annotation_file']) != dataset['annotation_sha256']:
        raise ValueError('Checkpoint or annotation changed during evaluation; refusing stale provenance')
    record = {'schema': SCHEMA, 'algorithm_suite': ALGORITHM_SUITE, 'model': model, 'seed': seed, 'split': split,
              'selected_epoch_zero_based': epoch, 'checkpoint': identity, 'dataset': dataset,
              'evaluation': copy.deepcopy(EVALUATION), 'device': str(device),
              'tool': {'name': 'faster-coco-eval', 'version': version('faster-coco-eval')},
              'evaluated_utc': datetime.now(timezone.utc).isoformat(), **result}
    if '_evaluation_provenance' in cfg:
        record['run_provenance'] = copy.deepcopy(cfg['_evaluation_provenance'])
    from tools.paper_v2.ap_summary import validate_record
    validate_record(record, split=split)
    _publish(path, record, predictions)
    return record
