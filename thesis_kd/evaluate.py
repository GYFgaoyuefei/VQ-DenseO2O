"""EMA Mixed Validation selection and frozen, single-process paper evaluation."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from engine.data.dataset.coco_eval import CocoEvaluator

from .data import build_frozen_test_loader, build_val_loader
from .factories import build_student, component, sha256_file


def selected_weights(cfg, run_dir):
    run_dir = Path(run_dir)
    selection = json.loads((run_dir / 'selection.json').read_text(encoding='utf8'))
    info = json.loads((run_dir / 'run_info.json').read_text(encoding='utf8'))
    for key in ('experiment', 'protocol_id', 'seed', 'run_tag'):
        if selection[key] != info[key]:
            raise ValueError(f'Selection/run metadata mismatch: {key}')
    if (cfg['experiment'], cfg['protocol_id'], cfg['arch']) != (
            info['experiment'], info['protocol_id'], info['arch']):
        raise ValueError('Selected checkpoint belongs to a different experiment or protocol')
    if selection['checkpoint'] != 'best_student.pth' or not 1 <= selection['epoch'] <= 120:
        raise ValueError('Invalid selected student or epoch')
    if (selection.get('source') != 'FP32 EMA Mixed Validation AP'
            or selection.get('tie_break') != 'earliest_epoch'
            or selection.get('teacher') != info.get('teacher')):
        raise ValueError('Run does not bind this student to FP32 EMA Mixed Validation selection')
    with (run_dir / 'validation_history.csv').open(encoding='utf8') as stream:
        history = [(float(row['AP']), int(row['epoch'])) for row in csv.DictReader(stream)]
    if not history or any(not 0 <= ap <= 1 for ap, _ in history):
        raise ValueError('Invalid Validation AP history')
    best = max(history, key=lambda record: (record[0], -record[1]))
    if best != (selection['mixed_ap'], selection['epoch']):
        raise ValueError('Selection does not match earliest maximum Validation AP')
    path = run_dir / 'best_student.pth'
    if sha256_file(path) != selection['sha256']:
        raise ValueError('Selected student SHA256 mismatch')
    return path, selection, info


def load_deployed_student(cfg, path, device):
    """Fresh native (unfused) student; no Teacher or pretraining weights needed."""
    model, _ = build_student(cfg, initialize=False)
    state = torch.load(path, weights_only=True, map_location='cpu')
    # FP32 EMA buffers (including RoPE periods) must retain their stored precision.
    model.float().load_state_dict(state, strict=True)
    return model.to(device).float().eval()


def _postprocessor(cfg, device):
    return component(cfg['native'], cfg['native']['postprocessor']).to(device).eval()


class StudentCocoEvaluator(CocoEvaluator):
    """Merge native per-image matches in image-ID order across padded DDP shards."""

    def synchronize_between_processes(self):
        from engine.misc import dist_utils

        shards = dist_utils.all_gather((self.img_ids, self.eval_imgs))
        ids = [image_id for local_ids, _ in shards for image_id in local_ids]
        ordered, first = np.unique(ids, return_index=True)
        if ordered.tolist() != sorted(self.coco_gt.getImgIds()):
            raise ValueError('Incomplete Validation image coverage')
        for kind, evaluator in self.coco_eval.items():
            blocks = [array for _, matches in shards for array in matches[kind]]
            combined = np.concatenate(blocks, axis=2)
            if combined.shape[2] != len(ids):
                raise ValueError('COCO evaluation results do not align with image IDs')
            evaluator.params.imgIds = ordered.tolist()
            evaluator._paramsEval = deepcopy(evaluator.params)
            evaluator._evalImgs_cpp = combined[:, :, first].ravel().tolist()


def validate_model(cfg, detector, device, seed):
    from engine.solver.det_engine import evaluate

    model = deepcopy(detector).to(device).float().eval()
    if torch.distributed.is_initialized():
        # Selection exports rank zero's EMA. DDP parameters already synchronize;
        # private EMA running statistics can differ when native BN is per rank.
        for buffer in model.buffers():
            torch.distributed.broadcast(buffer, src=0)
    loader = build_val_loader(cfg, seed)
    evaluator = StudentCocoEvaluator(loader.dataset.coco, ['bbox'])
    # evaluate() only invokes criterion.eval() during bbox validation.
    stats, _ = evaluate(model, torch.nn.Identity(), _postprocessor(cfg, device),
                        loader, evaluator, device, fp32=True)
    values = stats['coco_eval_bbox']
    if len(values) != 12 or not 0 <= values[0] <= 1:
        raise ValueError(f'Invalid Mixed Validation COCO statistics: {values}')
    return values


def freeze_selection(run_dir, selection):
    """Commit under the shared run writer lock before opening any Test input."""
    from .checkpoint import write_json
    marker = run_dir / 'frozen_selection.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf8')) != selection:
            raise ValueError('A different student selection was previously frozen for Test')
    else:
        write_json(marker, selection)


def evaluate_run(cfg, run_dir, device, *, split='val', confirm_frozen=False, illumination_map=None):
    if split not in ('val', 'test') or (split == 'test' and not confirm_frozen):
        raise ValueError('Select val, or test with explicit --confirm-frozen')
    if torch.distributed.is_initialized():
        raise ValueError('Run the frozen dataset report with single-process Python')
    from tools.paper_v2.ap_evaluate import _infer, _publish
    from tools.paper_v2.ap_metrics import evaluate_predictions

    run_dir = Path(run_dir)
    target = run_dir / f'evaluation_{split}' / 'metrics.json'
    if target.parent.exists() or target.parent.is_symlink():
        raise FileExistsError(f'Result directory already exists: {target.parent}')
    path, selection, info = selected_weights(cfg, run_dir)
    model = load_deployed_student(cfg, path, device)
    if split == 'test':
        freeze_selection(run_dir, selection)
        loader = build_frozen_test_loader(cfg, info['seed'], confirm_frozen=confirm_frozen)
    else:
        loader = build_val_loader(cfg, info['seed'], distributed=False)
    if illumination_map is not None and not isinstance(illumination_map, dict):
        illumination_map = json.loads(Path(illumination_map).read_text(encoding='utf8'))
    coco, predictions = _infer(model, _postprocessor(cfg, device), loader, device)
    record = evaluate_predictions(coco, predictions, illumination_map)
    map_digest = None if illumination_map is None else hashlib.sha256(
        json.dumps(illumination_map, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    record.update(split=split, selection=selection, frozen=bool(confirm_frozen),
                  illumination_map_sha256=map_digest,
                  precision='fp32', tf32=False, autocast=False,
                  annotation_sha256=sha256_file(loader.dataset.ann_file))
    _publish(target, record, predictions)
    return record
