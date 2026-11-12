"""Synthetic COCO/data contracts, real native deployment and statistics checks."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from thesis_kd.config import resolve_config
from thesis_kd.data import build_train_loader, build_val_loader
from thesis_kd.evaluate import evaluate_run, load_deployed_student, selected_weights
from thesis_kd.factories import build_student
from thesis_kd.summarize import AP_COLUMNS, CORE, SEEDS, summarize_runs


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding='utf8')


def _stats_for_ids(ids):
    from faster_coco_eval import COCO
    from thesis_kd.evaluate import StudentCocoEvaluator

    coco = COCO()
    coco.dataset = {'images': [{'id': i, 'width': 640, 'height': 640} for i in range(1, 6)],
                    'categories': [{'id': 1, 'name': 'immature'}, {'id': 3, 'name': 'ripe'}],
                    'annotations': []}
    for i in range(1, 6):
        size = 10 * i
        coco.dataset['annotations'].append(dict(id=i, image_id=i, category_id=1 if i < 3 else 3,
            bbox=[2*i, 1, size, size], area=size**2, iscrowd=0))
    coco.createIndex()
    evaluator = StudentCocoEvaluator(coco, ['bbox'])
    for i in ids:
        shift = 4*i if i % 2 else 0
        box = torch.tensor([[2*i+shift, 1, 12*i+shift, 1+10*i]], dtype=torch.float)
        evaluator.update({i: {'boxes': box, 'labels': torch.tensor([1 if i < 3 else 3]),
                              'scores': torch.tensor([0.4 + i/10])}})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return list(evaluator.coco_eval['bbox'].stats)


def _distributed_eval(rank, directory):
    import torch.distributed as dist
    directory = Path(directory)
    dist.init_process_group('gloo', store=dist.FileStore(str(directory / 'ddp_eval_store'), 2),
                            rank=rank, world_size=2)
    try:
        # DistributedSampler with 5 images, two ranks and padding.
        from torch.utils.data import DistributedSampler
        ids = [i + 1 for i in DistributedSampler(range(5), num_replicas=2, rank=rank, shuffle=False)]
        measured = _stats_for_ids(ids)
        save(directory / f'rank{rank}.json', measured)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason='Gloo unavailable')
def test_padded_distributed_validation_matches_single(tmp_path):
    expected = _stats_for_ids(range(1, 6))
    torch.multiprocessing.spawn(_distributed_eval, args=(str(tmp_path),), nprocs=2)
    for rank in (0, 1):
        assert json.loads((tmp_path / f'rank{rank}.json').read_text()) == pytest.approx(expected)


def test_report_guards_before_opening_test(tmp_path):
    with pytest.raises(ValueError, match='confirm-frozen'):
        evaluate_run({}, tmp_path, 'cpu', split='test')
    from thesis_kd.evaluate import freeze_selection
    frozen = {'sha256': 'a' * 64, 'epoch': 18}
    freeze_selection(tmp_path, frozen)
    freeze_selection(tmp_path, frozen)
    with pytest.raises(ValueError, match='previously frozen'):
        freeze_selection(tmp_path, dict(frozen, epoch=19))


def test_binding_rejects_tampered_student(tmp_path):
    cfg = resolve_config('m_base')
    keys = dict(experiment='m_base', seed=42, protocol_id=cfg['protocol_id'], run_tag='trial')
    save(tmp_path / 'run_info.json', dict(**keys, arch='m'))
    save(tmp_path / 'selection.json', dict(**keys, checkpoint='best_student.pth', epoch=42, sha256='0'*64,
         mixed_ap=.4, source='FP32 EMA Mixed Validation AP', tie_break='earliest_epoch'))
    (tmp_path / 'validation_history.csv').write_text('epoch,AP\n42,0.4\n')
    (tmp_path / 'best_student.pth').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        selected_weights(cfg, tmp_path)


def test_native_s_validation_inference(tmp_path, monkeypatch):
    # One labelled RGB fixture; the S detector, transforms and evaluators are native.
    root = tmp_path / 'rgb'
    (root / 'images/val').mkdir(parents=True)
    Image.new('RGB', (40, 32)).save(root / 'images/val/one.png')
    data = dict(images=[dict(id=1, width=40, height=32, file_name='one.png', illumination='natural')],
                categories=[dict(id=i, name=n) for i, n in enumerate(('immature', 'half', 'ripe'), 1)],
                annotations=[dict(id=1, image_id=1, category_id=3, bbox=[5, 5, 18, 10], area=180, iscrowd=0)])
    save(root / 'annotations/instances_val.json', data)
    monkeypatch.setenv('TOMATO_DATA_ROOT', str(root))
    cfg = resolve_config('s_kd')
    cfg['native']['val_dataloader'].update(num_workers=0, total_batch_size=1)
    cfg['resources']['THESIS_KD_TEACHER_CHECKPOINT'] = None
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        from thesis_kd.evaluate import validate_model
        model, _ = build_student(cfg, initialize=False)
        model.float().eval()
        ckpt = tmp_path / 'best_student.pth'
        torch.save(model.state_dict(), ckpt)
        fresh = load_deployed_student(cfg, ckpt, 'cpu')
        loader = build_val_loader(cfg, 42)
        sample, target = next(iter(loader))
        assert target[0]['labels'].tolist() == [2]
        with torch.no_grad():
            output = model(sample)
            reloaded = fresh(sample)
            torch.testing.assert_close(output['pred_boxes'], reloaded['pred_boxes'], rtol=0, atol=0)
            torch.testing.assert_close(output['pred_logits'], reloaded['pred_logits'], rtol=0, atol=0)
        stats = validate_model(cfg, fresh, 'cpu', 42)
        assert len(stats) == 12 and 0 <= stats[0] <= 1
        keys = dict(experiment='s_kd', seed=42, protocol_id=cfg['protocol_id'], run_tag='synthetic')
        save(tmp_path / 'run_info.json', dict(**keys, arch='s'))
        save(tmp_path / 'selection.json', dict(**keys, checkpoint=ckpt.name, epoch=1, mixed_ap=stats[0],
             source='FP32 EMA Mixed Validation AP', tie_break='earliest_epoch',
             sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()))
        (tmp_path / 'validation_history.csv').write_text(f'epoch,AP\n1,{stats[0]}\n')
        record = evaluate_run(cfg, tmp_path, 'cpu', split='val')
        assert record['metrics']['Mixed AP (%)'] == pytest.approx(100 * stats[0])
        assert record['metrics']['Sodium-light AP (%)'] is None
    finally:
        torch.set_num_threads(threads)


def test_summary_three_seeds_pairing_no_smoke_speed(tmp_path):
    campaign = tmp_path / 'feature_kd_ms_v1' / 'tag'
    for model in CORE:
        for i, seed in enumerate(SEEDS):
            run = campaign / model / f'seed_{seed}'
            teacher = {'id': 'a'*64 + '/ema.module'} if model.endswith('_kd') else None
            keys = dict(seed=seed, experiment=model, run_tag='tag', protocol_id='feature_kd_ms_v1', teacher=teacher)
            sel = dict(**keys, checkpoint='best_student.pth', epoch=3, mixed_ap=.4,
                       sha256=hashlib.sha256(model.encode()).hexdigest())
            save(run / 'selection.json', sel)
            save(run / 'run_info.json', dict(**keys, arch=model[0]))
            save(run / 'evaluation_val/metrics.json', dict(selection=sel, split='val', frozen=False,
                 annotation_sha256='b'*64, status='COMPLETE',
                 metrics={k: 40+i+(3 if teacher else 0) for k in AP_COLUMNS}))
            save(run / 'benchmark_smoke.json', {'formal': False, 'latency_ms': .01})
            save(run / 'kd_diagnostics.json', [dict(epoch=1, optimizer_updates=2, train_seconds=5, peak_memory_bytes=4)])
    summary = summarize_runs(campaign, split='val')
    dest = Path(summary['directory'])
    def read(filename):
        with (dest / filename).open(encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    assert len(read('per_seed.csv')) == 12
    paper = {r['Model']: r for r in read('paper.csv')}
    assert paper['T0']['Status'] == 'NOT RUN' and paper['T0']['Mixed AP (%)'] == ''
    assert paper['m_base']['Mixed AP (%)'] == '41.0000 ± 1.0000'
    assert paper['m_base']['Latency(ms)'] == ''
    assert {float(r['DeltaAP_pp']) for r in read('paired_ap.csv')} == {3.0}
    assert all(r['TrainSeconds'] == '5' for r in read('training_cost.csv'))


@pytest.mark.parametrize("arch", ["m", "s"])
def test_standalone_native_process(tmp_path, arch):
    """Use official config in a fresh Python process with no import of thesis_kd."""
    cfg = resolve_config(arch + '_kd')
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model, _ = build_student(cfg, initialize=False)
        model.float().eval()
        image = torch.zeros(1, 3, 640, 640)
        with torch.inference_mode():
            prediction = model(image)
        weights = tmp_path / 'deploy.pth'
        reference = tmp_path / 'reference.pth'
        torch.save(model.state_dict(), weights)
        torch.save({key: prediction[key] for key in ('pred_boxes', 'pred_logits')}, reference)
    finally:
        torch.set_num_threads(threads)
    child = """
import sys
import torch
from engine.core import YAMLConfig
arch, weights, reference = sys.argv[1:]
torch.set_num_threads(2)
cfg = YAMLConfig('configs/deimv2/deimv2_dinov3_' + arch + '_coco.yml',
                 num_classes=3, remap_mscoco_category=False, DINOv3STAs={'weights_path': None},
                 PostProcessor={'category_ids': [1, 2, 3]})
model = cfg.model.float().eval()
model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
with torch.inference_mode():
    predicted = model(torch.zeros(1, 3, 640, 640))
for key, tensor in torch.load(reference, map_location='cpu', weights_only=True).items():
    torch.testing.assert_close(tensor, predicted[key], rtol=0, atol=0)
assert not any(m.startswith('thesis_kd') for m in sys.modules)
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith('THESIS_KD_')}
    try:
        subprocess.run([sys.executable, '-c', child, arch, str(weights), str(reference)],
                       check=True, env=env, timeout=240)
    finally:
        weights.unlink(missing_ok=True)
        reference.unlink(missing_ok=True)


@pytest.mark.skipif(not os.environ.get('TOMATO_DATA_ROOT'), reason='Real RGB Train/Validation root not supplied')
def test_real_train_val_sample():
    cfg = resolve_config('m_base')
    for name, factory in [('train', build_train_loader), ('val', build_val_loader)]:
        cfg['native'][name + '_dataloader'].update(num_workers=0, total_batch_size=2)
        loader = factory(cfg, 42)
        loader.set_epoch(0)
        images, targets = next(iter(loader))
        assert images.shape[:2] == (2, 3) and images.dtype == torch.float32
        assert loader.dataset.category2label == {1: 0, 2: 1, 3: 2}
        assert all('masks' not in t and set(t['labels'].tolist()) <= {0, 1, 2} for t in targets)
