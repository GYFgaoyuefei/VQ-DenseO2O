"""Synthetic selected-checkpoint inference; never opens the real Test split."""
import copy
import json
from types import SimpleNamespace

from faster_coco_eval import COCO
import pytest
import torch
import yaml

from engine.deim.postprocessor import PostProcessor
from tools.paper_v2 import ap_evaluate, run
from tools.paper_v2.efficiency_protocol import AP_COLUMNS


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-8.))
        self.calls = []

    def forward(self, samples):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled('cpu')
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert samples.dtype == torch.float32
        self.calls.append((len(samples), self.weight.item()))
        logits = samples.new_full((len(samples), 100, 3), -10.)
        logits[:, :, 0] = self.weight - 4.
        logits[:, 0, 0] = self.weight
        boxes = samples.new_full((len(samples), 100, 4), .5)
        return {'pred_logits': logits, 'pred_boxes': boxes}


@pytest.fixture
def selected_run(tmp_path, monkeypatch):
    root = tmp_path / 'synthetic_dataset'
    (root / 'images/test').mkdir(parents=True)
    (root / 'annotations').mkdir()
    images, annotations = [], []
    for index, light in enumerate(('自然光', '人工光', '微弱光', '钠黄光'), 1):
        filename = f'{index}.jpg'
        (root / 'images/test' / filename).touch()
        images.append(dict(id=index, file_name=filename, width=20, height=10, light_condition=light))
        annotations.append(dict(id=index, image_id=index, category_id=1,
                                bbox=[5., 2.5, 10., 5.], area=50., iscrowd=0))
    data = dict(images=images, annotations=annotations,
                categories=[dict(id=index, name=str(index)) for index in (1, 2, 3)])
    annotation_file = root / 'annotations/instances_test.json'
    annotation_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    directory = tmp_path / 'run'
    directory.mkdir()
    cfg = run.load_resolved('baseline_native', expand_env=False)
    cfg.update(seed=42)
    (directory / 'config_resolved.yml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    info = dict(experiment='baseline_native', seed=42)
    (directory / 'run_info.json').write_text(json.dumps(info), encoding='utf-8')
    (directory / 'completed.json').write_text('{}', encoding='utf-8')
    selection = dict(model='ema', split='mixed_validation', precision='FP32',
                     autocast=False, tf32=False, epoch=87, ap=.65)
    (directory / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    checkpoint = {'last_epoch': 87, 'ema': {'module': {'weight': torch.tensor(2.)}},
                  'model': {'weight': torch.tensor(9.)}}
    torch.save(checkpoint, directory / 'best.pth')
    coco = COCO()
    coco.dataset = data
    coco.createIndex()

    class Loader:
        dataset = SimpleNamespace(coco=coco)
        ids = [1, 2, 3, 4]

        def __iter__(self):
            yield (torch.zeros(len(self.ids), 3, 640, 640),
                   [{'image_id': torch.tensor([index]), 'orig_size': torch.tensor([20, 10])}
                    for index in self.ids])

    detector, loader, seen_cfg = TinyDetector(), Loader(), []

    def create_config(config):
        seen_cfg.append(copy.deepcopy(config))
        assert config['DINOv3STAs']['weights_path'] is None
        return SimpleNamespace(model=detector,
                               postprocessor=PostProcessor(num_classes=3, num_top_queries=300,
                                                           category_ids=[1, 2, 3]),
                               val_dataloader=loader)

    monkeypatch.setattr(ap_evaluate, '_create_config', create_config)
    return SimpleNamespace(root=root, directory=directory, cfg=cfg, selection=selection,
                           checkpoint=checkpoint, annotation_file=annotation_file,
                           detector=detector, loader=loader, seen_cfg=seen_cfg,
                           output=tmp_path / 'ap',
                           args=dict(run_dir=directory, model='baseline_native', seed=42,
                                     output_dir=tmp_path / 'ap', data_root=root, device='cpu'))


def test_selected_ema_infers_once_and_reuses_predictions_for_all_ap(selected_run):
    item = selected_run
    with torch.autocast('cpu', dtype=torch.bfloat16):
        record = ap_evaluate.evaluate_run(**item.args)
    assert item.detector.calls == [(4, 2.)]
    assert record['status'] == 'COMPLETE'
    for key in (AP_COLUMNS[0], *AP_COLUMNS[-4:]):
        assert record['metrics'][key] == pytest.approx(100.)
    assert record['image_counts'] == dict(mixed=4, natural=1, artificial=1, low_light=1, sodium_light=1)
    assert record['selected_epoch_zero_based'] == 87
    assert record['checkpoint']['model_state'] == 'ema'
    assert record['evaluation'] == ap_evaluate.EVALUATION
    path = ap_evaluate.record_path(item.output, 'baseline_native', 42)
    predictions = json.loads((path.parent / 'predictions.json').read_text(encoding='utf-8'))
    assert len(predictions) == 1200  # Preserve native top-k, no confidence filtering.
    assert predictions[0]['bbox'] == [5., 2.5, 10., 5.]
    assert {row['category_id'] for row in predictions} == {1, 2, 3}
    assert ap_evaluate.reusable_record(**item.args) == record
    assert item.detector.calls == [(4, 2.)]
    with pytest.raises(FileExistsError):
        ap_evaluate.evaluate_run(**item.args)


def test_direct_api_rejects_historical_campaign_before_opening_resources(selected_run, monkeypatch):
    item = selected_run
    item.output.mkdir()
    marker = item.output / 'ap_protocol.json'
    marker.write_text(json.dumps({'schema': 'paper_v2_ap_summary_v1', 'split': 'test'}))
    csv = item.output / 'ap_summary.csv'
    csv.write_text('old measured summary')
    before = {path.name: path.read_bytes() for path in (marker, csv)}
    monkeypatch.setattr(ap_evaluate, '_binding', lambda *a, **kw: pytest.fail('Do not open data or checkpoints'))
    with pytest.raises(ValueError, match='Historical or unbound AP campaign'):
        ap_evaluate.evaluate_run(**item.args)
    assert {path.name: path.read_bytes() for path in item.output.iterdir()} == before
    assert item.detector.calls == item.seen_cfg == []


def test_signed_native_boxes_are_saved_and_diagnosed_without_filtering(selected_run, monkeypatch):
    item = selected_run
    original = item.detector.forward

    def signed_boxes(samples):
        result = original(samples)
        result['pred_boxes'][:, -1, 2] = -.25
        return result

    monkeypatch.setattr(item.detector, 'forward', signed_boxes)
    record = ap_evaluate.evaluate_run(**item.args)
    assert record['status'] == 'COMPLETE'
    assert record['prediction_diagnostics']['negative_size_count'] == 12  # 4 images x 3 classes.
    assert record['prediction_diagnostics']['negative_height_count'] == 0
    path = ap_evaluate.record_path(item.output, 'baseline_native', 42)
    predictions = json.loads((path.parent / 'predictions.json').read_text(encoding='utf-8'))
    assert len(predictions) == 1200
    assert sum(prediction['bbox'][2] == -5. for prediction in predictions) == 12
    assert ap_evaluate.reusable_record(**item.args) == record


@pytest.mark.parametrize('change', ['model', 'seed', 'recipe', 'switch', 'precision', 'epoch'])
def test_rejects_unbound_run_before_model_construction(selected_run, change):
    item = selected_run
    if change in ('recipe', 'switch'):
        if change == 'recipe':
            item.cfg['epoches'] = 10
        else:
            item.cfg['innovation']['vq_dense_o2o']['enabled'] = True
        (item.directory / 'config_resolved.yml').write_text(yaml.safe_dump(item.cfg), encoding='utf-8')
    elif change in ('precision', 'epoch'):
        item.selection.update({'precision': 'FP16'} if change == 'precision' else {'epoch': 120})
        (item.directory / 'selection.json').write_text(json.dumps(item.selection), encoding='utf-8')
    else:
        item.args[change] = 'vq_dense_o2o' if change == 'model' else 0
    with pytest.raises(ValueError):
        ap_evaluate.evaluate_run(**item.args)
    assert item.seen_cfg == []
    assert not item.output.exists()


@pytest.mark.parametrize('change', ['last_epoch', 'half_ema', 'missing_ema'])
def test_checkpoint_must_be_selected_fp32_ema(selected_run, change):
    item = selected_run
    if change == 'last_epoch':
        item.checkpoint['last_epoch'] = 86
    elif change == 'half_ema':
        item.checkpoint['ema']['module']['weight'] = torch.tensor(2., dtype=torch.float16)
    else:
        item.checkpoint.pop('ema')
    torch.save(item.checkpoint, item.directory / 'best.pth')
    with pytest.raises(ValueError):
        ap_evaluate.evaluate_run(**item.args)
    assert item.seen_cfg == []


@pytest.mark.parametrize('ids', [[1, 2, 3], [1, 2, 2, 4], [1, 2, 3, 9]])
def test_duplicate_missing_or_unknown_images_do_not_publish(selected_run, ids):
    item = selected_run
    item.loader.ids = ids
    with pytest.raises(ValueError, match='image'):
        ap_evaluate.evaluate_run(**item.args)
    assert not item.output.exists()


@pytest.mark.parametrize('change', ['checkpoint', 'annotation', 'map'])
def test_cached_ap_rejects_changed_source_provenance(selected_run, change):
    item = selected_run
    ap_evaluate.evaluate_run(**item.args)
    if change == 'checkpoint':
        item.checkpoint['extra'] = 'new checkpoint'
        torch.save(item.checkpoint, item.directory / 'best.pth')
    elif change == 'annotation':
        with item.annotation_file.open('a', encoding='utf-8') as stream:
            stream.write(' ')
    else:
        item.args['illumination_map'] = {'1': 'natural'}
    with pytest.raises(ValueError, match='provenance'):
        ap_evaluate.reusable_record(**item.args)


def test_missing_resources_and_distributed_fail_without_inference(selected_run, monkeypatch):
    item = selected_run
    monkeypatch.delenv('TOMATO_DATA_ROOT', raising=False)
    item.args['data_root'] = None
    with pytest.raises(FileNotFoundError, match='data-root'):
        ap_evaluate.evaluate_run(**item.args)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='single-process'):
        ap_evaluate.evaluate_run(**item.args)
    assert item.seen_cfg == []


def test_mutation_during_inference_does_not_publish(selected_run, monkeypatch):
    item = selected_run
    original = ap_evaluate._infer

    def mutate(*args):
        result = original(*args)
        with item.annotation_file.open('a', encoding='utf-8') as stream:
            stream.write(' ')
        return result

    monkeypatch.setattr(ap_evaluate, '_infer', mutate)
    with pytest.raises(ValueError, match='changed during'):
        ap_evaluate.evaluate_run(**item.args)
    assert not item.output.exists()


def test_validation_uses_saved_validation_paths_without_test_root(selected_run, monkeypatch):
    item = selected_run
    # Synthetic names retained to avoid duplicating images; source is explicitly
    # bound to saved validation configuration, not inferred from Test environment.
    item.cfg['val_dataloader']['dataset'].update(
        img_folder=str(item.root / 'images/test'), ann_file=str(item.annotation_file))
    (item.directory / 'config_resolved.yml').write_text(yaml.safe_dump(item.cfg), encoding='utf-8')
    monkeypatch.delenv('TOMATO_DATA_ROOT', raising=False)
    item.args.update(split='validation', data_root=None)
    assert ap_evaluate.evaluate_run(**item.args)['split'] == 'validation'


def test_mapping_digest_normalizes_json_keys_but_rejects_collisions():
    assert ap_evaluate._mapping_digest({1: 'natural'}) == ap_evaluate._mapping_digest({'1': 'natural'})
    with pytest.raises(ValueError, match='duplicate'):
        ap_evaluate._mapping_digest({1: 'natural', '1': 'artificial'})
