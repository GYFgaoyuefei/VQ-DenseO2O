"""Formal efficiency binds to the right completed run, without data/GPU access."""
import json
from types import SimpleNamespace

import pytest
import torch

from tools.paper_v2 import run
from tools.paper_v2.efficiency_protocol import MODELS, PAPER_COLUMNS, protocol, validate_protocol


def saved_run(tmp_path, model='vq_dense_o2o', seed=42):
    args = run.parser().parse_args(['benchmark', '--model', model, '--seed', str(seed),
                                   '--output-dir', str(tmp_path / 'run'),
                                   '--efficiency-dir', str(tmp_path / 'efficiency')])
    directory = run.output_path(args)
    directory.mkdir()
    cfg = run.load_resolved(model, expand_env=False)
    cfg['seed'] = seed
    run.write_yaml(directory / 'config_resolved.yml', cfg)
    (directory / 'run_info.json').write_text(json.dumps({'experiment': model, 'seed': seed}))
    (directory / 'completed.json').write_text('{}')
    (directory / 'selection.json').write_text(json.dumps({'model': 'ema', 'split': 'mixed_validation', 'epoch': 13,
                                                        'precision': 'FP32', 'autocast': False, 'tf32': False}))
    (directory / 'best.pth').touch()
    return args, directory, cfg


@pytest.mark.parametrize('model', MODELS)
def test_all_nine_models_bind_without_training_resources(tmp_path, model, monkeypatch):
    for key in ('TOMATO_DATA_ROOT', 'DINOV3_WEIGHTS', 'TOMATO_DETECTOR_CHECKPOINT'):
        monkeypatch.delenv(key, raising=False)
    args, directory, _ = saved_run(tmp_path, model)
    cfg, path, selection = run.benchmark_binding(args)
    assert cfg['epoches'] == 120 and selection['epoch'] == 13
    assert path == directory / 'best.pth'


@pytest.mark.parametrize('fault,match', [('seed', 'metadata'), ('model', 'metadata'),
    ('unfinished', 'completed'), ('switches', 'algorithm switches'), ('size', '640'),
    ('topk', 'top-k'), ('selection', 'Validation'), ('checkpoint', 'best.pth')])
def test_binding_rejects_cross_run_or_nonstandard_inputs(tmp_path, fault, match):
    args, directory, cfg = saved_run(tmp_path)
    if fault == 'seed':
        args.seed = 0
    elif fault == 'model':
        args.model = 'baseline_native'
    elif fault == 'unfinished':
        (directory / 'completed.json').unlink()
    elif fault == 'switches':
        cfg['innovation']['vq_dense_o2o']['enabled'] = False
    elif fault == 'size':
        cfg['eval_spatial_size'] = [128, 128]
    elif fault == 'topk':
        cfg['PostProcessor']['num_top_queries'] = 10
    elif fault == 'selection':
        (directory / 'selection.json').write_text(json.dumps({'model': 'ema', 'split': 'test'}))
    elif fault == 'checkpoint':
        args.checkpoint = str(directory / 'last.pth')
    run.write_yaml(directory / 'config_resolved.yml', cfg)
    with pytest.raises(ValueError, match=match):
        run.benchmark_binding(args)


@pytest.mark.parametrize('field,value', [('precision', 'FP16'), ('autocast', True), ('tf32', True),
                                        ('precision', None)])
def test_binding_requires_explicit_fp32_validation_selection(tmp_path, field, value):
    args, directory, _ = saved_run(tmp_path)
    path = directory / 'selection.json'
    selection = json.loads(path.read_text())
    if value is None:
        selection.pop(field)
    else:
        selection[field] = value
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='FP32'):
        run.benchmark_binding(args)


@pytest.mark.parametrize('iterations', ['300', '500'])
def test_formal_iteration_count_is_not_a_cli_override(iterations):
    with pytest.raises(SystemExit):
        run.parser().parse_args(['benchmark', '--model', 'dc_sta', '--iterations', iterations])


def test_protocol_snapshot_is_immutable_and_original_paper_prefix_preserved():
    snapshot = protocol()
    snapshot['timed_iterations'] = 300
    assert protocol()['timed_iterations'] == 500
    with pytest.raises(ValueError, match='timed_iterations'):
        validate_protocol(snapshot)
    assert PAPER_COLUMNS[:7] == ('Model', 'Epoch', 'Params(M)', 'GFLOPs', 'Latency(ms)', 'FPS', 'Hardware')
    assert PAPER_COLUMNS[7] == 'Mixed AP (%)'
    assert PAPER_COLUMNS[-4:] == ('Natural AP (%)', 'Artificial AP (%)', 'Low-light AP (%)', 'Sodium-light AP (%)')


def test_benchmark_constructs_only_inference_components_and_prevents_overwrite(tmp_path, monkeypatch):
    from tools.paper_v2 import efficiency_measure
    args, directory, _ = saved_run(tmp_path)
    calls = []
    class Detector:
        def load_state_dict(self, weights, strict):
            assert weights == {'sentinel': 123} and strict is True
            calls.append('load_ema')
    model, processor = Detector(), object()
    def build(cfg):
        assert cfg['DINOv3STAs']['weights_path'] is None
        calls.append('construct_inference')
        # Accessing any training property fails: this object has only two fields.
        return SimpleNamespace(model=model, postprocessor=processor)
    def measure(detector, postprocessor, device):
        assert detector is model and postprocessor is processor and device == 'cuda:0'
        calls.append('measure')
        return {'full_params_m': 1., 'deploy_params_m': .8}
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'last_epoch': 13, 'ema': {'module': {'sentinel': 123}}})
    monkeypatch.setattr(run, 'create_config', build)
    monkeypatch.setattr(run, 'git', lambda *a: 'fake_commit' if a[0] == 'rev-parse' else '')
    monkeypatch.setattr(efficiency_measure, 'measure', measure)
    monkeypatch.setattr(run, 'summarize_efficiency', lambda args: calls.append('summarize'))
    run.benchmark(args)
    assert calls == ['construct_inference', 'load_ema', 'measure', 'summarize']
    path = run.efficiency_path(args) / 'raw/vq_dense_o2o/seed_42/efficiency.json'
    record = json.loads(path.read_text())
    assert record['protocol'] == protocol() and record['epochs'] == 120
    assert record['selected_epoch_zero_based'] == 13 and record['checkpoint']['model_state'] == 'ema'
    with pytest.raises(FileExistsError):
        run.benchmark(args)
    assert calls == ['construct_inference', 'load_ema', 'measure', 'summarize']


def test_benchmark_rejects_mismatched_checkpoint_epoch_before_constructing(tmp_path, monkeypatch):
    args, _, _ = saved_run(tmp_path)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'last_epoch': 14, 'ema': {'module': {}}})
    monkeypatch.setattr(run, 'create_config', lambda cfg: pytest.fail('must reject before model construction'))
    with pytest.raises(ValueError, match='recorded selected EMA epoch'):
        run.benchmark(args)
    assert not run.efficiency_path(args).exists()


def test_changed_checkpoint_during_measurement_is_not_saved(tmp_path, monkeypatch):
    from tools.paper_v2 import efficiency_measure
    args, directory, _ = saved_run(tmp_path)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'last_epoch': 13, 'ema': {'module': {}}})
    model = SimpleNamespace(load_state_dict=lambda *a, **kw: None)
    monkeypatch.setattr(run, 'create_config', lambda cfg: SimpleNamespace(model=model, postprocessor=object()))
    def changed(*args):
        (directory / 'best.pth').write_bytes(b'replaced checkpoint')
        return {}
    monkeypatch.setattr(efficiency_measure, 'measure', changed)
    with pytest.raises(ValueError, match='checkpoint changed'):
        run.benchmark(args)
    assert not run.efficiency_path(args).exists()


def test_benchmark_rejects_historical_destination_before_inference(tmp_path, monkeypatch):
    args, _, _ = saved_run(tmp_path)
    destination = run.efficiency_path(args)
    destination.mkdir()
    report = destination / 'efficiency_protocol.json'
    report.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(run, 'benchmark_binding', lambda _: pytest.fail('must reject before inference'))
    with pytest.raises(ValueError, match='Historical efficiency campaign'):
        run.benchmark(args)
    assert report.read_text() == '{}'
    assert not (destination / 'raw').exists()
