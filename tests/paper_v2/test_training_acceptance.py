"""Focused CPU contracts for bounded native Train and FP32 selection integration."""
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import yaml

from engine.data.li_copyblend import STAT_NAMES
from engine.optim.ema import ModelEMA
from engine.solver import det_engine
from tools.paper_v2 import run
from tools.paper_v2 import train_smoke
from tools.paper_v2.train_smoke import BoundedLoader, TrainStepObserver


@pytest.mark.parametrize('fail_inside', [False, True])
def test_fp32_selection_disables_outer_cpu_autocast_and_restores_backend_flags(fail_inside):
    model = torch.nn.Linear(4, 2).float()
    sample = torch.ones(3, 4)
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                  torch.get_float32_matmul_precision())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            assert model(sample).dtype == torch.bfloat16
            try:
                with det_engine.fp32_evaluation(model, 'cpu', enabled=True):
                    assert model(sample).dtype == torch.float32
                    assert not torch.backends.cuda.matmul.allow_tf32
                    assert not torch.backends.cudnn.allow_tf32
                    assert torch.get_float32_matmul_precision() == 'highest'
                    if fail_inside:
                        raise RuntimeError('evaluation failed')
            except RuntimeError as error:
                assert fail_inside and str(error) == 'evaluation failed'
            assert model(sample).dtype == torch.bfloat16
        after = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.get_float32_matmul_precision())
        assert after == before
    finally:
        torch.set_float32_matmul_precision(original[2])
        torch.backends.cuda.matmul.allow_tf32 = original[0]
        torch.backends.cudnn.allow_tf32 = original[1]


def test_fp32_selection_rejects_half_precision_ema_before_evaluation():
    ema = ModelEMA(torch.nn.Linear(4, 2)).to(dtype=torch.float16)
    with pytest.raises(ValueError, match='FP32 EMA'):
        with det_engine.fp32_evaluation(ema.module, 'cpu', enabled=True):
            pytest.fail('half EMA was allowed into evaluation')


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))

    def forward(self, samples, targets=None):
        result = samples.mean((1, 2, 3))[:, None, None] * self.weight
        return {'pred_boxes': result.expand(-1, 1, 4)}


class TinyCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metas = []

    def forward(self, outputs, targets, **metas):
        self.metas.append(metas)
        return {'loss_boxes': outputs['pred_boxes'].square().mean()}


def stats_tensor(*, attempted, accepted, images, light, collision, invalid):
    # Derived fields are intentionally wrong: integration must use additive
    # counters and rederive ratios after accumulation across unequal batches.
    values = dict(attempted_pastes=attempted, accepted_pastes=accepted,
                  skipped_incompatible_light=light, skipped_collision=collision,
                  skipped_invalid_mask=invalid, image_count=images,
                  li_enabled_image_count=images, foreground_area_sum=.2 * images,
                  acceptance_ratio=111., added_instances_per_image=222.,
                  foreground_area_ratio=333.)
    return torch.tensor([values.get(name, 0.) for name in STAT_NAMES])


def test_native_bounded_train_records_unscaled_preclip_gradients_and_sums_counters(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('bounded Train must never call evaluation')

    monkeypatch.setattr(det_engine, 'evaluate', forbidden)
    monkeypatch.setattr(det_engine, '_evaluate', forbidden)
    model, criterion = TinyDetector(), TinyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    ema = ModelEMA(model, decay=.9, warmups=0)
    scaler = torch.amp.GradScaler('cpu', init_scale=32.)
    observer = TrainStepObserver('cpu')
    counters = [stats_tensor(attempted=10, accepted=2, images=2, light=7, collision=3, invalid=4),
                stats_tensor(attempted=2, accepted=2, images=2, light=1, collision=2, invalid=0)]

    class Loader:
        consumed = 0

        def __len__(self):
            return 5

        def __iter__(self):
            for index in range(len(self)):
                self.consumed += 1
                targets = [{'labels': torch.tensor([0]), 'boxes': torch.ones(1, 4)} for _ in range(2)]
                targets[0]['li_copyblend_stats'] = counters[index % 2]
                yield torch.full((2, 3, 2, 2), float(index + 1)), targets

    class Scheduler:
        def __init__(self):
            self.steps = []

        def step(self, index, opt):
            self.steps.append(index)
            return opt

    loader, scheduler = Loader(), Scheduler()
    bounded = BoundedLoader(loader, 2)
    assert len(bounded) == 5
    metrics = det_engine.train_one_epoch(True, scheduler, model, criterion, bounded,
                                         optimizer, torch.device('cpu'), 3, max_norm=.1,
                                         print_freq=1, scaler=scaler, ema=ema, step_observer=observer)
    assert loader.consumed == 2 and ema.updates == 2
    assert scheduler.steps == [15, 16]
    assert [meta['global_step'] for meta in criterion.metas] == [15, 16]
    assert all(meta['epoch_step'] == 5 for meta in criterion.metas)
    assert len(observer.rows) == 2
    for row in observer.rows:
        assert row['status'] == 'PASS' and row['gradients_finite']
        assert row['gradient_tensors'] == 1 and row['gradient_l2_before_clip'] > 0
        assert row['iteration_ms'] > 0 and row['loss'] > 0
        assert row['input_shape'] == [2, 3, 2, 2]
        assert row['gt_instances_per_image'] == [1, 1]
        assert row['peak_gpu_memory_bytes'] is None
    # d(weight**2)/dweight=4, before clip (.1) and after unscale (32).
    assert observer.rows[0]['gradient_l2_before_clip'] == pytest.approx(4.)
    assert model.weight.item() == pytest.approx(1.98, abs=1e-6)
    assert metrics['li_attempted_pastes'] == 12
    assert metrics['li_accepted_pastes'] == 4
    assert metrics['li_acceptance_ratio'] == pytest.approx(4 / 12)
    assert metrics['li_added_instances_per_image'] == 1
    assert metrics['li_skipped_by_light'] == 8
    assert metrics['li_skipped_by_collision'] == 5
    assert metrics['li_skipped_by_invalid_mask'] == 4
    assert metrics['copyblend_foreground_area_ratio'] == pytest.approx(.2)
    assert metrics['li_batches'] == 2


def test_smoke_train_missing_assets_writes_not_run_without_launch(tmp_path, monkeypatch):
    for key, path in (('TOMATO_DATA_ROOT', tmp_path / 'absent_train'),
                      ('DINOV3_WEIGHTS', tmp_path / 'absent_dino.pth'),
                      ('TOMATO_DETECTOR_CHECKPOINT', tmp_path / 'absent_detector.pth')):
        monkeypatch.setenv(key, str(path))
    monkeypatch.setattr(run, 'git', lambda *args: 'paper_v3.0')
    launched = []
    monkeypatch.setattr(run, 'stream_process', lambda *args, **kwargs: launched.append(args))
    destination = tmp_path / 'smoke'
    args = run.parser().parse_args(['smoke-train', '--model', 'vq_dense_o2o', '--seed', '42',
                                   '--device', 'cuda:0', '--output-dir', str(destination)])
    with pytest.raises(FileNotFoundError, match='Required paths are missing'):
        run.smoke_train(args)
    report = json.loads((destination / 'train_smoke_preflight.json').read_text(encoding='utf-8'))
    assert report['status'] == 'NOT RUN' and report['real_train_batches'] == 0
    assert report['test'] == 'NOT RUN' and not launched
    assert not (destination / 'config_resolved.yml').exists()


def test_smoke_train_launch_keeps_real_batch_recipe_and_passes_native_bound(tmp_path, monkeypatch):
    cfg = run.load_resolved('li_copyblend', expand_env=False)
    monkeypatch.setattr(run, 'git', lambda *args: 'paper_v3.0')
    monkeypatch.setattr(run, 'load_resolved', lambda *args: cfg)
    checked = []
    monkeypatch.setattr(run, 'check_assets', lambda value, **kwargs: checked.append(kwargs))
    monkeypatch.setenv('TOMATO_DETECTOR_CHECKPOINT', 'shared_pretrained.pth')
    launched = []
    monkeypatch.setattr(run, 'stream_process', lambda *args, **kwargs: launched.append(args))
    destination = tmp_path / 'bounded'
    args = run.parser().parse_args(['smoke-train', '--model', 'li_copyblend', '--seed', '3407',
                                   '--smoke-steps', '2', '--smoke-epoch', '10', '--nproc-per-node', '2',
                                   '--device', 'cuda:0', '--output-dir', str(destination)])
    run.smoke_train(args)
    assert checked == [{'train_only': True}]
    assert len(launched) == 1
    command = launched[0][0]
    assert command[command.index('--train-smoke-steps') + 1] == '2'
    assert command[command.index('--train-smoke-epoch') + 1] == '10'
    assert command[command.index('--seed') + 1] == '3407'
    assert command[command.index('--tuning') + 1] == 'shared_pretrained.pth'
    assert 'torch.distributed.run' in command and '--test-only' not in command
    saved = yaml.safe_load((destination / 'config_resolved.yml').read_text(encoding='utf-8'))
    assert saved['train_dataloader']['total_batch_size'] == 32
    assert saved['epoches'] == 120 and saved['use_amp'] and saved['use_ema']
    assert saved['flat_epoch'] == 60 and saved['no_aug_epoch'] == 14


def test_native_train_zero_smoke_steps_hits_bounded_guard_and_never_fits(monkeypatch):
    import train as native_train

    cfg = SimpleNamespace(yaml_cfg={'task': 'acceptance_fake'})
    solver = SimpleNamespace(cfg=cfg, fit=mock.Mock(), val=mock.Mock())
    monkeypatch.setattr(native_train.dist_utils, 'setup_distributed', lambda *args, **kwargs: None)
    monkeypatch.setattr(native_train, 'YAMLConfig', lambda *args, **kwargs: cfg)
    monkeypatch.setattr(native_train.yaml_utils, 'parse_cli', lambda value: {})
    monkeypatch.setitem(native_train.TASKS, 'acceptance_fake', lambda value: solver)
    args = SimpleNamespace(print_rank=0, print_method='builtin', seed=42,
                           tuning='pretrained.pth', resume=None, update=None,
                           config='unused.yml', test_only=False,
                           train_smoke_steps=0, train_smoke_epoch=0)
    with pytest.raises(ValueError, match='1-10 steps'):
        native_train.main(args)
    solver.fit.assert_not_called()
    solver.val.assert_not_called()


@pytest.mark.parametrize('resume,tuning', [('last.pth', None), (None, 'pretrained.pth'), (None, None)])
def test_native_paper_resume_does_not_reopen_backbone_initialization(monkeypatch, resume, tuning):
    import train as native_train

    cfg = SimpleNamespace(yaml_cfg={'task': 'acceptance_fake', 'paper_v2': True,
                                   'DINOv3STAs': {'weights_path': 'unmounted/dino.pth'}})
    solver = SimpleNamespace(fit=mock.Mock(), val=mock.Mock())
    monkeypatch.setattr(native_train.dist_utils, 'setup_distributed', lambda *a, **kw: None)
    monkeypatch.setattr(native_train.dist_utils, 'cleanup', lambda: None)
    monkeypatch.setattr(native_train, 'YAMLConfig', lambda *a, **kw: cfg)
    monkeypatch.setattr(native_train.yaml_utils, 'parse_cli', lambda value: {})

    def construct(value):
        expected = None if resume else 'unmounted/dino.pth'
        assert value.yaml_cfg['DINOv3STAs']['weights_path'] == expected
        return solver

    monkeypatch.setitem(native_train.TASKS, 'acceptance_fake', construct)
    args = SimpleNamespace(print_rank=0, print_method='builtin', seed=42,
                           tuning=tuning, resume=resume, update=None,
                           config='unused.yml', test_only=False, train_smoke_steps=None)
    native_train.main(args)
    solver.fit.assert_called_once()


@pytest.mark.parametrize('gate_bias,resume', [(None, None), (.25, None), (.25, 'last.pth')])
def test_native_training_requires_original_sigmoid_initialization(monkeypatch, gate_bias, resume):
    from engine.backbone.dc_sta import DCSTA
    from engine.solver._solver import BaseSolver

    model = DCSTA([4, 8, 8])
    if gate_bias is not None:
        with torch.no_grad():
            model.spatial[0].bias.fill_(gate_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loader = SimpleNamespace(shuffle=True)
    cfg = SimpleNamespace(yaml_cfg={'paper_v2': True}, resume=resume, optimizer=optimizer,
                           lr_scheduler=None, lr_warmup_scheduler=None,
                           train_dataloader=loader, val_dataloader=loader, evaluator=None)
    solver = BaseSolver(cfg)
    solver.model, solver.ema = model, ModelEMA(model)
    monkeypatch.setattr(solver, '_setup', lambda: None)
    monkeypatch.setattr(solver, 'load_resume_state', lambda path: None)
    monkeypatch.setattr(det_engine.dist_utils, 'warp_loader', lambda value, **kw: value)
    if gate_bias and not resume:
        with pytest.raises(RuntimeError, match='sigmoid'):
            solver.train()
    else:
        solver.train()
        assert solver.dc_sta_training_audit['status'] == 'PASS'


def test_observed_amp_train_rejects_nonfinite_boxes_without_saving_checkpoint(monkeypatch):
    class InvalidDetector(TinyDetector):
        def forward(self, samples, targets=None):
            result = super().forward(samples, targets)
            result['pred_boxes'] = result['pred_boxes'] * float('nan')
            return result

    model, criterion = InvalidDetector(), TinyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    observer = TrainStepObserver('cpu')
    saved = mock.Mock(side_effect=AssertionError('Train smoke must not save NaN.pth'))
    monkeypatch.setattr(torch, 'save', saved)
    monkeypatch.setattr(det_engine.dist_utils, 'save_on_master', saved)
    batch = (torch.ones(1, 3, 2, 2), [{'labels': torch.tensor([0])}])
    with pytest.raises(RuntimeError, match='nonfinite boxes; no checkpoint saved'):
        det_engine.train_one_epoch(False, None, model, criterion, [batch], optimizer,
                                   torch.device('cpu'), 0,
                                   scaler=torch.amp.GradScaler('cpu', init_scale=32.),
                                   step_observer=observer)
    saved.assert_not_called()
    assert criterion.metas == []
    assert model.weight.grad is None


def test_bounded_train_constructs_native_warmup_before_flatcosine_without_validation(tmp_path, monkeypatch):
    events = []
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)

    class Loader:
        shuffle, batch_size = True, 32

        def __len__(self):
            return 4

        def __iter__(self):
            return iter(range(4))

        def set_epoch(self, epoch):
            events.append(('loader_epoch', epoch))

    loader = Loader()

    class Config:
        yaml_cfg = {'paper_v2': True, 'tomato': {'detector_checkpoint': 'shared_detector.pth'},
                    'DINOv3STAs': {'weights_path': 'shared_dino.pth'},
                    'train_dataloader': {'total_batch_size': 32}}
        seed, resume, tuning = 42, None, 'shared_detector.pth'
        use_amp, use_ema, epoches = True, True, 120
        output_dir = str(tmp_path)
        lr_gamma, warmup_iter, flat_epoch, no_aug_epoch, clip_max_norm = .5, 1000, 60, 14, .1

        @property
        def optimizer(self):
            events.append('optimizer')
            return optimizer

        @property
        def lr_scheduler(self):
            events.append('lr_scheduler')
            optimizer.param_groups[0]['initial_lr'] = .1
            return object()

        @property
        def lr_warmup_scheduler(self):
            events.append('lr_warmup_scheduler')
            assert optimizer.param_groups[0]['initial_lr'] == .1
            optimizer.param_groups[0]['lr'] = .0001
            return object()

        @property
        def train_dataloader(self):
            return loader

        @property
        def val_dataloader(self):
            pytest.fail('Train smoke accessed validation data')

        @property
        def evaluator(self):
            pytest.fail('Train smoke constructed the COCO evaluator')

    solver = SimpleNamespace(cfg=Config(), model=model, criterion=TinyCriterion(),
                             ema=ModelEMA(model), scaler=object(), device=torch.device('cuda:0'),
                             _setup=lambda: events.append('setup'))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'mock-only; no GPU operations')
    monkeypatch.setattr(train_smoke.dist_utils, 'get_rank', lambda: 0)
    monkeypatch.setattr(train_smoke.dist_utils, 'get_world_size', lambda: 1)
    monkeypatch.setattr(train_smoke.dist_utils, 'is_dist_available_and_initialized', lambda: False)
    monkeypatch.setattr(train_smoke.dist_utils, 'warp_loader', lambda value, **kwargs: value)

    def build_flatcosine(*args):
        events.append('flatcosine')
        assert args[0] is optimizer and args[2] == 4 and args[3] == 120
        assert optimizer.param_groups[0]['lr'] == .0001
        return object()

    def native_train(*args, **kwargs):
        events.append('native_train')
        assert len(args[4]) == 4 and list(args[4]) == [0, 1]
        assert optimizer.param_groups[0]['lr'] == .0001
        kwargs['step_observer'].rows = [{'step': i, 'status': 'PASS'} for i in range(2)]
        return {'synthetic_control_flow_only': True}

    monkeypatch.setattr(train_smoke, 'FlatCosineLRScheduler', build_flatcosine)
    monkeypatch.setattr(train_smoke, 'train_one_epoch', native_train)
    report = train_smoke.run_bounded_train(solver, steps=2, epoch=10)
    relevant = [event for event in events if isinstance(event, str)]
    assert relevant == ['setup', 'optimizer', 'lr_scheduler', 'lr_warmup_scheduler',
                        'flatcosine', 'native_train']
    assert ('loader_epoch', 10) in events
    assert report['status'] == 'PASS' and report['completed_steps'] == 2
    assert report['validation'] == report['test'] == report['formal_training'] == 'NOT RUN'
    saved = json.loads((tmp_path / 'train_smoke_rank_0.json').read_text(encoding='utf-8'))
    assert saved['metrics'] == {'synthetic_control_flow_only': True}

