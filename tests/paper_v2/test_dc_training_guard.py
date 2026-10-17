"""Focused acceptance checks on DC training wiring and checkpoint architecture."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from engine.backbone.dc_sta import DCSTA, audit_dc_training
from engine.optim.ema import ModelEMA
from engine.solver._solver import BaseSolver
from tools.paper_v2 import run


class SmallDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.dc_sta = DCSTA([4, 8, 8])


def setup_training():
    model = SmallDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    ema = ModelEMA(model, decay=.9, warmups=0)
    return model, optimizer, ema


def gate_backward(model):
    images = torch.randn(2, 3, 32, 32)
    semantic = [torch.randn(2, 12, h, h) for h in (4, 2, 1)]
    detail = [torch.randn(2, c, h, h) for c, h in ((4, 4), (8, 2), (8, 1))]
    sum(value.square().mean() for value in model.dc_sta(images, semantic, detail)).backward()


def test_real_d1_optimizer_and_ema_include_every_gate_parameter():
    config = run.load_resolved('dc_sta', expand_env=False)
    config['DINOv3STAs']['weights_path'] = None
    obj = run.create_config(config)
    result = audit_dc_training(obj.model, obj.optimizer, obj.ema)
    assert result['status'] == 'PASS'
    assert result['parameter_count'] == 18
    assert result['identity_initialized']
    assert all(name.startswith('backbone.dc_sta.') for name in result['parameter_names'])
    assert result['optimizer_coverage_ok'] and result['ema_coverage_ok']


def test_gradient_audit_allows_first_step_zeros_and_detects_second_step_signal():
    model, optimizer, ema = setup_training()
    gate_backward(model)
    first = audit_dc_training(model, optimizer, ema, check_gradients=True)
    assert first['status'] == 'PASS' and first['gradient_l2_norm'] > 0
    assert set(first['gradient_zero']) == {
        f'dc_sta.channel.{level}.0.{kind}' for level in range(3) for kind in ('weight', 'bias')}
    optimizer.step()
    ema.update(model)
    optimizer.zero_grad(set_to_none=True)
    gate_backward(model)
    second = audit_dc_training(model, optimizer, ema, check_gradients=True)
    assert second['status'] == 'PASS'
    assert not second['identity_initialized']
    assert not second['gradient_zero']
    assert ema.module.dc_sta._gate_architecture.item() == 1
    assert ema.module.dc_sta._gate_architecture.dtype == torch.int64
    assert not torch.equal(ema.module.dc_sta.channel[0][-1].weight,
                           torch.zeros_like(ema.module.dc_sta.channel[0][-1].weight))


@pytest.mark.parametrize('damage', ['optimizer_missing', 'optimizer_duplicate', 'frozen', 'ema_missing', 'ema_half'])
def test_training_audit_detects_wiring_failures(damage):
    model, optimizer, ema = setup_training()
    parameter = model.dc_sta.channel[0][0].weight
    if damage == 'optimizer_missing':
        optimizer.param_groups[0]['params'] = [value for value in optimizer.param_groups[0]['params']
                                               if value is not parameter]
    elif damage == 'optimizer_duplicate':
        optimizer.param_groups[0]['params'].append(parameter)
    elif damage == 'frozen':
        parameter.requires_grad_(False)
    elif damage == 'ema_missing':
        ema.module.dc_sta = nn.Identity()
    elif damage == 'ema_half':
        ema.module.half()
    assert audit_dc_training(model, optimizer, ema)['status'] == 'FAIL'


@pytest.mark.parametrize('damage', ['missing', 'nonfinite', 'all_zero'])
def test_gradient_audit_detects_missing_nonfinite_or_dead_signal(damage):
    model, optimizer, ema = setup_training()
    gate_backward(model)
    if damage == 'missing':
        model.dc_sta.spatial[0].weight.grad = None
    elif damage == 'nonfinite':
        model.dc_sta.spatial[0].weight.grad.fill_(float('inf'))
    else:
        for parameter in model.dc_sta.parameters():
            parameter.grad.zero_()
    result = audit_dc_training(model, optimizer, ema, check_gradients=True)
    assert result['status'] == 'FAIL' and not result['gradients_ok']


@pytest.mark.parametrize('marker', [None, torch.tensor(0), torch.tensor(1.), torch.tensor([1])])
@pytest.mark.parametrize('strict', [False, True])
def test_direct_checkpoint_load_rejects_unmarked_or_wrong_gate_architecture(marker, strict):
    module = DCSTA([4, 8, 8])
    state = module.state_dict()
    if marker is None:
        del state['_gate_architecture']
    else:
        state['_gate_architecture'] = marker
    with pytest.raises(RuntimeError, match='Historical sigmoid'):
        module.load_state_dict(state, strict=strict)


def test_residual_tanh_checkpoint_roundtrip_includes_model_and_ema():
    model, optimizer, ema = setup_training()
    gate_backward(model)
    optimizer.step()
    ema.update(model)
    solver = BaseSolver.__new__(BaseSolver)
    solver.model, solver.ema = SmallDetector(), ModelEMA(SmallDetector())
    solver.load_state_dict({'model': model.state_dict(), 'ema': ema.state_dict()})
    for key, value in model.state_dict().items():
        assert torch.equal(solver.model.state_dict()[key], value)
    for key, value in ema.module.state_dict().items():
        assert torch.equal(solver.ema.module.state_dict()[key], value)


def test_tuning_rejects_legacy_keys_before_shape_matching_can_drop_them(tmp_path, monkeypatch):
    model = SmallDetector()
    legacy = {'stem.weight': model.stem.weight.detach(), 'stem.bias': model.stem.bias.detach(),
              'dc_sta.legacy_sigmoid_head.weight': torch.randn(1, 2)}
    path = tmp_path / 'legacy.pth'
    torch.save({'ema': {'module': legacy}}, path)
    solver = BaseSolver.__new__(BaseSolver)
    solver.model = model
    monkeypatch.setattr(solver, '_matched_state', lambda *_: pytest.fail('must validate before filtering'))
    with pytest.raises(RuntimeError, match='Historical sigmoid'):
        solver.load_tuning_state(str(path))


def test_native_pretrained_tuning_without_gates_keeps_new_gates_at_identity(tmp_path):
    model = SmallDetector()
    native = {key: torch.full_like(value, .25) for key, value in model.state_dict().items()
              if not key.startswith('dc_sta.')}
    path = tmp_path / 'native.pth'
    torch.save({'model': native}, path)
    solver = BaseSolver.__new__(BaseSolver)
    solver.model = model
    solver.load_tuning_state(str(path))
    assert torch.equal(model.stem.weight, torch.full_like(model.stem.weight, .25))
    result = audit_dc_training(model, torch.optim.AdamW(model.parameters()), ModelEMA(model))
    assert result['status'] == 'PASS' and result['identity_initialized']


def test_five_configs_share_detector_initialization_and_rng_for_same_seed():
    # Existing adapter identity tests cover gate math; this additionally checks
    # the later encoder/class heads, whose RNG must not shift when adding gates.
    with torch.random.fork_rng(devices=[]):
        baseline = None
        next_draw = None
        for name in run.MODELS:
            config = run.load_resolved(name, expand_env=False)
            config['DINOv3STAs']['weights_path'] = None
            torch.manual_seed(42)
            model = run.create_config(config).model
            state = {key: value for key, value in model.state_dict().items() if '.dc_sta.' not in key}
            draw = torch.rand(8)
            if baseline is None:
                baseline, next_draw = deepcopy(state), draw
            else:
                assert state.keys() == baseline.keys()
                assert all(torch.equal(value, baseline[key]) for key, value in state.items()), name
                assert torch.equal(draw, next_draw), name
            del model, state
