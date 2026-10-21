"""Existing-checkpoint binding regression cases, using tiny local fixtures only."""
import copy

import pytest
import torch
import yaml

from tools.paper_v2 import run
from tools.paper_v2.checkpoint_binding import bind_existing_checkpoint, inference_digest
from tools.paper_v2.evaluation_state import _EvaluationDCSTA, load_evaluation_state, prepare_inference_config


def save_checkpoint(directory):
    directory.mkdir()
    torch.save({'last_epoch': 12, 'ema': {'module': {'weight': torch.ones(1)}}}, directory / 'best.pth')


def test_best_only_does_not_create_training_records(tmp_path):
    directory = tmp_path / 'seed_0'
    save_checkpoint(directory)
    before = (directory / 'best.pth').read_bytes()
    cfg, path, selection, source = bind_existing_checkpoint(directory, 'vq_dense_o2o', 0)
    assert path.read_bytes() == before
    assert selection['epoch'] == 12 and cfg['seed'] == 0
    assert source['config_origin'] == 'repository_model_fallback'
    assert source['selection_verification'] == 'UNVERIFIED'
    assert [item.name for item in directory.iterdir()] == ['best.pth']


def test_saved_inference_config_survives_old_suite_tag(tmp_path):
    directory = tmp_path / 'seed_42'
    save_checkpoint(directory)
    cfg = run.load_resolved('baseline_maskaware', expand_env=False)
    cfg['seed'] = 42
    cfg['paper_v2']['algorithm_suite'] = 'unrestricted_historical_label'
    cfg['DEIMTransformer']['eval_idx'] = 1
    # A resolved config does not need the source tree of the training machine.
    cfg['__include__'] = ['/missing/old/training/config.yml']
    (directory / 'config_resolved.yml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    actual, _, selection, provenance = bind_existing_checkpoint(directory, 'baseline_maskaware', 42)
    assert actual['DEIMTransformer']['eval_idx'] == 1
    assert selection == {'epoch': 12}
    assert provenance['declared_algorithm_suite'] == 'unrestricted_historical_label'
    assert provenance['selection_verification'] == 'UNVERIFIED'


def test_cache_digest_tracks_inference_not_training_tags():
    first = run.load_resolved('baseline_native', expand_env=False)
    prepare_inference_config(first)
    second = copy.deepcopy(first)
    second['paper_v2']['algorithm_suite'] = 'any_other_label'
    second['seed'] = 3407
    second['train_dataloader']['total_batch_size'] = 8
    assert inference_digest(first) == inference_digest(second)
    second['DEIMTransformer']['eval_idx'] = 1
    assert inference_digest(first) != inference_digest(second)


@pytest.mark.parametrize('marker', [None, -123, 1, 2])
def test_eval_ignores_only_dc_bookkeeping_but_loads_all_weights(marker):
    model = torch.nn.Module()
    model.dc_sta = _EvaluationDCSTA([2, 4, 4], {'enabled': True})
    state = copy.deepcopy(model.state_dict())
    state['dc_sta.channel.0.2.bias'].fill_(.75)
    if marker is not None:
        state['dc_sta._gate_architecture'] = torch.tensor(marker)
    load_evaluation_state(model, state, 'existing-checkpoint')
    assert torch.equal(model.dc_sta.channel[0][2].bias, torch.full((2,), .75))
    assert ('dc_sta._gate_architecture' in state) == (marker is not None)
    state.pop('dc_sta.channel.0.2.bias')
    with pytest.raises(RuntimeError, match='no learned weights were dropped'):
        load_evaluation_state(model, state, 'existing-checkpoint')


def test_saved_residual_formula_uses_alpha_without_version_requirement():
    gate = _EvaluationDCSTA([2, 4, 4], {'enabled': True, 'alpha': .2})
    with torch.no_grad():
        gate.channel[0][2].weight.zero_()
        gate.channel[0][2].bias.fill_(1.)
        gate.spatial[0].weight.zero_()
        gate.spatial[0].bias.fill_(1.)
    channel, spatial = gate.compute_gates(torch.zeros(1, 8), torch.zeros(1, 3, 2, 2),
                                          torch.zeros(1, 2, 2, 2), 0)
    expected = 1 + .2 * torch.tanh(torch.tensor(1.))
    assert torch.allclose(channel, expected.expand_as(channel))
    assert torch.allclose(spatial, expected.expand_as(spatial))
    assert '_gate_architecture' not in gate.state_dict()
