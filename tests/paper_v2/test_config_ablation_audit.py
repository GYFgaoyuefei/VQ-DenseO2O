import copy

import pytest

from tools.paper_v2 import run
from tools.paper_v2.config_audit import inspect_configs


def configs():
    return {model: run.load_resolved(model, expand_env=False) for model in run.ALL_MODELS}


def test_full_factorial_effective_config_has_only_allowed_differences():
    result = inspect_configs(configs())
    assert result['status'] == 'PASS' and result['unexpected'] == []
    assert len(result['rows']) == 9 and len(result['comparisons']) == 8


def test_unexpected_learning_rate_and_augmentation_changes_are_detected():
    values = configs()
    values['li_copyblend']['optimizer']['lr'] *= 2
    values['li_copyblend']['train_dataloader']['collate_fn']['copyblend_prob'] = .1
    result = inspect_configs(values)
    assert result['status'] == 'FAIL'
    assert {'optimizer.lr', 'train_dataloader.collate_fn.copyblend_prob'} <= {item['field'] for item in result['unexpected']}


def test_management_fields_are_separate_and_old_suite_is_rejected():
    values = configs()
    values['baseline_maskaware']['output_dir'] = 'synthetic/another-run'
    assert inspect_configs(values)['status'] == 'PASS'
    old = copy.deepcopy(values['baseline_native'])
    old['paper_v2'].pop('algorithm_suite')
    with pytest.raises(ValueError, match='old runs'):
        run.verify_recipe(old)


@pytest.mark.parametrize('model', ['gr_vq', 'gr_vq_beta010', 'dc_sta_residual', 'dc_sta_v2', 'C4'])
def test_retired_model_ids_cannot_launch_new_training(model):
    with pytest.raises(ValueError, match='Unknown formal model'):
        run.config_path(model)


def test_enabled_vq_cannot_change_beta_or_enable_retired_formula():
    from engine.core.innovation_config import resolve_innovation
    from engine.core.yaml_utils import load_config
    raw = load_config(str(run.config_path('vq_dense_o2o')))
    raw['innovation']['vq_dense_o2o']['beta'] = .25
    with pytest.raises(ValueError, match='fixed at 0.10'):
        resolve_innovation(raw, expand_env=False)
    raw['innovation']['vq_dense_o2o']['beta'] = .10
    raw['innovation']['gr_vq'] = {'enabled': True}
    with pytest.raises(ValueError, match='historical innovation'):
        resolve_innovation(raw, expand_env=False)


def test_mask_only_audit_does_not_allow_arbitrary_transform_replacement():
    values = configs()
    operations = values['baseline_maskaware']['train_dataloader']['dataset']['transforms']['ops']
    index = next(i for i, op in enumerate(operations) if op['type'] == 'Resize')
    operations[index]['type'] = 'RandomCrop'
    result = inspect_configs(values)
    assert result['status'] == 'FAIL'
    assert any(item['pair'] == ['baseline_native', 'baseline_maskaware'] and
               item['field'] == f'train_dataloader.dataset.transforms.ops[{index}].type'
               for item in result['unexpected'])


def test_b0_must_not_silently_become_mask_aware():
    values = configs()
    values['baseline_native'] = copy.deepcopy(values['baseline_maskaware'])
    with pytest.raises(ValueError, match='B0 must use the native data path'):
        inspect_configs(values)


@pytest.mark.parametrize('target', ['loss', 'matcher'])
def test_derived_vq_criterion_cannot_hide_native_hyperparameter_changes(target):
    values = configs()
    criterion = values['vq_dense_o2o']['VQDEIMCriterion']
    if target == 'loss':
        criterion['weight_dict']['loss_mal'] *= 2
    else:
        criterion['matcher']['weight_dict']['cost_class'] *= 2
    with pytest.raises(ValueError, match='inherit every native criterion/matcher setting unchanged'):
        inspect_configs(values)


def test_a_shared_beta_change_is_rejected_even_without_pairwise_differences():
    values = configs()
    for value in values.values():
        value['innovation']['vq_dense_o2o']['beta'] = .2
    with pytest.raises(ValueError, match='beta must remain 0.10'):
        inspect_configs(values)
