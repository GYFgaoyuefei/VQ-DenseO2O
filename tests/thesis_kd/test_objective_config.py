"""Study02 reuses the M training recipe and routes outputs by protocol."""
import pytest

from thesis_kd.cli import export_commands, locate, parser
from thesis_kd.config import ROOT, kd_enabled, pair_diff, resolve_config


@pytest.mark.parametrize(('name', 'objective'), [
    ('m_mse_l5', 'mse'), ('m_cwd_l5_t1', 'cwd'),
])
def test_study02_matches_m_training_recipe(name, objective):
    reference = resolve_config('m_kd')
    candidate = resolve_config(name)
    assert candidate['native'] == reference['native']
    assert candidate['stage_policy'] == reference['stage_policy']
    assert candidate['selection'] == reference['selection'] == 'ema_mixed_validation_ap'
    assert candidate['tie_rule'] == reference['tie_rule'] == 'earliest_epoch'
    assert candidate['seeds'] == [42]
    assert candidate['snapshot_epochs'] == [5, 20, 60, 120]
    expected = dict(reference['kd'], lambda_kd=5.0, objective=objective)
    if objective == 'cwd':
        expected['temperature'] = 1.0
    assert candidate['kd'] == expected
    assert candidate['kd']['active_scales'] == ['P3', 'P4', 'P5']
    assert candidate['kd']['warmup_epochs'] == 5
    comparison = pair_diff(candidate)
    assert comparison['compared'] == [name, 'm_base']
    assert comparison['non_kd_differences'] == []
    assert comparison['protocol_metadata_differences']['protocol_id'] == [
        'feature_kd_objective_v1', 'feature_kd_ms_v1']
    candidate['native']['clip_max_norm'] = 2.0
    assert pair_diff(candidate)['non_kd_differences'] == ['native']


@pytest.mark.parametrize('action', ['resolve-config', 'preflight', 'smoke', 'train', 'resume', 'validate'])
@pytest.mark.parametrize('name', ['m_mse_l5', 'm_cwd_l5_t1'])
def test_study02_cli_output_path(action, name, monkeypatch):
    monkeypatch.delenv('THESIS_KD_OUTPUT_ROOT', raising=False)
    args = parser().parse_args([action, '--experiment', name, '--run-tag', 'study02'])
    cfg, path = locate(args)
    assert cfg['protocol_id'] == 'feature_kd_objective_v1'
    output = 'thesis_kd_smoke' if action == 'smoke' else 'thesis_kd'
    assert path == ROOT / 'outputs' / output / cfg['protocol_id'] / 'study02' / name / 'seed_42'


def test_study01_signatures_and_batches_stay_separate():
    kd = resolve_config('m_kd')
    base = resolve_config('m_base')
    assert kd['kd']['lambda_kd'] == 1.0 and kd_enabled(kd) and not kd_enabled(base)
    assert 'objective' not in kd['kd'] and 'temperature' not in kd['kd']
    assert 'snapshot_epochs' not in kd
    assert kd['protocol_id'] == base['protocol_id'] == 'feature_kd_ms_v1'
    assert len(export_commands('study01', sensitivity=True).splitlines()) == 18
    assert 'm_mse_l5' not in export_commands('study01', sensitivity=True)
    assert 'm_cwd_l5_t1' not in export_commands('study01', sensitivity=True)
