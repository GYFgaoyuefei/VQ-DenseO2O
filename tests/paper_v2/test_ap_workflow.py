"""AP orchestration uses selected completed runs; no training or real Test data."""
import json

import pytest

from tools.paper_v2 import ap_workflow, run
from tools.paper_v2.efficiency_protocol import ALGORITHM_SUITE


def arguments(tmp_path, *extra):
    return run.parser().parse_args(['evaluate-ap', '--runs-dir', str(tmp_path / 'runs'),
        '--ap-dir', str(tmp_path / 'ap'), '--model', 'vq_dense_o2o', *extra])


def prepare_run(tmp_path, model='vq_dense_o2o', seed=42):
    directory = tmp_path / 'runs' / model / f'seed_{seed}'
    directory.mkdir(parents=True)
    for name in ap_workflow.REQUIRED_RUN_FILES:
        (directory / name).touch()
    return directory


def bind_campaign(directory):
    (directory / 'evaluation_attempts.json').write_text(json.dumps({
        'algorithm_suite': ALGORITHM_SUITE, 'split': 'test', 'runs': []}))


def test_ap_cli_defaults_to_requested_test_three_seeds_and_independent_models():
    args = run.parser().parse_args(['evaluate-ap'])
    assert args.ap_split == 'test' and args.ap_seeds == [0, 42, 3407]
    assert ap_workflow.selected_models(args) == list(run.MODELS)
    args.ap_models = [*run.MODELS, *run.COMBINATIONS]
    assert len(ap_workflow.selected_models(args)) == 9
    args.model = 'vq_dense_o2o'
    with pytest.raises(ValueError, match='either'):
        ap_workflow.selected_models(args)


def test_campaign_records_missing_runs_and_evaluates_existing_only(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path)
    expected = prepare_run(tmp_path)
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {'status': 'COMPLETE'}

    monkeypatch.setattr(ap_evaluate, 'evaluate_run', evaluate)
    monkeypatch.setattr(ap_evaluate, 'reusable_record', lambda **kw: pytest.fail('No cache exists'))
    monkeypatch.setattr(ap_workflow, 'summarize', lambda args: 'summarized')
    assert ap_workflow.evaluate_and_summarize(args) == 'summarized'
    assert len(calls) == 1 and calls[0]['run_dir'] == expected
    assert calls[0]['model'] == 'vq_dense_o2o' and calls[0]['seed'] == 42
    assert calls[0]['split'] == 'test' and calls[0]['device'] == 'cuda:0'
    report = json.loads((tmp_path / 'ap/evaluation_attempts.json').read_text())
    assert [row['status'] for row in report['runs']] == ['NOT RUN', 'COMPLETE', 'NOT RUN']
    assert report['status'] == 'INCOMPLETE'
    assert report['algorithm_suite'] == ALGORITHM_SUITE


def test_existing_verified_record_reused_without_inference(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path, '--ap-seeds', '42')
    cached = tmp_path / 'ap/raw/vq_dense_o2o/seed_42/metrics.json'
    cached.parent.mkdir(parents=True)
    cached.write_text('{}')
    bind_campaign(tmp_path / 'ap')
    monkeypatch.setattr(ap_evaluate, 'reusable_record', lambda **kw: {'status': 'COMPLETE'})
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: pytest.fail('Must reuse'))
    monkeypatch.setattr(ap_workflow, 'summarize', lambda args: 'summarized')
    assert ap_workflow.evaluate_and_summarize(args) == 'summarized'
    report = json.loads((tmp_path / 'ap/evaluation_attempts.json').read_text())
    assert report['runs'][0]['reused'] is True


def test_invalid_cached_result_is_not_overwritten_or_published(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path, '--ap-seeds', '42')
    cached = tmp_path / 'ap/raw/vq_dense_o2o/seed_42/metrics.json'
    cached.parent.mkdir(parents=True)
    cached.write_text('{"existing":"preserve"}')
    bind_campaign(tmp_path / 'ap')

    def stale(**kwargs):
        raise ValueError('Checkpoint changed')

    monkeypatch.setattr(ap_evaluate, 'reusable_record', stale)
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: pytest.fail('Must not overwrite'))
    with pytest.raises(RuntimeError, match='1 AP runs failed'):
        ap_workflow.evaluate_and_summarize(args)
    assert json.loads(cached.read_text()) == {'existing': 'preserve'}
    report = json.loads((tmp_path / 'ap/evaluation_attempts.json').read_text())
    assert report['status'] == 'FAIL' and 'Checkpoint changed' in report['runs'][0]['reason']
    protocol = json.loads((tmp_path / 'ap/ap_protocol.json').read_text(encoding='utf-8'))
    assert protocol['status'] == 'FAIL' and protocol['measured_model_seed_count'] == 0
    assert (tmp_path / 'ap/ap_by_seed.csv').is_file()
    assert (tmp_path / 'ap/ap_summary.csv').is_file()


def test_one_seed_repair_keeps_other_unresolved_failures_excluded(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    from tools.paper_v2.ap_summary import load_exclusions
    args = arguments(tmp_path, '--ap-seeds', '42')
    ap_dir = tmp_path / 'ap'
    cached = ap_dir / 'raw/vq_dense_o2o/seed_42/metrics.json'
    cached.parent.mkdir(parents=True)
    cached.write_text('{}')
    # A current-suite retry ledger may still omit the optional excluded_runs field.
    (ap_dir / 'evaluation_attempts.json').write_text(json.dumps({
        'algorithm_suite': 'beta_select_algorithms_v1', 'split': 'test', 'status': 'FAIL', 'runs': [
            {'model': 'vq_dense_o2o', 'seed': 42, 'status': 'FAIL', 'reason': 'Old failure'},
            {'model': 'baseline_native', 'seed': 0, 'status': 'FAIL', 'reason': 'Unrepaired stale cache'},
        ]}))
    monkeypatch.setattr(ap_evaluate, 'reusable_record', lambda **kw: {'status': 'COMPLETE'})
    captured = []
    monkeypatch.setattr(ap_workflow, 'summarize', lambda summary_args:
                        captured.append(ap_workflow.selected_models(summary_args)))
    ap_workflow.evaluate_and_summarize(args)
    assert captured == [['baseline_native', 'vq_dense_o2o']]
    excluded = load_exclusions(ap_dir, 'test')
    assert list(excluded) == [('baseline_native', 0)]
    assert excluded[('baseline_native', 0)]['reason'] == 'Unrepaired stale cache'


def test_one_seed_repair_keeps_existing_other_models_in_shared_summary(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path, '--ap-seeds', '42')
    prepare_run(tmp_path)
    for model in ('baseline_native', 'vq_dense_o2o_li_copyblend_dc_sta'):
        (tmp_path / 'ap/raw' / model / 'seed_0').mkdir(parents=True)
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: {'status': 'COMPLETE'})
    captured = []
    monkeypatch.setattr(ap_workflow, 'summarize', lambda summary_args:
                        captured.append(ap_workflow.selected_models(summary_args)))
    ap_workflow.evaluate_and_summarize(args)
    assert captured == [['baseline_native', 'vq_dense_o2o', 'vq_dense_o2o_li_copyblend_dc_sta']]
    assert args.model == 'vq_dense_o2o' and args.ap_models is None


def test_offline_summary_never_invokes_evaluation(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_summary
    args = arguments(tmp_path)
    captured = []
    monkeypatch.setattr(ap_summary, 'summarize', lambda *a, **kw: captured.append((a, kw)) or {'status': 'NOT RUN'})
    assert ap_workflow.summarize(args) == {'status': 'NOT RUN'}
    assert captured == [((tmp_path / 'ap',), {'split': 'test', 'models': ['vq_dense_o2o']})]


@pytest.mark.parametrize('extra', [('--resume', 'last.pth'), ('--checkpoint', 'other.pth'),
                                 ('--synthetic',), ('--nproc-per-node', '2')])
def test_campaign_rejects_training_or_distributed_overrides(tmp_path, extra):
    with pytest.raises(ValueError):
        ap_workflow.evaluate_and_summarize(arguments(tmp_path, *extra))


def test_paper_efficiency_import_accepts_test_percent_ap_and_rejects_validation(tmp_path):
    from tools.paper_v2.efficiency_summary import _read_ap

    path = tmp_path / 'ap_by_seed.csv'
    path.write_text('Model,Seed,Split,Mixed AP (%),Algorithm suite\nvq_dense_o2o,42,test,65.25,beta_select_algorithms_v1\n', encoding='utf-8')
    assert _read_ap(path) == [{'Model': 'vq_dense_o2o', 'Seed': 42, 'Mixed AP (%)': 65.25, 'Algorithm suite': 'beta_select_algorithms_v1'}]
    path.write_text('Model,Seed,Split,Mixed AP (%),Algorithm suite\nvq_dense_o2o,42,validation,65.25,beta_select_algorithms_v1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Validation must remain separate'):
        _read_ap(path)


@pytest.mark.parametrize('marker', ['ap_protocol.json', 'evaluation_attempts.json'])
@pytest.mark.parametrize('suite', [None, 'paper_v2_gr_residual'])
def test_compatible_campaign_reuses_raw_records_without_training_suite_gate(tmp_path, monkeypatch, marker, suite):
    from tools.paper_v2 import ap_evaluate
    from test_ap_summary import make_record

    directory = tmp_path / 'ap'
    directory.mkdir()
    metadata = {'split': 'test', 'runs': []}
    if suite is not None:
        metadata['algorithm_suite'] = suite
    (directory / marker).write_text(json.dumps(metadata))
    cached = directory / 'raw/baseline_native/seed_42/metrics.json'
    cached.parent.mkdir(parents=True)
    record = make_record(seed=42)
    if suite is None:
        record.pop('algorithm_suite')
    else:
        record['algorithm_suite'] = suite
    cached.write_text(json.dumps(record))
    before = cached.read_bytes()
    monkeypatch.setattr(ap_evaluate, 'reusable_record', lambda **kw: record)
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: pytest.fail('Reject before inference'))
    args = arguments(tmp_path, '--ap-seeds', '42')
    args.model = 'baseline_native'
    result = ap_workflow.evaluate_and_summarize(args)
    assert result['measured_model_seed_count'] == 1
    assert result['sources'][0]['recorded_algorithm_suite'] == suite
    assert ap_workflow.summarize(args)['measured_model_seed_count'] == 1
    assert cached.read_bytes() == before
