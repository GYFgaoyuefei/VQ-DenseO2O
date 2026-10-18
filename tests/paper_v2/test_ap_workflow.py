"""AP orchestration uses selected completed runs; no training or real Test data."""
import json

import pytest

from tools.paper_v2 import ap_workflow, run


def arguments(tmp_path, *extra):
    return run.parser().parse_args(['evaluate-ap', '--runs-dir', str(tmp_path / 'runs'),
        '--ap-dir', str(tmp_path / 'ap'), '--model', 'gr_vq', *extra])


def prepare_run(tmp_path, model='gr_vq', seed=42):
    directory = tmp_path / 'runs' / model / f'seed_{seed}'
    directory.mkdir(parents=True)
    for name in ap_workflow.REQUIRED_RUN_FILES:
        (directory / name).touch()
    return directory


def test_ap_cli_defaults_to_requested_test_three_seeds_and_independent_models():
    args = run.parser().parse_args(['evaluate-ap'])
    assert args.ap_split == 'test' and args.ap_seeds == [0, 42, 3407]
    assert ap_workflow.selected_models(args) == list(run.MODELS)
    args.ap_models = [*run.MODELS, *run.COMBINATIONS]
    assert len(ap_workflow.selected_models(args)) == 9
    args.model = 'gr_vq'
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
    assert calls[0]['model'] == 'gr_vq' and calls[0]['seed'] == 42
    assert calls[0]['split'] == 'test' and calls[0]['device'] == 'cuda:0'
    report = json.loads((tmp_path / 'ap/evaluation_attempts.json').read_text())
    assert [row['status'] for row in report['runs']] == ['NOT RUN', 'COMPLETE', 'NOT RUN']
    assert report['status'] == 'INCOMPLETE'


def test_existing_verified_record_reused_without_inference(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path, '--ap-seeds', '42')
    cached = tmp_path / 'ap/raw/gr_vq/seed_42/metrics.json'
    cached.parent.mkdir(parents=True)
    cached.write_text('{}')
    monkeypatch.setattr(ap_evaluate, 'reusable_record', lambda **kw: {'status': 'COMPLETE'})
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: pytest.fail('Must reuse'))
    monkeypatch.setattr(ap_workflow, 'summarize', lambda args: 'summarized')
    assert ap_workflow.evaluate_and_summarize(args) == 'summarized'
    report = json.loads((tmp_path / 'ap/evaluation_attempts.json').read_text())
    assert report['runs'][0]['reused'] is True


def test_invalid_cached_result_is_not_overwritten_or_published(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path, '--ap-seeds', '42')
    cached = tmp_path / 'ap/raw/gr_vq/seed_42/metrics.json'
    cached.parent.mkdir(parents=True)
    cached.write_text('{"existing":"preserve"}')

    def stale(**kwargs):
        raise ValueError('Checkpoint changed')

    monkeypatch.setattr(ap_evaluate, 'reusable_record', stale)
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: pytest.fail('Must not overwrite'))
    monkeypatch.setattr(ap_workflow, 'summarize', lambda args: pytest.fail('Must not publish stale table'))
    with pytest.raises(RuntimeError, match='1 AP runs failed'):
        ap_workflow.evaluate_and_summarize(args)
    assert json.loads(cached.read_text()) == {'existing': 'preserve'}
    report = json.loads((tmp_path / 'ap/evaluation_attempts.json').read_text())
    assert report['status'] == 'FAIL' and 'Checkpoint changed' in report['runs'][0]['reason']


def test_one_seed_repair_keeps_existing_other_models_in_shared_summary(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_evaluate
    args = arguments(tmp_path, '--ap-seeds', '42')
    prepare_run(tmp_path)
    for model in ('baseline_native', 'C4'):
        (tmp_path / 'ap/raw' / model / 'seed_0').mkdir(parents=True)
    monkeypatch.setattr(ap_evaluate, 'evaluate_run', lambda **kw: {'status': 'COMPLETE'})
    captured = []
    monkeypatch.setattr(ap_workflow, 'summarize', lambda summary_args:
                        captured.append(ap_workflow.selected_models(summary_args)))
    ap_workflow.evaluate_and_summarize(args)
    assert captured == [['baseline_native', 'gr_vq', 'C4']]
    assert args.model == 'gr_vq' and args.ap_models is None


def test_offline_summary_never_invokes_evaluation(tmp_path, monkeypatch):
    from tools.paper_v2 import ap_summary
    args = arguments(tmp_path)
    captured = []
    monkeypatch.setattr(ap_summary, 'summarize', lambda *a, **kw: captured.append((a, kw)) or {'status': 'NOT RUN'})
    assert ap_workflow.summarize(args) == {'status': 'NOT RUN'}
    assert captured == [((tmp_path / 'ap',), {'split': 'test', 'models': ['gr_vq']})]


@pytest.mark.parametrize('extra', [('--resume', 'last.pth'), ('--checkpoint', 'other.pth'),
                                 ('--synthetic',), ('--nproc-per-node', '2')])
def test_campaign_rejects_training_or_distributed_overrides(tmp_path, extra):
    with pytest.raises(ValueError):
        ap_workflow.evaluate_and_summarize(arguments(tmp_path, *extra))


def test_paper_efficiency_import_accepts_test_percent_ap_and_rejects_validation(tmp_path):
    from tools.paper_v2.efficiency_summary import _read_ap

    path = tmp_path / 'ap_by_seed.csv'
    path.write_text('Model,Seed,Split,Mixed AP (%)\ngr_vq,42,test,65.25\n', encoding='utf-8')
    assert _read_ap(path) == [{'Model': 'gr_vq', 'Seed': 42, 'Mixed AP (%)': 65.25}]
    path.write_text('Model,Seed,Split,Mixed AP (%)\ngr_vq,42,validation,65.25\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Validation must remain separate'):
        _read_ap(path)
