"""Final campaign orchestration with temporary metadata and mocked execution.

These tests never load a detector/checkpoint tensor or access real Test data.
"""
import json
from pathlib import Path

import pytest

from tools.paper_v2 import ap_summary, efficiency_summary, final_test, final_test_report, run
from tools.paper_v2.efficiency_protocol import MODELS, SEEDS


def arguments(tmp_path, *extra):
    return ['--runs-dir', str(tmp_path / 'runs with spaces'),
            '--data-root', str(tmp_path / 'unopened dataset'),
            '--report-dir', str(tmp_path / 'report'), '--ap-dir', str(tmp_path / 'ap'),
            '--efficiency-dir', str(tmp_path / 'efficiency'), *extra]


def unexpected(*args, **kwargs):
    pytest.fail('This path must not perform data/model access or launch a process')


def test_command_arguments_preserve_all_models_seeds_and_paths(tmp_path):
    args = final_test.parser().parse_args(arguments(tmp_path, '--threads', '3',
                                                  '--illumination-map', str(tmp_path / 'lights map.json')))
    ap = run.parser().parse_args(final_test.ap_command(args)[2:])
    assert ap.command == 'evaluate-ap' and ap.ap_split == 'test'
    assert ap.run_policy == 'existing-checkpoint'
    assert ap.ap_models == list(MODELS) and ap.ap_seeds == list(SEEDS)
    assert ap.runs_dir == str(args.runs_dir)
    assert ap.illumination_map == str(args.illumination_map)
    for model in MODELS:
        efficiency = run.parser().parse_args(final_test.efficiency_command(args, model, 3407)[2:])
        assert efficiency.command == 'benchmark' and efficiency.model == model
        assert efficiency.seed == 3407 and efficiency.threads == 3
        assert efficiency.run_policy == 'existing-checkpoint'
        assert Path(efficiency.output_dir) == args.runs_dir / model / 'seed_3407'


def test_dry_run_never_starts_evaluation_or_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(final_test, 'check_test_lights', unexpected)
    monkeypatch.setattr(final_test, 'mark_ap_pending', unexpected)
    monkeypatch.setattr(final_test, 'reusable_efficiency', unexpected)
    monkeypatch.setattr(run, 'stream_process', unexpected)
    monkeypatch.setattr(final_test_report, 'export_report', unexpected)
    assert final_test.main(arguments(tmp_path, '--dry-run')) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['expected_runs'] == 27
    assert len(output['efficiency_commands']) == 27
    assert len(output['missing_run_files']) == 27
    assert output['required_run_files'] == ['best.pth']
    assert not (tmp_path / 'report').exists()
    assert not (tmp_path / 'ap').exists()


def test_existing_checkpoint_dry_run_does_not_require_wrapper_metadata(tmp_path, capsys):
    runs = tmp_path / 'runs with spaces'
    for model in MODELS:
        for seed in SEEDS:
            checkpoint = runs / model / f'seed_{seed}' / 'best.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'not opened during dry-run')
    assert final_test.main(arguments(tmp_path, '--dry-run')) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['missing_run_files'] == {}


def test_cached_efficiency_must_match_selected_checkpoint(tmp_path, monkeypatch):
    args = final_test.parser().parse_args(arguments(tmp_path))
    model, seed = MODELS[0], SEEDS[0]
    checkpoint = args.runs_dir / model / f'seed_{seed}/best.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'not a tensor checkpoint; only its file identity is used')
    stat = checkpoint.stat()
    cache = args.efficiency_dir / 'raw' / model / f'seed_{seed}/efficiency.json'
    cache.parent.mkdir(parents=True)
    record = {'model': model, 'seed': seed, 'selected_epoch_zero_based': 19,
              'checkpoint': {'path': str(checkpoint), 'size_bytes': stat.st_size,
                             'mtime_ns': stat.st_mtime_ns, 'model_state': 'ema'}}
    cache.write_text(json.dumps(record), encoding='utf-8')
    monkeypatch.setattr(efficiency_summary, 'validate_record', lambda value: value)

    def binding(bound):
        assert bound.model == model and bound.seed == seed
        assert Path(bound.output_dir) == checkpoint.parent
        return {}, checkpoint, {'epoch': 19}

    monkeypatch.setattr(run, 'benchmark_binding', binding)
    assert final_test.reusable_efficiency(args, model, seed)
    record['checkpoint']['mtime_ns'] += 1
    cache.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='Stale efficiency'):
        final_test.reusable_efficiency(args, model, seed)


def test_ap_preflight_failure_excludes_old_results_before_data_check(tmp_path, monkeypatch):
    def failed_lights(args):
        exclusions = ap_summary.load_exclusions(args.ap_dir, 'test')
        assert set(exclusions) == {(model, seed) for model in MODELS for seed in SEEDS}
        raise ValueError('Missing explicit illumination')

    captured = []

    def export(ap_dir, efficiency_dir, report_dir, **kwargs):
        captured.append(kwargs)
        assert len(ap_summary.load_exclusions(ap_dir, 'test')) == 27
        return {'status': 'FAIL', 'ap_status': 'FAIL'}

    monkeypatch.setattr(final_test, 'check_test_lights', failed_lights)
    monkeypatch.setattr(run, 'stream_process', unexpected)
    monkeypatch.setattr(final_test_report, 'export_report', export)
    assert final_test.main(arguments(tmp_path, '--mode', 'ap-only')) == 1
    assert 'Missing explicit illumination' in captured[0]['errors'][0]
    ledger = json.loads((tmp_path / 'report/final_test_attempts.json').read_text())
    assert ledger['status'] == 'FAIL' and ledger['ap'] == 'FAIL'


def test_ap_only_success_does_not_require_efficiency(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test, 'check_test_lights', lambda args: {'natural': 1})
    monkeypatch.setattr(run, 'stream_process', lambda *args, **kwargs: None)
    monkeypatch.setattr(final_test, 'reusable_efficiency', unexpected)
    monkeypatch.setattr(final_test_report, 'export_report', lambda *args, **kwargs:
                        {'status': 'INCOMPLETE', 'ap_status': 'COMPLETE'})
    assert final_test.main(arguments(tmp_path, '--mode', 'ap-only')) == 0


def test_interrupt_during_ap_keeps_all_requested_efficiency_unverified(tmp_path, monkeypatch):
    def interrupt(args):
        raise KeyboardInterrupt

    captured = []

    def export(ap_dir, efficiency_dir, report_dir, **kwargs):
        captured.append(kwargs)
        assert len(ap_summary.load_exclusions(ap_dir, 'test')) == 27
        return {'status': 'FAIL', 'ap_status': 'INCOMPLETE'}

    monkeypatch.setattr(final_test, 'check_test_lights', interrupt)
    monkeypatch.setattr(run, 'stream_process', unexpected)
    monkeypatch.setattr(final_test_report, 'export_report', export)
    assert final_test.main(arguments(tmp_path)) == 130
    assert captured[0]['excluded_efficiency'] == {(model, seed) for model in MODELS for seed in SEEDS}
    ledger = json.loads((tmp_path / 'report/final_test_attempts.json').read_text())
    assert ledger['status'] == 'INTERRUPTED'


def test_one_failed_efficiency_cache_does_not_stop_other_models(tmp_path, monkeypatch):
    rejected = (MODELS[0], SEEDS[0])
    checked, captured = [], []

    def reusable(args, model, seed):
        checked.append((model, seed))
        if (model, seed) == rejected:
            raise ValueError('Selected checkpoint changed')
        return True

    def export(*args, **kwargs):
        captured.append(kwargs)
        return {'status': 'FAIL', 'ap_status': 'NOT RUN', 'efficiency_status': 'FAIL'}

    monkeypatch.setattr(final_test, 'reusable_efficiency', reusable)
    monkeypatch.setattr(final_test, 'check_test_lights', unexpected)
    monkeypatch.setattr(run, 'stream_process', unexpected)
    monkeypatch.setattr(final_test_report, 'export_report', export)
    assert final_test.main(arguments(tmp_path, '--mode', 'efficiency-only')) == 1
    assert len(checked) == 27 and captured[0]['excluded_efficiency'] == {rejected}
    ledger = json.loads((tmp_path / 'report/final_test_attempts.json').read_text())
    assert len(ledger['efficiency']) == 27
    assert sum(row['status'] == 'REUSED' for row in ledger['efficiency']) == 26


def test_offline_summary_preserves_exclusions_without_model_or_data_access(tmp_path, monkeypatch):
    report = tmp_path / 'report'
    report.mkdir()
    excluded = [[MODELS[1], 42]]
    (report / 'final_test_attempts.json').write_text(json.dumps({'excluded_efficiency': excluded}))

    def export(*args, **kwargs):
        assert kwargs['excluded_efficiency'] == {(MODELS[1], 42)}
        return {'status': 'INCOMPLETE', 'ap_status': 'COMPLETE'}

    monkeypatch.setattr(final_test_report, 'export_report', export)
    monkeypatch.setattr(final_test, 'check_test_lights', unexpected)
    monkeypatch.setattr(final_test, 'mark_ap_pending', unexpected)
    monkeypatch.setattr(final_test, 'reusable_efficiency', unexpected)
    monkeypatch.setattr(run, 'stream_process', unexpected)
    assert final_test.main(arguments(tmp_path, '--mode', 'summarize-only')) == 1
