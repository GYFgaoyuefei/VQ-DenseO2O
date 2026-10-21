"""Final-paper joins use validated evidence and never impute absent results."""

import csv
import json
import statistics

import pytest

from test_ap_summary import make_record as ap_record, write_record as write_ap
from test_efficiency_summary import make_record as efficiency_record, write_record as write_efficiency
from tools.paper_v2.efficiency_protocol import AP_COLUMNS, MODELS, PAPER_COLUMNS, SEEDS
from tools.paper_v2.final_test_report import export_report


def write_pair(root, model, seed, ap_value=40., primary=10., mutate_efficiency=None):
    ap = ap_record(seed, model=model, value=ap_value)
    efficiency = efficiency_record(seed, primary=primary, model=model)
    efficiency['checkpoint'] = dict(ap['checkpoint'])
    efficiency['selected_epoch_zero_based'] = ap['selected_epoch_zero_based']
    if mutate_efficiency:
        mutate_efficiency(efficiency)
    write_ap(root / 'ap', ap)
    write_efficiency(root / 'efficiency', efficiency)


def read_table(root, name):
    with (root / 'final' / name).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def export(root, **kwargs):
    return export_report(root / 'ap', root / 'efficiency', root / 'final', **kwargs)


def test_complete_report_has_exact_paper_columns_three_seed_statistics_and_four_lights(tmp_path):
    for model in MODELS:
        for seed, value, latency in zip(SEEDS, (20., 30., 60.), (10., 20., 40.)):
            write_pair(tmp_path, model, seed, value, latency)
    result = export(tmp_path)
    assert result['status'] == result['ap_status'] == result['efficiency_status'] == 'COMPLETE'
    assert result['errors'] == []
    columns, by_seed = read_table(tmp_path, 'test_results_by_seed.csv')
    assert tuple(columns[:23]) == PAPER_COLUMNS
    assert len(by_seed) == 27
    assert by_seed[0]['Model'] == 'baseline_native'
    assert by_seed[0]['Epoch'] == '120' and by_seed[0]['Selected epoch'] == '77'
    assert float(by_seed[0]['Params(M)']) == 11.25
    assert float(by_seed[0]['Full Params(M)']) == 12.5

    columns, summaries = read_table(tmp_path, 'test_results_summary.csv')
    assert tuple(columns[:23]) == PAPER_COLUMNS and len(summaries) == 9
    row = summaries[0]
    latencies = [12.25, 22.25, 42.25]
    assert float(row['Latency(ms)']) == statistics.mean(latencies)
    assert float(row['Latency sample SD(ms)']) == statistics.stdev(latencies)
    assert float(row['FPS']) == 1000 / statistics.mean(latencies)
    assert float(row['FPS']) != statistics.mean(1000 / value for value in latencies)
    assert float(row['Mixed AP (%)']) == statistics.mean((20., 30., 60.))
    assert float(row['Mixed AP sample SD (%)']) == statistics.stdev((20., 30., 60.))
    for offset, column in enumerate(AP_COLUMNS[-4:], 12):
        assert float(row[column]) == statistics.mean(value + offset for value in (20., 30., 60.))

    columns, paper = read_table(tmp_path, 'paper_test_table.csv')
    assert tuple(columns) == PAPER_COLUMNS and len(paper) == 9
    assert '±' in paper[0]['Latency(ms)'] and 'PARTIAL' in paper[0]['GFLOPs']
    assert all('±' in paper[0][column] for column in AP_COLUMNS)
    manifest = json.loads((tmp_path / 'final/final_test_protocol.json').read_text(encoding='utf-8'))
    assert manifest['ap_status'] == manifest['efficiency_status'] == 'COMPLETE'
    assert len(manifest['image_counts_by_run']) == 27
    assert manifest['image_counts_by_run'][0]['sodium_light'] == 10


def test_missing_seed_keeps_existing_values_but_no_two_seed_aggregates(tmp_path):
    for seed in SEEDS[:2]:
        write_pair(tmp_path, MODELS[0], seed)
    result = export(tmp_path)
    assert result['status'] == result['ap_status'] == result['efficiency_status'] == 'INCOMPLETE'
    _, seeds = read_table(tmp_path, 'test_results_by_seed.csv')
    assert len(seeds) == 27 and float(seeds[0]['Mixed AP (%)']) == 40.
    assert seeds[2]['AP status'] == seeds[2]['Efficiency status'] == 'NOT RUN'
    assert seeds[2]['Hardware'] == ''
    _, summary = read_table(tmp_path, 'test_results_summary.csv')
    assert all(summary[0][column] == '' for column in (*AP_COLUMNS, 'Latency(ms)', 'FPS', 'Params(M)', 'GFLOPs'))


@pytest.mark.parametrize('mismatch', ['path', 'size_bytes', 'mtime_ns', 'model_state', 'selected_epoch'])
def test_checkpoint_mismatch_excludes_only_efficiency_and_preserves_test_ap(tmp_path, mismatch):
    def mutate(record):
        if mismatch == 'selected_epoch':
            record['selected_epoch_zero_based'] += 1
        else:
            record['checkpoint'][mismatch] = {'path': '/other/best.pth', 'size_bytes': 1001,
                                               'mtime_ns': 123456790, 'model_state': 'model'}[mismatch]
    for seed in SEEDS:
        write_pair(tmp_path, MODELS[0], seed, mutate_efficiency=mutate if seed == 42 else None)
    result = export(tmp_path)
    assert result['status'] == result['efficiency_status'] == 'FAIL'
    assert result['ap_record_count'] == 3 and result['efficiency_record_count'] == 2
    assert any('checkpoint identity or selected epoch' in reason for reason in result['errors'])
    _, seeds = read_table(tmp_path, 'test_results_by_seed.csv')
    assert seeds[1]['AP status'] == 'COMPLETE' and seeds[1]['Efficiency status'] == 'FAIL'
    assert float(seeds[1]['Mixed AP (%)']) == 40.
    assert seeds[1]['Latency(ms)'] == seeds[1]['Hardware'] == ''
    _, summary = read_table(tmp_path, 'test_results_summary.csv')
    assert float(summary[0]['Mixed AP (%)']) == 40. and summary[0]['Latency(ms)'] == ''


def test_ap_only_does_not_claim_protocol_gpu_was_measured(tmp_path):
    for model in MODELS:
        for seed in SEEDS:
            write_ap(tmp_path / 'ap', ap_record(seed, model=model))
    result = export(tmp_path)
    assert result['ap_status'] == 'COMPLETE' and result['efficiency_status'] == 'NOT RUN'
    assert result['status'] == 'INCOMPLETE'
    for name in ('test_results_by_seed.csv', 'test_results_summary.csv', 'paper_test_table.csv'):
        _, rows = read_table(tmp_path, name)
        assert all(row['Hardware'] == row['Latency(ms)'] == row['Params(M)'] == '' for row in rows)
        assert all(row['Natural AP (%)'] and row['Sodium-light AP (%)'] for row in rows)


def test_invalid_efficiency_campaign_still_exports_valid_ap(tmp_path):
    write_pair(tmp_path, MODELS[0], 0)
    malformed = tmp_path / 'efficiency/raw/baseline_native/seed_42/efficiency.json'
    malformed.parent.mkdir(parents=True)
    malformed.write_text('{"bad": true}', encoding='utf-8')
    result = export(tmp_path)
    assert result['status'] == result['efficiency_status'] == 'FAIL'
    assert result['ap_record_count'] == 1 and result['efficiency_record_count'] == 0
    _, rows = read_table(tmp_path, 'test_results_by_seed.csv')
    assert float(rows[0]['Mixed AP (%)']) == 40. and rows[0]['Hardware'] == ''
    assert any('Efficiency campaign validation failed' in reason for reason in result['errors'])


def test_efficiency_completion_is_independent_of_missing_ap(tmp_path):
    for model in MODELS:
        for seed in SEEDS:
            record = efficiency_record(seed, model=model)
            record['checkpoint'] = ap_record(seed, model=model)['checkpoint']
            write_efficiency(tmp_path / 'efficiency', record)
    result = export(tmp_path)
    assert result['efficiency_status'] == 'COMPLETE' and result['ap_status'] == 'NOT RUN'
    assert result['status'] == 'INCOMPLETE' and result['errors'] == []


def test_caller_exclusion_cannot_reintroduce_stale_efficiency(tmp_path):
    for seed in SEEDS:
        write_pair(tmp_path, MODELS[0], seed)
    result = export(tmp_path, excluded_efficiency={(MODELS[0], 42)}, errors=['Cached efficiency rejected'])
    assert result['status'] == result['efficiency_status'] == 'FAIL'
    assert result['efficiency_record_count'] == 2 and result['ap_record_count'] == 3
    _, rows = read_table(tmp_path, 'test_results_by_seed.csv')
    assert rows[1]['Efficiency status'] == 'FAIL' and rows[1]['Hardware'] == ''
