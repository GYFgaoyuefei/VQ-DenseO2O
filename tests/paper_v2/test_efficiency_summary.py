"""Evidence validation and hierarchical efficiency aggregation, entirely on CPU."""
import csv
import json
import math
import statistics

import pytest

from tools.paper_v2.efficiency_protocol import AP_COLUMNS, HARDWARE, PAPER_COLUMNS, SCOPES, SEEDS, protocol
from tools.paper_v2.efficiency_summary import FLOPS_COUNT_SCOPE, sample_statistics, summarize


def make_record(seed=0, primary=10.0, model='baseline_native'):
    repeats = []
    for repeat in range(1, 6):
        scopes = {}
        for name, base in zip(SCOPES, (primary, 2.0, 0.5)):
            samples = [base + repeat - 1 + (index % 5) * 0.125 for index in range(500)]
            scopes[name] = {**sample_statistics(samples), 'samples_ms': samples,
                            'warmup_iterations': 100, 'timed_iterations': 500,
                            'peak_gpu_memory_bytes': 1000000 + repeat * 1000}
        repeats.append({'repeat': repeat, 'scopes': scopes})
    return {'schema': 'paper_v2_efficiency_record_v1', 'algorithm_suite': 'beta_select_algorithms_v1', 'status': 'MEASURED',
            'protocol': protocol(), 'model': model, 'seed': seed, 'epochs': 120,
            'selected_epoch_zero_based': 76, 'checkpoint': {'path': f'model_{seed}.pth', 'sha256': 'abc'},
            'full_params_m': 12.5, 'deploy_params_m': 11.25,
            'hardware': HARDWARE, 'backend': protocol()['backend'], 'precision': 'FP32',
            'runtime': {'torch_version': '2.8.0', 'torchvision_version': '0.23.0',
                        'cuda_version': '12.8', 'cudnn_version': 91002,
                        'cpu_threads': 8, 'cudnn_deterministic': False,
                        'autocast': False, 'tf32_matmul': False, 'tf32_cudnn': False,
                        'float32_matmul_precision': 'highest', 'cudnn_benchmark': False},
            'flops': {'tool': 'calflops', 'tool_version': '0.3.2', 'status': 'PARTIAL',
                      'gflops': 20.5, 'coverage_status': 'PARTIAL',
                      'raw_flops': 20.5e9, 'raw_macs': 10.25e9, 'raw_params': 11.25e6,
                      'count_scope': FLOPS_COUNT_SCOPE,
                      'operator_coverage': {'status': 'PARTIAL', 'exhaustive': False,
                                            'method': 'Fixture source analysis and profiler diagnostics.',
                                            'unsupported_operator_enumeration': 'UNAVAILABLE',
                                            'interpretation': 'Source inspection cannot establish exhaustive executed coverage.'},
                      'oracle': {'operation': '[4,8] @ [8,12]', 'expected_macs': 384,
                                 'expected_flops': 768, 'raw_macs': 384, 'raw_flops': 768},
                      'coverage_notes': ['Unsupported functional operators may be omitted.']},
            'repeats': repeats}


def write_record(root, record, relative=None):
    path = root / (relative or f"raw/{record['model']}/seed_{record['seed']}/efficiency.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


def read_csv(root, filename):
    with (root / filename).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_ap(path, rows, columns=('Model', 'Seed', AP_COLUMNS[0])):
    columns = (*columns, 'Algorithm suite')
    rows = [dict(row, **{'Algorithm suite': 'beta_select_algorithms_v1'}) for row in rows]
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_complete_aggregation_uses_repeat_means_then_sample_sd_and_reciprocal_fps(tmp_path):
    for seed, primary in zip(SEEDS, (10.0, 20.0, 40.0)):
        write_record(tmp_path, make_record(seed, primary))
    result = summarize(tmp_path)
    assert result['complete_models'] == ['baseline_native']
    assert result['status'] == 'INCOMPLETE'  # Other eight models have not run.
    by_seed = read_csv(tmp_path, 'efficiency_by_seed.csv')
    seed_latencies = [12.25, 22.25, 42.25]
    for row, expected in zip(by_seed, seed_latencies):
        assert float(row['Latency(ms)']) == expected
        assert float(row['FPS']) == 1000 / expected
        assert int(row['Peak GPU memory(bytes)']) == 1005000
    row = read_csv(tmp_path, 'efficiency_summary.csv')[0]
    expected_mean = statistics.mean(seed_latencies)
    assert float(row['Latency(ms)']) == expected_mean
    assert float(row['Latency sample SD(ms)']) == statistics.stdev(seed_latencies)
    assert float(row['FPS']) == 1000 / expected_mean
    assert float(row['FPS']) != statistics.mean(1000 / value for value in seed_latencies)
    assert float(row['detector_plus_postprocessor_latency_ms']) != (
        float(row['detector_forward_latency_ms']) + float(row['postprocessor_latency_ms']))
    assert float(row['detector_plus_postprocessor_p95_ms']) == pytest.approx(expected_mean + 0.25)


def test_paper_prefix_deploy_params_and_partial_flops_are_explicit(tmp_path):
    for seed in SEEDS:
        write_record(tmp_path, make_record(seed))
    summarize(tmp_path)
    rows = read_csv(tmp_path, 'paper_efficiency_table.csv')
    assert tuple(rows[0])[:len(PAPER_COLUMNS)] == PAPER_COLUMNS
    assert rows[0]['Model'] == 'Baseline'
    assert rows[0]['Epoch'] == '120'
    assert rows[0]['Params(M)'] == '11.25'
    assert rows[0]['Full Params(M)'] == '12.5'
    assert rows[0]['GFLOPs'] == '20.5'
    assert rows[0]['GFLOPs status'] == 'PARTIAL'
    assert rows[0]['Hardware'] == HARDWARE
    assert all(rows[0][column] == '' for column in AP_COLUMNS)
    metadata = json.loads((tmp_path / 'efficiency_protocol.json').read_text(encoding='utf-8'))
    assert metadata['flops_coverage_status'] == 'PARTIAL'
    assert metadata['flops_scope'] == 'deploy detector forward only; postprocessor excluded'
    assert metadata['flops_tool_version'] == '0.3.2'
    assert metadata['protocol'] == protocol()


@pytest.mark.parametrize('count', [0, 1, 2])
def test_missing_seeds_never_look_like_complete_measurements(tmp_path, count):
    for seed in SEEDS[:count]:
        write_record(tmp_path, make_record(seed))
    summarize(tmp_path)
    measured = read_csv(tmp_path, 'efficiency_by_seed.csv')
    assert len(measured) == count
    summary = read_csv(tmp_path, 'efficiency_summary.csv')[0]
    paper = read_csv(tmp_path, 'paper_efficiency_table.csv')[0]
    assert int(summary['measured_seed_count']) == count
    assert summary['status'] == ('INCOMPLETE' if count else 'NOT RUN')
    for key in ('Params(M)', 'GFLOPs', 'Latency(ms)', 'FPS', 'Latency sample SD(ms)', 'Peak GPU memory(bytes)'):
        assert paper[key] == ''


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r['protocol'].update(timed_iterations=300), 'Nonstandard efficiency protocol'),
    (lambda r: r['repeats'].pop(), 'Exactly 5 repeats'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]].update(timed_iterations=300), '500 timed'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]].update(warmup_iterations=50), '100 warmup'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]]['samples_ms'].pop(), 'Exactly 500 raw'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]]['samples_ms'].__setitem__(0, math.nan), 'finite'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]]['samples_ms'].__setitem__(0, -1), 'positive'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]].update(latency_ms=999), 'disagrees with raw'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]].update(fps=999), 'disagrees with raw'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]].update(p95_ms=999), 'disagrees with raw'),
    (lambda r: r['repeats'][0]['scopes'].pop(SCOPES[2]), 'independently measured'),
    (lambda r: r['repeats'][4].update(repeat=1), 'ordered 1 through 5'),
    (lambda r: r.update(hardware='Other GPU'), 'Nonstandard hardware'),
    (lambda r: r.update(precision='FP16'), 'Nonstandard precision'),
    (lambda r: r.update(epochs=100), '120-epoch'),
    (lambda r: r.update(selected_epoch_zero_based=120), 'outside the training'),
    (lambda r: r['flops'].update(coverage_status='EXACT'), 'PARTIAL operator coverage'),
    (lambda r: r['flops'].update(status='COMPLETE'), 'Unrecognized FLOPs'),
    (lambda r: r['flops'].update(coverage_notes=[]), 'coverage notes'),
    (lambda r: r['flops'].update(status='FAILED'), 'must be null'),
    (lambda r: r['runtime'].update(tf32_matmul=True), 'violates the fixed protocol'),
    (lambda r: r['runtime'].update(torch_version=None), 'Missing runtime version'),
    (lambda r: r['repeats'][0].update(repeat=True), 'ordered 1 through 5'),
    (lambda r: r['repeats'][0]['scopes'][SCOPES[0]].update(timed_iterations=500.0), '500 timed'),
])
def test_invalid_evidence_is_rejected_without_replacing_reports(tmp_path, mutation, match):
    summarize(tmp_path)
    prior = (tmp_path / 'efficiency_summary.csv').read_bytes()
    record = make_record()
    mutation(record)
    write_record(tmp_path, record)
    with pytest.raises(ValueError, match=match):
        summarize(tmp_path)
    assert (tmp_path / 'efficiency_summary.csv').read_bytes() == prior


@pytest.mark.parametrize('with_protocol', [True, False])
def test_historical_summary_directory_cannot_be_overwritten(tmp_path, with_protocol):
    report = tmp_path / 'paper_efficiency_table.csv'
    report.write_text('historical results', encoding='utf-8')
    if with_protocol:
        (tmp_path / 'efficiency_protocol.json').write_text('{"schema":"paper_v2_efficiency_summary_v1"}')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='Nonstandard efficiency protocol|lack measurement protocol'):
        summarize(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_ap_csv_does_not_require_a_training_suite_tag(tmp_path):
    path = tmp_path / 'old_ap.csv'
    path.write_text('Model,Seed,Mixed AP (%)\nbaseline_native,0,42\n', encoding='utf-8')
    summarize(tmp_path / 'new_efficiency', ap_csv=path)
    rows = read_csv(tmp_path / 'new_efficiency', 'ap_by_seed_snapshot.csv')
    assert rows[0]['Mixed AP (%)'] == '42.0'


@pytest.mark.parametrize('tag', [None, 'paper_v2_old'])
def test_same_protocol_efficiency_accepts_any_training_suite_tag(tmp_path, tag):
    record = make_record()
    if tag is None:
        record.pop('algorithm_suite')
    else:
        record['algorithm_suite'] = tag
    write_record(tmp_path, record)
    summarize(tmp_path)
    assert len(read_csv(tmp_path, 'efficiency_by_seed.csv')) == 1


@pytest.mark.parametrize('location,value', [
    ('torch_version', '2.9.0'), ('torchvision_version', '0.24.0'),
    ('cuda_version', '13.0'), ('cudnn_version', 99999), ('tool_version', '99.0'),
    ('cpu_threads', 4), ('cudnn_deterministic', True),
])
def test_cross_model_runtime_and_profiler_versions_cannot_be_mixed(tmp_path, location, value):
    first, second = make_record(), make_record(42, model='vq_dense_o2o')
    second['flops' if location == 'tool_version' else 'runtime'][location] = value
    write_record(tmp_path, first)
    write_record(tmp_path, second)
    with pytest.raises(ValueError, match='Mixed runtime or FLOPs tool versions'):
        summarize(tmp_path)


def test_raw_records_are_never_selected_by_latest_timestamp(tmp_path):
    record = make_record()
    write_record(tmp_path, record)
    write_record(tmp_path, record, 'raw/baseline_native/seed_0/retry/efficiency.json')
    with pytest.raises(ValueError, match='Duplicate model/seed'):
        summarize(tmp_path)


def test_noncanonical_record_directory_is_rejected(tmp_path):
    write_record(tmp_path, make_record(), 'raw/old_run/efficiency.json')
    with pytest.raises(ValueError, match='Nonstandard raw record location'):
        summarize(tmp_path)


@pytest.mark.parametrize('key', ['full_params_m', 'deploy_params_m'])
def test_different_parameter_counts_in_same_model_are_rejected(tmp_path, key):
    first, second = make_record(), make_record(42)
    second[key] += 1
    write_record(tmp_path, first)
    write_record(tmp_path, second)
    with pytest.raises(ValueError, match=f'inconsistent {key}'):
        summarize(tmp_path)


def test_different_flops_counts_same_model_are_rejected(tmp_path):
    first, second = make_record(), make_record(42)
    second['flops']['gflops'] += 1
    second['flops']['raw_flops'] += 1e9
    write_record(tmp_path, first)
    write_record(tmp_path, second)
    with pytest.raises(ValueError, match='inconsistent deploy-forward GFLOPs'):
        summarize(tmp_path)


def test_unavailable_flops_keep_complete_latency_but_blank_flops(tmp_path):
    for seed in SEEDS:
        record = make_record(seed)
        record['flops'].update(status='UNAVAILABLE', gflops=None, tool_version=None,
                               raw_flops=None, raw_macs=None, raw_params=None)
        write_record(tmp_path, record)
    summarize(tmp_path)
    row = read_csv(tmp_path, 'paper_efficiency_table.csv')[0]
    assert row['status'] == 'COMPLETE'
    assert row['Latency(ms)'] == '12.25'
    assert row['GFLOPs'] == ''
    assert row['GFLOPs status'] == 'UNAVAILABLE'


def test_explicit_ap_is_averaged_by_seed_and_survives_automatic_refresh(tmp_path):
    source = tmp_path / 'original_ap.csv'
    write_ap(source, [{'Model': 'B0', 'Seed': seed, AP_COLUMNS[0]: value}
                      for seed, value in zip(SEEDS, [60, 70, 80])])
    summarize(tmp_path, ap_csv=source)
    source.unlink()
    summarize(tmp_path)
    table = read_csv(tmp_path, 'paper_efficiency_table.csv')
    assert table[0][AP_COLUMNS[0]] == '70.0'
    assert table[0]['Mixed AP sample SD (%)'] == '10.0'
    assert table[0][AP_COLUMNS[1]] == ''
    assert table[0]['Mixed AP50 sample SD (%)'] == ''
    assert table[1][AP_COLUMNS[0]] == ''
    assert table[1]['Mixed AP sample SD (%)'] == ''
    metadata = json.loads((tmp_path / 'efficiency_protocol.json').read_text(encoding='utf-8'))
    assert metadata['ap_source'] == str(source.resolve())
    assert metadata['ap_snapshot'] == 'ap_by_seed_snapshot.csv'


def test_partial_ap_does_not_produce_a_three_seed_mean(tmp_path):
    source = tmp_path / 'ap.csv'
    write_ap(source, [{'Model': 'baseline_native', 'Seed': seed, AP_COLUMNS[0]: 90} for seed in SEEDS[:2]])
    summarize(tmp_path, ap_csv=source)
    row = read_csv(tmp_path, 'paper_efficiency_table.csv')[0]
    assert row[AP_COLUMNS[0]] == ''
    assert row['Mixed AP sample SD (%)'] == ''


@pytest.mark.parametrize('value', ['101', '-1', 'nan', 'inf', 'not a number'])
def test_ap_percent_values_are_validated(tmp_path, value):
    source = tmp_path / 'ap.csv'
    write_ap(source, [{'Model': 'baseline_native', 'Seed': 0, AP_COLUMNS[0]: value}])
    with pytest.raises(ValueError):
        summarize(tmp_path, ap_csv=source)


def test_duplicate_ap_seed_rejected(tmp_path):
    source = tmp_path / 'ap.csv'
    write_ap(source, [{'Model': model, 'Seed': 0, AP_COLUMNS[0]: 90} for model in ['B0', 'Baseline']])
    with pytest.raises(ValueError, match='Duplicate AP'):
        summarize(tmp_path, ap_csv=source)


def test_explicit_missing_ap_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='AP CSV does not exist'):
        summarize(tmp_path, ap_csv=tmp_path / 'missing.csv')


def test_raw_evidence_is_unchanged_by_summary(tmp_path):
    path = write_record(tmp_path, make_record())
    before = path.read_bytes()
    summarize(tmp_path)
    assert path.read_bytes() == before


def test_quantile_definition_is_median_and_nearest_rank():
    result = sample_statistics(list(range(1, 501)))
    assert result['p50_ms'] == 250.5
    assert result['p95_ms'] == 475
    assert result['latency_ms'] == 250.5


@pytest.mark.parametrize('key', [
    'autocast', 'tf32_matmul', 'tf32_cudnn', 'float32_matmul_precision',
    'cudnn_benchmark', 'cudnn_deterministic', 'cpu_threads',
])
def test_missing_observed_runtime_settings_fail_closed(tmp_path, key):
    record = make_record()
    del record['runtime'][key]
    write_record(tmp_path, record)
    with pytest.raises(ValueError, match=key):
        summarize(tmp_path)


@pytest.mark.parametrize('key,value', [
    ('cpu_threads', 0), ('cpu_threads', 2.0), ('cudnn_deterministic', 0),
    ('cudnn_version', None), ('cudnn_version', '91002'), ('cudnn_version', 0),
    ('torch_version', 'unknown'), ('torchvision_version', 'UNAVAILABLE'),
    ('cuda_version', 'UNKNOWN'), ('autocast', 0), ('tf32_matmul', 0),
])
def test_invalid_runtime_setting_types_and_version_placeholders_fail_closed(tmp_path, key, value):
    record = make_record()
    record['runtime'][key] = value
    write_record(tmp_path, record)
    with pytest.raises(ValueError, match=key):
        summarize(tmp_path)


def test_ap_sample_sd_keeps_original_all_ap_column_positions(tmp_path):
    source = tmp_path / 'ap.csv'
    write_ap(source, [{'Model': 'D1', 'Seed': seed, column: value}
                      for seed, value in zip(SEEDS, [1, 5, 9])
                      for column in [AP_COLUMNS[-1]]], columns=('Model', 'Seed', AP_COLUMNS[-1]))
    summarize(tmp_path, ap_csv=source)
    row = read_csv(tmp_path, 'paper_efficiency_table.csv')[4]
    assert tuple(row)[:len(PAPER_COLUMNS)] == PAPER_COLUMNS
    assert list(row).index('Sodium-light AP sample SD (%)') > list(row).index('Peak GPU memory(bytes)')
    assert row[AP_COLUMNS[-1]] == '5.0'
    assert row['Sodium-light AP sample SD (%)'] == '4.0'


@pytest.mark.parametrize('mutation,match', [
    (lambda f: f.update(count_scope='full training forward and backward'), 'count_scope'),
    (lambda f: f.pop('count_scope'), 'count_scope'),
    (lambda f: f.update(raw_flops=1e9), 'gflops/raw_flops'),
    (lambda f: f.update(raw_flops=math.inf), 'raw_flops'),
    (lambda f: f.update(raw_macs=math.nan), 'raw_macs'),
    (lambda f: f.update(raw_params=0), 'raw_params'),
    (lambda f: f.update(raw_params=True), 'raw_params'),
    (lambda f: f['oracle'].update(raw_flops=384), 'FMA=2'),
    (lambda f: f['oracle'].update(expected_flops=384), 'FMA=2'),
    (lambda f: f['oracle'].update(raw_macs=768), 'FMA=2'),
    (lambda f: f.pop('oracle'), 'oracle evidence'),
    (lambda f: f['oracle'].update(operation='another shape'), 'oracle evidence'),
    (lambda f: f.pop('operator_coverage'), 'operator_coverage'),
    (lambda f: f['operator_coverage'].update(status='COMPLETE'), 'operator_coverage'),
    (lambda f: f['operator_coverage'].update(exhaustive=True), 'operator_coverage'),
    (lambda f: f['operator_coverage'].pop('interpretation'), 'interpretation evidence'),
    (lambda f: f['operator_coverage'].update(method=''), 'method evidence'),
    (lambda f: f['operator_coverage'].pop('unsupported_operator_enumeration'), 'unsupported_operator_enumeration'),
])
def test_profiler_scope_counts_convention_and_coverage_evidence_are_validated(tmp_path, mutation, match):
    record = make_record()
    mutation(record['flops'])
    write_record(tmp_path, record)
    with pytest.raises(ValueError, match=match):
        summarize(tmp_path)


@pytest.mark.parametrize('status', ['UNAVAILABLE', 'FAILED', 'CONVENTION_MISMATCH'])
def test_unavailable_profiler_does_not_retain_numeric_raw_counts(tmp_path, status):
    record = make_record()
    record['flops'].update(status=status, gflops=None)
    write_record(tmp_path, record)
    with pytest.raises(ValueError, match='raw counts must be null'):
        summarize(tmp_path)


def test_minimal_unavailable_profiler_coverage_evidence_is_valid(tmp_path):
    record = make_record()
    record['flops'].update(status='UNAVAILABLE', gflops=None, raw_flops=None,
                           raw_macs=None, raw_params=None, tool_version=None,
                           operator_coverage={'status': 'PARTIAL', 'exhaustive': False,
                                              'unsupported_operator_enumeration': 'NOT_RUN'})
    record['flops'].pop('oracle')
    write_record(tmp_path, record)
    assert summarize(tmp_path)['measured_model_seed_count'] == 1
