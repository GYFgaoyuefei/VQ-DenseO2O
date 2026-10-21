"""Validate raw efficiency evidence and aggregate repeats, then independent seeds.

No test-set inference happens here. AP values can only come from an explicitly
provided, percent-unit, per-seed CSV; its normalized snapshot survives refreshes.
"""
import csv
import json
import math
from pathlib import Path
import statistics

from .efficiency_protocol import (
    ALGORITHM_SUITE, AP_COLUMNS, HARDWARE, LABELS, MODELS, PAPER_COLUMNS, SCOPES, SEEDS,
    protocol, validate_protocol,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'outputs/paper_v2_beta_select_algorithms/efficiency'
RUNTIME_KEYS = ('torch_version', 'torchvision_version', 'cuda_version', 'cudnn_version')
RUNTIME_SETTINGS = ('cpu_threads', 'cudnn_deterministic')
FLOPS_STATUSES = ('PARTIAL', 'UNAVAILABLE', 'FAILED', 'CONVENTION_MISMATCH')
FLOPS_COUNT_SCOPE = 'deploy detector forward only; postprocessor, criterion and training augmentation excluded'
_BASE = ('Model', 'Seed', 'Epoch', 'selected_epoch_zero_based', 'status',
         'Full Params(M)', 'Deploy Params(M)', 'GFLOPs', 'GFLOPs status',
         'GFLOPs tool', 'GFLOPs tool version', 'GFLOPs coverage notes',
         'Latency(ms)', 'FPS', 'Peak GPU memory(bytes)', 'Hardware', 'Backend', 'Precision')
_METRICS = ('latency_ms', 'fps', 'p50_ms', 'p95_ms', 'peak_gpu_memory_bytes')
BY_SEED_COLUMNS = (*_BASE, *RUNTIME_KEYS,
                   *(f'{scope}_{metric}' for scope in SCOPES for metric in _METRICS))
SUMMARY_COLUMNS = (
    'Model', 'Epoch', 'status', 'measured_seed_count', 'seeds',
    'Full Params(M)', 'Deploy Params(M)', 'GFLOPs', 'GFLOPs status',
    'GFLOPs tool', 'GFLOPs tool version', 'GFLOPs coverage notes',
    'Latency(ms)', 'Latency sample SD(ms)', 'FPS', 'Peak GPU memory(bytes)',
    'Hardware', 'Backend', 'Precision', *RUNTIME_KEYS,
    *(f'{scope}_{metric}{suffix}' for scope in SCOPES
      for metric in ('latency_ms', 'p50_ms', 'p95_ms') for suffix in ('', '_sample_sd')),
    *(f'{scope}_{metric}' for scope in SCOPES for metric in ('fps', 'peak_gpu_memory_bytes')),
)
PAPER_OUTPUT_COLUMNS = (*PAPER_COLUMNS, 'status', 'measured_seed_count',
                        'Latency sample SD(ms)', 'GFLOPs status', 'Full Params(M)',
                        'Backend', 'Precision', 'Peak GPU memory(bytes)',
                        *(column.replace(' (%)', ' sample SD (%)') for column in AP_COLUMNS))


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{label} must be {"positive" if positive else "nonnegative"}')
    return value


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _matches(actual, expected, label):
    _number(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label} disagrees with raw samples: {actual} != {expected}')


def sample_statistics(samples):
    """The protocol's median and nearest-rank P95, without numpy dependencies."""
    ordered = sorted(samples)
    return {'latency_ms': statistics.mean(samples),
            'fps': 1000.0 / statistics.mean(samples),
            'p50_ms': statistics.median(samples),
            'p95_ms': ordered[math.ceil(0.95 * len(samples)) - 1]}


def validate_record(record):
    """Reject incompatible or internally inconsistent evidence before any writes."""
    if not isinstance(record, dict):
        raise ValueError('efficiency record must be a JSON object')
    if record.get('schema') != 'paper_v2_efficiency_record_v1' or record.get('status') != 'MEASURED':
        raise ValueError('Only MEASURED paper_v2_efficiency_record_v1 records can be summarized')
    if not isinstance(record.get('protocol'), dict):
        raise ValueError('Missing efficiency protocol')
    validate_protocol(record['protocol'])
    if record.get('model') not in MODELS or type(record.get('seed')) is not int or record['seed'] not in SEEDS:
        raise ValueError('Unknown model or seed in efficiency record')
    if type(record.get('epochs')) is not int or record['epochs'] != 120:
        raise ValueError('Efficiency records must use the 120-epoch training recipe')
    selected = _integer(record.get('selected_epoch_zero_based'), 'selected_epoch_zero_based')
    if selected >= record['epochs']:
        raise ValueError('selected_epoch_zero_based is outside the training recipe')
    if not isinstance(record.get('checkpoint'), dict) or not record['checkpoint']:
        raise ValueError('Missing checkpoint provenance')
    for key in ('full_params_m', 'deploy_params_m'):
        _number(record.get(key), key, positive=True)
    for key in ('hardware', 'backend', 'precision'):
        if record.get(key) != protocol()[key]:
            raise ValueError(f'Nonstandard {key}: {record.get(key)!r}')
    runtime = record.get('runtime')
    if not isinstance(runtime, dict) or any(key not in runtime for key in RUNTIME_KEYS):
        raise ValueError('Missing runtime version provenance')
    for key in ('torch_version', 'torchvision_version', 'cuda_version'):
        if not isinstance(runtime[key], str) or not runtime[key] or not runtime[key][0].isdigit():
            raise ValueError(f'Missing runtime version or placeholder version: {key}')
    _integer(runtime['cudnn_version'], 'cudnn_version', minimum=1)
    _integer(runtime.get('cpu_threads'), 'cpu_threads', minimum=1)
    if type(runtime.get('cudnn_deterministic')) is not bool:
        raise ValueError('Missing or invalid runtime cudnn_deterministic flag')
    for key, expected in (('autocast', False), ('tf32_matmul', False), ('tf32_cudnn', False),
                          ('float32_matmul_precision', 'highest'), ('cudnn_benchmark', False)):
        if key not in runtime or type(runtime[key]) is not type(expected) or runtime[key] != expected:
            raise ValueError(f'Runtime {key} violates the fixed protocol')
    flops = record.get('flops')
    if not isinstance(flops, dict) or flops.get('tool') != 'calflops' or flops.get('status') not in FLOPS_STATUSES:
        raise ValueError('Unrecognized FLOPs tool or status')
    if 'tool_version' not in flops or flops.get('coverage_status') != 'PARTIAL':
        raise ValueError('Missing FLOPs tool version or PARTIAL operator coverage')
    if flops.get('count_scope') != FLOPS_COUNT_SCOPE:
        raise ValueError('FLOPs count_scope must be the deployed detector forward only')
    coverage = flops.get('operator_coverage')
    if (not isinstance(coverage, dict) or coverage.get('status') != 'PARTIAL'
            or coverage.get('exhaustive') is not False):
        raise ValueError('FLOPs operator_coverage must explicitly document PARTIAL nonexhaustive coverage')
    if (not isinstance(coverage.get('unsupported_operator_enumeration'), str)
            or not coverage['unsupported_operator_enumeration'].strip()):
        raise ValueError('Missing FLOPs unsupported_operator_enumeration evidence')
    if not isinstance(flops.get('coverage_notes'), list) or not flops['coverage_notes'] or not all(
            isinstance(note, str) and note.strip() for note in flops['coverage_notes']):
        raise ValueError('FLOPs coverage notes must explain partial or unavailable coverage')
    if flops['status'] == 'PARTIAL':
        _number(flops.get('gflops'), 'gflops', positive=True)
        _number(flops.get('raw_flops'), 'raw_flops', positive=True)
        _number(flops.get('raw_macs'), 'raw_macs')
        _number(flops.get('raw_params'), 'raw_params', positive=True)
        _matches(flops['gflops'], flops['raw_flops'] / 1e9, 'gflops/raw_flops')
        if not isinstance(flops['tool_version'], str) or not flops['tool_version']:
            raise ValueError('Measured GFLOPs need a tool version')
        for key in ('method', 'interpretation'):
            if not isinstance(coverage.get(key), str) or not coverage[key].strip():
                raise ValueError(f'Missing FLOPs operator_coverage {key} evidence')
        oracle = flops.get('oracle')
        if not isinstance(oracle, dict) or oracle.get('operation') != '[4,8] @ [8,12]':
            raise ValueError('Missing FLOPs contraction oracle evidence')
        for key, expected in (('expected_macs', 384), ('expected_flops', 768),
                              ('raw_macs', 384), ('raw_flops', 768)):
            _number(oracle.get(key), f'oracle {key}', positive=True)
            if oracle[key] != expected:
                raise ValueError(f'FLOPs oracle {key} must equal {expected} to demonstrate FMA=2')
    elif any(key not in flops or flops[key] is not None
             for key in ('gflops', 'raw_flops', 'raw_macs', 'raw_params')):
        raise ValueError('Unavailable/failed/convention-mismatched GFLOPs and raw counts must be null')
    repeats = record.get('repeats')
    if not isinstance(repeats, list) or len(repeats) != protocol()['repeats_per_seed']:
        raise ValueError('Exactly 5 repeats are required for every model/seed')
    for index, repeat in enumerate(repeats, start=1):
        if type(repeat.get('repeat')) is not int or repeat['repeat'] != index:
            raise ValueError('Repeat identifiers must be ordered 1 through 5')
        scopes = repeat.get('scopes')
        if not isinstance(scopes, dict) or set(scopes) != set(SCOPES):
            raise ValueError('Each repeat must contain all independently measured scopes')
        for scope in SCOPES:
            data = scopes[scope]
            if (type(data.get('warmup_iterations')) is not int or data['warmup_iterations'] != 100
                    or type(data.get('timed_iterations')) is not int or data['timed_iterations'] != 500):
                raise ValueError('Each scope requires 100 warmup and 500 timed iterations')
            samples = data.get('samples_ms')
            if not isinstance(samples, list) or len(samples) != 500:
                raise ValueError('Exactly 500 raw samples are required per repeat/scope')
            for value in samples:
                _number(value, f'{scope} sample', positive=True)
            for key, value in sample_statistics(samples).items():
                _matches(data.get(key), value, f'{scope} {key}')
            _integer(data.get('peak_gpu_memory_bytes'), f'{scope} peak_gpu_memory_bytes', minimum=1)
    return record


def _signature(record):
    return tuple(record['runtime'].get(key) for key in (*RUNTIME_KEYS, *RUNTIME_SETTINGS)) + (
        record['flops']['tool'], record['flops']['tool_version'])


def _load_records(output_dir):
    records, seen = [], set()
    # Strictly one immutable-evidence location per pair; never select a latest run.
    for path in sorted((output_dir / 'raw').rglob('efficiency.json')):
        try:
            record = validate_record(json.loads(path.read_text(encoding='utf-8')))
            key = (record['model'], record['seed'])
            if key in seen:
                raise ValueError(f'Duplicate model/seed record: {key}')
            expected = output_dir / 'raw' / key[0] / f'seed_{key[1]}' / 'efficiency.json'
            if path != expected:
                raise ValueError(f'Nonstandard raw record location: expected {expected}')
            seen.add(key)
            records.append(record)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f'{path}: {error}') from error
    if records and any(_signature(record) != _signature(records[0]) for record in records):
        raise ValueError('Mixed runtime or FLOPs tool versions; use a separate output directory')
    for model in MODELS:
        group = [r for r in records if r['model'] == model]
        if group:
            for key in ('full_params_m', 'deploy_params_m'):
                if any(not math.isclose(r[key], group[0][key], rel_tol=1e-12) for r in group):
                    raise ValueError(f'{model}: inconsistent {key} across seeds')
            counts = [r['flops']['gflops'] for r in group if r['flops']['gflops'] is not None]
            if counts and any(not math.isclose(count, counts[0], rel_tol=1e-9) for count in counts):
                raise ValueError(f'{model}: inconsistent deploy-forward GFLOPs across seeds')
    return sorted(records, key=lambda r: (MODELS.index(r['model']), SEEDS.index(r['seed'])))


def _seed_row(record):
    flops = record['flops']
    row = {'Model': record['model'], 'Seed': record['seed'], 'Epoch': record['epochs'],
           'selected_epoch_zero_based': record['selected_epoch_zero_based'], 'status': 'MEASURED',
           'Full Params(M)': record['full_params_m'], 'Deploy Params(M)': record['deploy_params_m'],
           'GFLOPs': flops['gflops'], 'GFLOPs status': flops['status'], 'GFLOPs tool': flops['tool'],
           'GFLOPs tool version': flops['tool_version'],
           'GFLOPs coverage notes': '; '.join(flops['coverage_notes']),
           'Hardware': record['hardware'], 'Backend': record['backend'], 'Precision': record['precision'],
           **{key: record['runtime'][key] for key in RUNTIME_KEYS}}
    for scope in SCOPES:
        measured = [sample_statistics(repeat['scopes'][scope]['samples_ms']) for repeat in record['repeats']]
        for metric in ('latency_ms', 'p50_ms', 'p95_ms'):
            row[f'{scope}_{metric}'] = statistics.mean(item[metric] for item in measured)
        row[f'{scope}_fps'] = 1000.0 / row[f'{scope}_latency_ms']
        row[f'{scope}_peak_gpu_memory_bytes'] = max(
            repeat['scopes'][scope]['peak_gpu_memory_bytes'] for repeat in record['repeats'])
    row['Latency(ms)'] = row[f'{SCOPES[0]}_latency_ms']
    row['FPS'] = 1000.0 / row['Latency(ms)']
    row['Peak GPU memory(bytes)'] = row[f'{SCOPES[0]}_peak_gpu_memory_bytes']
    return row


def _summary_row(model, rows):
    row = {'Model': model, 'Epoch': 120, 'status': 'COMPLETE' if len(rows) == 3 else
           ('INCOMPLETE' if rows else 'NOT RUN'), 'measured_seed_count': len(rows),
           'seeds': ','.join(str(item['Seed']) for item in rows),
           'Hardware': HARDWARE, 'Backend': protocol()['backend'], 'Precision': 'FP32'}
    if rows:
        row.update({key: rows[0][key] for key in (*RUNTIME_KEYS, 'GFLOPs tool', 'GFLOPs tool version')})
        row['GFLOPs coverage notes'] = '; '.join(dict.fromkeys(item['GFLOPs coverage notes'] for item in rows))
        statuses = sorted({item['GFLOPs status'] for item in rows})
        row['GFLOPs status'] = statuses[0] if len(statuses) == 1 else 'MIXED:' + ','.join(statuses)
    if len(rows) != 3:
        return row
    for key in ('Full Params(M)', 'Deploy Params(M)'):
        row[key] = rows[0][key]
    if all(item['GFLOPs'] is not None for item in rows):
        row['GFLOPs'] = statistics.mean(item['GFLOPs'] for item in rows)
    for scope in SCOPES:
        for metric in ('latency_ms', 'p50_ms', 'p95_ms'):
            key = f'{scope}_{metric}'
            row[key] = statistics.mean(item[key] for item in rows)
            row[f'{key}_sample_sd'] = statistics.stdev(item[key] for item in rows)
        row[f'{scope}_fps'] = 1000.0 / row[f'{scope}_latency_ms']
        row[f'{scope}_peak_gpu_memory_bytes'] = max(item[f'{scope}_peak_gpu_memory_bytes'] for item in rows)
    row['Latency(ms)'] = row[f'{SCOPES[0]}_latency_ms']
    row['Latency sample SD(ms)'] = row[f'{SCOPES[0]}_latency_ms_sample_sd']
    row['FPS'] = 1000.0 / row['Latency(ms)']
    row['Peak GPU memory(bytes)'] = row[f'{SCOPES[0]}_peak_gpu_memory_bytes']
    return row


def _read_ap(path):
    if path is None or not path.exists():
        return []
    aliases = {**{model: model for model in MODELS}, **{label: model for model, label in LABELS.items()},
               **dict(zip(('B0', 'B1', 'V1', 'L1', 'D1', 'C1', 'C2', 'C3', 'C4'), MODELS))}
    rows, seen = [], set()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {'Model', 'Seed'}.issubset(reader.fieldnames):
            raise ValueError('AP CSV requires Model and Seed columns')
        if not set(AP_COLUMNS).intersection(reader.fieldnames):
            raise ValueError('AP CSV requires named percent-unit AP_COLUMNS')
        for source in reader:
            if 'Split' in reader.fieldnames and source.get('Split') != 'test':
                raise ValueError('Paper AP CSV must contain Test results; Validation must remain separate')
            model = aliases.get(source['Model'])
            try:
                seed = int(source['Seed'])
            except (ValueError, TypeError) as error:
                raise ValueError('AP CSV Seed must be an integer') from error
            if model is None or seed not in SEEDS:
                raise ValueError('AP CSV contains an unknown model or seed')
            key = (model, seed)
            if key in seen:
                raise ValueError(f'Duplicate AP model/seed: {key}')
            seen.add(key)
            row = {'Model': model, 'Seed': seed, 'Algorithm suite': source.get('Algorithm suite', '')}
            for column in AP_COLUMNS:
                raw = source.get(column, '')
                if raw is None or not raw.strip():
                    continue
                try:
                    value = float(raw)
                except ValueError as error:
                    raise ValueError(f'Invalid {column}: {raw!r}') from error
                _number(value, column)
                if value > 100:
                    raise ValueError(f'{column} must be a percentage between 0 and 100')
                row[column] = value
            rows.append(row)
    return sorted(rows, key=lambda row: (MODELS.index(row['Model']), SEEDS.index(row['Seed'])))


def _paper_row(summary, ap_rows):
    row = {key: summary.get(key, '') for key in PAPER_OUTPUT_COLUMNS}
    row['Model'] = LABELS[summary['Model']]
    row['Params(M)'] = summary.get('Deploy Params(M)', '')
    for column in AP_COLUMNS:
        values = [item[column] for item in ap_rows if item['Model'] == summary['Model'] and column in item]
        row[column] = statistics.mean(values) if len(values) == len(SEEDS) else ''
        row[column.replace(' (%)', ' sample SD (%)')] = statistics.stdev(values) if len(values) == len(SEEDS) else ''
    return row


def _write_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def check_campaign(output_dir):
    """Check measurement compatibility, without gating on training suite tags."""
    output_dir = Path(output_dir)
    metadata = output_dir / 'efficiency_protocol.json'
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding='utf-8'))
        validate_protocol(previous.get('protocol', {}))
    elif any((output_dir / name).exists() for name in (
            'efficiency_by_seed.csv', 'efficiency_summary.csv', 'paper_efficiency_table.csv',
            'ap_by_seed_snapshot.csv')):
        raise ValueError('Existing efficiency reports lack measurement protocol evidence; use a new directory')
    for path in output_dir.glob('raw/*/seed_*/efficiency.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        validate_protocol(record.get('protocol', {}))


def summarize(output_dir=DEFAULT_OUTPUT_DIR, ap_csv=None):
    """Rebuild four reports from explicit evidence; validation failure writes nothing."""
    output_dir = Path(output_dir)
    check_campaign(output_dir)
    records = _load_records(output_dir)
    snapshot = output_dir / 'ap_by_seed_snapshot.csv'
    if ap_csv is not None and not Path(ap_csv).is_file():
        raise ValueError(f'AP CSV does not exist: {ap_csv}')
    ap_rows = _read_ap(Path(ap_csv) if ap_csv is not None else snapshot)
    by_seed = [_seed_row(record) for record in records]
    summary = [_summary_row(model, [row for row in by_seed if row['Model'] == model]) for model in MODELS]
    paper = [_paper_row(row, ap_rows) for row in summary]
    metadata = {
        'schema': 'paper_v2_efficiency_summary_v1', 'algorithm_suite': ALGORITHM_SUITE, 'protocol': protocol(),
        'status': 'COMPLETE' if len(records) == len(MODELS) * len(SEEDS) else
                  ('INCOMPLETE' if records else 'NOT RUN'),
        'measured_model_seed_count': len(records), 'expected_model_seed_count': len(MODELS) * len(SEEDS),
        'model_status': {row['Model']: {'status': row['status'], 'measured_seed_count': row['measured_seed_count']}
                         for row in summary},
        'runtime': records[0]['runtime'] if records else None,
        'flops_tool': 'calflops', 'flops_tool_version': records[0]['flops']['tool_version'] if records else None,
        'flops_scope': protocol()['gflops_scope'], 'flops_coverage_status': 'PARTIAL',
        'flops_coverage_notes': sorted({note for record in records for note in record['flops']['coverage_notes']}),
        'raw_record_paths': [f"raw/{record['model']}/seed_{record['seed']}/efficiency.json" for record in records],
        'sources': [{'model': record['model'], 'seed': record['seed'],
                     'checkpoint': record['checkpoint'],
                     'algorithm_suite': record.get('algorithm_suite'),
                     'run_provenance': record.get('run_provenance')} for record in records],
        'aggregation': {'latency': protocol()['summary_latency'], 'fps': protocol()['fps'],
                        'p50_p95': 'mean of five repeat quantiles per seed, then mean and sample SD across three seeds',
                        'peak_memory': 'maximum over five repeats per seed, then maximum over three seeds',
                        'incomplete_models': 'numeric aggregate efficiency fields are blank until all three seeds exist',
                        'paper_params': 'Deploy Params(M)',
                        'paper_ap': 'mean and sample SD (ddof=1) of three explicitly supplied per-seed percent values; otherwise blank'},
        'ap_snapshot': snapshot.name if snapshot.exists() or ap_csv is not None else None,
        'ap_source': str(Path(ap_csv).resolve()) if ap_csv is not None else None,
    }
    previous_protocol = output_dir / 'efficiency_protocol.json'
    if ap_csv is None and snapshot.exists() and previous_protocol.exists():
        previous = json.loads(previous_protocol.read_text(encoding='utf-8'))
        metadata['ap_source'] = previous.get('ap_source')
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / 'efficiency_by_seed.csv', BY_SEED_COLUMNS, by_seed)
    _write_csv(output_dir / 'efficiency_summary.csv', SUMMARY_COLUMNS, summary)
    _write_csv(output_dir / 'paper_efficiency_table.csv', PAPER_OUTPUT_COLUMNS, paper)
    if ap_csv is not None:
        _write_csv(snapshot, ('Model', 'Seed', *AP_COLUMNS, 'Algorithm suite'), ap_rows)
    temporary = previous_protocol.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(previous_protocol)
    return {'output_dir': str(output_dir.resolve()), 'status': metadata['status'],
            'measured_model_seed_count': len(records),
            'complete_models': [row['Model'] for row in summary if row['status'] == 'COMPLETE']}
