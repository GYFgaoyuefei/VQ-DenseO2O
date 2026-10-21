"""Join validated Test AP and fixed-protocol efficiency without running inference.

The first 23 columns follow the paper exactly. Numeric per-seed/summary exports
retain machine-readable values; the compact paper table renders mean +/- sample
SD and explicitly labels partial FLOPs. Missing evidence is never imputed.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from . import ap_summary, efficiency_summary
from .efficiency_protocol import ALGORITHM_SUITE, AP_COLUMNS, MODELS, PAPER_COLUMNS, SCOPES, SEEDS, protocol


AP_SD_COLUMNS = tuple(column.replace(' (%)', ' sample SD (%)') for column in AP_COLUMNS)
EFFICIENCY_DETAILS = (
    'GFLOPs status', 'Full Params(M)', 'Backend', 'Precision', 'Peak GPU memory(bytes)',
    'GFLOPs tool', 'GFLOPs tool version', 'GFLOPs coverage notes',
)
SCOPE_DETAILS = tuple(f'{scope}_{metric}' for scope in SCOPES
                      for metric in ('latency_ms', 'fps', 'p50_ms', 'p95_ms', 'peak_gpu_memory_bytes'))
BY_SEED_COLUMNS = (*PAPER_COLUMNS, 'Seed', 'Selected epoch', 'selected_epoch_zero_based',
                   'AP status', 'Efficiency status', 'Configuration source', 'Selection verification',
                   *EFFICIENCY_DETAILS, *SCOPE_DETAILS)
SUMMARY_COLUMNS = (*PAPER_COLUMNS, 'AP status', 'Efficiency status',
                   'AP measured seed count', 'Efficiency measured seed count', 'Selected epochs',
                   'Latency sample SD(ms)', *AP_SD_COLUMNS, *EFFICIENCY_DETAILS, *SCOPE_DETAILS)
CHECKPOINT_KEYS = ('path', 'size_bytes', 'mtime_ns', 'model_state')


def _read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _numeric(value):
    # Values come from freshly rebuilt, validated AP tables, never arbitrary
    # caller CSV. Keep the unavailable COCO convention distinct from AP=0.
    return '' if value in ('', None) else float(value)


def _pair(row):
    return row['model'], row['seed']


def _checkpoint_matches(first, second):
    a, b = first['checkpoint'], second['checkpoint']
    return (all(key in a and key in b and a[key] == b[key] for key in CHECKPOINT_KEYS)
            and first['selected_epoch_zero_based'] == second['selected_epoch_zero_based'])


def _write_json(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)


def _paper_row(row):
    result = {column: row.get(column, '') for column in PAPER_COLUMNS}
    for column, sd_column in [('Latency(ms)', 'Latency sample SD(ms)'), *zip(AP_COLUMNS, AP_SD_COLUMNS)]:
        mean, sd = row.get(column), row.get(sd_column)
        result[column] = f'{mean:.3f} ± {sd:.3f}' if mean not in ('', None) and sd not in ('', None) else ''
    for column in ('Params(M)', 'FPS'):
        value = row.get(column)
        result[column] = f'{value:.3f}' if value not in ('', None) else ''
    gflops = row.get('GFLOPs')
    result['GFLOPs'] = f'{gflops:.3f} (PARTIAL)' if gflops not in ('', None) else ''
    return result


def export_report(ap_dir, efficiency_dir, output_dir, *, excluded_efficiency=(), errors=()):
    """Export all nine models x three seeds from saved, validated evidence.

    ``excluded_efficiency`` is an iterable of ``(model, seed)`` pairs rejected
    by the caller's run/checkpoint checks. ``errors`` contains prior campaign
    error strings. An invalid efficiency campaign leaves efficiency blank but
    still publishes valid AP. No model, checkpoint or dataset is opened here.
    """
    ap_dir, efficiency_dir, output_dir = map(Path, (ap_dir, efficiency_dir, output_dir))
    problems = [str(error) for error in errors]
    excluded = set(excluded_efficiency)
    if any(not isinstance(key, tuple) or len(key) != 2 or key[0] not in MODELS
           or type(key[1]) is not int or key[1] not in SEEDS for key in excluded):
        raise ValueError('excluded_efficiency requires registered (model, seed) pairs')

    ap_metadata, ap_by_key, ap_summary_by_model, ap_sources = None, {}, {}, {}
    ap_failed = False
    try:
        ap_metadata = ap_summary.summarize(ap_dir, split='test', models=MODELS)
        ap_by_key = {(row['Model'], int(row['Seed'])): row
                     for row in _read_csv(ap_dir / 'ap_by_seed.csv')}
        ap_summary_by_model = {row['Model']: row for row in _read_csv(ap_dir / 'ap_summary.csv')}
        ap_sources = {_pair(source): source for source in ap_metadata['sources']}
        for row in ap_metadata['excluded_runs']:
            if row['status'] == 'FAIL':
                problems.append(f"AP {row['model']} seed={row['seed']}: {row.get('reason', 'evaluation failed')}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Do not fall back to a stale CSV after raw-evidence validation fails.
        ap_failed = True
        ap_metadata, ap_by_key, ap_summary_by_model, ap_sources = None, {}, {}, {}
        problems.append(f'AP summary validation failed: {type(error).__name__}: {error}')

    efficiency_records, efficiency_failed = [], False
    try:
        efficiency_summary.check_campaign(efficiency_dir)
        efficiency_records = efficiency_summary._load_records(efficiency_dir)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        efficiency_failed = True
        problems.append(f'Efficiency campaign validation failed: {type(error).__name__}: {error}')

    accepted_efficiency, rejected_efficiency = [], []
    for record in efficiency_records:
        key = _pair(record)
        if key in excluded:
            rejected_efficiency.append({'model': key[0], 'seed': key[1],
                                        'reason': 'Excluded by campaign run/checkpoint validation'})
            continue
        if key in ap_sources and not _checkpoint_matches(record, ap_sources[key]):
            reason = 'Efficiency checkpoint identity or selected epoch differs from Test AP'
            problems.append(f'{key[0]} seed={key[1]}: {reason}')
            excluded.add(key)
            rejected_efficiency.append({'model': key[0], 'seed': key[1], 'reason': reason})
            continue
        accepted_efficiency.append(record)
    efficiency_by_key = {_pair(record): efficiency_summary._seed_row(record) for record in accepted_efficiency}
    efficiency_sources = {_pair(record): record for record in accepted_efficiency}
    ap_exclusions = ({(row['model'], row['seed']): row for row in ap_metadata['excluded_runs']}
                     if ap_metadata else {})

    by_seed, summary = [], []
    for model in MODELS:
        for seed in SEEDS:
            key = (model, seed)
            ap, efficiency = ap_by_key.get(key), efficiency_by_key.get(key)
            row = dict.fromkeys(BY_SEED_COLUMNS, '')
            row.update(Model=model, Epoch=120, Seed=seed)
            row['AP status'] = (ap['Status'] if ap else 'FAIL' if ap_failed else
                                ap_exclusions.get(key, {}).get('status', 'NOT RUN'))
            row['Efficiency status'] = ('MEASURED' if efficiency else 'FAIL'
                                        if efficiency_failed or key in excluded else 'NOT RUN')
            if ap:
                row.update({column: _numeric(ap.get(column)) for column in AP_COLUMNS})
                row['Selected epoch'] = int(ap['Selected epoch'])
                row['selected_epoch_zero_based'] = int(ap['selected_epoch_zero_based'])
            elif efficiency:
                row['selected_epoch_zero_based'] = efficiency['selected_epoch_zero_based']
                row['Selected epoch'] = efficiency['selected_epoch_zero_based'] + 1
            if efficiency:
                row['Params(M)'] = efficiency['Deploy Params(M)']
                row.update({column: efficiency.get(column, '') for column in
                            ('GFLOPs', 'Latency(ms)', 'FPS', 'Hardware', *EFFICIENCY_DETAILS, *SCOPE_DETAILS)})
            source = ap_sources.get(key) or efficiency_sources.get(key) or {}
            provenance = source.get('run_provenance') or {}
            row['Configuration source'] = provenance.get('config_source', '')
            row['Selection verification'] = provenance.get('selection_verification', 'UNRECORDED' if source else '')
            by_seed.append(row)

        ap = ap_summary_by_model.get(model)
        seed_rows = [row for row in by_seed if row['Model'] == model]
        efficiency_group = [efficiency_by_key[(model, seed)] for seed in SEEDS if (model, seed) in efficiency_by_key]
        efficiency = efficiency_summary._summary_row(model, efficiency_group)
        row = dict.fromkeys(SUMMARY_COLUMNS, '')
        row.update(Model=model, Epoch=120)
        row['AP status'] = ap['Status'] if ap else 'FAIL' if ap_failed else 'NOT RUN'
        row['Efficiency status'] = ('FAIL' if any(item['Efficiency status'] == 'FAIL' for item in seed_rows)
                                    else efficiency['status'])
        row['AP measured seed count'] = int(ap['measured_seed_count']) if ap else 0
        row['Efficiency measured seed count'] = len(efficiency_group)
        row['Selected epochs'] = '; '.join(f"{item['Seed']}:{item['Selected epoch']}" for item in seed_rows
                                           if item['Selected epoch'] != '')
        if ap:
            row.update({column: _numeric(ap.get(column)) for column in (*AP_COLUMNS, *AP_SD_COLUMNS)})
        # The helper supplies the official hardware even for an empty group.
        # Never publish that expected value as if a GPU had been measured.
        if efficiency_group:
            row['Params(M)'] = efficiency.get('Deploy Params(M)', '')
            row.update({column: efficiency.get(column, '') for column in
                        ('GFLOPs', 'Latency(ms)', 'Latency sample SD(ms)', 'FPS', 'Hardware',
                         *EFFICIENCY_DETAILS, *SCOPE_DETAILS)})
        summary.append(row)

    problems = list(dict.fromkeys(problems))
    has_failure = problems or any(row['AP status'] == 'FAIL' or row['Efficiency status'] == 'FAIL' for row in by_seed)
    complete = all(row['AP status'] == 'COMPLETE' and row['Efficiency status'] == 'COMPLETE'
                   and row['GFLOPs'] not in ('', None) for row in summary)
    ap_status = ap_metadata['status'] if ap_metadata is not None else 'FAIL'
    efficiency_status = ('FAIL' if any(row['Efficiency status'] == 'FAIL' for row in by_seed)
                         else 'COMPLETE' if len(efficiency_by_key) == len(MODELS) * len(SEEDS)
                         and all(row['GFLOPs'] not in ('', None) for row in summary)
                         else 'INCOMPLETE' if efficiency_by_key else 'NOT RUN')
    status = ('FAIL' if has_failure else 'COMPLETE' if complete else
              'INCOMPLETE' if ap_by_key or efficiency_by_key else 'NOT RUN')
    metadata = {
        'schema': 'paper_v3_final_test_report_v1', 'algorithm_suite': ALGORITHM_SUITE,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'split': 'test', 'status': status,
        'ap_status': ap_status, 'efficiency_status': efficiency_status,
        'models': list(MODELS), 'seeds': list(SEEDS), 'paper_columns': list(PAPER_COLUMNS),
        'ap_directory': str(ap_dir.resolve()), 'efficiency_directory': str(efficiency_dir.resolve()),
        'output_directory': str(output_dir.resolve()),
        'expected_model_seed_count': len(MODELS) * len(SEEDS),
        'ap_record_count': len(ap_by_key), 'efficiency_record_count': len(efficiency_by_key),
        'complete_models': [row['Model'] for row in summary
                            if row['AP status'] == row['Efficiency status'] == 'COMPLETE'
                            and row['GFLOPs'] not in ('', None)],
        'model_status': {row['Model']: {'ap': row['AP status'], 'efficiency': row['Efficiency status'],
                                      'gflops': row['GFLOPs status']} for row in summary},
        'errors': problems, 'excluded_efficiency': [{'model': model, 'seed': seed} for model, seed in sorted(excluded)],
        'rejected_efficiency': rejected_efficiency,
        'ap_protocol': ap_metadata,
        'image_counts_by_run': [{'model': source['model'], 'seed': source['seed'],
                                **source['image_counts']} for source in ap_sources.values()],
        'efficiency_protocol': protocol(),
        'efficiency_sources': [{'model': record['model'], 'seed': record['seed'],
                                'checkpoint': record['checkpoint'],
                                'selected_epoch_zero_based': record['selected_epoch_zero_based'],
                                'hardware': record['hardware'], 'runtime': record['runtime'],
                                'flops': record['flops'],
                                'run_provenance': record.get('run_provenance')} for record in efficiency_sources.values()],
        'aggregation': {
            'epoch': 'Epoch=120 is the requested training budget, not proof of training completion '
                     'or the selected checkpoint epoch. Selected epoch is one-based.',
            'ap': 'Arithmetic mean and sample SD (ddof=1) of seeds 0,42,3407, per metric; fewer than three valid values stay blank.',
            'latency': protocol()['summary_latency'], 'fps': protocol()['fps'],
            'params': 'Deploy Params(M), all parameters after deploy conversion, including frozen parameters.',
            'gflops': 'calflops deploy detector forward only; partial operator coverage is explicitly labeled PARTIAL.',
            'missing': 'Missing values are blank, never zero; no measured efficiency evidence means blank Hardware.',
            'mixed': 'Full Test COCO evaluation, not the arithmetic mean of illumination AP.',
        },
        'illumination_columns': dict(zip(AP_COLUMNS[-4:], ('自然光', '人工光', '微弱光', '钠黄光'))),
        'files': ['test_results_by_seed.csv', 'test_results_summary.csv', 'paper_test_table.csv',
                  'final_test_protocol.json'],
    }
    # Validate serialization before replacing any final-report artifact.
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    efficiency_summary._write_csv(output_dir / 'test_results_by_seed.csv', BY_SEED_COLUMNS, by_seed)
    efficiency_summary._write_csv(output_dir / 'test_results_summary.csv', SUMMARY_COLUMNS, summary)
    efficiency_summary._write_csv(output_dir / 'paper_test_table.csv', PAPER_COLUMNS,
                                  [_paper_row(row) for row in summary])
    _write_json(output_dir / 'final_test_protocol.json', metadata)
    return {'status': status, 'ap_status': ap_status, 'efficiency_status': efficiency_status,
            'errors': problems, 'output_dir': str(output_dir.resolve()),
            'ap_record_count': len(ap_by_key), 'efficiency_record_count': len(efficiency_by_key),
            'complete_models': metadata['complete_models'], 'files': metadata['files']}
