"""Aggregate saved AP evidence without opening images, annotations or checkpoints.

An output directory belongs to exactly one split. All AP/AR numbers are in
percent units; an unavailable COCO statistic is null, never a fabricated zero.
"""
import csv
import json
import math
from pathlib import Path
import re
import statistics

from .efficiency_protocol import ALGORITHM_SUITE, AP_COLUMNS, MODELS, SEEDS


REQUIRED_AP = (AP_COLUMNS[0], *AP_COLUMNS[-4:])
IMAGE_GROUPS = ('mixed', 'natural', 'artificial', 'low_light', 'sodium_light')
BY_SEED_COLUMNS = ('Model', 'Seed', 'Split', 'Selected epoch',
                   'selected_epoch_zero_based', 'Status', *AP_COLUMNS, 'Algorithm suite')
SUMMARY_COLUMNS = ('Model', 'Split', 'Status', 'measured_seed_count', 'seeds',
                   *AP_COLUMNS,
                   *(column.replace(' (%)', ' sample SD (%)') for column in AP_COLUMNS))


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
        raise ValueError(f'{label} must be a SHA256 hexadecimal digest')


def validate_record(record, split=None):
    """Check one measured record, without requiring its source files locally."""
    _object(record, 'AP record')
    if record.get('schema') != 'paper_v2_ap_record_v1':
        raise ValueError('Unknown AP record schema')
    if record.get('status') not in ('COMPLETE', 'INCOMPLETE'):
        raise ValueError('AP record status must be COMPLETE or INCOMPLETE')
    if record.get('model') not in MODELS:
        raise ValueError('Unknown model in AP record')
    if type(record.get('seed')) is not int or record['seed'] not in SEEDS:
        raise ValueError('Unknown seed in AP record')
    if record.get('split') not in ('test', 'validation'):
        raise ValueError('AP record split must be test or validation')
    if split is not None and record['split'] != split:
        raise ValueError(f'Mixed evaluation splits: expected {split}, found {record["split"]}')
    selected = record.get('selected_epoch_zero_based')
    _integer(selected, 'selected_epoch_zero_based')
    if selected >= 120:
        raise ValueError('selected_epoch_zero_based must be within the 120-epoch recipe')

    metrics = _object(record.get('metrics'), 'metrics')
    if not set(AP_COLUMNS).issubset(metrics):
        raise ValueError('metrics must contain every AP_COLUMNS key; use null for unavailable values')
    for column in AP_COLUMNS:
        value = metrics[column]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError(f'{column} must be a finite percentage in [0, 100] or null')
    if record['status'] == 'COMPLETE' and any(metrics[column] is None for column in REQUIRED_AP):
        raise ValueError('COMPLETE requires Mixed AP and all four illumination AP values')

    counts = _object(record.get('image_counts'), 'image_counts')
    for group in IMAGE_GROUPS:
        _integer(counts.get(group), f'image_counts.{group}')
    if sum(counts[group] for group in IMAGE_GROUPS[1:]) > counts['mixed']:
        raise ValueError('Illumination image counts cannot exceed the mixed image count')
    for group, columns in [('mixed', AP_COLUMNS[:12]), *zip(IMAGE_GROUPS[1:],
                                                         [(key,) for key in AP_COLUMNS[-4:]])]:
        if counts[group] == 0 and any(metrics[column] is not None for column in columns):
            raise ValueError(f'Empty image group {group} cannot contain measured AP values')

    checkpoint = _object(record.get('checkpoint'), 'checkpoint')
    _text(checkpoint.get('path'), 'checkpoint.path')
    _integer(checkpoint.get('size_bytes'), 'checkpoint.size_bytes', minimum=1)
    _integer(checkpoint.get('mtime_ns'), 'checkpoint.mtime_ns')
    if checkpoint.get('model_state') != 'ema':
        raise ValueError('AP records must evaluate the selected EMA checkpoint state')
    dataset = _object(record.get('dataset'), 'dataset')
    _text(dataset.get('annotation_file'), 'dataset.annotation_file')
    _digest(dataset.get('annotation_sha256'), 'dataset.annotation_sha256')
    if 'illumination_map_sha256' not in dataset:
        raise ValueError('dataset.illumination_map_sha256 is required; use null without an external map')
    if dataset['illumination_map_sha256'] is not None:
        _digest(dataset['illumination_map_sha256'], 'dataset.illumination_map_sha256')
    evaluation = _object(record.get('evaluation'), 'evaluation')
    if (evaluation.get('precision') != 'FP32' or evaluation.get('autocast') is not False
            or evaluation.get('tf32') is not False):
        raise ValueError('AP evaluation requires FP32, autocast=false and TF32=false')
    # Extra evaluator settings are compared verbatim across all included runs.
    json.dumps(evaluation, sort_keys=True, allow_nan=False)
    tool = _object(record.get('tool'), 'tool')
    if tool.get('name') != 'faster-coco-eval':
        raise ValueError('AP evaluation tool must be faster-coco-eval')
    _text(tool.get('version'), 'tool.version')
    return record


def _signature(record):
    dataset = record['dataset']
    return (dataset['annotation_sha256'].lower(),
            (dataset['illumination_map_sha256'] or '').lower(),
            tuple(record['image_counts'][group] for group in IMAGE_GROUPS),
            json.dumps(record['evaluation'], sort_keys=True, allow_nan=False),
            record['tool']['name'], record['tool']['version'])


def validate_campaign(output_dir, split):
    """Bind existing reports before exclusions or retry ledgers can be written.

    A failed cache may be skipped by the ordinary recovery path, so checking
    only included raw records is insufficient to protect an older campaign.
    """
    output_dir = Path(output_dir)
    markers = [output_dir / name for name in ('ap_protocol.json', 'evaluation_attempts.json')]
    existing = [path for path in markers if path.exists()]
    for path in existing:
        metadata = _object(json.loads(path.read_text(encoding='utf-8')), path.name)
        if metadata.get('split') != split:
            label = 'Evaluation attempt split' if path.name == 'evaluation_attempts.json' else 'AP protocol split'
            raise ValueError(f'{label} does not match the requested AP split')
    if not existing:
        # Compatible raw evidence can be summarized regardless of its training
        # suite label. CSV-only exports without binding must not be overwritten.
        raw_paths = list((output_dir / 'raw').rglob('metrics.json'))
        for path in raw_paths:
            record = _object(json.loads(path.read_text(encoding='utf-8')), str(path))
            if record.get('schema') != 'paper_v2_ap_record_v1':
                raise ValueError(f'Unknown AP record schema: {path}')
            if record.get('split') != split:
                raise ValueError(f'Mixed evaluation splits: expected {split}, found {record.get("split")}')
        if not raw_paths and any((output_dir / name).exists() for name in ('ap_by_seed.csv', 'ap_summary.csv')):
            raise ValueError(f'Historical or unbound AP tables in {output_dir}; use a new --ap-dir')


def load_exclusions(output_dir, split):
    """Keep failed/unverified runs out of reports, including offline summaries.

    ``excluded_runs`` retains unresolved failures across one-seed retries. The
    current attempt's rows supersede it, so a successful retry clears its key.
    """
    validate_campaign(output_dir, split)
    path = Path(output_dir) / 'evaluation_attempts.json'
    if not path.exists():
        return {}
    ledger = _object(json.loads(path.read_text(encoding='utf-8')), 'evaluation attempts')
    if ledger.get('split') != split:
        raise ValueError('Evaluation attempt split does not match the requested AP split')
    exclusions = {}
    for field in ('excluded_runs', 'runs'):
        rows = ledger.get(field, [])
        if not isinstance(rows, list):
            raise ValueError(f'evaluation attempts.{field} must be a list')
        seen = set()
        for row in rows:
            _object(row, f'evaluation attempts.{field} row')
            if (row.get('model') not in MODELS or type(row.get('seed')) is not int
                    or row['seed'] not in SEEDS):
                raise ValueError('Unknown model/seed in evaluation attempts')
            key = (row['model'], row['seed'])
            if key in seen:
                raise ValueError(f'Duplicate model/seed in evaluation attempts.{field}: {key}')
            seen.add(key)
            status = row.get('status')
            if status in ('COMPLETE', 'INCOMPLETE'):
                exclusions.pop(key, None)
            elif status in ('FAIL', 'NOT RUN', 'RUNNING'):
                exclusions[key] = {name: row[name] for name in ('model', 'seed', 'status')}
                exclusions[key]['reason'] = row.get('reason', 'Latest evaluation attempt did not finish successfully')
            else:
                raise ValueError('Unknown status in evaluation attempts')
    return exclusions


def _load_records(output_dir, split, models, exclusions):
    records, seen = [], set()
    excluded_paths = {output_dir / 'raw' / model / f'seed_{seed}' / 'metrics.json'
                      for model, seed in exclusions}
    for path in sorted((output_dir / 'raw').rglob('metrics.json')):
        # A rejected cache can itself be malformed. Skip by its canonical path
        # before opening it; preserve the file for diagnosis or explicit repair.
        if path in excluded_paths:
            continue
        try:
            record = validate_record(json.loads(path.read_text(encoding='utf-8')), split=split)
            key = (record['model'], record['seed'])
            if key in seen:
                raise ValueError(f'Duplicate model/seed AP record: {key}')
            expected = output_dir / 'raw' / key[0] / f'seed_{key[1]}' / 'metrics.json'
            if path != expected:
                raise ValueError(f'Nonstandard raw AP record location: expected {expected}')
            seen.add(key)
            if record['model'] in models:
                records.append(record)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{path}: {error}') from error
    if records and any(_signature(record) != _signature(records[0]) for record in records):
        raise ValueError('Inconsistent dataset digest, illumination map, image counts, '
                         'evaluation settings or evaluator versions across AP records')
    return sorted(records, key=lambda record: (MODELS.index(record['model']), SEEDS.index(record['seed'])))


def _write_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(output_dir, split='test', models=None):
    """Write per-seed AP and three-seed statistics for an explicit split directory.

    ``output_dir`` is ``ap/<split>``, not its parent. ``models=None`` includes
    all nine configurations; a requested subset has its own completion status.
    Validation errors leave any existing summary files unchanged.
    """
    if split not in ('test', 'validation'):
        raise ValueError('split must be test or validation')
    requested = MODELS if models is None else tuple(models)
    if not requested or len(set(requested)) != len(requested) or any(model not in MODELS for model in requested):
        raise ValueError('models must be a nonempty selection of distinct known model names')
    models = tuple(model for model in MODELS if model in requested)
    output_dir = Path(output_dir)
    exclusions = load_exclusions(output_dir, split)
    records = _load_records(output_dir, split, models, exclusions)
    excluded = [row for (model, seed), row in exclusions.items() if model in models]
    by_seed = [dict({'Model': record['model'], 'Seed': record['seed'], 'Split': split,
                     'Selected epoch': record['selected_epoch_zero_based'] + 1,
                     'selected_epoch_zero_based': record['selected_epoch_zero_based'],
                     'Status': record['status'], 'Algorithm suite': ALGORITHM_SUITE},
                    **{column: record['metrics'][column] for column in AP_COLUMNS})
               for record in records]
    summary, model_status = [], {}
    for model in models:
        group = [record for record in records if record['model'] == model]
        missing_seeds = [seed for seed in SEEDS if all(record['seed'] != seed for record in group)]
        failed_seeds = [seed for seed in SEEDS if exclusions.get((model, seed), {}).get('status') == 'FAIL']
        complete = not missing_seeds and all(record['status'] == 'COMPLETE' for record in group)
        status = 'FAIL' if failed_seeds else 'COMPLETE' if complete else ('INCOMPLETE' if group else 'NOT RUN')
        row = {'Model': model, 'Split': split, 'Status': status, 'measured_seed_count': len(group),
               'seeds': ','.join(str(record['seed']) for record in group)}
        valid_counts = {}
        for column in AP_COLUMNS:
            values = [record['metrics'][column] for record in group if record['metrics'][column] is not None]
            valid_counts[column] = len(values)
            if len(values) == len(SEEDS):
                row[column] = statistics.mean(values)
                row[column.replace(' (%)', ' sample SD (%)')] = statistics.stdev(values)
        summary.append(row)
        model_status[model] = {'status': status, 'measured_seed_count': len(group),
                               'missing_seeds': missing_seeds, 'failed_seeds': failed_seeds,
                               'metric_valid_seed_count': valid_counts,
                               'incomplete_record_seeds': [record['seed'] for record in group
                                                           if record['status'] != 'COMPLETE'],
                               'missing_metrics_by_seed': {
                                   str(record['seed']): [column for column in AP_COLUMNS
                                                        if record['metrics'][column] is None]
                                   for record in group}}
    status = ('FAIL' if any(row['Status'] == 'FAIL' for row in summary)
              else 'COMPLETE' if all(row['Status'] == 'COMPLETE' for row in summary)
              else ('INCOMPLETE' if records else 'NOT RUN'))
    metadata = {
        'schema': 'paper_v2_ap_summary_v1', 'algorithm_suite': ALGORITHM_SUITE, 'split': split, 'status': status,
        'output_dir': str(output_dir.resolve()), 'models': list(models), 'seeds': list(SEEDS),
        'measured_model_seed_count': len(records), 'expected_model_seed_count': len(models) * len(SEEDS),
        'complete_models': [row['Model'] for row in summary if row['Status'] == 'COMPLETE'],
        'excluded_runs': excluded,
        'failed_model_seed_count': sum(row['status'] == 'FAIL' for row in excluded),
        'model_status': model_status,
        'required_ap_columns': list(REQUIRED_AP), 'metric_columns': list(AP_COLUMNS),
        'illumination_columns': dict(zip(AP_COLUMNS[-4:], ('自然光', '人工光', '微弱光', '钠黄光'))),
        'units': 'percent',
        'aggregation': 'Arithmetic mean and sample standard deviation (ddof=1) across seeds 0, 42, 3407; '
                       'each metric requires all three valid values; missing values remain blank, never zero.',
        'selected_epoch': 'Selected epoch is one-based; selected_epoch_zero_based preserves the checkpoint index.',
        'checkpoint_selection': 'Evaluates the EMA in each supplied best.pth; Test does not select checkpoints. '
                                'Existing-checkpoint runs report selection verification in each source run_provenance; '
                                'missing training metadata does not establish EMA FP32 Mixed Validation AP selection.',
        'dataset': records[0]['dataset'] if records else None,
        'evaluation': records[0]['evaluation'] if records else None,
        'tool': records[0]['tool'] if records else None,
        'raw_record_paths': [f"raw/{record['model']}/seed_{record['seed']}/metrics.json" for record in records],
        'sources': [{'model': record['model'], 'seed': record['seed'], 'checkpoint': record['checkpoint'],
                     'recorded_algorithm_suite': record.get('algorithm_suite'),
                     'selected_epoch_zero_based': record['selected_epoch_zero_based'],
                     'image_counts': record['image_counts'],
                     'run_provenance': record.get('run_provenance')} for record in records],
    }
    # Serialize before any report writes, so invalid provenance cannot leave a
    # new CSV beside an old protocol manifest.
    payload = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    # Only summarized JSON evidence is read. No data/model access or inference.
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / 'ap_by_seed.csv', BY_SEED_COLUMNS, by_seed)
    _write_csv(output_dir / 'ap_summary.csv', SUMMARY_COLUMNS, summary)
    target = output_dir / 'ap_protocol.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(target)
    return metadata
