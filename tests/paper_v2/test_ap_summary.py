"""AP aggregation is split-aware and never invents missing seed/group results."""
import csv
import json
import math
import statistics

import pytest

from tools.paper_v2.ap_summary import REQUIRED_AP, summarize, validate_record
from tools.paper_v2.efficiency_protocol import AP_COLUMNS, MODELS, SEEDS
from tools.paper_v2.efficiency_summary import _read_ap


def make_record(seed=0, model='baseline_native', value=40.0, split='test'):
    return {
        'schema': 'paper_v2_ap_record_v1', 'status': 'COMPLETE', 'model': model,
        'seed': seed, 'split': split, 'selected_epoch_zero_based': 76,
        'metrics': {column: value + index for index, column in enumerate(AP_COLUMNS)},
        'image_counts': {'mixed': 40, 'natural': 10, 'artificial': 10,
                         'low_light': 10, 'sodium_light': 10},
        'checkpoint': {'path': f'/remote/{model}/seed_{seed}/best.pth',
                       'size_bytes': 1000, 'mtime_ns': 123456789, 'model_state': 'ema'},
        'dataset': {'annotation_file': '/remote/instances_test.json',
                    'annotation_sha256': 'a' * 64, 'illumination_map_sha256': None},
        'evaluation': {'precision': 'FP32', 'autocast': False, 'tf32': False,
                       'backend': 'PyTorch eager eval', 'input_size': [640, 640],
                       'num_top_queries': 300, 'iou_type': 'bbox', 'max_dets': [1, 10, 100]},
        'tool': {'name': 'faster-coco-eval', 'version': '1.6.7'},
    }


def write_record(root, record, relative=None):
    path = root / (relative or f"raw/{record['model']}/seed_{record['seed']}/metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


def read_csv(root, filename):
    with (root / filename).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def test_three_seed_mean_sample_sd_and_efficiency_join(tmp_path):
    values = (20.0, 30.0, 60.0)
    for seed, value in zip(SEEDS, values):
        write_record(tmp_path, make_record(seed, value=value))
    result = summarize(tmp_path, models=['baseline_native'])
    assert result['status'] == 'COMPLETE'
    assert result['complete_models'] == ['baseline_native']
    row = read_csv(tmp_path, 'ap_summary.csv')[0]
    assert row['Status'] == 'COMPLETE'
    assert float(row['Mixed AP (%)']) == statistics.mean(values)
    assert float(row['Mixed AP sample SD (%)']) == statistics.stdev(values)
    assert float(row['Mixed AP sample SD (%)']) != statistics.pstdev(values)
    assert float(row['Natural AP (%)']) == statistics.mean(value + 12 for value in values)
    seed_rows = read_csv(tmp_path, 'ap_by_seed.csv')
    assert [int(row['Seed']) for row in seed_rows] == list(SEEDS)
    assert all(row['Selected epoch'] == '77' and row['selected_epoch_zero_based'] == '76'
               for row in seed_rows)
    # This is the exact existing efficiency import path, not a second CSV parser.
    imported = _read_ap(tmp_path / 'ap_by_seed.csv')
    assert [row['Mixed AP (%)'] for row in imported] == list(values)
    metadata = json.loads((tmp_path / 'ap_protocol.json').read_text(encoding='utf-8'))
    assert metadata == result
    assert len(metadata['sources']) == 3
    assert metadata['sources'][0]['checkpoint']['path'].startswith('/remote/')


def test_real_coco_evaluator_to_three_seed_summary_to_paper_table(tmp_path):
    """Exercise the full saved-results path using actual COCO matching on CPU."""
    import hashlib
    from importlib.metadata import version

    from faster_coco_eval import COCO
    from tools.paper_v2.ap_metrics import evaluate_predictions
    from tools.paper_v2.efficiency_protocol import PAPER_COLUMNS
    from tools.paper_v2.efficiency_summary import summarize as summarize_efficiency

    dataset = {
        'images': [{'id': index, 'file_name': f'synthetic_{index}.jpg',
                    'width': 640, 'height': 640, 'light_condition': light}
                   for index, light in enumerate(('自然光', '人工光', '微弱光', '钠黄光'), 1)],
        'categories': [{'id': 1, 'name': 'tomato'}],
        'annotations': [{'id': index, 'image_id': index, 'category_id': 1,
                         'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}
                        for index in range(1, 5)],
    }
    digest = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode('utf-8')).hexdigest()
    ap_dir = tmp_path / 'ap' / 'test'
    for seed in SEEDS:
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        predictions = [{'image_id': index, 'category_id': 1,
                        'bbox': [10, 10, 20, 20], 'score': 0.9}
                       for index in range(1, 5)]
        evaluated = evaluate_predictions(coco, predictions)
        record = make_record(seed)
        record.update(evaluated)
        record['dataset']['annotation_sha256'] = digest
        record['tool']['version'] = version('faster-coco-eval')
        write_record(ap_dir, record)

    assert summarize(ap_dir, models=['baseline_native'])['status'] == 'COMPLETE'
    efficiency_dir = tmp_path / 'efficiency'
    summarize_efficiency(efficiency_dir, ap_csv=ap_dir / 'ap_by_seed.csv')
    paper = read_csv(efficiency_dir, 'paper_efficiency_table.csv')[0]
    assert tuple(paper)[:len(PAPER_COLUMNS)] == PAPER_COLUMNS
    assert paper['Model'] == 'Baseline'
    for column in REQUIRED_AP:
        assert float(paper[column]) == pytest.approx(100.)
        assert float(paper[column.replace(' (%)', ' sample SD (%)')]) == pytest.approx(0.)
    # COCO has no eligible medium/large objects. Nulls must survive both CSV hops.
    assert paper['Mixed AP-M (%)'] == paper['Mixed AP-L (%)'] == ''
    assert paper['Mixed AP-M sample SD (%)'] == paper['Mixed AP-L sample SD (%)'] == ''
    # AP import must not invent separate, unmeasured efficiency evidence.
    assert paper['Latency(ms)'] == ''


@pytest.mark.parametrize('count', [0, 1, 2])
def test_missing_seed_retains_individual_results_but_no_mean(tmp_path, count):
    for seed in SEEDS[:count]:
        write_record(tmp_path, make_record(seed))
    result = summarize(tmp_path, models=['baseline_native'])
    assert result['status'] == ('INCOMPLETE' if count else 'NOT RUN')
    rows = read_csv(tmp_path, 'ap_by_seed.csv')
    assert len(rows) == count
    row = read_csv(tmp_path, 'ap_summary.csv')[0]
    assert all(row[column] == '' for column in AP_COLUMNS)
    assert result['model_status']['baseline_native']['missing_seeds'] == list(SEEDS[count:])


def test_missing_light_does_not_suppress_available_three_seed_mixed_ap(tmp_path):
    for seed in SEEDS:
        record = make_record(seed)
        if seed == 3407:
            record['metrics']['Sodium-light AP (%)'] = None
            record['status'] = 'INCOMPLETE'
        write_record(tmp_path, record)
    result = summarize(tmp_path, models=['baseline_native'])
    row = read_csv(tmp_path, 'ap_summary.csv')[0]
    assert row['Status'] == 'INCOMPLETE'
    assert float(row['Mixed AP (%)']) == 40.0
    assert float(row['Mixed AP sample SD (%)']) == 0.0
    assert row['Sodium-light AP (%)'] == row['Sodium-light AP sample SD (%)'] == ''
    assert result['model_status']['baseline_native']['metric_valid_seed_count']['Sodium-light AP (%)'] == 2
    assert read_csv(tmp_path, 'ap_by_seed.csv')[-1]['Mixed AP (%)'] == '40.0'
    assert read_csv(tmp_path, 'ap_by_seed.csv')[-1]['Sodium-light AP (%)'] == ''


def test_absent_coco_size_bin_is_null_without_making_five_ap_incomplete(tmp_path):
    for seed in SEEDS:
        record = make_record(seed)
        record['metrics']['Mixed AP-S (%)'] = None
        write_record(tmp_path, record)
    result = summarize(tmp_path, models=['baseline_native'])
    assert result['status'] == 'COMPLETE'
    assert read_csv(tmp_path, 'ap_summary.csv')[0]['Mixed AP-S (%)'] == ''


def test_default_includes_nine_models_and_requested_scope_excludes_other_raw(tmp_path):
    for model in ('baseline_native', 'gr_vq'):
        for seed in SEEDS:
            write_record(tmp_path, make_record(seed, model=model))
    all_result = summarize(tmp_path)
    assert all_result['status'] == 'INCOMPLETE'
    assert [row['Model'] for row in read_csv(tmp_path, 'ap_summary.csv')] == list(MODELS)
    result = summarize(tmp_path, models=['gr_vq'])
    assert result['status'] == 'COMPLETE'
    assert result['models'] == ['gr_vq']
    assert all(row['Model'] == 'gr_vq' for row in read_csv(tmp_path, 'ap_by_seed.csv'))
    assert (tmp_path / 'raw/baseline_native/seed_0/metrics.json').exists()


def test_validation_split_is_separate_from_test(tmp_path):
    write_record(tmp_path, make_record(split='validation'))
    with pytest.raises(ValueError, match='Mixed evaluation splits'):
        summarize(tmp_path, split='test')
    assert not (tmp_path / 'ap_summary.csv').exists()
    summarize(tmp_path, split='validation')
    assert read_csv(tmp_path, 'ap_by_seed.csv')[0]['Split'] == 'validation'


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r.update(seed=123), 'Unknown seed'),
    (lambda r: r.update(seed=False), 'Unknown seed'),
    (lambda r: r.update(model='unknown'), 'Unknown model'),
    (lambda r: r.update(selected_epoch_zero_based=120), '120-epoch'),
    (lambda r: r.update(selected_epoch_zero_based=True), 'integer'),
    (lambda r: r['metrics'].update({'Mixed AP (%)': math.nan}), 'finite percentage'),
    (lambda r: r['metrics'].update({'Mixed AP (%)': math.inf}), 'finite percentage'),
    (lambda r: r['metrics'].update({'Mixed AP (%)': -1}), 'finite percentage'),
    (lambda r: r['metrics'].update({'Mixed AP (%)': 100.1}), 'finite percentage'),
    (lambda r: r['metrics'].update({'Mixed AP (%)': True}), 'finite percentage'),
    (lambda r: r['metrics'].update({'Mixed AP (%)': None}), 'COMPLETE requires'),
    (lambda r: r['metrics'].pop('Mixed AP-S (%)'), 'every AP_COLUMNS'),
    (lambda r: r['checkpoint'].update(model_state='model'), 'EMA'),
    (lambda r: r['checkpoint'].update(size_bytes=0), 'integer'),
    (lambda r: r['dataset'].update(annotation_sha256='abc'), 'SHA256'),
    (lambda r: r['dataset'].pop('illumination_map_sha256'), 'required'),
    (lambda r: r['evaluation'].update(autocast=True), 'FP32'),
    (lambda r: r['evaluation'].update(tf32=0), 'FP32'),
    (lambda r: r['evaluation'].update(precision='FP16'), 'FP32'),
    (lambda r: r['tool'].update(name='pycocotools'), 'faster-coco-eval'),
    (lambda r: r['image_counts'].update(mixed=1), 'cannot exceed'),
    (lambda r: r['image_counts'].update(natural=0), 'Empty image group'),
])
def test_invalid_evidence_is_rejected_before_summary_writes(tmp_path, mutation, match):
    record = make_record()
    mutation(record)
    write_record(tmp_path, record)
    (tmp_path / 'ap_summary.csv').write_text('previous report', encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        summarize(tmp_path)
    assert (tmp_path / 'ap_summary.csv').read_text(encoding='utf-8') == 'previous report'
    assert not (tmp_path / 'ap_protocol.json').exists()


@pytest.mark.parametrize('mutation', [
    lambda r: r['dataset'].update(annotation_sha256='b' * 64),
    lambda r: r['dataset'].update(illumination_map_sha256='b' * 64),
    lambda r: r['image_counts'].update(mixed=41),
    lambda r: r['evaluation'].update(num_top_queries=100),
    lambda r: r['tool'].update(version='1.7.0'),
])
def test_mismatched_dataset_or_evaluation_cannot_be_averaged(tmp_path, mutation):
    write_record(tmp_path, make_record(0))
    record = make_record(42)
    mutation(record)
    write_record(tmp_path, record)
    with pytest.raises(ValueError, match='Inconsistent dataset digest'):
        summarize(tmp_path)


@pytest.mark.parametrize('relative', ['raw/backup/metrics.json',
                                      'raw/baseline_native/seed_42/metrics.json'])
def test_duplicate_or_noncanonical_locations_rejected(tmp_path, relative):
    record = make_record()
    write_record(tmp_path, record)
    write_record(tmp_path, record, relative=relative)
    with pytest.raises(ValueError, match='Duplicate|Nonstandard'):
        summarize(tmp_path)


def test_empty_light_group_is_incomplete_without_discarding_other_metrics(tmp_path):
    record = make_record()
    record['image_counts']['natural'] = 0
    record['metrics']['Natural AP (%)'] = None
    record['status'] = 'INCOMPLETE'
    assert validate_record(record) is record
    write_record(tmp_path, record)
    result = summarize(tmp_path, models=['baseline_native'])
    assert result['status'] == 'INCOMPLETE'
    assert result['model_status']['baseline_native']['missing_metrics_by_seed']['0'] == ['Natural AP (%)']


@pytest.mark.parametrize('models', [[], ['unknown'], ['gr_vq', 'gr_vq']])
def test_invalid_model_scope_rejected(tmp_path, models):
    with pytest.raises(ValueError, match='models must'):
        summarize(tmp_path, models=models)
