"""Synthetic fixtures ONLY: never use the generated images as paper figures."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from PIL import Image, ImageDraw

from tools.paper_v2 import qualitative_fig8_data as data


MODEL_NAMES = {
    'B1': 'baseline_maskaware',
    'VQ': 'vq_dense_o2o',
    'LI': 'li_copyblend',
    'DC': 'dc_sta',
    'Full': 'vq_dense_o2o_li_copyblend_dc_sta',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def case_files(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    # IDs, file positions, and category positions are deliberately unrelated.
    records = [
        {'id': 91, 'file_name': 'z_first.png', 'width': 160, 'height': 100,
         'illumination': 'natural'},
        {'id': 7, 'file_name': 'a_second.png', 'width': 120, 'height': 150,
         'illumination': 'sodium_light'},
    ]
    for record, color in zip(records, ('#ddeeff', '#ffffcc')):
        image = Image.new('RGB', (record['width'], record['height']), color)
        ImageDraw.Draw(image).text((2, 2), 'SYNTHETIC TEST ONLY', fill='black')
        image.save(images / record['file_name'])
    categories = [
        {'id': 14, 'name': 'Ripe'},
        {'id': 3, 'name': 'Immature'},
        # Actual Test vocabulary; deliberately not category_id=2.
        {'id': 28, 'name': 'half'},
    ]
    annotation = tmp_path / 'instances_test.json'
    coco = {'images': records, 'categories': categories, 'annotations': [
        {'id': 901, 'image_id': 91, 'category_id': 14,
         'bbox': [20, 25, 30, 25], 'area': 750, 'iscrowd': 0},
        {'id': 902, 'image_id': 91, 'category_id': 3,
         'bbox': [80, 25, 30, 25], 'area': 750, 'iscrowd': 0},
        {'id': 701, 'image_id': 7, 'category_id': 28,
         'bbox': [30, 40, 25, 35], 'area': 875, 'iscrowd': 0},
    ]}
    write_json(annotation, coco)
    pred_root = tmp_path / 'raw'
    base = [
        {'image_id': 7, 'category_id': 28, 'bbox': [30, 40, 25, 35], 'score': .9},
        {'image_id': 91, 'category_id': 14, 'bbox': [20, 25, 30, 25], 'score': .93},
    ]
    digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
    for alias, model in MODEL_NAMES.items():
        predictions = copy.deepcopy(base)
        if alias in ('VQ', 'Full'):
            predictions.append({'image_id': 91, 'category_id': 3,
                                'bbox': [80, 25, 30, 25], 'score': .8})
        if alias == 'Full':
            predictions = [item for item in predictions if item['image_id'] != 7]
        folder = pred_root / model / 'seed_42'
        write_json(folder / 'predictions.json', predictions)
        write_json(folder / 'metrics.json', {
            'model': model, 'seed': 42, 'split': 'test',
            'dataset': {'annotation_sha256': digest},
        })
    return {'annotations': annotation, 'images': images, 'pred_root': pred_root,
            'seed': 42, 'expected_images': 2}


def test_explicit_image_ids_xywh_and_category_mapping(case_files):
    bundle = data.load_inputs(**case_files)
    assert list(bundle['images']) == [91, 7]
    assert bundle['images'][91]['file_name'] == 'z_first.png'
    assert bundle['images'][7]['file_name'] == 'a_second.png'
    assert bundle['categories'][3]['name'] == 'Immature'
    assert bundle['categories'][14]['name'] == 'Ripe'
    assert bundle['categories'][28]['name'] == 'Half-ripe'
    assert bundle['categories'][28]['original_name'] == 'half'
    assert bundle['categories'][28]['color'] == data.COLORS['Half-ripe']
    assert bundle['predictions']['B1'][91][0]['bbox'] == [20, 25, 30, 25]
    assert bundle['predictions']['B1'][7][0]['category_id'] == 28
    assert data.bbox_iou([20, 25, 30, 25], [20, 25, 30, 25]) == 1
    assert data.bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)
    assert data.bbox_iou([0, 0, 1, 1], [2, 2, 1, 1]) == 0


@pytest.mark.parametrize('name', ['Half-ripe', ' half ', 'HALF'])
def test_half_class_spelling_preserves_source_and_does_not_depend_on_id(case_files, name):
    coco = json.loads(case_files['annotations'].read_text(encoding='utf-8'))
    next(c for c in coco['categories'] if c['id'] == 28)['name'] = name
    write_json(case_files['annotations'], coco)
    digest = hashlib.sha256(case_files['annotations'].read_bytes()).hexdigest()
    for path in case_files['pred_root'].rglob('metrics.json'):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata['dataset']['annotation_sha256'] = digest
        write_json(path, metadata)
    originals = {p: p.read_bytes() for p in case_files['pred_root'].rglob('predictions.json')}
    bundle = data.load_inputs(**case_files)
    assert bundle['categories'][28] == {
        'name': 'Half-ripe', 'original_name': name, 'color': data.COLORS['Half-ripe']}
    assert all(p.read_bytes() == original for p, original in originals.items())


def test_one_to_one_class_mismatch_threshold_and_ignored_crowd():
    gt = [
        {'id': 1, 'category_id': 14, 'bbox': [0, 0, 10, 10], 'iscrowd': 0},
        {'id': 2, 'category_id': 3, 'bbox': [20, 0, 10, 10], 'iscrowd': 0},
        {'id': 3, 'category_id': 14, 'bbox': [40, 0, 10, 10], 'iscrowd': 1},
    ]
    pred = [
        {'_index': 0, 'category_id': 14, 'bbox': [0, 0, 10, 10], 'score': .9},
        {'_index': 1, 'category_id': 14, 'bbox': [0, 0, 10, 10], 'score': .8},
        {'_index': 2, 'category_id': 14, 'bbox': [20, 0, 10, 10], 'score': .7},
        {'_index': 3, 'category_id': 14, 'bbox': [40, 0, 10, 10], 'score': .6},
        {'_index': 4, 'category_id': 3, 'bbox': [20, 0, 10, 10], 'score': .299},
    ]
    original = copy.deepcopy((gt, pred))
    diagnosis = data.diagnose(gt, pred, .3, .5)
    assert (diagnosis['tp'], diagnosis['fp'], diagnosis['fn']) == (1, 2, 1)
    assert diagnosis['class_errors'] == 1
    assert diagnosis['unmatched_gt_ids'] == [2]
    assert diagnosis['ignored_gt_count'] == 1
    assert diagnosis['mean_iou'] == 1
    assert len(diagnosis['matches']) == 1
    assert (gt, pred) == original
    assert data.diagnose(gt, pred, .299, .5)['fn'] == 0


@pytest.mark.parametrize('bbox', [
    [1, 2, -3, 4], [1, 2, 3, 0], [float('nan'), 2, 3, 4],
    [1, float('inf'), 3, 4], [1, 2, 3],
])
def test_invalid_box_is_audited_never_fixed_or_written(case_files, bbox):
    path = case_files['pred_root'] / MODEL_NAMES['B1'] / 'seed_42/predictions.json'
    predictions = json.loads(path.read_text(encoding='utf-8'))
    predictions.append({'image_id': 91, 'category_id': 14, 'bbox': bbox, 'score': .99})
    write_json(path, predictions)
    original = path.read_bytes()
    bundle = data.load_inputs(**case_files)
    assert len(bundle['predictions']['B1'][91]) == 1
    assert len(bundle['invalid']) == 1
    assert bundle['invalid'][0]['image_id'] == 91
    assert path.read_bytes() == original


@pytest.mark.parametrize('field,value', [('image_id', 0), ('category_id', 1)])
def test_unknown_image_or_category_fails_instead_of_position_mapping(case_files, field, value):
    path = case_files['pred_root'] / MODEL_NAMES['B1'] / 'seed_42/predictions.json'
    predictions = json.loads(path.read_text(encoding='utf-8'))
    predictions[0][field] = value
    write_json(path, predictions)
    with pytest.raises(ValueError):
        data.load_inputs(**case_files)


@pytest.mark.parametrize('change', ['hash', 'split', 'seed', 'model', 'missing'])
def test_prediction_provenance_must_match_same_test_dataset(case_files, change):
    path = case_files['pred_root'] / MODEL_NAMES['DC'] / 'seed_42/metrics.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if change == 'missing':
        path.unlink()
    else:
        if change == 'hash':
            metadata['dataset']['annotation_sha256'] = '0' * 64
        else:
            metadata[change] = {'split': 'validation', 'seed': 0, 'model': 'baseline_native'}[change]
        write_json(path, metadata)
    with pytest.raises(ValueError):
        data.load_inputs(**case_files)


def test_missing_light_metadata_uses_explicit_id_map_not_id_ranges(case_files):
    annotation = case_files['annotations']
    coco = json.loads(annotation.read_text(encoding='utf-8'))
    for image in coco['images']:
        del image['illumination']
    write_json(annotation, coco)
    for model in MODEL_NAMES.values():
        path = case_files['pred_root'] / model / 'seed_42/metrics.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata['dataset']['annotation_sha256'] = hashlib.sha256(annotation.read_bytes()).hexdigest()
        write_json(path, metadata)
    unknown = data.load_inputs(**case_files)
    assert set(unknown['illumination'].values()) == {'unknown'}
    assert not any(row['B_supported'] for row in data.candidate_rows(unknown, .3))
    bundle = data.load_inputs(**case_files, illumination_map={'91': '自然光', '7': '钠黄光'})
    rows = {row['image_id']: row for row in data.candidate_rows(bundle, .3)}
    assert rows[91]['illumination'] == 'natural'
    assert rows[7]['illumination'] == 'sodium_light'


def cli_args(case_files, out, mode):
    return ['--mode', mode, '--annotations', str(case_files['annotations']),
            '--images', str(case_files['images']), '--pred-root', str(case_files['pred_root']),
            '--seed', '42', '--expected-images', '2', '--out', str(out)]


def test_candidate_ranking_has_all_five_models_and_evidence(case_files):
    bundle = data.load_inputs(**case_files)
    rows = {row['image_id']: row for row in data.candidate_rows(bundle, .3)}
    assert set(rows) == {91, 7}
    assert rows[91]['B1_FN'] == 1
    assert rows[91]['VQ_FN'] == 0
    assert rows[91]['A_supported']
    assert rows[7]['B_supported']
    for row in rows.values():
        for alias in MODEL_NAMES:
            for metric in ('TP', 'FP', 'FN'):
                assert f'{alias}_{metric}' in row
        assert row['recommendation_reason']


def test_rank_cli_writes_all_candidates_and_audited_contact_sheet(case_files, tmp_path):
    from tools.paper_v2 import qualitative_fig8 as fig8
    out = tmp_path / 'rank'
    assert fig8.main(cli_args(case_files, out, 'rank') + ['--contact-count', '2']) == 0
    with (out / 'fig8_candidate_ranking.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row['image_id']) for row in rows} == {7, 91}
    for row in rows:
        assert row['seed'] == '42'
        assert row['file_name'] and row['illumination'] and row['recommendation_reason']
    audit = json.loads((out / 'Fig8_rank_audit.json').read_text(encoding='utf-8'))
    assert audit['candidate_count'] == 2
    assert audit['A_candidate_count'] >= 1 and audit['B_candidate_count'] >= 1
    assert len(audit['panels']) == 12
    with Image.open(out / 'fig8_candidate_contact_sheet.png') as image:
        assert image.width >= 2100
        assert image.info['dpi'][0] >= 299.9


def test_render_outputs_uniform_roi_dpi_vectors_and_complete_audit(case_files, tmp_path):
    from tools.paper_v2 import qualitative_fig8 as fig8
    out = tmp_path / 'render'
    # Keep both explained GT objects within A and the missed object within B.
    rois = {'91': [10, 15, 120, 70], '7': [10, 20, 90, 120]}
    roi_path = tmp_path / 'rois.json'
    write_json(roi_path, rois)
    invalid_path = case_files['pred_root'] / MODEL_NAMES['B1'] / 'seed_42/predictions.json'
    invalid_predictions = json.loads(invalid_path.read_text(encoding='utf-8'))
    invalid_predictions.extend([
        {'image_id': 91, 'category_id': 14, 'bbox': [1, 2, -3, 4], 'score': .99},
        {'image_id': 91, 'category_id': 14, 'bbox': [float('nan'), 2, 3, 4], 'score': .99},
    ])
    write_json(invalid_path, invalid_predictions)
    prediction_paths = list(case_files['pred_root'].rglob('predictions.json'))
    originals = {p: p.read_bytes() for p in prediction_paths}
    argv = cli_args(case_files, out, 'render') + ['--image-ids', '91', '7',
        '--roi-json', str(roi_path), '--score-thr', '0.30']
    assert fig8.main(argv) == 0
    expected = {
        'Fig8_Qualitative_Comparison.png', 'Fig8_Qualitative_Comparison.tif',
        'Fig8_Qualitative_Comparison.svg', 'Fig8_selected_cases.csv',
        'Fig8_render_audit.json', 'Fig8_caption_en.txt', 'Fig8_caption_zh.txt',
        'Fig8_full_image_contact_sheet.png',
    }
    assert {path.name for path in out.iterdir()} == expected
    audit = json.loads((out / 'Fig8_render_audit.json').read_text(encoding='utf-8'))
    assert audit['selected_image_ids'] == [91, 7]
    assert audit['score_threshold'] == .3 and audit['seed'] == 42
    assert set(audit['categories']) == {'14', '3', '28'}
    assert audit['categories']['28']['original_name'] == 'half'
    assert audit['categories']['28']['name'] == 'Half-ripe'
    assert audit['sources'] and len(audit['invalid_predictions']) == 2
    assert 'not coco ap' in audit['diagnostic_protocol'].lower()
    assert len(audit['panels']) == 12
    for image_id in (91, 7):
        panels = [panel for panel in audit['panels'] if panel['image_id'] == image_id]
        assert [panel['model'] for panel in panels] == ['GT / Input', *MODEL_NAMES]
        assert all(panel['roi'] == rois[str(image_id)] for panel in panels)
        assert all(panel['score_threshold'] == .3 for panel in panels)
        assert len({panel['image_sha256'] for panel in panels}) == 1
        assert len({panel['pixels_per_inch_in_figure'] for panel in panels}) == 1
    for case in audit['cases']:
        assert case['file_name'] and case['illumination'] and case['gt_count'] > 0
        assert set(case['diagnostics_full_image']) == set(MODEL_NAMES)
        assert set(case['diagnostics_fully_inside_roi']) == set(MODEL_NAMES)
        assert case['interpretation_supported']
    with (out / 'Fig8_selected_cases.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    required_fields = {
        'image_id', 'file_name', 'illumination', 'seed', 'model', 'prediction_json',
        'original_width', 'original_height', 'roi_xyxy', 'score_threshold',
        'category_mapping', 'gt_count', 'drawn_prediction_count',
        'invalid_prediction_count', 'gt_matching',
    }
    assert required_fields <= rows[0].keys()
    for row in rows:
        assert Path(row['prediction_json']).is_file()
        assert set(json.loads(row['gt_matching'])) >= {'tp', 'fp', 'fn', 'matches'}
        if row['image_id'] == '91' and row['alias'] == 'B1':
            assert row['invalid_prediction_count'] == '2'
            assert row['drawn_prediction_count'] == '1'
    for suffix in ('png', 'tif'):
        with Image.open(out / f'Fig8_Qualitative_Comparison.{suffix}') as image:
            assert list(image.size) == audit['output_pixel_size']
            assert image.width == 2160  # 7.2 inch double-column export at 300 dpi.
            assert image.info['dpi'][0] >= 299.9
    svg = ET.parse(out / 'Fig8_Qualitative_Comparison.svg').getroot()
    assert svg.findall('.//{http://www.w3.org/2000/svg}image')
    for image in svg.findall('.//{http://www.w3.org/2000/svg}image'):
        assert image.attrib['{http://www.w3.org/1999/xlink}href'].startswith('data:image/png;base64,')
    assert svg.findall('.//{http://www.w3.org/2000/svg}path')
    text = ' '.join(svg.itertext())
    assert 'Ripe 0.93' in text and 'Half-ripe 0.90' in text and 'GT / Input' in text
    english = (out / 'Fig8_caption_en.txt').read_text(encoding='utf-8')
    assert '(A) Example of improved detection' in english
    assert '(B) Example illustrating a limitation' in english
    assert all(p.read_bytes() == content for p, content in originals.items())
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError):
        fig8.main(argv)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_unsupported_case_titles_are_neutral_and_roi_cannot_change_per_model(case_files):
    from tools.paper_v2 import qualitative_fig8 as fig8
    bundle = data.load_inputs(**case_files)
    a = fig8.diagnostics(bundle, 91, .3, roi=[0, 0, 10, 10])
    assert not data.case_evidence(a, 'natural')['A_supported']
    b = fig8.diagnostics(bundle, 91, .3)
    assert not data.case_evidence(b, 'natural')['B_supported']
    args = fig8.parser().parse_args(cli_args(case_files, Path('unused'), 'render'))
    english, chinese = fig8.captions(args, [91, 7], bundle, [False, False])
    assert 'Example of improved detection' not in english
    assert 'Example illustrating a limitation' not in english
    assert 'Selected comparison' in english and '未支持' in chinese
    with pytest.raises(ValueError):
        fig8.validate_roi({'B1': [0, 0, 10, 10]}, 160, 100)


@pytest.mark.parametrize('roi', [[0, 0, 161, 100], [4, 4, 3, 9],
                                 [0, 0, float('nan'), 4], [0, 0, 5]])
def test_roi_must_be_finite_nonempty_and_inside_original_image(roi):
    from tools.paper_v2 import qualitative_fig8 as fig8
    with pytest.raises(ValueError):
        fig8.validate_roi(roi, 160, 100)


def test_mixed_tradeoff_does_not_receive_an_improvement_or_limitation_title():
    # Predicting nothing reduces FP but increases FN: not evidence of better detection.
    baseline = {'tp': 1, 'fp': 1, 'fn': 0, 'class_errors': 0,
                'matches': [{'gt_id': 1, 'iou': .9}]}
    empty = {'tp': 0, 'fp': 0, 'fn': 1, 'class_errors': 0, 'matches': []}
    diags = {'B1': baseline, 'VQ': empty, 'LI': baseline, 'DC': baseline, 'Full': baseline}
    evidence = data.case_evidence(diags, 'natural')
    assert not evidence['A_supported']
    diags['Full'] = empty
    diags['VQ'] = baseline
    evidence = data.case_evidence(diags, 'sodium_light')
    assert not evidence['B_supported']


def test_artist_geometry_labels_and_colors_use_original_coco_data(case_files):
    from matplotlib.colors import to_rgba
    from tools.paper_v2 import qualitative_fig8 as fig8
    bundle = data.load_inputs(**case_files)
    fig, panels, _ = fig8.comparison_figure(
        bundle, [91], case_files['images'], {91: [10., 15., 120., 70.]},
        ['SYNTHETIC TEST ONLY'], .3,
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            boundary = ax.get_window_extent(renderer)
            for label in ax.texts:
                text_box = label.get_window_extent(renderer)
                assert text_box.x0 >= boundary.x0
                assert text_box.x1 <= boundary.x1
                assert text_box.y0 >= boundary.y0
                assert text_box.y1 <= boundary.y1
        gt_ax, b1_ax = fig.axes[:2]
        assert [label.get_text() for label in gt_ax.texts] == ['Ripe', 'Immature']
        assert [label.get_text() for label in b1_ax.texts] == ['Ripe 0.93']
        box = b1_ax.patches[0]
        assert [box.get_x(), box.get_y(), box.get_width(), box.get_height()] == [20, 25, 30, 25]
        assert box.get_edgecolor() == to_rgba(bundle['categories'][14]['color'])
        assert box.get_linestyle() == '-'
        assert gt_ax.patches[0].get_linestyle() != '-'
        assert all(panel['roi'] == [10., 15., 120., 70.] for panel in panels)
    finally:
        fig8.close_figure(fig)

