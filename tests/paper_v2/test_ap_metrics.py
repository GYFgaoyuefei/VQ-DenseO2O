"""AP fixtures are synthetic COCO dictionaries; no dataset files are read."""
import copy

import pytest
from faster_coco_eval import COCO

from tools.paper_v2.ap_metrics import evaluate_predictions
from tools.paper_v2.efficiency_protocol import AP_COLUMNS


LIGHTS = ('natural', 'artificial', 'low_light', 'sodium_light')


def fixture(lights=LIGHTS):
    coco = COCO()
    coco.dataset = {
        'images': [{'id': index, 'file_name': f'{index}.jpg', 'width': 100, 'height': 100,
                    'illumination': light} for index, light in enumerate(lights, 1)],
        'categories': [{'id': 1, 'name': 'tomato'}],
        'annotations': [{'id': index, 'image_id': index, 'category_id': 1,
                         'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}
                        for index in range(1, len(lights) + 1)],
    }
    coco.createIndex()
    predictions = [{'image_id': index, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'score': .9}
                   for index in range(1, len(lights) + 1)]
    return coco, predictions


def test_perfect_predictions_and_no_input_mutations():
    coco, predictions = fixture()
    original_gt, original_predictions = copy.deepcopy(coco.dataset), copy.deepcopy(predictions)
    result = evaluate_predictions(coco, predictions)
    assert result['status'] == 'COMPLETE'
    assert list(result['metrics']) == list(AP_COLUMNS)
    assert result['image_counts'] == {'mixed': 4, **dict.fromkeys(LIGHTS, 1)}
    for key in (AP_COLUMNS[0], AP_COLUMNS[1], AP_COLUMNS[2], *AP_COLUMNS[-4:]):
        assert result['metrics'][key] == pytest.approx(100.)
    assert result['metrics']['Mixed AP-M (%)'] is None
    assert result['metrics']['Mixed AP-L (%)'] is None
    assert result['metrics']['Mixed AR100 (%)'] == pytest.approx(100.)
    assert coco.dataset == original_gt and predictions == original_predictions


def test_empty_predictions_are_zero_ap_for_nonempty_gt_groups():
    coco, _ = fixture()
    result = evaluate_predictions(coco, [])
    assert result['status'] == 'COMPLETE'
    for key in (AP_COLUMNS[0], *AP_COLUMNS[-4:]):
        assert result['metrics'][key] == 0.


def test_mixed_is_evaluated_over_all_images_not_mean_of_light_ap():
    coco, predictions = fixture(('natural', 'natural', 'natural', *LIGHTS[1:]))
    result = evaluate_predictions(coco, predictions[:3])
    light_ap = [result['metrics'][key] for key in AP_COLUMNS[-4:]]
    assert light_ap == pytest.approx([100., 0., 0., 0.])
    assert result['metrics'][AP_COLUMNS[0]] == pytest.approx(100 * 51 / 101)
    assert result['metrics'][AP_COLUMNS[0]] != pytest.approx(sum(light_ap) / 4)


def test_chinese_metadata_and_consistent_explicit_maps():
    coco, predictions = fixture()
    for metadata, label in zip(coco.dataset['images'], ('自然光', '人工光', '微弱光', '钠黄光')):
        metadata['light_condition'] = label
        del metadata['illumination']
    result = evaluate_predictions(coco, predictions, {1: 0, '2': 'artificial', '3.jpg': '微弱光', '4': 3})
    assert result['status'] == 'COMPLETE'
    assert result['image_counts'] == {'mixed': 4, **dict.fromkeys(LIGHTS, 1)}


def test_map_supplies_missing_metadata_without_filename_inference():
    coco, predictions = fixture()
    for image in coco.dataset['images']:
        del image['illumination']
    mapping = {str(index): name for index, name in enumerate(LIGHTS, 1)}
    assert evaluate_predictions(coco, predictions, mapping)['status'] == 'COMPLETE'
    with pytest.raises(ValueError, match='Missing explicit illumination'):
        evaluate_predictions(coco, predictions)


@pytest.mark.parametrize('edit,mapping,match', [
    ({'illumination': 'unknown'}, None, 'Unknown illumination'),
    ({'illumination': -1}, None, 'Unknown illumination'),
    ({'light_condition': '人工光'}, None, 'Conflicting illumination'),
    ({}, {'1': 'artificial'}, 'Conflicting illumination'),
    ({}, {1: 'natural', '1': 'artificial'}, 'Conflicting illumination'),
    ({}, {'1.jpg': 'not-a-label'}, 'Unknown illumination'),
])
def test_unknown_or_conflicting_labels_fail(edit, mapping, match):
    coco, predictions = fixture()
    coco.imgs[1].update(edit)
    with pytest.raises(ValueError, match=match):
        evaluate_predictions(coco, predictions, mapping)


def test_empty_light_group_is_none_and_incomplete():
    coco, predictions = fixture(LIGHTS[:3])
    result = evaluate_predictions(coco, predictions)
    assert result['status'] == 'INCOMPLETE'
    assert result['image_counts']['sodium_light'] == 0
    assert result['metrics']['Sodium-light AP (%)'] is None
    assert result['metrics']['Mixed AP (%)'] == pytest.approx(100.)


def test_group_with_images_but_no_gt_is_none_not_negative_percent():
    coco, predictions = fixture()
    coco.dataset['annotations'] = coco.dataset['annotations'][:-1]
    coco.createIndex()
    result = evaluate_predictions(coco, predictions[:-1])
    assert result['status'] == 'INCOMPLETE'
    assert result['image_counts']['sodium_light'] == 1
    assert result['metrics']['Sodium-light AP (%)'] is None
    assert not any(value is not None and value < 0 for value in result['metrics'].values())


def test_empty_dataset_has_no_metrics():
    coco, predictions = fixture(())
    result = evaluate_predictions(coco, predictions)
    assert result['status'] == 'INCOMPLETE'
    assert all(value is None for value in result['metrics'].values())
    assert all(value == 0 for value in result['image_counts'].values())


@pytest.mark.parametrize('edit,match', [
    ({'image_id': 99}, 'unknown image/category'),
    ({'category_id': 99}, 'unknown image/category'),
    ({'image_id': True}, 'must be an integer'),
    ({'category_id': 1.1}, 'must be an integer'),
    ({'bbox': [0, 0, -1, 20]}, 'negative bbox size'),
    ({'bbox': [0, 0, float('nan'), 20]}, 'finite numbers'),
    ({'bbox': [0, 0, 20]}, 'four xywh'),
    ({'score': float('inf')}, 'finite numbers'),
    ({'score': 1.1}, 'outside'),
])
def test_invalid_predictions_fail(edit, match):
    coco, predictions = fixture()
    predictions[0].update(edit)
    with pytest.raises(ValueError, match=match):
        evaluate_predictions(coco, predictions)


def test_duplicate_record_ids_fail_but_duplicate_boxes_remain_false_positives():
    coco, predictions = fixture()
    predictions[0]['id'] = 9
    predictions[1]['id'] = 9
    with pytest.raises(ValueError, match='Duplicate prediction id'):
        evaluate_predictions(coco, predictions)
    for prediction in predictions:
        prediction.pop('id', None)
    # Repeated boxes are valid COCO detections, not records to silently remove.
    predictions.insert(0, copy.deepcopy(predictions[0]))
    result = evaluate_predictions(coco, predictions)
    assert 0 < result['metrics'][AP_COLUMNS[0]] < 100


@pytest.mark.parametrize('field', ['images', 'categories'])
def test_duplicate_gt_ids_are_not_silently_collapsed(field):
    coco, predictions = fixture()
    coco.dataset[field].append(copy.deepcopy(coco.dataset[field][0]))
    coco.createIndex()
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate_predictions(coco, predictions)
