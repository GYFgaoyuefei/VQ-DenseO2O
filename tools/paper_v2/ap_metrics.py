"""COCO AP on the mixed split and explicit acquisition-light subsets.

All subsets reuse the same predictions; Mixed AP is evaluated over all images,
not averaged from the four subset AP values. Public metric values are percentages.
"""
import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real

from faster_coco_eval import COCO, COCOeval_faster

from engine.data.dataset.visible_coco_dataset import ILLUMINATION_IDS, illumination_id
from tools.paper_v2.efficiency_protocol import AP_COLUMNS, COCO_NAMES


LIGHT_FIELDS = ('illumination', 'illumination_label', 'source_illumination_id', 'light_condition')


def _integer(value, description):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f'{description} must be an integer: {value!r}')
    return int(value)


def _unique_ids(records, description):
    ids = [_integer(record['id'], f'{description} id') for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate {description} ids in COCO metadata')
    return set(ids)


def _light_groups(coco_gt, illumination_map):
    if illumination_map is None:
        illumination_map = {}
    if not isinstance(illumination_map, Mapping):
        raise ValueError('illumination_map must map image_id or file_name to an explicit light label')
    groups = {name: [] for name in ILLUMINATION_IDS}
    names = {value: key for key, value in ILLUMINATION_IDS.items()}
    for image_id, metadata in coco_gt.imgs.items():
        candidates = [(key, metadata[key]) for key in LIGHT_FIELDS if metadata.get(key) is not None]
        # Check every matching mapping key, so conflicting integer/string IDs or
        # filename entries cannot be hidden by precedence rules.
        for key in dict.fromkeys((image_id, str(image_id), metadata.get('file_name'))):
            if key is not None and key in illumination_map:
                candidates.append((f'illumination_map[{key!r}]', illumination_map[key]))
        parsed = []
        for field, value in candidates:
            light_id = illumination_id(value)
            if light_id < 0:
                raise ValueError(f'Unknown illumination for image_id={image_id}: {field}={value!r}')
            parsed.append(light_id)
        if not parsed:
            raise ValueError(f'Missing explicit illumination for image_id={image_id}')
        if len(set(parsed)) != 1:
            raise ValueError(f'Conflicting illumination for image_id={image_id}: {candidates!r}')
        groups[names[parsed[0]]].append(image_id)
    return groups


def _validated_predictions(predictions, image_ids, category_ids):
    """Validate records without dropping false-positive or repeated detections."""
    result = []
    explicit_ids = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            raise ValueError(f'Prediction {index} must be a mapping')
        try:
            image_id = _integer(prediction['image_id'], f'Prediction {index} image_id')
            category_id = _integer(prediction['category_id'], f'Prediction {index} category_id')
            bbox = prediction['bbox']
            score = prediction['score']
        except KeyError as exc:
            raise ValueError(f'Prediction {index} is missing {exc.args[0]}') from exc
        if image_id not in image_ids or category_id not in category_ids:
            raise ValueError(f'Prediction {index} references unknown image/category id: {image_id}/{category_id}')
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f'Prediction {index} bbox must contain four xywh values')
        values = (*bbox, score)
        if any(not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
               for value in values):
            raise ValueError(f'Prediction {index} bbox/score must contain finite numbers')
        if bbox[2] < 0 or bbox[3] < 0 or not 0 <= score <= 1:
            raise ValueError(f'Prediction {index} has negative bbox size or score outside [0, 1]')
        if 'id' in prediction:
            detection_id = _integer(prediction['id'], f'Prediction {index} id')
            if detection_id in explicit_ids:
                raise ValueError(f'Duplicate prediction id: {detection_id}')
            explicit_ids.add(detection_id)
        # COCO.loadRes mutates its argument; use fresh dictionaries and lists.
        result.append({'image_id': image_id, 'category_id': category_id,
                       'bbox': [float(value) for value in bbox], 'score': float(score)})
    return result


def _evaluate_subset(coco_gt, coco_dt, image_ids):
    if not image_ids:
        return [None] * len(COCO_NAMES)
    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType='bbox')
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    if len(evaluator.stats) < len(COCO_NAMES):
        raise RuntimeError('COCO evaluator did not produce the twelve standard bbox metrics')
    # COCO uses -1 when a metric has no eligible ground truth. Preserve that
    # absence as null rather than incorrectly publishing -100%.
    return [float(value) * 100 if math.isfinite(value) and value >= 0 else None
            for value in evaluator.stats[:len(COCO_NAMES)]]


def evaluate_predictions(coco_gt, predictions, illumination_map=None):
    """Evaluate one split using native COCO matching and explicit light metadata.

    ``predictions`` is a COCO detection iterable (xywh, image_id, category_id,
    score). Multiple detections of the same image/category, including repeated
    boxes, remain in the evaluation. If optional detection ``id`` fields are
    supplied they must be unique. The mapping accepts integer/string image IDs
    or exact filenames and uses the same light vocabulary as the dataset.
    Missing, unknown, or conflicting acquisition labels raise ``ValueError``.
    Empty or ground-truth-free groups have null AP and yield INCOMPLETE status.
    """
    image_ids = _unique_ids(coco_gt.dataset.get('images', []), 'image')
    category_ids = _unique_ids(coco_gt.dataset.get('categories', []), 'category')
    if image_ids != set(coco_gt.imgs) or category_ids != set(coco_gt.cats):
        raise ValueError('COCO metadata and indexes are inconsistent; rebuild the COCO index')
    groups = _light_groups(coco_gt, illumination_map)
    results = _validated_predictions(predictions, image_ids, category_ids)
    # COCO evaluation can attach ignore fields to GT annotations. Do not modify
    # the caller's dataset or prediction cache when evaluating several scopes.
    evaluation_gt = copy.deepcopy(coco_gt)
    if results:
        coco_dt = evaluation_gt.loadRes(results)
    else:
        # Older faster-coco-eval releases cannot loadRes([]).
        coco_dt = COCO()
        coco_dt.dataset = {'images': copy.deepcopy(evaluation_gt.dataset.get('images', [])),
                           'categories': copy.deepcopy(evaluation_gt.dataset.get('categories', [])),
                           'annotations': []}
        coco_dt.createIndex()
    mixed = _evaluate_subset(evaluation_gt, coco_dt, image_ids)
    light_values = [_evaluate_subset(evaluation_gt, coco_dt, groups[name])[0]
                    for name in ILLUMINATION_IDS]
    metrics = dict(zip(AP_COLUMNS, [*mixed, *light_values]))
    required = [mixed[0], *light_values]
    return {'metrics': metrics,
            'image_counts': {'mixed': len(image_ids), **{key: len(value) for key, value in groups.items()}},
            'status': 'COMPLETE' if all(value is not None for value in required) else 'INCOMPLETE'}
