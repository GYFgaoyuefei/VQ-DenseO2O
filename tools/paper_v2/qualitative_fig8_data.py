"""Read-only COCO inputs and transparent, non-AP diagnostics for Fig. 8.

No training, inference, checkpoint, torch, or engine dependencies are imported.
The light vocabulary and conflict policy mirror ``ap_metrics._light_groups``;
the few constants are repeated here to avoid importing its inference stack.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PureWindowsPath


MODELS = {
    'B1': 'baseline_maskaware',
    'VQ': 'vq_dense_o2o',
    'LI': 'li_copyblend',
    'DC': 'dc_sta',
    'Full': 'vq_dense_o2o_li_copyblend_dc_sta',
}
COLORS = {'Immature': '#228B22', 'Half-ripe': '#E58B16', 'Ripe': '#D62728'}
LIGHTS = ('natural', 'artificial', 'low_light', 'sodium_light')
LIGHT_FIELDS = ('illumination', 'illumination_label', 'source_illumination_id', 'light_condition')
LIGHT_ALIASES = {'自然光': 'natural', '人工光': 'artificial', '微弱光': 'low_light', '钠黄光': 'sodium_light',
                 'low-light': 'low_light', 'low light': 'low_light',
                 'sodium-light': 'sodium_light', 'sodium light': 'sodium_light'}
CATEGORY_ALIASES = {
    'immature': 'Immature', 'unripe': 'Immature', 'green': 'Immature',
    '未成熟': 'Immature', '未成熟果': 'Immature', '青果': 'Immature',
    'half-ripe': 'Half-ripe', 'half ripe': 'Half-ripe', 'half_ripe': 'Half-ripe',
    'half': 'Half-ripe', 'halfripe': 'Half-ripe', 'turning': 'Half-ripe', 'semi-ripe': 'Half-ripe',
    '半成熟': 'Half-ripe', '半成熟果': 'Half-ripe', '转色果': 'Half-ripe',
    'ripe': 'Ripe', 'red': 'Ripe', '成熟': 'Ripe', '成熟果': 'Ripe',
}
DIAGNOSTIC_DESCRIPTION = (
    'Auxiliary score-thresholded diagnostic, not COCO AP. Descending-score, '
    'class-consistent, one-to-one IoU matching. Crowd/ignore GT are excluded; '
    'otherwise-unmatched predictions overlapping same-class ignored GT are ignored. '
    'Class errors are cross-class matches among remaining FP and FN, not extra counts.'
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON object key: {key!r}')
        result[key] = value
    return result


def _json(path):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        return json.load(stream, object_pairs_hook=_object_pairs)


def _mapping(value, name):
    if value is None:
        return {}
    result = _json(value) if isinstance(value, (str, Path)) else value
    if not isinstance(result, Mapping):
        raise ValueError(f'{name} must be a JSON object')
    return result


def _integer(value, description):
    if type(value) is not int:
        raise ValueError(f'{description} must be an integer, got {value!r}')
    return value


def _index(records, description):
    if not isinstance(records, list):
        raise ValueError(f'COCO {description} must be a list')
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f'COCO {description} entries must be objects')
        key = _integer(record.get('id'), f'{description} id')
        if key in indexed:
            raise ValueError(f'Duplicate {description} id: {key}')
        indexed[key] = dict(record)
    return indexed


def _bbox_error(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 'bbox must contain exactly four xywh coordinates'
    if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) for v in bbox):
        return 'bbox coordinates must be finite numbers'
    if bbox[2] <= 0 or bbox[3] <= 0:
        return 'bbox width and height must be strictly positive'
    if not math.isfinite(bbox[0] + bbox[2]) or not math.isfinite(bbox[1] + bbox[3]):
        return 'bbox corner calculation is not finite'
    return None


def _json_safe(value):
    """Keep rejected values legible without writing non-standard NaN JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {key: _json_safe(v) for key, v in value.items()}
    return value


def _light(value):
    if type(value) is int and 0 <= value < len(LIGHTS):
        return LIGHTS[value]
    if isinstance(value, str):
        normalized = value.strip().lower()
        normalized = LIGHT_ALIASES.get(normalized, normalized)
        if normalized in LIGHTS:
            return normalized
    raise ValueError(f'Unrecognized explicit illumination label: {value!r}')


def _illuminations(images, mapping):
    result = {}
    for image_id, metadata in images.items():
        candidates = [metadata[key] for key in LIGHT_FIELDS if metadata.get(key) is not None]
        for key in dict.fromkeys((image_id, str(image_id), metadata['file_name'])):
            if key in mapping:
                candidates.append(mapping[key])
        parsed = [_light(value) for value in candidates]
        if len(set(parsed)) > 1:
            raise ValueError(f'Conflicting illumination for image_id={image_id}: {candidates!r}')
        result[image_id] = parsed[0] if parsed else 'unknown'
    return result


def _provenance(prediction_path, alias, model, seed, annotation_hash, manifest):
    metrics_path = prediction_path.with_name('metrics.json')
    source = {}
    if metrics_path.is_file():
        record = _json(metrics_path)
        if not isinstance(record, dict):
            raise ValueError(f'Metrics must be a JSON object: {metrics_path}')
        if record.get('model') != model or type(record.get('seed')) is not int or record.get('seed') != seed or record.get('split') != 'test':
            raise ValueError(f'Metrics model/seed/Test split mismatch: {metrics_path}')
        dataset = record.get('dataset') or {}
        if not isinstance(dataset, Mapping) or dataset.get('annotation_sha256') != annotation_hash:
            raise ValueError(f'Metrics annotation SHA256 differs from supplied COCO Test data: {metrics_path}')
        source = {'kind': 'saved_evaluation_metrics', 'path': str(metrics_path),
                  'sha256': sha256_file(metrics_path), 'annotation_sha256': annotation_hash,
                  'model': model, 'seed': seed, 'split': 'test'}
    explicit = [manifest[key] for key in dict.fromkeys((alias, model)) if key in manifest]
    for entry in explicit:
        if not isinstance(entry, Mapping) or entry.get('annotation_sha256') != annotation_hash:
            raise ValueError(f'Dataset manifest annotation SHA256 mismatch for {alias}')
        if any(key in entry and entry[key] != expected
               for key, expected in (('model', model), ('seed', seed), ('split', 'test'))):
            raise ValueError(f'Dataset manifest model/seed/Test split mismatch for {alias}')
        if entry.get('predictions_sha256') and entry['predictions_sha256'] != sha256_file(prediction_path):
            raise ValueError(f'Dataset manifest predictions SHA256 mismatch for {alias}')
    if not source:
        if not explicit:
            raise ValueError(f'Cannot establish Test dataset identity for {alias}: missing {metrics_path}. '
                             'Supply --dataset-manifest with independently verified annotation_sha256 '
                             'for every model lacking saved metrics. Image IDs alone do not prove identity.')
        source = {'kind': 'explicit_user_dataset_manifest', 'annotation_sha256': annotation_hash,
                  'model': model, 'seed': seed, 'split': 'test'}
    return source


def load_inputs(annotations, images, pred_root, seed=42, illumination_map=None,
                expected_images=400, dataset_manifest=None):
    """Load and validate saved Test inputs without modifying any source file.

    ``dataset_manifest`` is optional when all five sibling ``metrics.json`` files
    identify the supplied annotation hash, model, seed and Test split. Otherwise
    it maps model names/aliases to objects containing ``annotation_sha256``.
    Such entries are explicit user assertions, distinguished in provenance.
    Missing prediction image IDs mean zero detections, never a position mapping.
    """
    _integer(seed, 'seed')
    annotation_path, image_root, prediction_root = (Path(p).expanduser().resolve()
                                                  for p in (annotations, images, pred_root))
    coco = _json(annotation_path)
    if not isinstance(coco, dict):
        raise ValueError('COCO annotation JSON must be an object')
    image_index = _index(coco.get('images'), 'images')
    if expected_images is not None and len(image_index) != expected_images:
        raise ValueError(f'Expected {expected_images} Test images, found {len(image_index)}')
    if not image_index:
        raise ValueError('COCO Test images cannot be empty')
    category_index = _index(coco.get('categories'), 'categories')
    categories = {}
    for key, category in category_index.items():
        original = category.get('name')
        name = CATEGORY_ALIASES.get(original.strip().lower()) if isinstance(original, str) else None
        if name is None:
            raise ValueError(f'Unknown COCO category name {original!r} for category_id={key}; '
                             'category IDs are never used to guess maturity classes')
        if name in [entry['name'] for entry in categories.values()]:
            raise ValueError(f'Two category IDs map to the same class {name!r}')
        categories[key] = {'name': name, 'original_name': original, 'color': COLORS[name]}
    if not categories:
        raise ValueError('COCO categories cannot be empty')
    for image_id, metadata in image_index.items():
        filename = metadata.get('file_name')
        if not isinstance(filename, str) or not filename or '\\' in filename or PureWindowsPath(filename).is_absolute():
            raise ValueError(f'Invalid relative COCO file_name for image_id={image_id}: {filename!r}')
        path = (image_root / filename).resolve()
        if not path.is_relative_to(image_root) or Path(filename).is_absolute():
            raise ValueError(f'COCO file_name leaves the image root: {filename!r}')
        if not path.is_file():
            raise FileNotFoundError(f'COCO image_id={image_id} source image missing: {path}')
        for field in ('width', 'height'):
            if _integer(metadata.get(field), f'image_id={image_id} {field}') <= 0:
                raise ValueError(f'image_id={image_id} {field} must be positive')
        metadata['_path'] = str(path)
    gt = {image_id: [] for image_id in image_index}
    for annotation in _index(coco.get('annotations'), 'annotations').values():
        image_id = _integer(annotation.get('image_id'), 'annotation image_id')
        category_id = _integer(annotation.get('category_id'), 'annotation category_id')
        if image_id not in image_index or category_id not in categories:
            raise ValueError(f'GT annotation {annotation["id"]} has unknown image/category ID')
        error = _bbox_error(annotation.get('bbox'))
        if error:
            raise ValueError(f'GT annotation {annotation["id"]}: {error}')
        for key in ('iscrowd', 'ignore'):
            if annotation.get(key, 0) not in (0, 1):
                raise ValueError(f'GT annotation {annotation["id"]}: {key} must be 0 or 1')
        gt[image_id].append(annotation)
    annotation_hash = sha256_file(annotation_path)
    lights = _illuminations(image_index, _mapping(illumination_map, 'illumination_map'))
    manifest = _mapping(dataset_manifest, 'dataset_manifest')
    invalid, predictions, paths = [], {}, {}
    sources = {'annotations': {'path': str(annotation_path), 'sha256': annotation_hash},
               'prediction_sets': {}, 'warnings': [], 'diagnostic': DIAGNOSTIC_DESCRIPTION}
    for label, argument in (('illumination_map', illumination_map), ('dataset_manifest', dataset_manifest)):
        if isinstance(argument, (str, Path)):
            sources[label] = {'path': str(Path(argument).resolve()), 'sha256': sha256_file(argument)}
        elif argument is not None:
            sources[label] = {'inline': _json_safe(dict(argument))}
    unknown_count = sum(value == 'unknown' for value in lights.values())
    if unknown_count:
        sources['warnings'].append(f'{unknown_count} images lack explicit illumination metadata; '
                                   'they cannot support a sodium-light limitation claim.')
    for alias, model in MODELS.items():
        path = prediction_root / model / f'seed_{seed}' / 'predictions.json'
        paths[alias] = str(path)
        provenance = _provenance(path, alias, model, seed, annotation_hash, manifest)
        records = _json(path)
        if not isinstance(records, list):
            raise ValueError(f'COCO detection predictions must be a list: {path}')
        grouped = {image_id: [] for image_id in image_index}
        present_ids = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'{alias} prediction {index} must be a JSON object')
            image_id = _integer(record.get('image_id'), f'{alias} prediction {index} image_id')
            category_id = _integer(record.get('category_id'), f'{alias} prediction {index} category_id')
            if image_id not in image_index or category_id not in categories:
                raise ValueError(f'{alias} prediction {index} has unknown image/category ID: {image_id}/{category_id}')
            present_ids.add(image_id)
            reasons = []
            bbox_error = _bbox_error(record.get('bbox'))
            if bbox_error:
                reasons.append(bbox_error)
            score = record.get('score')
            if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                reasons.append('score must be finite and in [0, 1]')
            if reasons:
                invalid.append({'alias': alias, 'model': model, 'image_id': image_id,
                                'prediction_index': index, 'category_id': category_id,
                                'reason': '; '.join(reasons), 'bbox': _json_safe(record.get('bbox')),
                                'score': _json_safe(score), 'prediction_json': str(path)})
                continue
            grouped[image_id].append(dict(record, _index=index))
        predictions[alias] = grouped
        sources['prediction_sets'][alias] = {
            'path': str(path), 'sha256': sha256_file(path), 'provenance': provenance,
            'record_count': len(records), 'valid_record_count': sum(map(len, grouped.values())),
            'invalid_record_count': sum(item['alias'] == alias for item in invalid),
            'images_with_records': len(present_ids),
            'image_ids_without_records': sorted(set(image_index) - present_ids),
        }
    return {'images': image_index, 'images_root': str(image_root), 'gt': gt,
            'predictions': predictions, 'categories': categories, 'illumination': lights,
            'sources': sources, 'invalid': invalid, 'prediction_paths': paths, 'seed': seed}


def bbox_iou(first, second):
    """IoU of two original-coordinate, positive-width/height COCO xywh boxes."""
    for box in (first, second):
        error = _bbox_error(box)
        if error:
            raise ValueError(error)
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def diagnose(gt, predictions, score_thr=.30, iou_thr=.50):
    """Deterministic one-to-one assistance for case selection, never COCO AP."""
    if not 0 <= score_thr <= 1 or not 0 < iou_thr <= 1:
        raise ValueError('score_thr must be in [0,1]; iou_thr in (0,1]')
    eligible = sorted((g for g in gt if not (g.get('iscrowd') or g.get('ignore'))), key=lambda g: g['id'])
    ignored = [g for g in gt if g.get('iscrowd') or g.get('ignore')]
    filtered = [(p.get('_index', index), p) for index, p in enumerate(predictions) if p['score'] >= score_thr]
    filtered.sort(key=lambda pair: (-pair[1]['score'], pair[0]))
    unmatched = {g['id']: g for g in eligible}
    matches, remaining, ignored_predictions = [], [], []
    for index, pred in filtered:
        candidates = [(bbox_iou(pred['bbox'], g['bbox']), gid) for gid, g in unmatched.items()
                      if g['category_id'] == pred['category_id']]
        best = max(candidates, key=lambda pair: (pair[0], -pair[1]), default=(0.0, None))
        if best[1] is not None and best[0] >= iou_thr:
            matches.append({'gt_id': best[1], 'prediction_index': index, 'iou': best[0],
                            'category_id': pred['category_id'], 'score': pred['score']})
            del unmatched[best[1]]
        elif any(g['category_id'] == pred['category_id'] and bbox_iou(pred['bbox'], g['bbox']) >= iou_thr for g in ignored):
            ignored_predictions.append(index)
        else:
            remaining.append((index, pred))
    wrong_category_targets = dict(unmatched)
    class_error_matches = []
    for index, pred in remaining:
        candidates = [(bbox_iou(pred['bbox'], g['bbox']), gid) for gid, g in wrong_category_targets.items()
                      if g['category_id'] != pred['category_id']]
        best = max(candidates, key=lambda pair: (pair[0], -pair[1]), default=(0.0, None))
        if best[1] is not None and best[0] >= iou_thr:
            target = wrong_category_targets.pop(best[1])
            class_error_matches.append({'gt_id': best[1], 'prediction_index': index, 'iou': best[0],
                                        'gt_category_id': target['category_id'], 'predicted_category_id': pred['category_id']})
    return {'tp': len(matches), 'fp': len(remaining), 'fn': len(unmatched),
            'mean_iou': sum(item['iou'] for item in matches) / len(matches) if matches else None,
            'matches': matches, 'class_errors': len(class_error_matches),
            'class_error_matches': class_error_matches, 'unmatched_gt_ids': sorted(unmatched),
            'unmatched_prediction_indices': [index for index, _ in remaining],
            'ignored_gt_count': len(ignored), 'ignored_prediction_indices': ignored_predictions,
            'ignored_prediction_count': len(ignored_predictions), 'prediction_count': len(filtered),
            'gt_count': len(eligible), 'score_threshold': score_thr, 'iou_threshold': iou_thr}


def _comparison(better, reference, better_name, reference_name):
    """Return positive diagnostic evidence; localization uses shared GT only."""
    reasons, tradeoffs, score = [], [], 0.0
    for key, direction, weight in (('tp', 1, 3), ('fp', -1, 2), ('fn', -1, 3), ('class_errors', -1, 2)):
        delta = direction * (better[key] - reference[key])
        if delta > 0:
            reasons.append(f'{better_name} vs {reference_name}: {key.upper()} {reference[key]} -> {better[key]}')
            score += delta * weight
        elif delta < 0:
            tradeoffs.append(f'{better_name} vs {reference_name}: {key.upper()} worsens {reference[key]} -> {better[key]}')
    better_iou = {m['gt_id']: m['iou'] for m in better['matches']}
    reference_iou = {m['gt_id']: m['iou'] for m in reference['matches']}
    common = sorted(better_iou.keys() & reference_iou.keys())
    gain = sum(better_iou[key] - reference_iou[key] for key in common) / len(common) if common else None
    if gain is not None and gain >= .05 - 1e-12:
        reasons.append(f'{better_name} vs {reference_name}: mean IoU on {len(common)} shared matched GT +{gain:.3f}')
        score += gain * 5
    elif gain is not None and gain <= -.05 + 1e-12:
        tradeoffs.append(f'{better_name} vs {reference_name}: mean IoU on {len(common)} shared matched GT {gain:.3f}')
    return reasons, score, {'common_gt_ids': common, 'mean_iou_delta_on_common_gt': gain,
                            'non_worse': not tradeoffs, 'tradeoffs': tradeoffs}


def case_evidence(diags, illumination):
    """Evidence is an invitation to manual review, not a success/failure verdict."""
    reasons_a, score_a, localization_a = _comparison(diags['VQ'], diags['B1'], 'VQ', 'B1')
    reasons_b, scores_b, localization_b, supported_b = [], [], {}, False
    if illumination == 'sodium_light':
        for reference in ('B1', 'VQ'):
            reasons, score, localization = _comparison(diags[reference], diags['Full'], reference, 'Full')
            reasons_b.extend(reasons)
            scores_b.append(score)
            localization_b[reference] = localization
            supported_b = supported_b or (bool(reasons) and localization['non_worse'])
    return {'A_supported': bool(reasons_a) and localization_a['non_worse'], 'B_supported': supported_b,
            'A_candidate': bool(reasons_a), 'B_candidate': bool(reasons_b),
            'A_score': score_a, 'B_score': max(scores_b, default=0.0),
            'reasons': reasons_a + reasons_b, 'A_reasons': reasons_a, 'B_reasons': reasons_b,
            'A_localization': localization_a, 'B_localization': localization_b,
            'A_tradeoffs': localization_a['tradeoffs'],
            'B_tradeoffs': {ref: item['tradeoffs'] for ref, item in localization_b.items()},
            'qualification_note': 'Formal titles require an improvement with no worse TP, FP, FN or class-error count '
                                  'and no mean IoU decrease of 0.05 or more on shared matched GT. '
                                  'Mixed tradeoffs remain candidates for manual review, not automatic success/failure labels.',
            'requires_manual_review': True}


def candidate_rows(bundle, score_thr=.30):
    """Rank every Test image; preserve no-evidence rows for a complete scan audit."""
    rows = []
    invalid_counts = Counter((item['alias'], item['image_id']) for item in bundle['invalid'])
    for image_id, metadata in bundle['images'].items():
        diagnostics = {alias: diagnose(bundle['gt'][image_id], bundle['predictions'][alias][image_id], score_thr)
                       for alias in MODELS}
        evidence = case_evidence(diagnostics, bundle['illumination'][image_id])
        row = {'image_id': image_id, 'file_name': metadata['file_name'],
               'illumination': bundle['illumination'][image_id], 'seed': bundle['seed'],
               'score_threshold': score_thr, 'iou_threshold': .50}
        for alias, diagnostic in diagnostics.items():
            for key in ('tp', 'fp', 'fn'):
                row[f'{alias}_{key.upper()}'] = diagnostic[key]
            row[f'{alias}_mean_matched_IoU'] = diagnostic['mean_iou']
            row[f'{alias}_class_errors'] = diagnostic['class_errors']
            row[f'{alias}_unmatched_GT'] = diagnostic['fn']
            row[f'{alias}_unmatched_GT_ids'] = json.dumps(diagnostic['unmatched_gt_ids'])
            row[f'{alias}_matches'] = json.dumps(diagnostic['matches'], separators=(',', ':'))
            row[f'{alias}_invalid_predictions'] = invalid_counts[(alias, image_id)]
        for key in ('A_supported', 'B_supported', 'A_candidate', 'B_candidate', 'A_score', 'B_score'):
            row[key] = evidence[key]
        row['recommendation_reason'] = '; '.join(evidence['reasons']) if evidence['reasons'] else 'No directional evidence at this threshold'
        row['A_tradeoffs'] = '; '.join(evidence['A_tradeoffs'])
        row['B_tradeoffs'] = json.dumps(evidence['B_tradeoffs'], ensure_ascii=False)
        row['qualification_note'] = evidence['qualification_note']
        row['review_status'] = 'REQUIRES_MANUAL_REVIEW'
        rows.append(row)
    return sorted(rows, key=lambda row: (-max(row['A_score'], row['B_score']), -row['B_score'], row['image_id']))
