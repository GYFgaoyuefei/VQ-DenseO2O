"""Instance metadata geometry. Masks are binary support, never opacity.

Image-level tensors are explicitly excluded even when their first dimension
coincidentally equals the number of instances (e.g. two instances/orig_size).
"""
import itertools
import os

import torch
from torch.nn import functional as F

IMAGE_FIELDS = frozenset(('image_id', 'idx', 'orig_size', 'size', 'padding',
                         'image_illumination_id', 'image_is_train', 'li_copyblend_stats'))
INSTANCE_FIELDS = frozenset(('boxes', 'labels', 'masks', 'visible_mask_valid',
    'annotation_ids', 'visibility_group', 'area', 'iscrowd',
    'source_illumination_id', 'illumination_source_id', 'source_is_train',
    'mixup', 'blend_opacity', 'area_from_mask', 'keypoints'))
_synthetic_counter = itertools.count(1)


def synthetic_annotation_id():
    """Negative IDs distinguish synthetic annotations; PID/counter separate workers."""
    return -((os.getpid() << 32) + next(_synthetic_counter))


def instance_fields(target, instance_count=None):
    n = len(target['labels']) if instance_count is None else instance_count
    return [k for k, v in target.items() if k not in IMAGE_FIELDS and
            isinstance(v, torch.Tensor) and v.ndim > 0 and
            (k in INSTANCE_FIELDS or len(v) == n)]


def filter_instances(target, keep):
    # Validate before mutating anything: malformed input must not partially
    # filter labels while leaving some metadata at the old instance count.
    assert_instance_alignment(target)
    for key in instance_fields(target):
        value = target[key]
        if len(value) != len(keep):
            raise ValueError(f'Instance field {key} has {len(value)} rows; expected {len(keep)}')
        target[key] = value[keep]
    return target


def assert_instance_alignment(target):
    n = len(target['labels'])
    for key in instance_fields(target):
        if len(target[key]) != n:
            raise ValueError(f'Instance field {key} does not align with labels ({n})')


def assert_visible_alignment(target, height, width):
    assert_instance_alignment(target)
    n = len(target['labels'])
    if not torch.isfinite(target['boxes']).all():
        raise ValueError('Non-finite target boxes')
    if 'masks' not in target:
        return
    masks = target['masks']
    if tuple(masks.shape) != (n, height, width):
        raise ValueError(f'Stale instance masks: {tuple(masks.shape)}; expected {(n, height, width)}')
    if masks.requires_grad or ((masks != 0) & (masks != 1)).any():
        raise ValueError('Instance masks must be binary geometric support without gradients')
    nonempty = masks.flatten(1).bool().any(1)
    target['visible_mask_valid'] = target.get('visible_mask_valid', nonempty).bool() & nonempty


def concatenate_instances(targets):
    """Preserve the anchor's image metadata; concatenate every instance tensor."""
    for target in targets:
        assert_instance_alignment(target)
    result = dict(targets[0])
    fields = set().union(*(instance_fields(t) for t in targets))
    for key in fields:
        if any(key not in t for t in targets):
            raise ValueError(f'Cannot combine targets with inconsistent instance field {key}')
        result[key] = torch.cat([t[key] for t in targets], 0)
    if 'image_illumination_id' in result:
        # Neither Mosaic nor MixUp has a reliable single image light provenance.
        result['image_illumination_id'] = torch.full_like(result['image_illumination_id'], -1)
    return result


def mosaic_masks(masks, quadrant, height, width):
    top, left = quadrant // 2 * height, quadrant % 2 * width
    h, w = masks.shape[-2:]
    return F.pad(masks, (left, width * 2 - left - w, top, height * 2 - top - h))


def resize_masks(target, size):
    size = (size, size) if isinstance(size, int) else tuple(size)
    masks = target['masks']
    if not len(masks):
        target['masks'] = masks.new_zeros((0, *size))
    elif (masks.ndim == 3 and tuple(masks.shape[-2:]) == size
          and all(isinstance(dim, int) and dim > 0 for dim in size)
          and masks.dtype in (torch.uint8, torch.bool)):
        # Identity nearest resize needs no FP32 intermediate. Keep a separate
        # plain tensor, as interpolation does, so later edits cannot affect a
        # donor/cache reference. Other dtypes retain the original FP32 path.
        target['masks'] = masks.as_subclass(torch.Tensor).clone(memory_format=torch.contiguous_format)
    else:
        target['masks'] = F.interpolate(masks[:, None].float(), size=size, mode='nearest')[:, 0].to(masks.dtype)
    if 'area_from_mask' in target:
        from_mask = target['area_from_mask'].bool()
        target['area'][from_mask] = target['masks'][from_mask].flatten(1).sum(1).to(target['area'].dtype)
    assert_visible_alignment(target, *size)


def translated_record(source, instance, source_rect, destination_rect, image_size):
    """Metadata for one translation; geometric bbox/area are supplied by caller."""
    sx0, sy0, sx1, sy1 = source_rect
    x0, y0, x1, y1 = destination_rect
    fields = instance_fields(source)
    # The translated mask is built below; cloning its full source first would
    # only allocate a second mask that is immediately discarded.
    record = {k: source[k][instance:instance + 1].clone() for k in fields
              if k not in ('boxes', 'labels', 'area', 'mixup', 'masks')}
    if 'masks' in fields:
        mask = source['masks'].new_zeros((1, *image_size))
        mask[:, y0:y1, x0:x1] = source['masks'][instance:instance + 1, sy0:sy1, sx0:sx1]
        record['masks'] = mask
        if 'visible_mask_valid' in record:
            record['visible_mask_valid'] &= mask.flatten(1).bool().any(1)
    if 'annotation_ids' in record:
        record['annotation_ids'].fill_(synthetic_annotation_id())
    if 'keypoints' in record:
        record['keypoints'][..., :2] += record['keypoints'].new_tensor((x0 - sx0, y0 - sy0))
    return record


def append_records(target, records):
    if not records:
        return
    original_count = len(target['labels']) - len(records)
    fields = set(instance_fields(target, original_count)) - {'boxes', 'labels', 'area', 'mixup'}
    for key in fields:
        if len(target[key]) != original_count or any(key not in record or len(record[key]) != 1 for record in records):
            raise ValueError(f'Cannot append incomplete instance field {key}')
    if any(set(record) != fields for record in records):
        raise ValueError('Pasted records must contain every propagated instance field')
    for key in fields:
        target[key] = torch.cat([target[key]] + [r[key] for r in records], 0)
    assert_instance_alignment(target)
