"""Illumination-consistent foreground CopyBlend at the native batch-collate stage.

Boxes entering/leaving this module are normalized CXCYWH. RGB blending may take
place in the shared affine normalization domain, which is mathematically equal
to RGB blending. Background photometry always uses explicitly inverse-normalized
[0, 1] RGB. No appearance-based acquisition-class inference is performed.
"""
import math
import random
from collections import Counter
from copy import deepcopy

import torch

from .visible_geometry import (assert_visible_alignment, instance_fields,
                               translated_record)

STAT_NAMES = ('attempted_pastes', 'accepted_pastes', 'skipped_invalid_mask',
              'skipped_incompatible_light', 'skipped_collision',
              'skipped_no_valid_location', 'foreground_area_ratio',
              'acceptance_ratio', 'added_instances_per_image', 'skipped_by_light',
              'skipped_by_collision', 'skipped_by_invalid_mask', 'image_count',
              'foreground_area_sum', 'li_enabled_image_count')
ADDITIVE_STAT_NAMES = (STAT_NAMES[:6] +
                      ('image_count', 'foreground_area_sum', 'li_enabled_image_count'))


def summarize_copyblend_stats(totals):
    """Derive ratios from summed counters, never average batch-wise ratios.

    Attempts count donor/destination trials; a trial may retry several locations.
    Skip counters count rejection events (including donor filtering and location
    retries), so they are not disjoint outcomes of attempted_pastes. image_count
    includes batches where augmentation was disabled or lost the random draw.
    The same counters describe native CopyBlend for a quantity-control baseline.
    """
    values = dict(zip(STAT_NAMES, totals)) if not isinstance(totals, dict) else totals
    result = {name: float(values.get(name, 0.0)) for name in ADDITIVE_STAT_NAMES}
    attempted, images = result['attempted_pastes'], result['image_count']
    result.update(
        acceptance_ratio=result['accepted_pastes'] / attempted if attempted else 0.0,
        added_instances_per_image=result['accepted_pastes'] / images if images else 0.0,
        foreground_area_ratio=result['foreground_area_sum'] / images if images else 0.0,
        skipped_by_light=result['skipped_incompatible_light'],
        skipped_by_collision=result['skipped_collision'],
        skipped_by_invalid_mask=result['skipped_invalid_mask'])
    return result


def boxes_xyxy(target, height, width):
    boxes = target['boxes'].float()
    xy = boxes[:, :2] * boxes.new_tensor((width, height))
    wh = boxes[:, 2:] * boxes.new_tensor((width, height))
    return torch.cat((xy - wh / 2, xy + wh / 2), -1)


def to_rgb(image, normalized, mean, std):
    image = image.float()
    if normalized:
        image = image * image.new_tensor(std)[:, None, None] + image.new_tensor(mean)[:, None, None]
    return image.clamp(0, 1)


def background_statistics(rgb, target, rect, ring=4):
    """Local neighborhood excluding all labeled fruit boxes, not tomato RGB means."""
    _, height, width = rgb.shape
    x0, y0, x1, y1 = rect
    l, t, r, b = max(0, x0-ring), max(0, y0-ring), min(width, x1+ring), min(height, y1+ring)
    allowed = torch.ones((b-t, r-l), dtype=torch.bool, device=rgb.device)
    # Exclude the candidate itself and all known fruit, even those with bad masks.
    allowed[max(y0-t, 0):min(y1-t, b-t), max(x0-l, 0):min(x1-l, r-l)] = False
    for box in boxes_xyxy(target, height, width):
        bx0, by0, bx1, by1 = box.tolist()
        a, c = max(0, math.floor(bx0)-l), min(r-l, math.ceil(bx1)-l)
        d, e = max(0, math.floor(by0)-t), min(b-t, math.ceil(by1)-t)
        if c > a and e > d:
            allowed[d:e, a:c] = False
    if int(allowed.sum()) < 8:
        return None
    pixels = rgb[:, t:b, l:r][:, allowed]
    luminance = (pixels * pixels.new_tensor((0.2126, 0.7152, 0.0722))[:, None]).sum(0)
    return torch.cat((pixels.mean(1), luminance.std(unbiased=False)[None]))


class LICopyBlend:
    def __init__(self, enabled=False, illumination_sampling=True, foreground_only=True,
                 light_policy='strict', local_photometry=True, photometry_threshold=0.20,
                 max_retries=20, collision_tolerance=0.0, num_classes=3,
                 normalized=False, normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225), neighborhood=4, max_donor_attempts=12):
        if light_policy not in ('strict', 'prefer'):
            raise ValueError('LI-CopyBlend light_policy must be strict or prefer')
        if max_retries < 1 or max_donor_attempts < 1 or not 0 <= collision_tolerance <= 1:
            raise ValueError('Invalid LI-CopyBlend retry count or collision tolerance')
        if len(normalize_mean) != 3 or len(normalize_std) != 3 or min(normalize_std) <= 0:
            raise ValueError('LI-CopyBlend normalization must contain three positive standard deviations')
        self.enabled = enabled
        self.illumination_sampling = illumination_sampling
        self.foreground_only = foreground_only
        self.light_policy = light_policy
        self.local_photometry = local_photometry
        self.photometry_threshold = photometry_threshold
        self.max_retries = max_retries
        self.max_donor_attempts = max_donor_attempts
        self.collision_tolerance = collision_tolerance
        self.num_classes = num_classes
        self.normalized = normalized
        self.normalize_mean, self.normalize_std = tuple(normalize_mean), tuple(normalize_std)
        self.neighborhood = neighborhood
        self.last_stats = dict.fromkeys(STAT_NAMES, 0.0)
        self.cumulative_stats = Counter()
        self._totals = Counter()

    def record_stats(self, images, targets, stats=None):
        """Attach exactly one batch payload, including zero-attempt batches."""
        counters = Counter(stats or {})
        counters['image_count'] = len(targets)
        counters['li_enabled_image_count'] = len(targets) if self.enabled else 0
        self.last_stats = summarize_copyblend_stats(counters)
        self._totals.update({name: self.last_stats[name] for name in ADDITIVE_STAT_NAMES})
        self.cumulative_stats = Counter(summarize_copyblend_stats(self._totals))
        if targets:
            targets[0]['li_copyblend_stats'] = images.new_tensor([self.last_stats[k] for k in STAT_NAMES])
        return images, targets

    def _collision(self, target, rect, height, width):
        if not len(target['labels']):
            return False
        existing = boxes_xyxy(target, height, width)
        new = existing.new_tensor(rect)
        overlap = (torch.minimum(existing[:, 2:], new[2:]) - torch.maximum(existing[:, :2], new[:2])).clamp_min(0).prod(-1)
        denominator = torch.minimum((existing[:, 2:] - existing[:, :2]).clamp_min(0).prod(-1),
                                    (new[2:] - new[:2]).prod()).clamp_min(1)
        if bool((overlap / denominator > self.collision_tolerance).any()):
            return True
        # A genuine visible mask can extend outside the annotated bbox. Check
        # support too, so we never silently overwrite such an existing fruit.
        if 'masks' in target:
            x0, y0, x1, y1 = (int(v) for v in rect)
            mask_overlap = target['masks'][:, y0:y1, x0:x1].flatten(1).sum(1)
            mask_overlap *= target['visible_mask_valid'].to(mask_overlap.dtype)
            if self.collision_tolerance == 0:
                return bool((mask_overlap > 0).any())
            intersects = mask_overlap > 0
            if intersects.any():
                mask_area = target['masks'][intersects].flatten(1).sum(1)
                denominator = mask_area.clamp_max((x1-x0) * (y1-y0)).clamp_min(1)
                return bool((mask_overlap[intersects] / denominator > self.collision_tolerance).any())
        return False

    def _compatible(self, source, instance, destination, src_rgb, dst_rgb, src_rect, dst_rect, src_stats):
        if not self.illumination_sampling:
            return True
        source_light = int(source['source_illumination_id'][instance])
        if source_light < 0:
            return False
        dst_light = int(destination.get('image_illumination_id', torch.tensor([-1]))[0])
        if dst_light >= 0 and source_light != dst_light and self.light_policy == 'strict':
            return False
        # Mixed/unknown destination provenance requires a reliable local comparison.
        need_local = dst_light < 0 or source_light != dst_light or self.local_photometry
        if need_local:
            if not self.local_photometry:
                return False
            dst_stats = background_statistics(dst_rgb, destination, dst_rect, self.neighborhood)
            return (src_stats is not None and dst_stats is not None and
                    bool((src_stats - dst_stats).abs().max() <= self.photometry_threshold))
        return True

    def __call__(self, images, targets, *, alpha, area_threshold, num_objects,
                 random_num_objects=False, with_expand=False, expand_ratios=(0.1, 0.25)):
        stats = Counter({key: 0.0 for key in STAT_NAMES})
        height, width = images.shape[-2:]
        out_images, out_targets = images.clone(), deepcopy(targets)
        for target in out_targets:
            target.setdefault('blend_opacity', images.new_ones(len(target['labels'])))
            target.setdefault('area_from_mask', torch.zeros_like(target['labels'], dtype=torch.bool))
        rgb = [to_rgb(image, self.normalized, self.normalize_mean, self.normalize_std) for image in images]
        destination_rgb = [image.clone() for image in rgb]
        pool = []
        for image_index, target in enumerate(targets):
            assert_visible_alignment(target, height, width)
            if 'masks' not in target:
                raise ValueError('LI-CopyBlend requires instance masks and instance_mask_sync=True')
            pixel_boxes = boxes_xyxy(target, height, width)
            for instance in range(len(target['labels'])):
                if not bool(target.get('source_is_train', torch.zeros_like(target['labels'], dtype=torch.bool))[instance]):
                    continue
                valid = (bool(target['visible_mask_valid'][instance]) and bool(target['masks'][instance].any()))
                if not valid:
                    stats['skipped_invalid_mask'] += 1
                    continue
                if not 0 <= int(target['labels'][instance]) < self.num_classes or float(target['area'][instance]) < area_threshold:
                    continue
                box = pixel_boxes[instance]
                if not torch.isfinite(box).all() or not bool((box[2:] > box[:2]).all()):
                    continue
                sx0, sy0 = max(0, math.floor(float(box[0]))), max(0, math.floor(float(box[1])))
                sx1, sy1 = min(width, math.ceil(float(box[2]))), min(height, math.ceil(float(box[3])))
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                if self.illumination_sampling and int(target['source_illumination_id'][instance]) < 0:
                    stats['skipped_incompatible_light'] += 1
                    continue
                if not bool(target['masks'][instance, sy0:sy1, sx0:sx1].any()):
                    stats['skipped_invalid_mask'] += 1
                    continue
                pool.append((image_index, instance, (sx0, sy0, sx1, sy1)))
        for destination_index, destination in enumerate(out_targets):
            # Defense in depth: a val/test target cannot enter the train augmentation path.
            if not bool(destination.get('image_is_train', torch.tensor([False]))[0]):
                continue
            desired = min(num_objects, len(pool))
            if random_num_objects and desired:
                desired = random.randint(1, desired)
            candidates = list(pool)
            random.shuffle(candidates)
            light = int(destination.get('image_illumination_id', torch.tensor([-1]))[0])
            if self.illumination_sampling and self.light_policy == 'strict' and light >= 0:
                compatible_candidates = [entry for entry in candidates if int(targets[entry[0]]['source_illumination_id'][entry[1]]) == light]
                stats['skipped_incompatible_light'] += len(candidates) - len(compatible_candidates)
                candidates = compatible_candidates
            if self.illumination_sampling and self.light_policy == 'prefer':
                candidates.sort(key=lambda entry: int(targets[entry[0]]['source_illumination_id'][entry[1]]) != light)
            candidates = candidates[:max(desired, 1) * self.max_donor_attempts]
            accepted_here = 0
            for source_index, instance, original_rect in candidates:
                if accepted_here >= desired:
                    break
                stats['attempted_pastes'] += 1
                source = targets[source_index]
                sx0, sy0, sx1, sy1 = original_rect
                # Expansion remains configurable; hard foreground support prevents RGB spill.
                if with_expand:
                    expansion = random.uniform(*expand_ratios)
                    ew, eh = int((sx1-sx0) * expansion), int((sy1-sy0) * expansion)
                    sx0, sy0, sx1, sy1 = max(0, sx0-ew), max(0, sy0-eh), min(width, sx1+ew), min(height, sy1+eh)
                patch_mask = source['masks'][instance, sy0:sy1, sx0:sx1].bool()
                ph, pw = patch_mask.shape
                yy, xx = patch_mask.nonzero(as_tuple=True)
                source_statistics = (background_statistics(rgb[source_index], source, (sx0, sy0, sx1, sy1), self.neighborhood)
                                     if self.illumination_sampling and self.local_photometry else None)
                success = False
                for _ in range(self.max_retries):
                    x0, y0 = random.randint(0, width-pw), random.randint(0, height-ph)
                    rect = (x0, y0, x0+pw, y0+ph)
                    # Tight inclusive-exclusive support box: no artificial 1px expansion.
                    tight = (x0+int(xx.min()), y0+int(yy.min()), x0+int(xx.max())+1, y0+int(yy.max())+1)
                    if self._collision(destination, tight if self.foreground_only else rect, height, width):
                        stats['skipped_collision'] += 1
                        continue
                    if not self._compatible(source, instance, destination, rgb[source_index],
                                            destination_rgb[destination_index], (sx0, sy0, sx1, sy1), rect, source_statistics):
                        stats['skipped_incompatible_light'] += 1
                        continue
                    source_patch = images[source_index, :, sy0:sy1, sx0:sx1]
                    destination_patch = out_images[destination_index, :, y0:y0+ph, x0:x0+pw]
                    blend = destination_patch * (1-alpha) + source_patch * alpha
                    # torch.where guarantees exact preservation outside the hard mask.
                    out_images[destination_index, :, y0:y0+ph, x0:x0+pw] = torch.where(patch_mask[None], blend, destination_patch) if self.foreground_only else blend
                    destination_rgb[destination_index][:, y0:y0+ph, x0:x0+pw] = to_rgb(
                        out_images[destination_index, :, y0:y0+ph, x0:x0+pw], self.normalized, self.normalize_mean, self.normalize_std)
                    record = translated_record(source, instance, (sx0, sy0, sx1, sy1), rect, (height, width))
                    record['labels'] = source['labels'][instance:instance+1].clone()
                    bx0, by0, bx1, by1 = tight
                    record['boxes'] = destination['boxes'].new_tensor([[(bx0+bx1)/2/width, (by0+by1)/2/height, (bx1-bx0)/width, (by1-by0)/height]])
                    record['area'] = destination['area'].new_tensor([int(patch_mask.sum())])
                    record['area_from_mask'] = torch.ones(1, dtype=torch.bool, device=destination['labels'].device)
                    record['iscrowd'] = torch.zeros_like(source['iscrowd'][instance:instance+1])
                    record['blend_opacity'] = destination['blend_opacity'].new_tensor([alpha])
                    # Preserve original target weights; new instance opacity is not mask area.
                    if 'mixup' not in destination:
                        destination['mixup'] = images.new_ones(len(destination['labels']))
                    record['mixup'] = destination['mixup'].new_tensor([alpha])
                    for key in instance_fields(destination):
                        if key not in record:
                            raise ValueError(f'LI-CopyBlend cannot append missing instance metadata {key}')
                        destination[key] = torch.cat((destination[key], record[key]), 0)
                    stats['accepted_pastes'] += 1
                    stats['foreground_area_sum'] += int(patch_mask.sum()) / (height * width)
                    accepted_here += 1
                    success = True
                    break
                if not success:
                    stats['skipped_no_valid_location'] += 1
            assert_visible_alignment(destination, height, width)
        return self.record_stats(out_images, out_targets, stats)
