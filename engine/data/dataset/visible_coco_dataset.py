"""COCO detection with optional genuine instance-mask metadata, independent of losses."""
import hashlib
import json
from pathlib import Path

import torch
from faster_coco_eval.core import mask as coco_mask

from .coco_dataset import CocoDetection
from .._misc import convert_to_tv_tensor
from ..visible_geometry import assert_visible_alignment
from ...core import register

ILLUMINATION_IDS = {'natural': 0, 'artificial': 1, 'low_light': 2, 'sodium_light': 3}
# Explicit labels in the real Train COCO light_condition field. These aliases
# describe acquisition metadata; no image/filename/color classification is used.
ILLUMINATION_LABEL_ALIASES = {
    '自然光': 'natural', '人工光': 'artificial', '微弱光': 'low_light', '钠黄光': 'sodium_light'}


def illumination_id(value):
    """Only explicit, documented acquisition metadata; never infer from RGB or filenames."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        return ILLUMINATION_IDS.get(ILLUMINATION_LABEL_ALIASES.get(normalized, normalized), -1)
    if isinstance(value, int) and not isinstance(value, bool) and value in ILLUMINATION_IDS.values():
        return value
    return -1


def read_sidecar(path, ann_file, coco):
    path = Path(path).expanduser()
    with path.open('r', encoding='utf-8') as handle:
        sidecar = json.load(handle)
    if sidecar.get('version') != 'visible-mask-sidecar-v1':
        raise ValueError(f'Unsupported visible-mask sidecar format: {path}')
    digest = hashlib.sha256(Path(ann_file).read_bytes()).hexdigest()
    if sidecar.get('coco_annotation_sha256') != digest:
        raise ValueError('Visible-mask sidecar does not match the COCO annotation SHA256')
    for key, record in sidecar.get('annotations', {}).items():
        original = coco.anns.get(int(key))
        if original is None or any(int(record.get(k, -1)) != int(original[k]) for k in ('image_id', 'category_id')):
            raise ValueError(f'Visible-mask sidecar identity mismatch for annotation_id={key}')
    return sidecar


def decode_mask(segmentation, height, width):
    if not segmentation:
        return torch.zeros((height, width), dtype=torch.uint8)
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get('counts'), list):
            rle = coco_mask.frPyObjects(rle, height, width)
        elif isinstance(rle.get('counts'), str):
            rle = dict(rle, counts=rle['counts'].encode('ascii'))
    else:
        rle = coco_mask.frPyObjects(segmentation, height, width)
    decoded = torch.as_tensor(coco_mask.decode(rle), dtype=torch.uint8)
    if decoded.ndim == 3:
        decoded = decoded.bool().any(-1).to(torch.uint8)
    if tuple(decoded.shape) != (height, width):
        raise ValueError('Mask segmentation dimensions differ from the COCO image')
    return decoded


@register()
class VisibleMaskCocoDetection(CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False,
                 remap_mscoco_category=False, visible_mask_sidecar=None,
                 visible_mask_required=False, category_ids=None, split='unknown', illumination_map=None):
        # Native bbox/category preparation remains the authoritative detection path.
        super().__init__(img_folder, ann_file, transforms, False, remap_mscoco_category, category_ids)
        if split not in ('train', 'val', 'validation', 'test', 'unknown'):
            raise ValueError(f'Unrecognized dataset split: {split}')
        self.split = split
        self.visible_mask_sidecar = visible_mask_sidecar
        if visible_mask_required and not visible_mask_sidecar:
            raise ValueError('visible_mask_required=True requires visible_mask_sidecar')
        self.visible_data = read_sidecar(visible_mask_sidecar, ann_file, self.coco) if visible_mask_sidecar else None
        if isinstance(illumination_map, (str, Path)):
            with Path(illumination_map).expanduser().open('r', encoding='utf-8') as handle:
                illumination_map = json.load(handle)
        self.illumination_map = illumination_map or {}

    def load_item(self, idx):
        image, target = super().load_item(idx)
        w, h = image.size
        image_id = int(self.ids[idx])
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        # Match the native COCO converter's crowd and clipped-box filtering exactly.
        kept = []
        for obj in annotations:
            x, y, bw, bh = obj['bbox']
            if not obj.get('iscrowd', 0) and min(x + bw, w) > max(x, 0) and min(y + bh, h) > max(y, 0):
                kept.append(obj)
        if len(kept) != len(target['labels']):
            raise ValueError('Native detection and sidecar annotation order do not align')
        image_meta = self.coco.imgs[image_id]
        explicit_light = next((image_meta[key] for key in
            ('illumination', 'illumination_label', 'source_illumination_id', 'light_condition')
            if image_meta.get(key) is not None), None)
        mapped_light = self.illumination_map.get(str(image_id), self.illumination_map.get(image_meta.get('file_name', '')))
        light = illumination_id(mapped_light if mapped_light is not None else explicit_light)
        records = self.visible_data.get('annotations', {}) if self.visible_data else None
        masks, validity, groups = [], [], []
        for annotation in kept:
            record = records.get(str(annotation['id'])) if records is not None else annotation
            segmentation = record.get('segmentation') if record else None
            valid = bool(record and record.get('visible_mask_valid', records is None) and segmentation)
            try:
                mask = decode_mask(segmentation, h, w) if valid else torch.zeros((h, w), dtype=torch.uint8)
            except (ValueError, TypeError, KeyError, RuntimeError):
                # Invalid optional masks never remove a valid detection target.
                mask = torch.zeros((h, w), dtype=torch.uint8)
                valid = False
            valid = valid and bool(mask.any())
            masks.append(mask)
            validity.append(valid)
            group = (record or {}).get('visibility_group', (record or {}).get('visibility_level', 'unknown'))
            groups.append({'easy': 0, 'mild': 1, 'moderate': 2}.get(group, -1))
        n = len(kept)
        target.update(
            masks=convert_to_tv_tensor(torch.stack(masks) if masks else torch.zeros((0, h, w), dtype=torch.uint8), 'masks'),
            visible_mask_valid=torch.tensor(validity, dtype=torch.bool),
            annotation_ids=torch.tensor([a['id'] for a in kept], dtype=torch.int64),
            visibility_group=torch.tensor(groups, dtype=torch.int64),
            source_illumination_id=torch.full((n,), light, dtype=torch.int64),
            illumination_source_id=torch.full((n,), light, dtype=torch.int64),
            source_is_train=torch.full((n,), self.split == 'train', dtype=torch.bool),
            image_illumination_id=torch.tensor([light], dtype=torch.int64),
            image_is_train=torch.tensor([self.split == 'train'], dtype=torch.bool),
            blend_opacity=torch.ones(n, dtype=torch.float32),
            area_from_mask=torch.zeros(n, dtype=torch.bool))
        assert_visible_alignment(target, h, w)
        return image, target

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        size = image.size[::-1] if hasattr(image, 'size') and not isinstance(image, torch.Tensor) else image.shape[-2:]
        assert_visible_alignment(target, *size)
        return image, target
