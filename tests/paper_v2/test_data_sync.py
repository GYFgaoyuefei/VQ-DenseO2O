import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
import torchvision.transforms.v2 as T
from PIL import Image

from engine.data._misc import convert_to_tv_tensor
from engine.data.dataset import CocoDetection, VisibleMaskCocoDetection
from engine.data.transforms import PadToSize, SanitizeBoundingBoxes, SanitizeVisibleBoundingBoxes
from engine.data.transforms.functional import crop
from engine.data.transforms.mosaic import Mosaic
from engine.data.visible_geometry import (assert_visible_alignment, concatenate_instances,
                                          filter_instances, resize_masks)


def xyxy_target(boxes, size=16, light=0):
    n = len(boxes)
    masks = torch.zeros((n, size, size), dtype=torch.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        masks[i, int(y0):int(y1), int(x0):int(x1)] = 1
    return dict(boxes=convert_to_tv_tensor(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), 'boxes', spatial_size=(size, size)),
                labels=torch.arange(n, dtype=torch.int64), masks=convert_to_tv_tensor(masks, 'masks'),
                visible_mask_valid=torch.ones(n, dtype=torch.bool), annotation_ids=torch.arange(n),
                visibility_group=torch.zeros(n, dtype=torch.int64), area=torch.ones(n),
                iscrowd=torch.zeros(n, dtype=torch.int64), source_illumination_id=torch.full((n,), light),
                illumination_source_id=torch.full((n,), light), source_is_train=torch.ones(n, dtype=torch.bool),
                blend_opacity=torch.ones(n), image_illumination_id=torch.tensor([light]),
                image_is_train=torch.tensor([True]), image_id=torch.tensor([12]), idx=torch.tensor([7]),
                orig_size=torch.tensor([size, size]), custom_instance_weight=torch.arange(n, dtype=torch.float32))


class DataGeometryTests(unittest.TestCase):
    def test_same_size_mask_resize_preserves_values_and_independent_storage(self):
        for dtype in (torch.uint8, torch.bool):
            for wrapped in (False, True):
                with self.subTest(dtype=dtype, wrapped=wrapped):
                    target = xyxy_target([[1, 1, 5, 5], [6, 6, 9, 9], [10, 10, 14, 14]])
                    masks = target['masks'].as_subclass(torch.Tensor).to(dtype)
                    masks[2].zero_()
                    target['masks'] = convert_to_tv_tensor(masks, 'masks') if wrapped else masks
                    target['visible_mask_valid'][1] = False
                    target['area_from_mask'] = torch.tensor([True, False, True])
                    target['area'] = torch.tensor([999., 888., 777.])
                    source_masks = target['masks']
                    expected_masks = source_masks.clone()
                    metadata = {k: v.clone() for k, v in target.items()
                                if k not in ('masks', 'area', 'visible_mask_valid')}

                    resize_masks(target, (16, 16))

                    self.assertIs(type(target['masks']), torch.Tensor)
                    self.assertEqual(target['masks'].dtype, dtype)
                    self.assertTrue(torch.equal(target['masks'], expected_masks))
                    self.assertNotEqual(target['masks'].data_ptr(), source_masks.data_ptr())
                    self.assertEqual(target['visible_mask_valid'].tolist(), [True, False, False])
                    self.assertEqual(target['area'].tolist(), [16., 888., 0.])
                    for key, value in metadata.items():
                        self.assertTrue(torch.equal(target[key], value), key)
                    target['masks'].zero_()
                    self.assertTrue(torch.equal(source_masks, expected_masks))

    def test_same_size_mask_resize_still_rejects_invalid_targets(self):
        for invalid_field, message in (('masks', 'binary'),
                                       ('annotation_ids', 'annotation_ids'),
                                       ('boxes', 'Non-finite')):
            with self.subTest(invalid_field=invalid_field):
                target = xyxy_target([[1, 1, 5, 5], [6, 6, 9, 9]])
                if invalid_field == 'masks':
                    target['masks'][0, 0, 0] = 2
                elif invalid_field == 'annotation_ids':
                    target['annotation_ids'] = target['annotation_ids'][:1]
                else:
                    target['boxes'][0, 0] = float('nan')
                with self.assertRaisesRegex(ValueError, message):
                    resize_masks(target, 16)

    def test_pad_to_size_supports_current_torchvision_and_binary_masks(self):
        for fill in (0, 114):
            with self.subTest(fill=fill):
                target = xyxy_target([[2, 2, 6, 6]], size=8)
                image, result = PadToSize((16, 12), fill=fill)(Image.new('RGB', (8, 8)), target)
                self.assertEqual(image.size, (16, 12))
                self.assertEqual(image.getpixel((15, 11)), (fill, fill, fill))
                self.assertEqual(result['padding'].tolist(), [0, 0, 8, 4])
                self.assertTrue(torch.equal(result['boxes'], target['boxes']))
                self.assertEqual(result['masks'].sum().item(), 16)
                assert_visible_alignment(result, 12, 16)

    def test_malformed_instance_metadata_rejected_before_filter_or_concatenation(self):
        target = xyxy_target([[1, 1, 5, 5], [6, 6, 9, 9]])
        target['annotation_ids'] = target['annotation_ids'][:1]
        labels = target['labels'].clone()
        with self.assertRaisesRegex(ValueError, 'annotation_ids'):
            filter_instances(target, torch.tensor([True, False]))
        self.assertTrue(torch.equal(target['labels'], labels))
        with self.assertRaisesRegex(ValueError, 'annotation_ids'):
            concatenate_instances([target, xyxy_target([])])

    def test_invalid_mask_does_not_remove_legal_bbox(self):
        target = xyxy_target([[1, 1, 5, 5], [6, 6, 9, 9], [4, 4, 4, 4]])
        target['masks'][0].zero_()
        image, result = SanitizeVisibleBoundingBoxes()(Image.new('RGB', (16, 16)), target)
        self.assertEqual(result['labels'].tolist(), [0, 1])
        self.assertEqual(result['visible_mask_valid'].tolist(), [False, True])
        for key in ('area', 'iscrowd', 'annotation_ids', 'visibility_group', 'source_illumination_id', 'custom_instance_weight'):
            self.assertEqual(len(result[key]), 2, key)
        self.assertEqual(result['image_id'].tolist(), [12])
        self.assertEqual(result['orig_size'].tolist(), [16, 16])

    def test_native_sanitize_has_identical_bbox_predicate_and_aligned_metadata(self):
        target = xyxy_target([[1, 1, 5, 5], [6, 6, 6, 9]])
        native_target = {k: v for k, v in target.items() if k not in ('masks', 'visible_mask_valid')}
        a = T.SanitizeBoundingBoxes()(Image.new('RGB', (16, 16)), copy.deepcopy(native_target))[1]
        b = SanitizeBoundingBoxes()(Image.new('RGB', (16, 16)), native_target)[1]
        self.assertTrue(torch.equal(a['boxes'], b['boxes']))
        self.assertTrue(torch.equal(a['labels'], b['labels']))
        self.assertEqual(len(b['area']), 1)
        self.assertEqual(len(b['iscrowd']), 1)
        self.assertEqual(b['orig_size'].tolist(), [16, 16])

    def test_crop_filters_all_metadata_but_not_image_fields(self):
        target = xyxy_target([[1, 1, 4, 4], [10, 10, 13, 13]])
        image, result = crop(Image.new('RGB', (16, 16)), target, (0, 0, 8, 8))
        self.assertEqual(result['labels'].tolist(), [0])
        self.assertEqual(result['custom_instance_weight'].tolist(), [0])
        self.assertEqual(result['size'].tolist(), [8, 8])
        self.assertEqual(result['orig_size'].tolist(), [16, 16])
        assert_visible_alignment(result, 8, 8)

    def test_mosaic_spatial_masks_and_sources_align_without_fake_image_label(self):
        targets = [xyxy_target([[2, 2, 4, 4]], size=8, light=i) for i in range(4)]
        images = [Image.new('RGB', (8, 8), (i*20, 0, 0)) for i in range(4)]
        mosaic = Mosaic(output_size=8, use_cache=False)
        image, result = mosaic.create_mosaic_from_dataset(images, targets, 8, 8)
        self.assertEqual(result['source_illumination_id'].tolist(), [0, 1, 2, 3])
        self.assertEqual(result['image_illumination_id'].tolist(), [-1])
        self.assertEqual(result['image_id'].tolist(), [12])
        self.assertEqual(result['orig_size'].tolist(), [8, 8])
        for i in range(4):
            y, x = i//2*8+2, i%2*8+2
            self.assertEqual(result['masks'][i, y:y+2, x:x+2].sum(), 4)
            self.assertTrue(torch.equal(result['boxes'][i], torch.tensor([x, y, x+2, y+2])))
        assert_visible_alignment(result, 16, 16)

    def test_mosaic_nonzero_image_fill_never_becomes_mask_support(self):
        image = Image.new('RGB', (8, 8))
        target = xyxy_target([[2, 2, 6, 6]], size=8)
        mosaic = Mosaic(output_size=8, rotation_range=0, translation_range=(0, 0),
                        scaling_range=(.5, .5), fill_value=114)
        result_image, result, _ = mosaic(image, target, None)
        self.assertEqual(result_image.getpixel((0, 0)), (114, 114, 114))
        self.assertEqual(result['masks'][:, 0, 0].sum().item(), 0)
        assert_visible_alignment(result, 16, 16)

    def test_cache_mosaic_matches_dataset_mosaic(self):
        targets = [xyxy_target([[2, 2, 4, 4]], size=8, light=i) for i in range(4)]
        images = [Image.new('RGB', (8, 8), (i*20, 0, 0)) for i in range(4)]
        mosaic = Mosaic(output_size=8)
        _, a = mosaic.create_mosaic_from_dataset(images, copy.deepcopy(targets), 8, 8)
        _, b = mosaic.create_mosaic_from_cache([dict(img=i, labels=t) for i, t in zip(images, targets)], 8, 8)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_b0_b1_same_random_geometry_rgb_bbox_and_class(self):
        image = Image.fromarray(torch.arange(16*16*3, dtype=torch.int64).remainder(255).byte().reshape(16, 16, 3).numpy())
        visible = xyxy_target([[2, 2, 8, 8], [10, 10, 14, 14]])
        native = {k: v for k, v in visible.items() if k not in ('masks', 'visible_mask_valid')}
        transform = T.Compose([T.RandomHorizontalFlip(.5), T.RandomZoomOut(p=.5), T.Resize((24, 24)), SanitizeBoundingBoxes()])
        torch.manual_seed(42)
        a, ta = transform(image.copy(), copy.deepcopy(native))
        torch.manual_seed(42)
        b, tb = transform(image.copy(), copy.deepcopy(visible))
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertTrue(torch.equal(ta['boxes'], tb['boxes']))
        self.assertTrue(torch.equal(ta['labels'], tb['labels']))
        assert_visible_alignment(tb, 24, 24)

    def test_mixed_instance_count_never_concatenates_image_fields(self):
        targets = [xyxy_target([[2, 2, 4, 4]], light=0), xyxy_target([], light=1)]
        result = concatenate_instances(targets)
        self.assertEqual(result['orig_size'].tolist(), [16, 16])
        self.assertEqual(result['image_id'].tolist(), [12])
        self.assertEqual(result['image_illumination_id'].tolist(), [-1])


class DatasetSidecarTests(unittest.TestCase):
    def setUp(self):
        root = Path('outputs/paper_v2_checks')
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root, prefix='data_')
        self.path = Path(self.tmp.name)
        Image.new('RGB', (16, 16)).save(self.path/'sample.png')
        self.coco = dict(images=[dict(id=12, file_name='sample.png', width=16, height=16, illumination='natural')],
            categories=[dict(id=1, name='green'), dict(id=2, name='turning'), dict(id=3, name='red')],
            annotations=[dict(id=3, image_id=12, category_id=3, bbox=[2, 2, 5, 5], area=20, iscrowd=0),
                         dict(id=4, image_id=12, category_id=1, bbox=[10, 10, 3, 3], area=9, iscrowd=0)])
        self.ann = self.path/'coco.json'
        self.ann.write_text(json.dumps(self.coco), encoding='utf-8')
        self.sidecar = dict(version='visible-mask-sidecar-v1',
            coco_annotation_sha256=hashlib.sha256(self.ann.read_bytes()).hexdigest(),
            annotations={'3': dict(image_id=12, category_id=3, visible_mask_valid=True,
                segmentation=[[2, 2, 7, 2, 2, 7]], visibility_group='easy'),
                '4': dict(image_id=12, category_id=1, visible_mask_valid=False, segmentation=[])})
        self.side = self.path/'masks.json'
        self.side.write_text(json.dumps(self.sidecar), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def make_dataset(self, **kwargs):
        return VisibleMaskCocoDetection(str(self.path), str(self.ann), None, category_ids=[1, 2, 3],
            visible_mask_sidecar=str(self.side), visible_mask_required=True, split='train', **kwargs)

    def test_b0_has_no_masks_and_b1_preserves_detection_targets(self):
        native = CocoDetection(str(self.path), str(self.ann), None, category_ids=[1, 2, 3])
        visible = self.make_dataset()
        image0, a = native[0]
        image1, b = visible[0]
        self.assertNotIn('masks', a)
        self.assertEqual(image0.tobytes(), image1.tobytes())
        for key in ('boxes', 'labels', 'area', 'iscrowd'):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertEqual(b['labels'].tolist(), [2, 0])
        self.assertEqual(b['visible_mask_valid'].tolist(), [True, False])
        self.assertEqual(b['source_illumination_id'].tolist(), [0, 0])
        self.assertEqual(b['source_is_train'].tolist(), [True, True])

    def test_sidecar_annotation_hash_mismatch_fails(self):
        self.sidecar['coco_annotation_sha256'] = 'not-a-matching-hash'
        self.side.write_text(json.dumps(self.sidecar), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.make_dataset()

    def test_sidecar_annotation_identity_mismatch_fails(self):
        self.sidecar['annotations']['3']['category_id'] = 1
        self.side.write_text(json.dumps(self.sidecar), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.make_dataset()

    def test_unknown_illumination_is_unknown_not_guessed_from_rgb(self):
        self.coco['images'][0].pop('illumination')
        self.ann.write_text(json.dumps(self.coco), encoding='utf-8')
        dataset = VisibleMaskCocoDetection(str(self.path), str(self.ann), None, category_ids=[1, 2, 3])
        _, target = dataset[0]
        self.assertEqual(target['source_illumination_id'].tolist(), [-1, -1])

    def test_real_train_light_condition_metadata_aliases(self):
        self.coco['images'][0].pop('illumination')
        for expected, label in enumerate(('自然光', '人工光', '微弱光', '钠黄光')):
            with self.subTest(light_condition=label):
                self.coco['images'][0]['light_condition'] = label
                self.ann.write_text(json.dumps(self.coco), encoding='utf-8')
                dataset = VisibleMaskCocoDetection(str(self.path), str(self.ann), None,
                    split='train', category_ids=[1, 2, 3])
                _, target = dataset[0]
                self.assertEqual(target['source_illumination_id'].tolist(), [expected, expected])
                self.assertEqual(target['image_illumination_id'].tolist(), [expected])

    def test_explicit_source_mapping_supported(self):
        _, target = self.make_dataset(illumination_map={'sample.png': 'sodium_light'})[0]
        self.assertEqual(target['source_illumination_id'].tolist(), [3, 3])

    def test_validation_split_cannot_provide_train_donor(self):
        dataset = VisibleMaskCocoDetection(str(self.path), str(self.ann), None, split='validation', category_ids=[1, 2, 3])
        _, target = dataset[0]
        self.assertFalse(target['source_is_train'].any())
        self.assertFalse(target['image_is_train'].any())

    def test_malformed_mask_keeps_bbox_and_marks_invalid(self):
        self.sidecar['annotations']['3']['segmentation'] = [[2, 2, 7]]
        self.side.write_text(json.dumps(self.sidecar), encoding='utf-8')
        _, target = self.make_dataset()[0]
        self.assertEqual(len(target['labels']), 2)
        self.assertFalse(target['visible_mask_valid'][0])


if __name__ == '__main__':
    unittest.main()
