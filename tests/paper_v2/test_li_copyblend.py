import random
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch

from engine.data.dataloader import BatchImageCollateFunction
from engine.data.li_copyblend import (LICopyBlend, STAT_NAMES, background_statistics,
                                      summarize_copyblend_stats, to_rgb)
from engine.data.visible_geometry import assert_visible_alignment, resize_masks


def target(n=1, light=0, train=True, size=16):
    masks = torch.zeros((n, size, size), dtype=torch.uint8)
    masks[:, 2:4, 2:4] = 1
    masks[:, 2, 3] = 0  # Genuine nonrectangular foreground.
    return dict(boxes=torch.tensor([[3/size, 3/size, 2/size, 2/size]]*n).reshape(n, 4),
                labels=torch.full((n,), 2, dtype=torch.int64), masks=masks,
                visible_mask_valid=torch.ones(n, dtype=torch.bool),
                area=torch.full((n,), 3., dtype=torch.float32), iscrowd=torch.zeros(n, dtype=torch.int64),
                annotation_ids=torch.arange(1, n+1, dtype=torch.int64), visibility_group=torch.zeros(n, dtype=torch.int64),
                source_illumination_id=torch.full((n,), light, dtype=torch.int64),
                illumination_source_id=torch.full((n,), light, dtype=torch.int64),
                source_is_train=torch.full((n,), train, dtype=torch.bool), blend_opacity=torch.ones(n),
                image_illumination_id=torch.tensor([light]), image_is_train=torch.tensor([train]),
                image_id=torch.tensor([5]), idx=torch.tensor([0]), orig_size=torch.tensor([size, size]))


def batch(light=0, train=True):
    images = torch.full((2, 3, 16, 16), 0.2)
    images[1, :, 2:4, 2:4] = 0.8
    return images, [target(0, light=light), target(1, train=train)]


class LICopyBlendTests(unittest.TestCase):
    def run_blend(self, images=None, targets=None, **options):
        if images is None:
            images, targets = batch()
        augmenter = LICopyBlend(enabled=True, **options)
        with patch('engine.data.li_copyblend.random.randint', return_value=10):
            result = augmenter(images, targets, alpha=0.25, area_threshold=1, num_objects=1)
        return (*result, augmenter)

    def test_foreground_formula_and_exact_outside_preservation(self):
        original, targets = batch()
        images, result, augmenter = self.run_blend(original, targets)
        mask = result[0]['masks'][0].bool()
        self.assertTrue(torch.equal(images[0, :, ~mask], original[0, :, ~mask]))
        self.assertTrue(torch.allclose(images[0, :, mask], torch.full((3, 3), 0.35)))
        self.assertGreater(augmenter.last_stats['accepted_pastes'], 0)

    def test_diagnostics_count_zero_attempt_images_and_derive_weighted_ratios(self):
        _, result, augmenter = self.run_blend()
        first = result[0]['li_copyblend_stats'].clone()
        accepted = augmenter.last_stats['accepted_pastes']
        self.assertEqual(augmenter.last_stats['image_count'], 2)
        self.assertEqual(augmenter.last_stats['acceptance_ratio'], 1)
        images = torch.zeros(3, 3, 16, 16)
        _, empty = augmenter(images, [target(0) for _ in range(3)], alpha=.5, area_threshold=1, num_objects=1)
        summary = summarize_copyblend_stats(first + empty[0]['li_copyblend_stats'])
        self.assertEqual(summary['image_count'], 5)
        self.assertEqual(summary['accepted_pastes'], accepted)
        self.assertEqual(summary['acceptance_ratio'], 1)
        self.assertEqual(summary['added_instances_per_image'], accepted / 5)
        self.assertAlmostEqual(summary['foreground_area_ratio'], accepted * 3 / 256 / 5)
        self.assertEqual(augmenter.cumulative_stats['added_instances_per_image'], accepted / 5)
        self.assertEqual(augmenter.last_stats['acceptance_ratio'], 0)
        self.assertEqual(augmenter.last_stats['added_instances_per_image'], 0)

    def test_collate_off_epoch_and_mixup_still_emit_zero_attempt_denominator(self):
        for epoch, mixup_prob in ((100, 0), (1, 1)):
            images, targets = batch()
            collate = BatchImageCollateFunction(copyblend_prob=1, copyblend_epochs=[0, 10],
                mixup_prob=mixup_prob, mixup_epochs=[0, 10], li_copyblend={'enabled': True})
            collate.set_epoch(epoch)
            _, result = collate.apply_mixup(images, targets)
            stats = summarize_copyblend_stats(result[0]['li_copyblend_stats'])
            self.assertEqual(stats['attempted_pastes'], 0)
            self.assertEqual(stats['accepted_pastes'], 0)
            self.assertEqual(stats['image_count'], 2)
            self.assertEqual(stats['li_enabled_image_count'], 2)

    def test_native_copyblend_diagnostic_matches_actual_gt_additions(self):
        images, targets = batch()
        collate = BatchImageCollateFunction(copyblend_prob=1, copyblend_epochs=[0, 10],
            area_threshold=1, num_objects=1, li_copyblend={'enabled': False})
        collate.set_epoch(1)
        _, result = collate.apply_mixup(images, targets)
        stats = summarize_copyblend_stats(result[0]['li_copyblend_stats'])
        added = sum(len(t['labels']) for t in result) - sum(len(t['labels']) for t in targets)
        self.assertEqual(stats['accepted_pastes'], added)
        self.assertEqual(stats['added_instances_per_image'], added / len(targets))
        self.assertEqual(stats['li_enabled_image_count'], 0)

    def test_requested_skip_names_preserve_legacy_event_counts(self):
        images, targets = batch()
        targets[1]['visible_mask_valid'].fill_(False)
        _, _, augmenter = self.run_blend(images, targets)
        stats = augmenter.last_stats
        self.assertEqual(stats['skipped_by_invalid_mask'], stats['skipped_invalid_mask'])
        self.assertEqual(stats['skipped_by_light'], stats['skipped_incompatible_light'])
        self.assertEqual(stats['skipped_by_collision'], stats['skipped_collision'])
        self.assertGreater(stats['skipped_by_invalid_mask'], 0)
        self.assertEqual(set(stats), set(STAT_NAMES))

    def test_new_geometry_label_area_crowd_light_ids_and_opacity(self):
        _, targets, _ = self.run_blend()
        t = targets[0]
        self.assertTrue(torch.equal(t['boxes'], torch.tensor([[11/16, 11/16, 2/16, 2/16]])))
        self.assertEqual(t['labels'].tolist(), [2])
        self.assertEqual(t['area'].tolist(), [3])
        self.assertEqual(t['iscrowd'].tolist(), [0])
        self.assertEqual(t['source_illumination_id'].tolist(), [0])
        self.assertEqual(t['illumination_source_id'].tolist(), [0])
        self.assertLess(int(t['annotation_ids'][0]), 0)
        self.assertEqual(t['blend_opacity'].tolist(), [0.25])
        self.assertEqual(t['masks'].sum().item(), 3)
        assert_visible_alignment(t, 16, 16)

    def test_pasted_area_tracks_multiscale_support_without_changing_native_area(self):
        _, targets, _ = self.run_blend()
        self.assertTrue(targets[0]['area_from_mask'][0])
        native_area = targets[1]['area'][0].clone()
        for t in targets:
            resize_masks(t, 64)
            selected = t['area_from_mask']
            self.assertTrue(torch.equal(t['area'][selected], t['masks'][selected].flatten(1).sum(1).float()))
        self.assertEqual(targets[0]['area'].tolist(), [48])
        self.assertEqual(targets[1]['area'][0], native_area)

    def test_candidate_budget_limits_dense_pool_work(self):
        images, targets = batch()
        targets = [target(20), target(20)]
        for t in targets:
            t['boxes'][:] = torch.tensor([.5, .5, 1., 1.])
        augmenter = LICopyBlend(enabled=True, max_retries=2, max_donor_attempts=1)
        augmenter(images, targets, alpha=.5, area_threshold=1, num_objects=1)
        self.assertLessEqual(augmenter.last_stats['attempted_pastes'], 2)
        self.assertLessEqual(augmenter.last_stats['skipped_collision'], 4)

    def test_val_or_test_donor_rejected(self):
        for split in ('validation', 'test'):
            with self.subTest(split=split):
                images, targets = batch(train=False)
                out, result, augmenter = self.run_blend(images, targets)
                self.assertTrue(torch.equal(images, out))
                self.assertEqual(len(result[0]['labels']), 0)
                self.assertEqual(augmenter.last_stats['accepted_pastes'], 0)

    def test_invalid_and_empty_mask_do_not_paste(self):
        for invalid in (True, False):
            images, targets = batch()
            if invalid:
                targets[1]['visible_mask_valid'].fill_(False)
            else:
                targets[1]['masks'].zero_()
            out, _, augmenter = self.run_blend(images, targets)
            self.assertTrue(torch.equal(images, out))
            self.assertGreater(augmenter.last_stats['skipped_invalid_mask'], 0)

    def test_unknown_light_donor_rejected(self):
        images, targets = batch()
        targets[1]['source_illumination_id'].fill_(-1)
        out, _, augmenter = self.run_blend(images, targets)
        self.assertTrue(torch.equal(images, out))
        self.assertGreater(augmenter.last_stats['skipped_incompatible_light'], 0)

    def test_strict_mismatch_rejected(self):
        images, targets = batch(light=3)
        _, result, augmenter = self.run_blend(images, targets)
        self.assertEqual(len(result[0]['labels']), 0)
        self.assertGreater(augmenter.last_stats['skipped_incompatible_light'], 0)

    def test_mixed_destination_requires_local_photometry(self):
        images, targets = batch(light=-1)
        _, result, _ = self.run_blend(images, targets, local_photometry=False)
        self.assertEqual(len(result[0]['labels']), 0)
        _, result, _ = self.run_blend(images, targets, local_photometry=True)
        self.assertEqual(len(result[0]['labels']), 1)

    def test_local_photometry_uses_background_not_fruit_color(self):
        images, targets = batch()
        images[1, :, 2:4, 2:4] = torch.tensor([1., 0., 0.])[:, None, None]
        _, result, _ = self.run_blend(images, targets, photometry_threshold=0.001)
        self.assertEqual(len(result[0]['labels']), 1)
        images[0].fill_(0.7)
        _, result, _ = self.run_blend(images, targets, photometry_threshold=0.01)
        self.assertEqual(len(result[0]['labels']), 0)

    def test_normalized_rgb_statistics_and_blending_equivalence(self):
        images, targets = batch()
        mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
        rgb_result, _, _ = self.run_blend(images, targets)
        norm_result, _, _ = self.run_blend((images-mean)/std, targets, normalized=True)
        self.assertTrue(torch.allclose(norm_result*std+mean, rgb_result, atol=1e-7))

    def test_collision_has_bounded_retries_and_skip(self):
        images, targets = batch()
        targets[0] = target()
        targets[0]['boxes'][:] = torch.tensor([0.5, 0.5, 1., 1.])
        targets[0]['masks'].fill_(1)
        _, result, augmenter = self.run_blend(images, targets, max_retries=3)
        self.assertEqual(len(result[0]['labels']), 1)
        self.assertGreater(augmenter.last_stats['skipped_collision'], 0)
        self.assertGreater(augmenter.last_stats['skipped_no_valid_location'], 0)

    def test_collision_checks_visible_mask_outside_existing_bbox(self):
        images, targets = batch()
        targets[0] = target()
        targets[0]['masks'].zero_()
        targets[0]['masks'][:, 10:12, 10:12] = 1
        _, result, augmenter = self.run_blend(images, targets, max_retries=2)
        self.assertEqual(len(result[0]['labels']), 1)
        self.assertTrue(torch.equal(result[0]['masks'], targets[0]['masks']))
        self.assertGreater(augmenter.last_stats['skipped_collision'], 0)

    def test_empty_pool_returns_without_error(self):
        images = torch.ones((2, 3, 16, 16))
        out, targets, augmenter = self.run_blend(images, [target(0), target(0)])
        self.assertTrue(torch.equal(out, images))
        self.assertEqual(augmenter.last_stats['accepted_pastes'], 0)

    def test_class_range_and_area_threshold(self):
        images, targets = batch()
        targets[1]['labels'].fill_(3)
        _, result, _ = self.run_blend(images, targets)
        self.assertEqual(len(result[0]['labels']), 0)
        targets[1]['labels'].fill_(0)
        targets[1]['area'].fill_(0.5)
        _, result, _ = self.run_blend(images, targets)
        self.assertEqual(len(result[0]['labels']), 0)

    def test_components_can_be_ablated_independently(self):
        images, targets = batch(light=3)
        _, result, _ = self.run_blend(images, targets, illumination_sampling=False)
        self.assertEqual(len(result[0]['labels']), 1)
        images, targets = batch()
        out, result, _ = self.run_blend(images, targets, foreground_only=False)
        mask = result[0]['masks'][0].bool()
        self.assertTrue((out[0, :, ~mask] != images[0, :, ~mask]).any())

    def test_seed_reproducible_images_geometry_and_classes(self):
        images, targets = batch()
        augmenter = LICopyBlend(enabled=True)
        random.seed(42)
        a, ta = augmenter(images, targets, alpha=.5, area_threshold=1, num_objects=1)
        random.seed(42)
        b, tb = augmenter(images, targets, alpha=.5, area_threshold=1, num_objects=1)
        self.assertTrue(torch.equal(a, b))
        for first, second in zip(ta, tb):
            for key in ('boxes', 'labels', 'masks', 'area', 'source_illumination_id'):
                self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_disabled_uses_native_rectangle_copyblend(self):
        images, targets = batch()
        collate = BatchImageCollateFunction(copyblend_prob=1, copyblend_epochs=[0, 10],
            area_threshold=1, num_objects=1, li_copyblend={'enabled': False})
        collate.set_epoch(1)
        with patch.object(collate.li_copyblend, '__call__', side_effect=AssertionError('LI called')), \
             patch('engine.data.dataloader.random.randint', return_value=10):
            out, result = collate.apply_mixup(images, targets)
        mask = result[0]['masks'][0].bool()
        self.assertTrue((out[0, :, ~mask] != images[0, :, ~mask]).any())
        assert_visible_alignment(result[0], 16, 16)

    def test_native_coco_masks_do_not_require_sidecar_validity_metadata(self):
        images, targets = batch()
        native_fields = ('boxes', 'labels', 'masks', 'area', 'iscrowd', 'image_id', 'idx', 'orig_size')
        targets = [{key: value for key, value in t.items() if key in native_fields} for t in targets]
        collate = BatchImageCollateFunction(copyblend_prob=1, copyblend_epochs=[0, 10],
            area_threshold=1, num_objects=1)
        collate.set_epoch(1)
        _, result = collate.apply_mixup(images, targets)
        self.assertEqual(sum(len(t['labels']) for t in result), 3)
        for t in result:
            assert_visible_alignment(t, 16, 16)
            self.assertTrue(t['visible_mask_valid'].all())

    def test_b0_b1_native_copyblend_same_rgb_bbox_labels_and_random_choices(self):
        images, visible = batch()
        native = [{k: v for k, v in t.items() if k in ('boxes', 'labels', 'area', 'iscrowd', 'image_id', 'idx', 'orig_size')}
                  for t in visible]
        for kind in ('blend', 'copy'):
            collate = BatchImageCollateFunction(copyblend_prob=1, copyblend_epochs=[0, 10],
                copyblend_type=kind, area_threshold=1, num_objects=1, with_expand=True)
            collate.set_epoch(1)
            random.seed(42)
            a, ta = collate.apply_mixup(images.clone(), deepcopy(native))
            state_a = random.getstate()
            random.seed(42)
            b, tb = collate.apply_mixup(images.clone(), deepcopy(visible))
            self.assertEqual(state_a, random.getstate())
            self.assertTrue(torch.equal(a, b))
            for first, second in zip(ta, tb):
                for key in ('boxes', 'labels', 'area', 'iscrowd', 'mixup'):
                    self.assertTrue(torch.equal(first[key], second[key]), key)
                assert_visible_alignment(second, 16, 16)

    def test_b0_b1_native_mixup_same_rgb_bbox_labels_and_random_choices(self):
        images, visible = batch()
        native = [{k: v for k, v in t.items() if k in ('boxes', 'labels', 'area', 'iscrowd', 'image_id', 'idx', 'orig_size')}
                  for t in visible]
        collate = BatchImageCollateFunction(mixup_prob=1, mixup_epochs=[0, 10])
        collate.set_epoch(1)
        random.seed(42)
        a, ta = collate.apply_mixup(images.clone(), deepcopy(native))
        random.seed(42)
        b, tb = collate.apply_mixup(images.clone(), deepcopy(visible))
        self.assertTrue(torch.equal(a, b))
        for first, second in zip(ta, tb):
            for key in ('boxes', 'labels', 'area', 'iscrowd', 'mixup'):
                self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_mixup_preserves_all_instance_fields_and_mixed_light(self):
        images, targets = batch()
        collate = BatchImageCollateFunction(mixup_prob=1, mixup_epochs=[0, 10])
        collate.set_epoch(1)
        _, result = collate.apply_mixup(images, targets)
        for t in result:
            assert_visible_alignment(t, 16, 16)
            self.assertEqual(t['image_illumination_id'].tolist(), [-1])
            self.assertEqual(len(t['iscrowd']), len(t['labels']))
            self.assertEqual(len(t['source_is_train']), len(t['labels']))

    def test_collate_multiscale_resizes_mask_binary_support(self):
        images, targets = batch()
        collate = BatchImageCollateFunction(base_size=64, base_size_repeat=1)
        collate.set_epoch(0)
        with patch('engine.data.dataloader.random.choice', return_value=64):
            out, result = collate(list(zip(images, targets)))
        self.assertEqual(out.shape[-2:], (64, 64))
        for t in result:
            assert_visible_alignment(t, 64, 64)


if __name__ == '__main__':
    unittest.main()
