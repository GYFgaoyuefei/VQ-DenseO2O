"""CPU synthetic tests for GR-VQ's geometry and detached quality contract."""

import unittest

import torch

from engine.deim.gr_vq_quality import aligned_box_iou, gr_vq_quality


def exact_intersection(mask, box):
    """Independent direct overlap sum over unit pixels (float64 reference)."""
    height, width = mask.shape
    x = torch.arange(width, dtype=torch.float64)
    y = torch.arange(height, dtype=torch.float64)
    dx = (torch.minimum(x + 1, box[2]) - torch.maximum(x, box[0])).clamp_min(0)
    dy = (torch.minimum(y + 1, box[3]) - torch.maximum(y, box[1])).clamp_min(0)
    return (mask.double() * dy[:, None] * dx[None, :]).sum()


class GRVQGeometryTests(unittest.TestCase):
    def setUp(self):
        # Nonrectangular L support with area 12 inside a 4x4 annotation box.
        self.mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        self.mask[:, 2:6, 2:4] = True
        self.mask[:, 4:6, 4:6] = True
        self.gt = torch.tensor([[2., 2., 6., 6.]])

    def test_identity_nonrectangular_mask(self):
        q = gr_vq_quality(self.gt, self.gt, self.mask)
        self.assertEqual(q['q_gr'].item(), 1.)
        self.assertTrue(torch.equal(q['q_grvq'], q['q_iou']))
        self.assertEqual(q['purity_gt'].item(), .75)

    def test_identity_rectangle_and_fractional_box(self):
        for mask in (torch.ones_like(self.mask), self.mask):
            box = torch.tensor([[2.3, 1.7, 5.6, 6.2]])
            q = gr_vq_quality(box, box, mask)
            self.assertTrue(torch.equal(q['q_gr'], torch.ones(1)))
            self.assertTrue(torch.equal(q['q_grvq'], q['q_iou']))

    def test_enlarged_prediction_reduces_relative_purity(self):
        q = gr_vq_quality(torch.tensor([[1., 1., 7., 7.]]), self.gt, self.mask)
        self.assertEqual(q['coverage_rel'].item(), 1.)
        self.assertAlmostEqual(q['purity_rel'].item(), 16 / 36, places=6)
        self.assertLess(q['q_gr'].item(), 1.)

    def test_partial_prediction_reduces_coverage(self):
        q = gr_vq_quality(torch.tensor([[2., 2., 4., 4.]]), self.gt, self.mask)
        self.assertAlmostEqual(q['coverage_rel'].item(), 1 / 3, places=6)
        self.assertEqual(q['purity_rel'].item(), 1.)

    def test_outside_image_retains_full_box_area(self):
        q = gr_vq_quality(torch.tensor([[-2., -2., 10., 10.]]), self.gt, self.mask)
        self.assertAlmostEqual(q['purity_pred'].item(), 12 / 144, places=7)
        self.assertAlmostEqual(q['purity_rel'].item(), 16 / 144, places=7)
        self.assertAlmostEqual(q['q_iou'].item(), 16 / 144, places=7)

    def test_empty_mask_falls_back(self):
        q = gr_vq_quality(self.gt, self.gt, torch.zeros_like(self.mask))
        self.assertFalse(q['valid_mask'].item())
        self.assertTrue(torch.equal(q['q_grvq'], q['q_iou']))

    def test_invalid_mask_flag_falls_back(self):
        q = gr_vq_quality(self.gt + 1, self.gt, self.mask, torch.tensor([False]))
        self.assertFalse(q['valid_mask'].item())
        self.assertTrue(torch.equal(q['q_grvq'], q['q_iou']))

    def test_nonbinary_and_nonfinite_masks_fall_back(self):
        for corrupt in (float('nan'), float('inf'), .3, 2.):
            mask = self.mask.float()
            mask[0, 0, 0] = corrupt
            q = gr_vq_quality(self.gt, self.gt, mask)
            self.assertFalse(q['valid_mask'].item())
            self.assertTrue(torch.equal(q['q_grvq'], q['q_iou']))
            self.assertTrue(all(torch.isfinite(value).all() for value in q.values()))

    def test_invalid_gt_reference_falls_back(self):
        for gt in (torch.tensor([[0., 0., 1., 1.]]), torch.tensor([[2., 2., 2., 3.]])):
            q = gr_vq_quality(self.gt, gt, self.mask)
            self.assertFalse(q['valid_mask'].item())
            self.assertTrue(torch.equal(q['q_grvq'], q['q_iou']))

    def test_empty_collection(self):
        q = gr_vq_quality(torch.empty(0, 4), torch.empty(0, 4), torch.empty(0, 8, 8))
        self.assertTrue(all(value.shape == (0,) for value in q.values()))

    def test_continuous_coordinates_match_independent_reference(self):
        generator = torch.Generator().manual_seed(101)
        masks = torch.rand((17, 8, 9), generator=generator) > .45
        starts = torch.rand((17, 2), generator=generator) * 8 - 2
        ends = starts + torch.rand((17, 2), generator=generator) * 6 + .125
        boxes = torch.cat((starts, ends), -1)
        gts = torch.tensor([[0., 0., 9., 8.]]).repeat(17, 1)
        q = gr_vq_quality(boxes, gts, masks, chunk_size=3)
        for i in range(len(boxes)):
            intersection = exact_intersection(masks[i], boxes[i].double())
            area = (boxes[i, 2:].double() - boxes[i, :2].double()).prod()
            self.assertAlmostEqual(q['coverage_pred'][i].item(),
                                   (intersection / masks[i].sum()).item(), delta=2e-6)
            self.assertAlmostEqual(q['purity_pred'][i].item(),
                                   (intersection / area).item(), delta=3e-6)

    def test_quality_bounds_all_beta_and_chunk_sizes(self):
        predictions = torch.tensor([[1., 1., 7., 7.], [2., 2., 4., 4.], [-1., 0., 8., 9.]])
        for beta in (0., .1, .5, 1.):
            q = gr_vq_quality(predictions, self.gt.repeat(3, 1), self.mask.repeat(3, 1, 1), beta=beta)
            self.assertTrue(((q['q_gr'] >= 0) & (q['q_gr'] <= 1)).all())
            self.assertTrue((q['q_grvq'] >= (1 - beta) * q['q_iou']).all())
            self.assertTrue((q['q_grvq'] <= q['q_iou']).all())
            alternative = gr_vq_quality(predictions, self.gt.repeat(3, 1), self.mask.repeat(3, 1, 1),
                                       beta=beta, chunk_size=1)
            for key in q:
                self.assertTrue(torch.equal(q[key], alternative[key]))

    def test_quality_detached_from_prediction_and_mask(self):
        pred = self.gt.clone().requires_grad_()
        mask = self.mask.float().requires_grad_()
        q = gr_vq_quality(pred, self.gt, mask)
        self.assertTrue(all(not value.requires_grad for value in q.values()))
        self.assertIsNone(pred.grad)
        self.assertIsNone(mask.grad)

    def test_fp32_and_cpu_amp_finite(self):
        for enabled in (False, True):
            with torch.autocast('cpu', enabled=enabled, dtype=torch.bfloat16):
                q = gr_vq_quality(self.gt.bfloat16(), self.gt.bfloat16(), self.mask)
            self.assertEqual(q['q_grvq'].dtype, torch.float32)
            self.assertTrue(all(torch.isfinite(value).all() for value in q.values()))

    def test_ordinary_iou_is_zero_for_disjoint_boxes(self):
        q = aligned_box_iou(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([[3., 3., 4., 4.]]))
        self.assertEqual(q.item(), 0.)

    def test_illegal_config_rejected(self):
        for beta in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                gr_vq_quality(self.gt, self.gt, self.mask, beta=beta)


if __name__ == '__main__':
    unittest.main()
