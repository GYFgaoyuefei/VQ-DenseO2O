"""Native criterion parity and synthetic all-branch integration tests."""

import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.vq_criterion import VQDEIMCriterion
from engine.deim.vq_quality import visible_quality, calibrate_quality
from engine.deim.matcher import HungarianMatcher


class RecordingMatcher(HungarianMatcher):
    def __init__(self):
        super().__init__({'cost_class': 2., 'cost_bbox': 5., 'cost_giou': 2.}, use_focal_loss=True)
        self.calls = []

    def forward(self, outputs, targets, **kwargs):
        result = super().forward(outputs, targets, **kwargs)
        self.calls.append([(src.clone(), dst.clone()) for src, dst in result['indices']])
        return result


def make_criterion(beta=None, local=False, boxes_weight_format=None):
    kwargs = dict(matcher=RecordingMatcher(), num_classes=3, gamma=1.5,
                  losses=['mal', 'boxes'] + (['local'] if local else []), reg_max=4,
                  weight_dict={'loss_mal': 1., 'loss_bbox': 5., 'loss_giou': 2.,
                               'loss_fgl': .15, 'loss_ddf': 1.5},
                  boxes_weight_format=boxes_weight_format)
    return DEIMCriterion(**kwargs) if beta is None else VQDEIMCriterion(**kwargs, vq_dense_o2o={'beta': beta})


def make_sample(local=False, empty=False):
    generator = torch.Generator().manual_seed(23)
    boxes = torch.tensor([[[.5, .5, .75, .75], [.15, .15, .1, .1], [.85, .85, .1, .1]]])
    logits = torch.tensor([[[-1., 2., -.5], [-3., -3., -3.], [-3., -3., -3.]]])

    def branch():
        item = {'pred_boxes': boxes.clone().requires_grad_(), 'pred_logits': logits.clone().requires_grad_()}
        if local:
            item.update(pred_corners=torch.randn(1, 3, 20, generator=generator, requires_grad=True),
                        ref_points=boxes.clone(),
                        teacher_corners=torch.randn(1, 3, 20, generator=generator),
                        teacher_logits=logits.clone())
        return item

    outputs = branch()
    outputs.update(aux_outputs=[branch()], pre_outputs=branch(), enc_aux_outputs=[branch()],
                   enc_meta={'class_agnostic': False}, dn_outputs=[branch()], dn_pre_outputs=branch(),
                   dn_meta={'dn_positive_idx': [torch.tensor([], dtype=torch.long) if empty else torch.tensor([0])],
                            'dn_num_group': 1})
    if local:
        outputs.update(up=torch.tensor([1.]), reg_scale=torch.tensor([4.]))
        # Encoder has no D-FINE head; DN-pre needs the same scalar inputs as native output.
        outputs['enc_aux_outputs'] = [{key: value for key, value in branch().items()
                                       if key in ('pred_boxes', 'pred_logits')}]
        outputs['pre_outputs'].update(up=outputs['up'], reg_scale=outputs['reg_scale'])
        outputs['dn_pre_outputs'].update(up=outputs['up'], reg_scale=outputs['reg_scale'], is_dn=True)
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[:, 2:6, 2:4], mask[:, 4:6, 4:6] = True, True
    targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[.5, .5, .5, .5]]),
                'masks': mask, 'visible_mask_valid': torch.tensor([True])}]
    if empty:
        targets = [{'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0, 4),
                    'masks': torch.empty(0, 8, 8, dtype=torch.bool),
                    'visible_mask_valid': torch.empty(0, dtype=torch.bool)}]
    return outputs, targets


class VQCriterionTests(unittest.TestCase):
    def test_beta_zero_exact_native_loss_and_gradient_all_branches(self):
        source, targets = make_sample(local=True)
        native, calibrated = make_criterion(local=True), make_criterion(beta=0., local=True)
        out_native, out_calibrated = copy.deepcopy(source), copy.deepcopy(source)
        # beta=0 must not even inspect masks.
        targets = [{key: value for key, value in target.items() if key not in ('masks', 'visible_mask_valid')}
                   for target in targets]
        a, b = native(out_native, targets), calibrated(out_calibrated, targets)
        self.assertEqual(a.keys(), b.keys())
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)
        sum(a.values()).backward()
        sum(b.values()).backward()
        self.assertTrue(torch.equal(out_native['pred_boxes'].grad, out_calibrated['pred_boxes'].grad))
        self.assertTrue(torch.equal(out_native['pred_logits'].grad, out_calibrated['pred_logits'].grad))
        self.assertEqual(calibrated.diagnostic_summary(), {})

    def test_only_ordinary_mal_changes_all_other_losses_and_dn_equal(self):
        source, targets = make_sample(local=True)
        native, calibrated = make_criterion(local=True), make_criterion(beta=.1, local=True)
        a, b = native(copy.deepcopy(source), targets), calibrated(copy.deepcopy(source), targets)
        changed = {'loss_mal', 'loss_mal_aux_0', 'loss_mal_pre', 'loss_mal_enc_0'}
        for name in a:
            if name in changed:
                self.assertFalse(torch.equal(a[name], b[name]), name)
            else:
                self.assertTrue(torch.equal(a[name], b[name]), name)
        self.assertEqual(set(calibrated.diagnostics), {'main', 'aux', 'pre', 'encoder'})

    def test_hungarian_matching_identical(self):
        source, targets = make_sample()
        native, calibrated = make_criterion(), make_criterion(beta=.1)
        native(copy.deepcopy(source), targets)
        calibrated(copy.deepcopy(source), targets)
        self.assertEqual(len(native.matcher.calls), 4)
        self.assertEqual(len(native.matcher.calls), len(calibrated.matcher.calls))
        for a, b in zip(native.matcher.calls, calibrated.matcher.calls):
            for (sa, ga), (sb, gb) in zip(a, b):
                self.assertTrue(torch.equal(sa, sb))
                self.assertTrue(torch.equal(ga, gb))

    def test_positive_target_formula_negative_and_class_identity_preserved(self):
        outputs, targets = make_sample()
        criterion = make_criterion(beta=.1)
        idx = [(torch.tensor([0]), torch.tensor([0]))]
        q_fg, valid = visible_quality(targets[0]['masks'], outputs['pred_boxes'][0, :1],
                                      targets[0]['visible_mask_valid'])
        q_iou = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0, :1]),
                        box_cxcywh_to_xyxy(targets[0]['boxes']))[0].diag()
        q_vq = calibrate_quality(q_iou, q_fg, valid, .1)
        original_bce = F.binary_cross_entropy_with_logits
        captured = []

        def capture(logits, target, **kwargs):
            captured.append((target.detach().clone(), kwargs['weight'].detach().clone()))
            return original_bce(logits, target, **kwargs)

        with mock.patch('torch.nn.functional.binary_cross_entropy_with_logits', side_effect=capture):
            loss = criterion.get_loss('mal', outputs, targets, idx, 1., supervision_branch='main')['loss_mal']
            make_criterion().get_loss('mal', outputs, targets, idx, 1.)
        target, weight = captured[0]
        self.assertAlmostEqual(target[0, 0, 1].item(), q_vq.pow(1.5).item(), places=7)
        expected = torch.zeros_like(target)
        expected[0, 0, 1] = q_vq.pow(1.5)[0]
        self.assertTrue(torch.equal(target, expected))
        self.assertEqual(weight[0, 0, 1].item(), 1.)
        # Compare the *actual powered BCE targets*, not an unpowered q delta.
        native_target = captured[1][0]
        summary = criterion.diagnostic_summary()['main']
        delta = (target[0, 0, 1] - native_target[0, 0, 1]).abs().item()
        self.assertAlmostEqual(summary['target_delta_from_native_mal'], delta, places=7)
        self.assertGreater(delta, 0.)
        loss.backward()
        self.assertIsNone(outputs['pred_boxes'].grad)
        self.assertTrue(torch.isfinite(outputs['pred_logits'].grad).all())

    def test_giou_metadata_is_explicitly_rejected_not_redefined(self):
        outputs, targets = make_sample()
        with self.assertRaisesRegex(ValueError, 'true IoU'):
            make_criterion(beta=.1, boxes_weight_format='giou')
        idx = [(torch.tensor([0]), torch.tensor([0]))]
        with self.assertRaisesRegex(ValueError, r'\[0,1\]'):
            make_criterion(beta=.1).get_loss('mal', outputs, targets, idx, 1.,
                                             supervision_branch='main', values=torch.tensor([-.8]))

    def test_dn_preserved_without_local_loss_marker(self):
        outputs, targets = make_sample(local=False)
        a = make_criterion()(copy.deepcopy(outputs), targets)
        b = make_criterion(beta=.1)(copy.deepcopy(outputs), targets)
        for name in ('loss_mal_dn_0', 'loss_mal_dn_pre'):
            self.assertTrue(torch.equal(a[name], b[name]))

    def test_empty_targets_forward_and_backward(self):
        outputs, targets = make_sample(empty=True)
        criterion = make_criterion(beta=.1)
        losses = criterion(outputs, targets)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(outputs['pred_logits'].grad).all())
        # An empty-GT rank must still participate in the same epoch diagnostic
        # collective as a nonempty rank, with every ordinary branch present.
        self.assertEqual(set(criterion.diagnostics), {'main', 'aux', 'pre', 'encoder'})
        with mock.patch('torch.distributed.is_available', return_value=True), \
                mock.patch('torch.distributed.is_initialized', return_value=True), \
                mock.patch('torch.distributed.all_reduce') as reduce:
            summary = criterion.diagnostic_summary()
            self.assertEqual(reduce.call_count, 1)
            self.assertEqual(reduce.call_args.args[0].shape, (4, 8))
            self.assertEqual(summary['all']['matched_positive_count'], 0)

    def test_invalid_mask_fallback_equals_native_positive_target(self):
        outputs, targets = make_sample()
        targets[0]['visible_mask_valid'].fill_(False)
        a = make_criterion()(copy.deepcopy(outputs), targets)
        criterion = make_criterion(beta=.1)
        b = criterion(copy.deepcopy(outputs), targets)
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)
        summary = criterion.diagnostic_summary()['all']
        self.assertEqual(summary['iou_fallback_count'], 4)
        self.assertEqual(summary['valid_mask_ratio'], 0)
        self.assertEqual(summary['fallback_ratio'], 1)
        self.assertEqual(summary['q_fg_mean'], 0)
        self.assertEqual(summary['q_fg_near_one_ratio'], 0)
        self.assertEqual(summary['target_delta_from_native_mal'], 0)

    def test_amp_box_dtype_preserves_native_iou_and_reports_actual_target_change(self):
        # Pre/encoder branches can produce half-precision boxes during AMP.
        # Converting CXCYWH to XYXY *after* an FP32 cast changes native IoU;
        # fallback must preserve the actual native MAL target nonetheless.
        outputs = {'pred_boxes': torch.tensor([[[.501, .497, .026, .017]]], dtype=torch.float16),
                   'pred_logits': torch.tensor([[[-1., 2., -.5]]], dtype=torch.float16, requires_grad=True)}
        mask = torch.zeros(1, 640, 640, dtype=torch.bool)
        mask[:, 314:326, 312:328] = True
        targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[.5, .5, .027, .02]]),
                    'masks': mask, 'visible_mask_valid': torch.tensor([False])}]
        idx = [(torch.tensor([0]), torch.tensor([0]))]
        native_iou = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0]),
                             box_cxcywh_to_xyxy(targets[0]['boxes']))[0].diag()
        original_bce = F.binary_cross_entropy_with_logits
        for valid in (False, True):
            targets[0]['visible_mask_valid'].fill_(valid)
            captured = []

            def capture(logits, target, **kwargs):
                captured.append(target.detach().clone())
                return original_bce(logits, target, **kwargs)

            criterion = make_criterion(beta=.1)
            with mock.patch('torch.nn.functional.binary_cross_entropy_with_logits', side_effect=capture):
                actual = criterion.get_loss('mal', outputs, targets, idx, 1., supervision_branch='pre')
                native = make_criterion().get_loss('mal', outputs, targets, idx, 1.)
            summary = criterion.diagnostic_summary()['pre']
            self.assertEqual(summary['q_iou_mean'], native_iou.item())
            actual_delta = (captured[0][0, 0, 1].float() - captured[1][0, 0, 1].float()).abs().item()
            self.assertEqual(summary['target_delta_from_native_mal'], actual_delta)
            if not valid:
                self.assertTrue(torch.equal(actual['loss_mal'], native['loss_mal']))
                self.assertTrue(torch.equal(captured[0], captured[1]))
                self.assertEqual(actual_delta, 0)
            else:
                self.assertGreater(actual_delta, 0)

    def test_diagnostics_tensor_streaming_summary_and_reset(self):
        outputs, targets = make_sample()
        criterion = make_criterion(beta=.1)
        criterion(outputs, targets)
        self.assertTrue(all(torch.is_tensor(record) and not record.requires_grad
                            for record in criterion.diagnostics.values()))
        summary = criterion.diagnostic_summary(reset=True)
        self.assertEqual(summary['all']['matched_positive_count'], 4)
        self.assertEqual(summary['all']['valid_mask_ratio'], 1)
        self.assertEqual(summary['all']['fallback_ratio'], 0)
        self.assertEqual(summary['all']['q_fg_near_one_ratio'], 0)
        self.assertGreater(summary['all']['target_delta_from_native_mal'], 0)
        self.assertLess(summary['all']['q_vq_mean'], summary['all']['q_iou_mean'])
        self.assertEqual(criterion.diagnostics, {})

    def test_diagnostics_distinguish_near_neutral_valid_masks_from_fallback(self):
        q_iou = torch.tensor([1., .8, .7, .7])
        q_fg = torch.tensor([1., .995, 2 / 3, 2 / 3])
        valid = torch.tensor([True, True, True, False])
        q_vq = calibrate_quality(q_iou, q_fg, valid, .1)
        criterion = make_criterion(beta=.1)
        criterion._record('main', q_iou, q_fg, q_vq, valid, torch.float32)
        summary = criterion.diagnostic_summary()['main']
        self.assertEqual(summary['matched_positive_count'], 4)
        self.assertEqual(summary['valid_mask_count'], 3)
        self.assertEqual(summary['valid_mask_ratio'], .75)
        self.assertEqual(summary['fallback_ratio'], .25)
        self.assertAlmostEqual(summary['q_fg_mean'], (1 + .995 + 2 / 3) / 3, places=6)
        self.assertAlmostEqual(summary['q_fg_near_one_ratio'], 2 / 3)
        expected_delta = (q_vq.pow(1.5) - q_iou.pow(1.5)).abs().mean()
        self.assertAlmostEqual(summary['target_delta_from_native_mal'], expected_delta.item(), places=7)
        # Low-precision rounding can eliminate a small effect. The diagnostic
        # must expose that actual target change, not just the FP32 quality gap.
        criterion.diagnostic_summary(reset=True)
        criterion._record('main', torch.tensor([.75]), torch.tensor([1.]),
                           torch.tensor([.74999]), torch.tensor([True]), torch.bfloat16)
        self.assertEqual(criterion.diagnostic_summary()['main']['target_delta_from_native_mal'], 0)

    def test_cpu_amp_forward_backward_finite(self):
        outputs, targets = make_sample(local=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses = make_criterion(beta=.1, local=True)(outputs, targets)
            loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(outputs['pred_boxes'].grad).all())
        self.assertTrue(torch.isfinite(outputs['pred_logits'].grad).all())

    def test_distributed_diagnostics_reduce_once_without_mutating_accumulators(self):
        outputs, targets = make_sample()
        criterion = make_criterion(beta=.1)
        criterion(outputs, targets)
        local = {key: value.clone() for key, value in criterion.diagnostics.items()}
        expected_mean = criterion.diagnostic_summary()['all']['q_vq_mean']
        with mock.patch('torch.distributed.is_available', return_value=True), \
                mock.patch('torch.distributed.is_initialized', return_value=True), \
                mock.patch('torch.distributed.all_reduce', side_effect=lambda tensor, op: tensor.mul_(2)) as reduce:
            summary = criterion.diagnostic_summary()
            self.assertEqual(reduce.call_count, 1)
            self.assertEqual(reduce.call_args.args[0].shape, (4, 8))
            self.assertEqual(summary['all']['matched_positive_count'], 8)
            self.assertEqual(summary['all']['q_vq_mean'], expected_mean)
        for branch in local:
            self.assertTrue(torch.equal(criterion.diagnostics[branch], local[branch]))

    def test_initial_empty_diagnostics_reset_never_reduces(self):
        criterion = make_criterion(beta=.1)
        with mock.patch('torch.distributed.all_reduce') as reduce:
            self.assertEqual(criterion.diagnostic_summary(reset=True), {})
            reduce.assert_not_called()

    def test_class_agnostic_encoder_preserves_mask_alignment(self):
        outputs, targets = make_sample()
        targets[0]['labels'].zero_()
        outputs['enc_meta']['class_agnostic'] = True
        outputs['enc_aux_outputs'][0]['pred_logits'] = torch.zeros(1, 3, 1, requires_grad=True)
        criterion = make_criterion(beta=.1)
        losses = criterion(outputs, targets)
        self.assertTrue(torch.isfinite(losses['loss_mal_enc_0']))
        self.assertEqual(criterion.num_classes, 3)
        self.assertEqual(criterion.diagnostic_summary()['encoder']['valid_mask_ratio'], 1)


if __name__ == '__main__':
    unittest.main()
