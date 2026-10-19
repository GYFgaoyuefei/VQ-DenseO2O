"""Reference-source VQ parity on synthetic masks; no dataset or checkpoint IO."""

import ast
import copy
from functools import lru_cache
from pathlib import Path
import subprocess
import types
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou
from engine.deim.vq_quality import visible_quality, calibrate_quality
from engine.deim.vq_criterion import VQDEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.core.workspace import GLOBAL_CONFIG


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_COMMIT = 'f92edb021fd468e354f5767bb81ac745e0347cfe'  # Audited beta_select HEAD.


@lru_cache(maxsize=None)
def reference_module(filename, components=False):
    """Read beta_select with git show only; strip registration, never mathematics.

    The quality module's optional instrumentation exposes existing C/P locals
    for deterministic comparison without replacing any expression.
    """
    result = subprocess.run(
        ['git', 'show', f'{REFERENCE_COMMIT}:engine/deim/{filename}.py'], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        pytest.skip(f'Audited beta_select reference {REFERENCE_COMMIT} is unavailable; fetch its history')
    source = result.stdout
    if components:
        source = source.replace('    mask_bool = masks.to(dtype=torch.bool)',
                                '    components = []\n    mask_bool = masks.to(dtype=torch.bool)')
        source = source.replace('        quality[start:start + len(m)] = (coverage * purity).sqrt()',
                                '        components.append((coverage, purity))\n'
                                '        quality[start:start + len(m)] = (coverage * purity).sqrt()')
        source = source.replace('    return quality.detach(), usable.detach()',
                                '    return quality.detach(), usable.detach(), components')
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []  # Keep the real registry untouched.
    module = types.ModuleType(f'engine.deim._beta_reference_{filename}_{components}')
    module.__package__ = 'engine.deim'
    # Use the original quality/base criterion too; no migrated implementation
    # can serve as its own reference.
    if filename == 'vq_criterion':
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in ('vq_quality', 'deim_criterion'):
                node.module = None
                node.level = 0
        # Replace only dependency imports with explicit reference globals.
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.ImportFrom) or node.module is not None]
        module.DEIMCriterion = reference_module('deim_criterion').DEIMCriterion
        quality = reference_module('vq_quality')
        module.visible_quality, module.calibrate_quality = quality.visible_quality, quality.calibrate_quality
    exec(compile(tree, f'beta_select:engine/deim/{filename}.py', 'exec'), module.__dict__)
    return module


def geometry_case():
    # Fractional boundaries, off-canvas area, holey foreground, explicit
    # invalidity, empty masks, zero-area boxes and empty intersection.
    generator = torch.Generator().manual_seed(246)
    masks = (torch.rand(9, 11, 13, generator=generator) > .45).to(torch.uint8)
    masks[5].zero_()
    boxes = torch.tensor([[.5, .5, .8, .8], [.1, .8, .4, .5], [.5, .5, 1.7, 1.6],
                          [.4, .3, .31, .24], [.5, .5, .5, .5], [.4, .4, .7, .7],
                          [.5, .5, 0., .5], [2., 2., .2, .2], [.5, .5, 1., 1.]])
    valid = torch.ones(9, dtype=torch.bool)
    valid[4] = False
    return masks, boxes, valid


@pytest.mark.parametrize('chunk_size', [1, 3, 32])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_reference_coverage_purity_fg_quality_and_positive_target(chunk_size, dtype):
    masks, boxes, valid = geometry_case()
    boxes = boxes.to(dtype).requires_grad_()
    reference = reference_module('vq_quality', components=True)
    fg, usable, parts = reference.visible_quality(masks, boxes, valid, chunk_size)
    actual = visible_quality(masks, boxes, valid, chunk_size, return_components=True)
    expected_c, expected_p = (torch.cat([part[index] for part in parts]) for index in (0, 1))
    for key, expected in [('coverage', expected_c), ('purity', expected_p),
                          ('q_fg', fg), ('valid_mask', usable)]:
        torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
        assert not actual[key].requires_grad
    iou = torch.linspace(0, 1, len(boxes), dtype=dtype, requires_grad=True)
    expected = reference.calibrate_quality(iou, fg, usable, .1)
    result = calibrate_quality(iou, actual['q_fg'], actual['valid_mask'], .1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.pow(1.5), expected.pow(1.5), rtol=0, atol=0)
    assert result.dtype == dtype and not result.requires_grad
    assert actual['q_fg'].dtype == torch.float32


def test_fractional_pixel_integral_and_unclipped_box_purity():
    mask = torch.zeros(1, 4, 4, dtype=torch.uint8)
    mask[:, 1:3, 1:3] = 1
    # [1.5, 1.5, 2.5, 2.5] intersects exactly one pixel area.
    details = visible_quality(mask, torch.tensor([[.5, .5, .25, .25]]),
                              torch.tensor([True]), return_components=True)
    torch.testing.assert_close(details['coverage'], torch.tensor([.25]))
    torch.testing.assert_close(details['purity'], torch.tensor([1.]))
    torch.testing.assert_close(details['q_fg'], torch.tensor([.5]))
    details = visible_quality(mask, torch.tensor([[.5, .5, 2., 2.]]),
                              torch.tensor([True]), return_components=True)
    torch.testing.assert_close(details['coverage'], torch.tensor([1.]))
    torch.testing.assert_close(details['purity'], torch.tensor([4. / 64.]))


def test_zero_beta_empty_and_invalid_mask_fallback():
    masks, boxes, valid = geometry_case()
    foreground, usable = visible_quality(masks, boxes, valid)
    assert not usable[4] and not usable[5]
    iou = torch.linspace(0, 1, len(boxes))
    torch.testing.assert_close(calibrate_quality(iou, foreground, usable, 0.), iou, rtol=0, atol=0)
    quality = calibrate_quality(iou, foreground, usable, .1)
    torch.testing.assert_close(quality[~usable], iou[~usable], rtol=0, atol=0)
    assert ((quality >= .9 * iou) & (quality <= iou)).all()
    q, valid_empty = visible_quality(masks[:0], boxes[:0], valid[:0])
    assert q.shape == valid_empty.shape == (0,)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 2.])
def test_nonbinary_masks_fail_explicitly(bad):
    mask = torch.ones(1, 4, 4)
    mask[0, 0, 0] = bad
    with pytest.raises(ValueError, match='binary'):
        visible_quality(mask, torch.tensor([[.5, .5, 1., 1.]]), torch.tensor([True]))


def make_criterion(cls):
    return cls(matcher=HungarianMatcher({'cost_class': 2., 'cost_bbox': 5., 'cost_giou': 2.},
                                       use_focal_loss=True),
               weight_dict={'loss_mal': 1., 'loss_bbox': 5., 'loss_giou': 2.,
                            'loss_fgl': .15, 'loss_ddf': 1.5},
               losses=['mal', 'boxes', 'local'], reg_max=4, num_classes=3, gamma=1.5,
               vq_dense_o2o={'beta': .1})


def all_branch_case():
    generator = torch.Generator().manual_seed(421)
    base = torch.tensor([[[.30, .30, .25, .25], [.70, .68, .30, .30], [.5, .5, .7, .7]]])
    def head(local=True):
        out = {'pred_boxes': base.clone().requires_grad_(),
               'pred_logits': torch.randn(1, 3, 3, generator=generator, requires_grad=True)}
        if local:
            out.update(pred_corners=torch.randn(1, 3, 20, generator=generator, requires_grad=True),
                       ref_points=base.clone(), teacher_corners=torch.randn(1, 3, 20, generator=generator),
                       teacher_logits=torch.randn(1, 3, 3, generator=generator))
        return out
    output = head()
    output.update(aux_outputs=[head()], pre_outputs=head(False), enc_aux_outputs=[head(False)],
                  enc_meta={'class_agnostic': False}, dn_outputs=[head()], dn_pre_outputs=head(False),
                  dn_meta={'dn_positive_idx': [torch.tensor([0, 1])], 'dn_num_group': 1},
                  up=torch.tensor([1.]), reg_scale=torch.tensor([4.]))
    masks = torch.zeros(2, 20, 20, dtype=torch.uint8)
    masks[0, 4:7, 4:7], masks[1, 11:16, 11:16] = 1, 1
    targets = [{'boxes': torch.tensor([[.3, .3, .2, .2], [.7, .68, .25, .25]]),
                'labels': torch.tensor([0, 2]), 'masks': masks,
                'visible_mask_valid': torch.tensor([True, False])}]
    return output, targets


@pytest.mark.parametrize('epoch', [0, 88])
def test_reference_ordinary_mal_all_losses_targets_and_gradients(epoch):
    reference = reference_module('vq_criterion').VQDEIMCriterion
    source, targets = all_branch_case()
    original_bce = F.binary_cross_entropy_with_logits

    def execute(cls):
        output = copy.deepcopy(source)
        bce_calls = []
        def capture(logits, target, **kwargs):
            # Reference-only diagnostic calls are [N], while actual MAL is [B,Q,C].
            if logits.ndim == 3:
                bce_calls.append((target.detach().clone(), kwargs['weight'].detach().clone()))
            return original_bce(logits, target, **kwargs)
        with mock.patch.object(F, 'binary_cross_entropy_with_logits', side_effect=capture):
            losses = make_criterion(cls)(output, copy.deepcopy(targets), epoch=epoch)
        sum(losses.values()).backward()
        branches = [output, output['aux_outputs'][0], output['pre_outputs'],
                    output['enc_aux_outputs'][0], output['dn_outputs'][0], output['dn_pre_outputs']]
        gradients = [tensor.grad for branch in branches for key, tensor in branch.items()
                     if key in ('pred_logits', 'pred_boxes', 'pred_corners')]
        return losses, bce_calls, gradients

    expected, target_reference, grad_reference = execute(reference)
    actual, target_actual, grad_actual = execute(VQDEIMCriterion)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)
    assert len(target_actual) == len(target_reference) == 6
    for actual_pair, reference_pair in zip(target_actual, target_reference):
        for tensor, expected_tensor in zip(actual_pair, reference_pair):
            torch.testing.assert_close(tensor, expected_tensor, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(grad_actual, grad_reference):
        assert actual_grad is not None and expected_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_formal_registry_defaults_and_absence_of_gr_algorithm():
    criterion = VQDEIMCriterion(matcher=None, weight_dict={}, losses=['mal'])
    assert criterion.beta == .1 and criterion.gamma == 1.5
    assert 'VQDEIMCriterion' in GLOBAL_CONFIG and 'GRVQDEIMCriterion' not in GLOBAL_CONFIG
    source = (ROOT / 'engine/deim/vq_criterion.py').read_text(encoding='utf-8')
    assert not any(token in source for token in ('gr_vq', 'q_gr', 'q_grvq', 'C_pred/C_gt', 'P_pred/P_gt'))


def test_original_reference_giou_hazard_is_reported_by_explicit_rejection():
    # The reference accepts this construction and can use GIoU as `values`.
    # Formal migration rejects the setting rather than redefining VQ silently.
    reference = reference_module('vq_criterion').VQDEIMCriterion
    reference(matcher=None, weight_dict={}, losses=['mal'], boxes_weight_format='giou')
    with pytest.raises(ValueError, match='true IoU'):
        VQDEIMCriterion(matcher=None, weight_dict={}, losses=['mal'], boxes_weight_format='giou')
