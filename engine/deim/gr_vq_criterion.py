"""Historical Paper v2 GR-VQ; excluded from registration and formal experiments.

Kept for source inspection only. Paper v3 uses VQDEIMCriterion.
"""

import math

import torch

from .box_ops import box_cxcywh_to_xyxy, box_iou
from .deim_criterion import DEIMCriterion
from .gr_vq_quality import gr_vq_quality


class GRVQDEIMCriterion(DEIMCriterion):
    """Use the native criterion's matcher, branch orchestration, and MAL loss.

    Only ordinary positive quality values are replaced. DN and DN-pre,
    negatives, class identity, and every localization/distillation loss are
    handled by the original DEIMCriterion. beta=0 takes the original path
    before inspecting any masks or constructing any quality tensor.
    """

    # A diagnostic threshold, not a training hyperparameter. Unlike numerical
    # saturation (1 - 1e-6), this detects weak modulation near the neutral value.
    Q_GR_NEAR_ONE_THRESHOLD = .99

    def __init__(self, matcher, weight_dict, losses, alpha=.2, gamma=1.5,
                 num_classes=80, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, mal_alpha=None, use_uni_set=True,
                 gr_vq=None):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes,
                         reg_max, boxes_weight_format, share_matched_indices,
                         mal_alpha, use_uni_set)
        options = gr_vq or {}
        self.beta = float(options.get('beta', .10))
        self.chunk_size = int(options.get('chunk_size', 16))
        self.eps = float(options.get('eps', 1e-12))
        self.enabled = bool(options.get('enabled', True))
        if not math.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError('GR-VQ beta must be finite and in [0,1]')
        if self.chunk_size < 1 or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('GR-VQ chunk_size and epsilon must be positive')
        self.diagnostics = {}

    def get_loss(self, loss, outputs, targets, indices, num_boxes,
                 supervision_branch=None, **kwargs):
        ordinary = supervision_branch in ('main', 'aux', 'pre', 'encoder')
        if (loss != 'mal' or not ordinary or self.beta == 0
                or not self.enabled or not self.training):
            return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
        with torch.no_grad():
            chunks = []
            for target, (pred_indices, gt_indices), predictions in zip(targets, indices, outputs['pred_boxes']):
                if len(gt_indices) == 0:
                    empty_boxes = predictions.new_empty((0, 4))
                    chunks.append(gr_vq_quality(empty_boxes, empty_boxes,
                                                predictions.new_empty((0, 0, 0)), beta=self.beta))
                    continue
                if 'masks' not in target or 'visible_mask_valid' not in target:
                    raise ValueError('GR-VQ requires instance_mask_sync with masks and visible_mask_valid')
                masks, flags = target['masks'], target['visible_mask_valid']
                if len(masks) != len(target['labels']) or flags.shape != target['labels'].shape:
                    raise ValueError('GR-VQ target instance masks/validity/labels are not aligned')
                height, width = masks.shape[-2:]
                matched_predictions = predictions[pred_indices].detach()
                matched_targets = target['boxes'][gt_indices].detach()
                pred = box_cxcywh_to_xyxy(matched_predictions.float())
                gt = box_cxcywh_to_xyxy(matched_targets.float())
                scale = pred.new_tensor([width, height, width, height])
                quality = gr_vq_quality(pred * scale, gt * scale, masks[gt_indices],
                                        flags[gt_indices], beta=self.beta,
                                        eps=self.eps, chunk_size=self.chunk_size)
                # Preserve native MAL's ordinary IoU, including the original
                # box-conversion dtype under AMP. Recomputing it after FP32
                # conversion/pixel scaling changes fallback supervision even
                # when every mask is invalid. Only mask integration needs the
                # FP32 pixel geometry above. Never reuse signed GIoU metadata.
                native_iou = box_iou(box_cxcywh_to_xyxy(matched_predictions),
                                     box_cxcywh_to_xyxy(matched_targets))[0].diag()
                native_iou = torch.nan_to_num(native_iou.float(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
                quality['q_iou'] = native_iou
                quality['q_grvq'] = torch.where(
                    quality['valid_mask'], native_iou * ((1 - self.beta) + self.beta * quality['q_gr']),
                    native_iou)
                chunks.append(quality)
            quality = {name: torch.cat([part[name] for part in chunks]) for name in chunks[0]}
            # Deliberately override values even if boxes_weight_format='giou':
            # GR-VQ must always start from ordinary IoU, never signed GIoU.
            kwargs['values'] = quality['q_grvq']
            self._record(supervision_branch, quality, outputs['pred_logits'].dtype)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def _record(self, branch, quality, logits_dtype):
        valid = quality['valid_mask']
        iou, gr, grvq = (quality[name] for name in ('q_iou', 'q_gr', 'q_grvq'))
        # Native MAL casts quality to the logits dtype before exponentiation.
        positive_target = grvq.to(dtype=logits_dtype).pow(self.gamma).float()
        native_positive_target = iou.to(dtype=logits_dtype).pow(self.gamma).float()
        target_delta = (positive_target - native_positive_target).abs()
        record = torch.stack((iou.new_tensor(iou.numel()), valid.sum().float(), iou.sum(),
                              torch.where(valid, gr, 0.).sum(), grvq.sum(),
                              (valid & (gr >= 1 - 1e-6)).sum().float(), positive_target.sum(),
                              (valid & (gr >= self.Q_GR_NEAR_ONE_THRESHOLD)).sum().float(),
                              target_delta.sum()))
        if branch not in self.diagnostics:
            self.diagnostics[branch] = torch.zeros_like(record, dtype=torch.float64)
        self.diagnostics[branch].add_(record.double())

    @torch.no_grad()
    def diagnostic_summary(self, reset=False):
        """Copy only nine accumulated scalars per branch when explicitly read.

        Intended for epoch logging, not per-step logging. All distributed
        ranks must call this together. One compact SUM collective produces
        global counts and weighted means without modifying local accumulators.
        The initial empty reset performs no collective or device transfer.

        Counts cover ordinary matched positives in main/aux/pre/encoder (DN
        is excluded). ``all`` pools these assignments, not unique annotations.
        valid_mask_ratio and fallback_ratio use every matched positive as the
        denominator. q_gr_mean and q_gr_near_one_ratio use valid masks only;
        near-one means q_gr >= 0.99. target_delta_from_native_mal is the mean
        absolute difference from the uncalibrated IoU-based MAL positive
        target, after the actual logits-dtype cast and gamma exponentiation,
        over all matched positives, including unchanged fallback targets.
        Empty denominators return 0 alongside explicit zero counts.
        """
        def summarize(record):
            count, valid, iou, gr, grvq, saturated, positive, near_one, target_delta = record
            return {'matched_positive_count': int(count), 'valid_mask_count': int(valid),
                    'q_iou_mean': iou / count if count else 0.,
                    'q_gr_mean': gr / valid if valid else 0.,
                    'q_grvq_mean': grvq / count if count else 0.,
                    'valid_mask_ratio': valid / count if count else 0.,
                    'iou_fallback_count': int(count - valid),
                    'fallback_ratio': (count - valid) / count if count else 0.,
                    'q_gr_near_one_ratio': near_one / valid if valid else 0.,
                    'gr_quality_saturated_ratio': saturated / valid if valid else 0.,
                    'positive_target_mean': positive / count if count else 0.,
                    'target_delta_from_native_mal': target_delta / count if count else 0.}
        result = {}
        if self.diagnostics:
            branches = sorted(self.diagnostics)
            records = torch.stack([self.diagnostics[branch] for branch in branches]).detach().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(records, op=torch.distributed.ReduceOp.SUM)
            # Single compact transfer at the explicit epoch boundary.
            records = records.cpu()
            result = {branch: summarize(record) for branch, record in zip(branches, records.tolist())}
            result['all'] = summarize(records.sum(0).tolist())
        if reset:
            self.diagnostics.clear()
        return result
