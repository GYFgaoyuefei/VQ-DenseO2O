"""Original beta_select VQ-DenseO2O on ordinary MAL positive targets.

The reference quality, fallback, branch routing and MAL implementation are
retained. Paper defaults are beta=0.10 and gamma=1.5. Compact diagnostics are
an engineering adaptation for the current epoch logger, outside autograd.
"""

import math
import torch
from ..core import register
from .deim_criterion import DEIMCriterion
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .vq_quality import visible_quality, calibrate_quality


@register()
class VQDEIMCriterion(DEIMCriterion):
    """Reuse original MAL code by providing detached calibrated IoU values."""

    def __init__(self, matcher, weight_dict, losses, alpha=.2, gamma=1.5, num_classes=80,
                 reg_max=32, boxes_weight_format=None, share_matched_indices=False,
                 mal_alpha=None, use_uni_set=True, vq_dense_o2o=None) -> None:
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes, reg_max,
                         boxes_weight_format, share_matched_indices, mal_alpha, use_uni_set)
        options = vq_dense_o2o or {}
        self.beta = float(options.get('beta', .10))
        self.enabled = bool(options.get('enabled', True))
        self.branches = set(options.get('apply_to', {}).get('branches', ['main', 'aux', 'pre', 'encoder']))
        if not math.isfinite(self.beta) or not 0 <= self.beta <= 1 or not self.branches <= {'main', 'aux', 'pre', 'encoder'}:
            raise ValueError('VQ is restricted to ordinary matched-positive MAL branches')
        # beta_select can route box-weight metadata through `values`. Its
        # formal VQ-only recipe uses None, and IoU metadata is also valid.
        # Explicitly reject GIoU rather than silently changing that formula.
        if boxes_weight_format not in (None, 'iou'):
            raise ValueError('VQ requires true IoU; boxes_weight_format must be None or iou, never giou')
        self.diagnostics: dict = {}

    def get_loss(self, loss: str, outputs: dict, targets: list[dict], indices: list,
                 num_boxes: float, supervision_branch: str | None = None, **kwargs) -> dict:
        """DN/DN-pre and all non-MAL losses always use the original implementation."""
        if (loss != 'mal' or supervision_branch not in self.branches or self.beta == 0
                or not self.enabled or not self.training):
            return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
        with torch.no_grad():
            idx = self._get_src_permutation_idx(indices)
            if 'values' in kwargs:
                q_iou = kwargs['values'].detach()
            else:
                target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)])
                q_iou = torch.diag(box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]),
                                         box_cxcywh_to_xyxy(target_boxes))[0]).detach()
            fg, validity = [], []
            for target, (prediction, ground_truth), boxes in zip(targets, indices, outputs['pred_boxes']):
                if 'masks' not in target or 'visible_mask_valid' not in target:
                    raise ValueError('VQ training requires synchronized visible masks')
                q, usable = visible_quality(target['masks'][ground_truth], boxes[prediction],
                                            target['visible_mask_valid'][ground_truth])
                fg.append(q)
                validity.append(usable)
            q_fg = torch.cat(fg).to(device=q_iou.device, dtype=q_iou.dtype)
            valid = torch.cat(validity).to(device=q_iou.device, dtype=torch.bool)
            q_vq = calibrate_quality(q_iou, q_fg, valid, self.beta)
            kwargs['values'] = q_vq
        result = super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
        self._record(supervision_branch, q_iou, q_fg, q_vq, valid, outputs['pred_logits'].dtype)
        return result

    @torch.no_grad()
    def _record(self, branch, qiou, qfg, qvq, valid, logits_dtype):
        """Accumulate detached scalars; no step-wise CPU/device synchronization."""
        target = qvq.to(logits_dtype).pow(self.gamma).float()
        native_target = qiou.to(logits_dtype).pow(self.gamma).float()
        qiou, qfg, qvq = qiou.float(), qfg.float(), qvq.float()
        record = torch.stack((qiou.new_tensor(qiou.numel()), valid.sum().float(),
                              qiou.sum(), torch.where(valid, qfg, 0.).sum(), qvq.sum(),
                              target.sum(), (target - native_target).abs().sum(),
                              (valid & (qfg >= .99)).sum().float()))
        if branch not in self.diagnostics:
            self.diagnostics[branch] = torch.zeros_like(record, dtype=torch.float64)
        self.diagnostics[branch].add_(record.double())

    @torch.no_grad()
    def diagnostic_summary(self, reset=False):
        """Epoch-boundary pooled matched-positive diagnostics (ordinary only).

        q_fg moments exclude invalid masks; fallback/target delta count every
        assignment, including unchanged fallback targets. Empty counts report
        zero. All distributed ranks call together; one SUM is performed on a
        compact tensor, without modifying the local accumulators.
        """
        def summarize(record):
            count, valid, qiou, qfg, qvq, positive, delta, near_one = record
            return {'matched_positive_count': int(count), 'valid_mask_count': int(valid),
                    'valid_mask_ratio': valid / count if count else 0.,
                    'iou_fallback_count': int(count - valid),
                    'fallback_ratio': (count - valid) / count if count else 0.,
                    'q_iou_mean': qiou / count if count else 0.,
                    'q_fg_mean': qfg / valid if valid else 0.,
                    'q_vq_mean': qvq / count if count else 0.,
                    'q_fg_near_one_ratio': near_one / valid if valid else 0.,
                    'positive_target_mean': positive / count if count else 0.,
                    'target_delta_from_native_mal': delta / count if count else 0.}
        result = {}
        if self.diagnostics:
            branches = sorted(self.diagnostics)
            records = torch.stack([self.diagnostics[branch] for branch in branches]).detach().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(records, op=torch.distributed.ReduceOp.SUM)
            records = records.cpu()
            result = {branch: summarize(record) for branch, record in zip(branches, records.tolist())}
            result['all'] = summarize(records.sum(0).tolist())
        if reset:
            self.diagnostics.clear()
        return result
