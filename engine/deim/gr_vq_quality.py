"""Historical GR-VQ geometry; excluded from formal Paper v3 experiments.

Each binary pixel occupies [x, x+1] x [y, y+1]. Bilinear interpolation of
the padded summed-area table therefore integrates fractional pixel edges
exactly (up to FP32 arithmetic). Box area is NEVER clipped to the canvas.
This module is training supervision only; every public function is detached.
"""

import math

import torch
from torch.nn import functional as F


def _rectangle_area(boxes):
    return (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)


def _integral_at(prefix, x, y):
    _, height, width = prefix.shape
    x, y = x.clamp(0, width - 1), y.clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    dx, dy = x - x0, y - y0
    batch = torch.arange(len(prefix), device=prefix.device)
    return ((1 - dx) * (1 - dy) * prefix[batch, y0, x0]
            + dx * (1 - dy) * prefix[batch, y0, x1]
            + (1 - dx) * dy * prefix[batch, y1, x0]
            + dx * dy * prefix[batch, y1, x1])


def _intersection(prefix, boxes):
    x0, y0, x1, y1 = boxes.unbind(-1)
    value = (_integral_at(prefix, x1, y1) - _integral_at(prefix, x0, y1)
             - _integral_at(prefix, x1, y0) + _integral_at(prefix, x0, y0))
    nondegenerate = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    return torch.where(nondegenerate, value.clamp_min(0), 0.)


@torch.no_grad()
def aligned_box_iou(pred_boxes, gt_boxes, eps=1e-12):
    """True ordinary IoU for aligned XYXY boxes, not generalized IoU."""
    if pred_boxes.shape != gt_boxes.shape or pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
        raise ValueError('Aligned IoU expects two [N,4] XYXY tensors')
    pred, gt = pred_boxes.detach().float(), gt_boxes.detach().to(pred_boxes.device).float()
    finite = torch.isfinite(pred).all(-1) & torch.isfinite(gt).all(-1)
    pred, gt = torch.nan_to_num(pred), torch.nan_to_num(gt)
    overlap = (torch.minimum(pred[:, 2:], gt[:, 2:])
               - torch.maximum(pred[:, :2], gt[:, :2])).clamp_min(0).prod(-1)
    union = _rectangle_area(pred) + _rectangle_area(gt) - overlap
    iou = torch.nan_to_num(overlap / union.clamp_min(eps), nan=0., posinf=0., neginf=0.)
    return torch.where(finite, iou.clamp(0, 1), 0.).detach()


@torch.no_grad()
def gr_vq_quality(pred_boxes, gt_boxes, masks, valid=None, beta=0.10,
                  eps=1e-12, chunk_size=16):
    """Compute GR-VQ for aligned pixel-coordinate XYXY boxes and masks.

    Args:
        pred_boxes, gt_boxes: [N,4], SAME pixel coordinates as the masks.
        masks: [N,H,W], genuine binary instance supports. Invalid, empty,
            nonbinary, nonfinite or externally invalid masks use IoU fallback.
        valid: optional [N] sidecar/synchronization validity flags.

    Returns a dict of detached FP32 [N] components plus bool ``valid_mask``.
    Invalid GR values are neutral (one) and excluded from diagnostic means.
    Empty target collections are supported. A GT reference with zero
    coverage/purity, zero box area, or nonfinite geometry uses IoU fallback.
    """
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('GR-VQ beta must be finite and in [0,1]')
    if not math.isfinite(eps) or eps <= 0 or chunk_size < 1:
        raise ValueError('GR-VQ epsilon and chunk_size must be positive')
    if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4 or gt_boxes.shape != pred_boxes.shape:
        raise ValueError('GR-VQ requires aligned [N,4] prediction and GT boxes')
    if masks.ndim != 3 or len(masks) != len(pred_boxes):
        raise ValueError('GR-VQ requires one aligned [H,W] instance mask per box')
    if valid is not None and valid.shape != (len(pred_boxes),):
        raise ValueError('GR-VQ mask validity flags must align with boxes')
    device = pred_boxes.device
    pred, gt = pred_boxes.detach().float(), gt_boxes.detach().to(device).float()
    q_iou = aligned_box_iou(pred, gt, eps)
    count = len(pred)
    components = {name: torch.zeros(count, device=device, dtype=torch.float32)
                  for name in ('coverage_pred', 'purity_pred', 'coverage_gt', 'purity_gt',
                               'coverage_rel', 'purity_rel')}
    q_gr = torch.ones(count, device=device, dtype=torch.float32)
    usable = torch.zeros(count, device=device, dtype=torch.bool)
    supplied_valid = (torch.ones(count, device=device, dtype=torch.bool) if valid is None
                      else valid.detach().to(device=device, dtype=torch.bool))
    height, width = masks.shape[-2:]
    if height and width:
        # Explicit FP32 disables any surrounding CPU/CUDA autocast policy.
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                stored = masks[start:end].detach().to(device=device)
                binary = ((stored == 0) | (stored == 1)).flatten(1).all(-1)
                mask = (stored == 1).float()
                mask_area = mask.sum((1, 2))
                prefix = F.pad(mask.cumsum(1, dtype=torch.float32).cumsum(2, dtype=torch.float32),
                               (1, 0, 1, 0))
                p, g = pred[start:end], gt[start:end]
                finite = torch.isfinite(p).all(-1) & torch.isfinite(g).all(-1)
                # Sanitizing invalid coordinates keeps every intermediate finite;
                # their validity flag still forces unchanged-IoU fallback.
                p = torch.where(torch.isfinite(p), p, 0.)
                g = torch.where(torch.isfinite(g), g, 0.)
                pred_area, gt_area = _rectangle_area(p), _rectangle_area(g)
                pred_inter, gt_inter = _intersection(prefix, p), _intersection(prefix, g)
                c_pred = (pred_inter / mask_area.clamp_min(eps)).clamp(0, 1)
                p_pred = (pred_inter / pred_area.clamp_min(eps)).clamp(0, 1)
                c_gt = (gt_inter / mask_area.clamp_min(eps)).clamp(0, 1)
                p_gt = (gt_inter / gt_area.clamp_min(eps)).clamp(0, 1)
                c_rel = (c_pred / c_gt.clamp_min(eps)).clamp(0, 1)
                p_rel = (p_pred / p_gt.clamp_min(eps)).clamp(0, 1)
                valid_chunk = (supplied_valid[start:end] & binary & finite & (mask_area > eps)
                               & torch.isfinite(gt_area) & (gt_area > eps)
                               & (c_gt > eps) & (p_gt > eps))
                usable[start:end] = valid_chunk
                q_gr[start:end] = torch.where(valid_chunk, (c_rel * p_rel).sqrt(), 1.)
                for name, value in zip(components, (c_pred, p_pred, c_gt, p_gt, c_rel, p_rel)):
                    components[name][start:end] = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
    q_grvq = q_iou if beta == 0 else torch.where(usable, q_iou * ((1 - beta) + beta * q_gr), q_iou)
    return dict(q_iou=q_iou, q_gr=q_gr, q_grvq=q_grvq.detach(), valid_mask=usable, **components)
