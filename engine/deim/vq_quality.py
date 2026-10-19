"""Original beta_select VQ geometry; detached FP32 continuous mask integration.

Only optional component reporting is added to the reference algorithm. In
particular, its numerator is clipped to the mask canvas while the predicted
box area in the purity denominator retains the original, unclipped extent.
"""

import torch
from torch.nn import functional as F
from .box_ops import box_cxcywh_to_xyxy


def _integral_at(prefix: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bilinearly integrate piecewise-constant pixel occupancy at real corners."""
    n, hp, wp = prefix.shape
    x = x.clamp(0, wp - 1)
    y = y.clamp(0, hp - 1)
    left, top = x.floor().long(), y.floor().long()
    right, bottom = (left + 1).clamp_max(wp - 1), (top + 1).clamp_max(hp - 1)
    batch = torch.arange(n, device=prefix.device)
    dx, dy = x - left, y - top
    return ((1-dx)*(1-dy)*prefix[batch, top, left] + dx*(1-dy)*prefix[batch, top, right]
            + (1-dx)*dy*prefix[batch, bottom, left] + dx*dy*prefix[batch, bottom, right])


@torch.no_grad()
def visible_quality(masks: torch.Tensor, boxes: torch.Tensor, valid: torch.Tensor,
                    chunk_size: int = 32, *, return_components: bool = False):
    """Return [N] q_fg and validity; masks [N,H,W], boxes normalized cxcywh.

    Box area and intersection use continuous pixel coordinates, including
    fractional boundary pixels; no floor/ceil-induced purity inflation.
    Invalid/empty masks are explicitly marked for unchanged-IoU fallback.
    Storage may be uint8; logical support/validity is bool. Raw quality is
    float32 on the boxes device; the criterion aligns it to the IoU dtype.
    """
    if masks.ndim != 3 or boxes.shape != (len(masks), 4) or valid.shape != (len(masks),):
        raise ValueError('VQ mask/box/valid shapes are inconsistent')
    if not boxes.is_floating_point():
        raise TypeError('VQ prediction boxes must be floating point')
    if masks.requires_grad or valid.requires_grad or not torch.isfinite(boxes).all():
        raise ValueError('Masks must not require gradients and boxes must be finite')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    h, w = masks.shape[-2:]
    if not h or not w:
        raise ValueError('Mask canvas cannot be empty')
    quality = boxes.new_zeros(len(boxes), dtype=torch.float32)
    if return_components:
        coverages, purities = torch.zeros_like(quality), torch.zeros_like(quality)
    mask_bool = masks.to(dtype=torch.bool)
    usable = (valid.to(device=boxes.device, dtype=torch.bool)
              & mask_bool.flatten(1).any(dim=1).to(device=boxes.device))
    for start in range(0, len(boxes), chunk_size):
        stored = masks[start:start + chunk_size]
        if not torch.isfinite(stored).all() or ((stored != 0) & (stored != 1)).any():
            raise ValueError('Visible masks must be binary')
        m = mask_bool[start:start + chunk_size].to(device=boxes.device, dtype=torch.float32)
        corners = box_cxcywh_to_xyxy(boxes[start:start + chunk_size].detach().float())
        corners = corners * corners.new_tensor([w, h, w, h])
        x0, y0, x1, y1 = corners.unbind(1)
        x0, x1 = x0.clamp(0, w), x1.clamp(0, w)
        y0, y1 = y0.clamp(0, h), y1.clamp(0, h)
        prefix = F.pad(m.cumsum(1).cumsum(2), (1, 0, 1, 0))
        intersection = (_integral_at(prefix, x1, y1) - _integral_at(prefix, x0, y1)
                        - _integral_at(prefix, x1, y0) + _integral_at(prefix, x0, y0)).clamp_min(0)
        mask_area = m.sum((1, 2))
        box_area = ((corners[:, 2]-corners[:, 0]).clamp_min(0)
                    * (corners[:, 3]-corners[:, 1]).clamp_min(0))
        coverage = (intersection / mask_area.clamp_min(1e-12)).clamp(0, 1)
        purity = (intersection / box_area.clamp_min(1e-12)).clamp(0, 1)
        quality[start:start + len(m)] = (coverage * purity).sqrt()
        if return_components:
            coverages[start:start + len(m)] = coverage
            purities[start:start + len(m)] = purity
    if not torch.isfinite(quality).all():
        raise FloatingPointError('Nonfinite visible quality')
    if return_components:
        return {'coverage': coverages.detach(), 'purity': purities.detach(),
                'q_fg': quality.detach(), 'valid_mask': usable.detach()}
    return quality.detach(), usable.detach()


@torch.no_grad()
def calibrate_quality(q_iou: torch.Tensor, q_fg: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    """Bool conditions and IoU-typed quality; beta zero does no quality arithmetic."""
    if not 0 <= beta <= 1:
        raise ValueError('beta must be in [0,1]')
    if q_iou.shape != q_fg.shape or valid.shape != q_iou.shape:
        raise ValueError('Quality vectors must have identical shapes')
    if not q_iou.is_floating_point() or not q_fg.is_floating_point():
        raise TypeError('IoU and foreground quality must be floating point')
    if not torch.isfinite(q_iou).all() or not torch.isfinite(q_fg).all():
        raise FloatingPointError('Quality vectors must be finite')
    if ((q_fg < 0) | (q_fg > 1)).any() or ((q_iou < 0) | (q_iou > 1)).any():
        raise ValueError('Quality vectors must lie in [0,1]')
    if beta == 0:
        return q_iou.detach()
    valid = valid.to(device=q_iou.device, dtype=torch.bool)
    q_fg = q_fg.to(device=q_iou.device, dtype=q_iou.dtype)
    return torch.where(valid, q_iou * ((1-beta) + beta*q_fg), q_iou).detach()
