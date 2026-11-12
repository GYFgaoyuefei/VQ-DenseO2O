"""FP32 active_scales_mean, without spatial alignment or feature normalization."""
import torch


def feature_loss(projected, teacher):
    if not projected:
        raise ValueError("Feature loss requires at least one active projector")
    with torch.autocast(device_type=next(iter(projected.values())).device.type, enabled=False):
        losses = {}
        for scale, student in projected.items():
            reference = teacher[scale]
            if student.ndim != 4 or student.shape != reference.shape:
                raise ValueError(f"{scale} shape mismatch {student.shape} vs {reference.shape}")
            losses[scale] = (student.float() - reference.detach().float()).square().mean()
        return torch.stack(list(losses.values())).mean(), losses


def effective_lambda(weight, warmup_epochs, updates, steps_per_epoch):
    if warmup_epochs == 0:
        return float(weight)
    return float(weight) * min(1.0, updates / (warmup_epochs * steps_per_epoch))
