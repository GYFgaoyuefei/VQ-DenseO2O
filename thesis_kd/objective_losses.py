"""Selectable encoder feature objectives for student distillation."""
import math
from numbers import Real

import torch

from .losses import feature_loss


def objective_loss(projected, teacher, kd):
    """Return the equal-scale mean and the unweighted loss for each active scale."""
    objective = kd.get("objective", "mse")
    if objective == "mse":
        return feature_loss(projected, teacher)
    if objective != "cwd":
        raise ValueError(f"Unknown feature objective: {objective!r}")

    temperature = kd.get("temperature")
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("CWD temperature must be a finite positive number")
    if not projected:
        raise ValueError("CWD requires at least one active projector")

    losses = {}
    with torch.autocast(device_type=next(iter(projected.values())).device.type, enabled=False):
        for scale, student in projected.items():
            reference = teacher[scale]
            if student.ndim != 4 or student.shape != reference.shape or student.numel() == 0:
                raise ValueError(f"{scale}: expected matching nonempty BCHW shapes, got "
                                 f"{student.shape} and {reference.shape}")
            # Each sample/channel defines its own distribution over spatial positions.
            student_logp = (student.float().flatten(2) / temperature).log_softmax(dim=-1)
            teacher_logp = (reference.detach().float().flatten(2) / temperature).log_softmax(dim=-1)
            channel_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
            losses[scale] = channel_kl.mean() * temperature ** 2
        return torch.stack(list(losses.values())).mean(), losses
