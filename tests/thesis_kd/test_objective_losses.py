"""CWD numerical and autograd regressions with deterministic synthetic features."""
import pytest
import torch
from torch import nn

from thesis_kd.losses import feature_loss
from thesis_kd.objective_losses import objective_loss


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def device(request):
    return torch.device(request.param)


def test_cwd_kl_direction_temperature_and_scale_reduction(device):
    temperature = 2.5
    kd = {"objective": "cwd", "temperature": temperature}
    # Explicit normalized, nonuniform distributions for two samples and channels.
    p = torch.tensor([[[0.70, 0.20, 0.10], [0.20, 0.45, 0.35]],
                      [[0.05, 0.20, 0.75], [0.40, 0.30, 0.30]]], device=device)
    q = torch.tensor([[[0.20, 0.20, 0.60], [0.50, 0.40, 0.10]],
                      [[0.20, 0.55, 0.25], [0.30, 0.30, 0.40]]], device=device)
    teacher = (temperature * p.log()).unsqueeze(2)
    student = (temperature * q.log()).unsqueeze(2)
    torch.testing.assert_close((teacher.flatten(2) / temperature).softmax(-1).sum(-1),
                               torch.ones_like(p[..., 0]))
    mean, scales = objective_loss(
        {"P3": student, "P4": teacher, "P5": student.repeat(1, 1, 4, 2)},
        {"P3": teacher, "P4": teacher, "P5": teacher.repeat(1, 1, 4, 2)}, kd)
    expected = ((p * (p.log() - q.log())).sum(-1).mean() * temperature ** 2)
    torch.testing.assert_close(scales["P3"], expected)
    torch.testing.assert_close(scales["P4"], torch.zeros_like(expected))
    # Repeating every position equally preserves spatial KL.
    torch.testing.assert_close(scales["P5"], expected)
    torch.testing.assert_close(mean, 2 * expected / 3)


def test_cwd_per_channel_shift_and_local_perturbation(device):
    kd = {"objective": "cwd", "temperature": 1.0}
    feature = torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 2, 2) / 4
    shift = torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3, 1, 1) - 3
    shifted = feature + shift
    same = objective_loss({"P3": shifted}, {"P3": feature}, kd)[0]
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-6, rtol=0)
    shifted[0, 1, 1, 1] += 2
    assert objective_loss({"P3": shifted}, {"P3": feature}, kd)[0].item() > 0.01


def test_cwd_autocast_fp32_and_projector_gradients(device):
    projector = nn.Conv2d(2, 3, kernel_size=1, bias=False).to(device)
    with torch.no_grad():
        projector.weight.copy_(torch.arange(1, 7, device=device).view(3, 2, 1, 1) / 8)
    x = (torch.arange(16, device=device, dtype=torch.float32) / 8).view(1, 2, 2, 4).requires_grad_()
    teacher = torch.zeros(1, 3, 2, 4, device=device, requires_grad=True)
    kd = {"objective": "cwd", "temperature": 2.0}
    amp_type = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_type):
        y = projector(x)
        mean, scales = objective_loss({"P3": y}, {"P3": teacher}, kd)
    assert y.dtype == amp_type
    assert mean.dtype == scales["P3"].dtype == torch.float32
    reference = objective_loss({"P3": y.detach().float()}, {"P3": teacher}, kd)[0]
    torch.testing.assert_close(mean.detach(), reference, atol=0, rtol=0)
    mean.backward()
    for grad in (x.grad, projector.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert teacher.grad is None


def test_old_mse_dispatch_matches_loss_and_gradient(device):
    student = torch.arange(8, device=device, dtype=torch.float32).view(1, 2, 2, 2).requires_grad_()
    teacher = torch.full_like(student, 1.5, requires_grad=True)
    projected, references = {"P3": student}, {"P3": teacher}
    expected, expected_scales = feature_loss(projected, references)
    expected_grad, = torch.autograd.grad(expected, student)
    for kd in ({}, {"objective": "mse"}):
        loss, scales = objective_loss(projected, references, kd)
        assert torch.equal(loss, expected) and torch.equal(scales["P3"], expected_scales["P3"])
        grad, = torch.autograd.grad(loss, student)
        assert torch.equal(grad, expected_grad)


@pytest.mark.parametrize("temperature", [None, True, "1", 0, -1, float("inf"), float("nan")])
def test_cwd_requires_valid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        objective_loss({}, {}, {"objective": "cwd", "temperature": temperature})


def test_cwd_invalid_objective_or_features():
    kd = {"objective": "cwd", "temperature": 1.0}
    with pytest.raises(ValueError, match="objective"):
        objective_loss({}, {}, {"objective": "unknown"})
    with pytest.raises(ValueError, match="projector"):
        objective_loss({}, {}, kd)
    with pytest.raises(ValueError, match="BCHW"):
        objective_loss({"P3": torch.zeros(1, 1, 3, 3)}, {"P3": torch.zeros(1, 1, 3)}, kd)
