"""DC-STA: degradation-conditioned residual gates for native STA details.

Only the detail features are modulated.  The caller retains the original
semantic-first concatenation, 1x1 projection and SyncBatchNorm.
"""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


GATE_ARCHITECTURE = 1  # 1 + alpha * tanh(logit), not the historical sigmoid gate.


def validate_dc_checkpoint(model, state_dict):
    """Validate gate identity before tuning discards unknown or mismatched keys.

    A native detector checkpoint has no DC subtree and can initialize a new gate.
    An existing DC subtree must identify the residual-tanh architecture explicitly.
    This deliberately does not guess the architecture of unmarked checkpoints.
    """
    for name, module in model.named_modules():
        if isinstance(module, DCSTA):
            module.validate_checkpoint(state_dict, name + "." if name else "")


def audit_dc_training(model, optimizer, ema, check_gradients=False):
    """Inspect real gate parameters, optimizer membership and EMA (no forward).

    ``check_gradients`` is for a post-backward, unscaled/pre-clip observation.
    Zero gradients in the first channel MLP layer at identity are expected;
    missing/nonfinite gradients or an entirely zero gate signal are failures.
    """
    from ..misc.dist_utils import de_parallel

    model = de_parallel(model)
    gates = [(name, module) for name, module in model.named_modules() if isinstance(module, DCSTA)]
    if not gates:
        return {"enabled": False, "status": "NOT APPLICABLE", "parameter_names": []}
    parameters = {f"{name}.{key}" if name else key: value
                  for name, module in gates for key, value in module.named_parameters()}
    occurrences = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            occurrences[id(parameter)] = occurrences.get(id(parameter), 0) + 1
    missing = [name for name, parameter in parameters.items() if occurrences.get(id(parameter), 0) == 0]
    duplicates = [name for name, parameter in parameters.items() if occurrences.get(id(parameter), 0) > 1]
    frozen = [name for name, parameter in parameters.items() if not parameter.requires_grad]
    ema_module = getattr(ema, "module", None)
    ema_state = ema_module.state_dict() if ema_module is not None else {}
    ema_missing = [name for name, parameter in parameters.items()
                   if name not in ema_state or ema_state[name].shape != parameter.shape]
    ema_non_fp32 = [name for name in parameters if name in ema_state and ema_state[name].dtype != torch.float32]
    marker_names = [(name + "." if name else "") + "_gate_architecture" for name, _ in gates]
    ema_marker_ok = all(name in ema_state and torch.equal(ema_state[name].cpu(), torch.tensor(GATE_ARCHITECTURE))
                        for name in marker_names)
    identity = all(torch.count_nonzero(parameter.detach()).item() == 0
                   for _, module in gates
                   for layer in [*(mlp[-1] for mlp in module.channel), *module.spatial]
                   for parameter in layer.parameters())
    report = {"enabled": True, "parameter_names": sorted(parameters),
              "parameter_count": len(parameters), "optimizer_missing": missing,
              "optimizer_duplicates": duplicates, "frozen_parameters": frozen,
              "optimizer_coverage_ok": not (missing or duplicates or frozen),
              "ema_missing": ema_missing, "ema_non_fp32": ema_non_fp32,
              "ema_marker_ok": ema_marker_ok,
              "ema_coverage_ok": not ema_missing and not ema_non_fp32 and ema_marker_ok,
              "identity_initialized": identity}
    if check_gradients:
        missing_grad, nonfinite_grad, zero_grad = [], [], []
        norm_sq = 0.0
        for name, parameter in parameters.items():
            gradient = parameter.grad
            if gradient is None:
                missing_grad.append(name)
            elif not torch.isfinite(gradient).all().item():
                nonfinite_grad.append(name)
            else:
                norm_sq += gradient.detach().double().square().sum().item()
                if not torch.count_nonzero(gradient).item():
                    zero_grad.append(name)
        report.update(gradient_missing=missing_grad, gradient_nonfinite=nonfinite_grad,
                      gradient_zero=zero_grad, gradient_l2_norm=math.sqrt(norm_sq) if not nonfinite_grad else None,
                      gradients_ok=not missing_grad and not nonfinite_grad and norm_sq > 0)
    report["status"] = "PASS" if (report["optimizer_coverage_ok"] and report["ema_coverage_ok"]
                                   and report.get("gradients_ok", True)) else "FAIL"
    return report


class DCSTA(nn.Module):
    """Three-scale, exactly identity-initialized ``1 + alpha*tanh(.)`` gates.

    The descriptor uses current augmented RGB recovered from normalization. Its
    last coordinate is a spatial finite-difference statistic, not an FFT feature.
    ``input_mean`` / ``input_std`` must match the input Normalize transform.
    """

    def __init__(self, detail_channels: Sequence[int], options: Mapping | None = None):
        super().__init__()
        options = dict(options or {})
        supported = {"enabled", "alpha", "input_mean", "input_std"}
        unknown = set(options) - supported
        if unknown:
            raise ValueError(f"Unknown DC-STA options: {sorted(unknown)}")
        if len(detail_channels) != 3 or any(int(c) <= 0 for c in detail_channels):
            raise ValueError("DC-STA requires three positive detail channel counts")
        self.detail_channels = tuple(int(c) for c in detail_channels)
        # An integer buffer is copied into EMA/checkpoints but never EMA-averaged.
        self.register_buffer("_gate_architecture", torch.tensor(GATE_ARCHITECTURE, dtype=torch.int64))
        self.alpha = float(options.get("alpha", 0.1))
        if not math.isfinite(self.alpha) or not 0 <= self.alpha < 1:
            raise ValueError("DC-STA alpha must be finite and in [0, 1)")
        mean = torch.as_tensor(options.get("input_mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
        std = torch.as_tensor(options.get("input_std", [0.229, 0.224, 0.225]), dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("DC-STA input_mean/input_std must be finite RGB triples with positive std")
        self.register_buffer("input_mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", std.view(1, 3, 1, 1), persistent=False)
        self.channel = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, channels))
            for channels in self.detail_channels
        ])
        self.spatial = nn.ModuleList([nn.Conv2d(10, 1, 3, padding=1) for _ in self.detail_channels])
        for mlp in self.channel:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        for conv in self.spatial:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.capture_diagnostics = False
        self.last_statistics = {}

    @staticmethod
    def validate_checkpoint(state_dict, prefix=""):
        present = any(key.startswith(prefix) for key in state_dict)
        if not present:
            return  # Native pretrained detector: initialize new residual gates.
        marker = state_dict.get(prefix + "_gate_architecture")
        if (not isinstance(marker, torch.Tensor) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() != GATE_ARCHITECTURE):
            raise RuntimeError("DC-STA checkpoint is unmarked or uses a different gate architecture; "
                               "residual-tanh weights require _gate_architecture=1. "
                               "Historical sigmoid checkpoints must not be loaded as residual-tanh gates.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.validate_checkpoint(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def descriptor(self, images: torch.Tensor) -> torch.Tensor:
        """Return float32 [B,8] statistics after inverse Normalize and RGB clamp."""
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("DC-STA expects [B,3,H,W] normalized RGB images")
        with torch.autocast(device_type=images.device.type, enabled=False):
            rgb = (images.float() * self.input_std.float() + self.input_mean.float()).clamp(0, 1)
            rgb = F.adaptive_avg_pool2d(rgb, (80, 80))
            luminance = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1)
            finite_difference = 0.5 * (
                (luminance[:, 1:, :] - luminance[:, :-1, :]).abs().mean((1, 2))
                + (luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean((1, 2))
            )
            luminance_statistics = torch.stack([
                luminance.mean((1, 2)),
                luminance.std((1, 2), unbiased=False),
                torch.sigmoid(12 * (0.2 - luminance)).mean((1, 2)),
                torch.sigmoid(12 * (luminance - 0.85)).mean((1, 2)),
            ], dim=1)
            return torch.cat([luminance_statistics, rgb.mean((2, 3)), finite_difference[:, None]], dim=1)

    def compute_gates(self, descriptor, semantic, detail, level):
        """Return channel [B,C,1,1] and spatial [B,1,H,W] residual gates."""
        mlp = self.channel[level]
        channel = 1 + self.alpha * torch.tanh(mlp(descriptor.to(mlp[0].weight.dtype)))
        local_input = torch.cat([
            semantic.mean(1, keepdim=True),
            detail.mean(1, keepdim=True),
            descriptor.to(detail.dtype)[:, :, None, None].expand(-1, -1, *detail.shape[-2:]),
        ], dim=1)
        spatial = 1 + self.alpha * torch.tanh(self.spatial[level](local_input))
        return channel[:, :, None, None].to(detail.dtype), spatial.to(detail.dtype)

    def forward(self, images, semantic, detail):
        """Gate detail only; preserve semantic tensors and native fusion layout."""
        if len(semantic) != 3 or len(detail) != 3:
            raise ValueError("DC-STA requires exactly three semantic/detail scales")
        descriptor = self.descriptor(images)
        result = []
        self.last_statistics = {}
        for level, (sem, sta) in enumerate(zip(semantic, detail)):
            if sem.ndim != 4 or sta.ndim != 4 or sem.shape[0] != images.shape[0] or sta.shape[0] != images.shape[0] or sem.shape[-2:] != sta.shape[-2:]:
                raise ValueError("DC-STA semantic/detail batch or spatial shapes differ")
            if sta.shape[1] != self.detail_channels[level]:
                raise ValueError("DC-STA detail channels do not match the configured pyramid")
            channel, spatial = self.compute_gates(descriptor, sem, sta, level)
            result.append(sta * channel * spatial)
            if self.capture_diagnostics:
                # Keep reduced, detached values on device; no per-step CPU copies.
                self.last_statistics[f"c{level + 2}"] = {
                    "channel_mean": channel.detach().float().mean(),
                    "spatial_mean": spatial.detach().float().mean(),
                }
        return result
