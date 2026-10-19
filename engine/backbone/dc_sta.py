"""DC-STA: beta_select sigmoid gates for native STA details.

Only the detail features are modulated.  The caller retains the original
semantic-first concatenation, 1x1 projection and SyncBatchNorm.
"""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


GATE_ARCHITECTURE = 2  # Restored beta_select sigmoid; 1 denotes retired residual-tanh.


def validate_dc_checkpoint(model, state_dict):
    """Validate gate identity before tuning discards unknown or mismatched keys.

    A native detector checkpoint has no DC subtree and can initialize a new gate.
    An existing DC subtree must identify the restored sigmoid architecture explicitly.
    This deliberately does not guess the architecture of unmarked checkpoints.
    """
    for name, module in model.named_modules():
        if isinstance(module, DCSTA):
            module.validate_checkpoint(state_dict, name + "." if name else "")


def audit_dc_training(model, optimizer, ema, check_gradients=False):
    """Inspect real gate parameters, optimizer membership and EMA (no forward).

    ``check_gradients`` is for a post-backward, unscaled/pre-clip observation.
    Zero gradients in the first channel MLP layer at initialization are expected;
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
    initialized = all(torch.count_nonzero(layer.weight.detach()).item() == 0
                      and torch.equal(layer.bias.detach(), torch.full_like(layer.bias, module.initial_bias))
                      for _, module in gates
                      for layer in [*(mlp[-1] for mlp in module.channel), *module.spatial])
    report = {"enabled": True, "parameter_names": sorted(parameters),
              "parameter_count": len(parameters), "optimizer_missing": missing,
              "optimizer_duplicates": duplicates, "frozen_parameters": frozen,
              "optimizer_coverage_ok": not (missing or duplicates or frozen),
              "ema_missing": ema_missing, "ema_non_fp32": ema_non_fp32,
              "ema_marker_ok": ema_marker_ok,
              "ema_coverage_ok": not ema_missing and not ema_non_fp32 and ema_marker_ok,
              "gate_architecture": "beta_select_sigmoid",
              "initial_detail_gain": gates[0][1].initial_detail_gain,
              "near_identity_initialized": initialized}
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
    """Original global-channel/local-spatial sigmoid gates, initially gain .999.

    The descriptor uses current augmented RGB recovered from ImageNet Normalize.
    Its last coordinate is a spatial finite-difference statistic, not an FFT.
    This module accepts no targets or instance masks.
    """

    def __init__(self, detail_channels: Sequence[int], options: Mapping | None = None):
        super().__init__()
        options = dict(options or {})
        supported = {"enabled", "descriptor_dim", "levels", "gate_init", "init_detail_gain",
                     "use_global_gate", "channel_gate", "use_local_reliability", "spatial_gate",
                     "use_luminance_stats", "use_color_stats", "use_high_frequency_proxy"}
        unknown = set(options) - supported
        if unknown:
            raise ValueError(f"Unknown DC-STA options: {sorted(unknown)}")
        if len(detail_channels) != 3 or any(int(c) <= 0 for c in detail_channels):
            raise ValueError("DC-STA requires three positive detail channel counts")
        self.detail_channels = tuple(int(c) for c in detail_channels)
        if options.get("descriptor_dim", 8) != 8 or options.get("levels", ["c2", "c3", "c4"]) != ["c2", "c3", "c4"]:
            raise ValueError("DC-STA requires the eight-statistic descriptor and native c2/c3/c4 levels")
        if options.get("gate_init", "identity") != "identity":
            raise ValueError("Only beta_select near-identity initialization is supported")
        self.options = options
        self.global_on = options.get("use_global_gate", True) and options.get("channel_gate", True)
        self.local_on = options.get("use_local_reliability", True) and options.get("spatial_gate", True)
        active = int(self.global_on) + int(self.local_on)
        gain = float(options.get("init_detail_gain", .999))
        if not 0 < gain < 1 or active == 0:
            raise ValueError("Enable a gate and use an initial detail gain strictly between 0 and 1")
        probability = gain ** (1 / active)
        self.initial_detail_gain = gain
        self.initial_bias = math.log(probability / (1 - probability))
        # An integer buffer is copied into EMA/checkpoints but never EMA-averaged.
        self.register_buffer("_gate_architecture", torch.tensor(GATE_ARCHITECTURE, dtype=torch.int64))
        self.channel = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, channels))
            for channels in self.detail_channels
        ]) if self.global_on else nn.ModuleList()
        self.spatial = nn.ModuleList([nn.Conv2d(10, 1, 3, padding=1) for _ in self.detail_channels]) if self.local_on else nn.ModuleList()
        for mlp in self.channel:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.constant_(mlp[-1].bias, self.initial_bias)
        for conv in self.spatial:
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, self.initial_bias)
        self.capture_diagnostics = False
        self.last_statistics = {}

    @staticmethod
    def validate_checkpoint(state_dict, prefix=""):
        present = any(key.startswith(prefix) for key in state_dict)
        if not present:
            return  # Native pretrained detector: initialize new sigmoid gates.
        marker = state_dict.get(prefix + "_gate_architecture")
        if (not isinstance(marker, torch.Tensor) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() != GATE_ARCHITECTURE):
            raise RuntimeError("DC-STA checkpoint is unmarked or uses a different gate architecture; "
                               "restored beta_select sigmoid weights require _gate_architecture=2. "
                               "Residual-tanh and historical unmarked checkpoints cannot resume new experiments; "
                               "start from the shared native detector pretraining.")

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
            x = images.float()
            rgb = (x * x.new_tensor([.229, .224, .225])[None, :, None, None]
                   + x.new_tensor([.485, .456, .406])[None, :, None, None]).clamp(0, 1)
            rgb = F.adaptive_avg_pool2d(rgb, (80, 80))
            luminance = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1)
            finite_difference = 0.5 * (
                (luminance[:, 1:, :] - luminance[:, :-1, :]).abs().mean((1, 2))
                + (luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean((1, 2))
            )
            luminance_statistics = torch.stack([
                luminance.mean((1, 2)),
                (luminance.var((1, 2), unbiased=False) + 1e-8).sqrt(),
                torch.sigmoid(12 * (0.2 - luminance)).mean((1, 2)),
                torch.sigmoid(12 * (luminance - 0.85)).mean((1, 2)),
            ], dim=1)
            if not self.options.get("use_luminance_stats", True):
                luminance_statistics = torch.zeros_like(luminance_statistics)
            color = rgb.mean((2, 3))
            if not self.options.get("use_color_stats", True):
                color = torch.zeros_like(color)
            if not self.options.get("use_high_frequency_proxy", True):
                finite_difference = torch.zeros_like(finite_difference)
            descriptor = torch.cat([luminance_statistics, color, finite_difference[:, None]], dim=1)
            if not torch.isfinite(descriptor).all():
                raise FloatingPointError("Nonfinite DC descriptor")
            return descriptor

    def compute_gates(self, descriptor, semantic, detail, level):
        """Return original sigmoid channel [B,C,1,1] and spatial [B,1,H,W]."""
        channel = (torch.sigmoid(self.channel[level](descriptor))[:, :, None, None]
                   if self.global_on else detail.new_ones((detail.shape[0], 1, 1, 1)))
        if self.local_on:
            local_input = torch.cat([
                semantic.mean(1, keepdim=True),
                detail.mean(1, keepdim=True),
                descriptor.to(detail.dtype)[:, :, None, None].expand(-1, -1, *detail.shape[-2:]),
            ], dim=1)
            spatial = torch.sigmoid(self.spatial[level](local_input))
        else:
            spatial = detail.new_ones((detail.shape[0], 1, *detail.shape[-2:]))
        return channel.to(detail.dtype), spatial.to(detail.dtype)

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
                    "global_mean": channel.detach().float().flatten(1).mean(1),
                    "global_std": channel.detach().float().flatten(1).std(1, unbiased=False),
                    "local_mean": spatial.detach().float().flatten(1).mean(1),
                    "local_std": spatial.detach().float().flatten(1).std(1, unbiased=False),
                }
        return result
