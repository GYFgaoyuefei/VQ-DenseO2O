"""Capture exactly one live encoder output from this student instance."""
import torch
from torch import nn


class StudentWrapper(nn.Module):
    def __init__(self, detector, active_scales=(), teacher_channels=None):
        super().__init__()
        self.detector = detector
        self.projectors = nn.ModuleDict()
        self._handle = None
        self._features = None
        self._capturing = False
        scales = tuple(active_scales)
        if not scales:
            return
        if len(scales) != len(set(scales)) or set(scales) - {"P3", "P4", "P5"}:
            raise ValueError(f"Invalid active scales {scales}")
        encoder = detector.encoder
        if tuple(encoder.out_strides) != (8, 16, 32) or len(encoder.out_channels) != 3:
            raise ValueError("Student encoder must expose P3/P4/P5 at strides 8/16/32")
        for scale in scales:
            index = int(scale[1:]) - 3
            self.projectors[scale] = nn.Conv2d(
                encoder.out_channels[index], teacher_channels[scale], 1, bias=False
            )
        self._handle = encoder.register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        if self._capturing:
            if self._features is not None or not isinstance(output, (tuple, list)) or len(output) != 3:
                raise RuntimeError("Expected one three-level encoder output")
            self._features = output

    def forward(self, samples, targets=None):
        self._features = None
        self._capturing = bool(self.projectors)
        try:
            outputs = self.detector(samples, targets=targets)
            projected = {}
            with torch.autocast(device_type=samples.device.type, enabled=False):
                for scale, projector in self.projectors.items():
                    if self._features is None:
                        raise RuntimeError("Encoder did not run inside StudentWrapper.forward")
                    index = int(scale[1:]) - 3
                    stride = 2 ** int(scale[1:])
                    feature = self._features[index]
                    spatial = tuple(n // stride for n in samples.shape[-2:])
                    if feature.shape[-2:] != spatial:
                        raise ValueError(f"{scale}: {feature.shape[-2:]} differs from {spatial}")
                    projected[scale] = projector(feature.float())
            return outputs, projected
        finally:
            self._capturing = False
            self._features = None

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._features = None
