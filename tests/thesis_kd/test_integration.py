"""Native DETR integration on synthetic tensors. No training assets or Test split used."""
import os
import subprocess
import sys

import pytest
import torch

from thesis_kd.config import ROOT, resolve_config
from thesis_kd.factories import build_student, component
from thesis_kd.losses import feature_loss
from thesis_kd.student_wrapper import StudentWrapper
from thesis_kd.teacher import Teacher, teacher_config
from thesis_kd.trainer import optimizer_for


def test_native_detector_loss_gradients_with_empty_image():
    # Small valid multiple of all strides, retaining at least 300 encoder candidates.
    cfg = resolve_config("m_kd")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    student, _ = build_student(cfg, initialize=False)
    # Native DDF skips identical corner logits. Give scratch heads distinct
    # outputs so this synthetic batch actually exercises that native branch.
    with torch.no_grad():
        for head in student.decoder.dec_bbox_head:
            head.layers[-1].bias.normal_(0., .005)
    l_cfg = teacher_config(ROOT / "configs/tomato/paper_v2/vq_dense_o2o.yml")
    teacher = Teacher(component(l_cfg, l_cfg["model"])).to(device)
    wrapper = StudentWrapper(student, cfg["kd"]["active_scales"], teacher.channels).to(device)
    optimizer = optimizer_for(wrapper, cfg["native"]["optimizer"])
    criterion = component(cfg["native"], cfg["native"]["criterion"]).to(device)
    x = torch.randn(2, 3, 192, 192, device=device)
    targets = [
        {"labels": torch.tensor([2], device=device), "boxes": torch.tensor([[.5, .5, .4, .3]], device=device)},
        {"labels": torch.empty(0, dtype=torch.long, device=device), "boxes": torch.empty(0, 4, device=device)}
    ]
    try:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output, projected = wrapper(x, targets=targets)
        native = criterion(output, targets, epoch=0, step=0, global_step=0, epoch_step=1)
        assert all(any(key.startswith(term) for key in native) for term in
                   ("loss_mal", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"))
        feature, _ = feature_loss(projected, teacher(x))
        loss = sum(native.values()) + feature
        assert torch.isfinite(loss)
        loss.backward()
        for proj in wrapper.projectors.values():
            assert torch.isfinite(proj.weight.grad).all() and proj.weight.grad.abs().max() > 0
        assert any(p.grad is not None for p in student.encoder.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), .1, error_if_nonfinite=True)
        assert norm > 0
        optimizer.step()
    finally:
        wrapper.close()


def test_native_student_strict_deployment_fresh_interpreter(tmp_path, request):
    old_threads = torch.get_num_threads()
    old_precision = torch.get_float32_matmul_precision()
    request.addfinalizer(lambda: torch.set_num_threads(old_threads))
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(old_precision))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    cfg = resolve_config("s_base")
    torch.manual_seed(0)
    model, _ = build_student(cfg, initialize=False)
    model.float().eval()
    # For this test the exact output of the original native detector is the reference.
    x = torch.zeros(1, 3, 640, 640)
    with torch.inference_mode():
        output = model(x)
    torch.save(model.state_dict(), tmp_path / "student.pth")
    torch.save({k: output[k] for k in ("pred_logits", "pred_boxes")}, tmp_path / "ref.pth")
    code = """
import sys, torch
from engine.core import YAMLConfig
from pathlib import Path
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
p = Path(sys.argv[1])
cfg = YAMLConfig('configs/deimv2/deimv2_dinov3_s_coco.yml')
cfg.yaml_cfg['num_classes'] = 3
cfg.yaml_cfg['DINOv3STAs']['weights_path'] = None
model = cfg.model.float().eval()
model.load_state_dict(torch.load(p / 'student.pth', map_location='cpu', weights_only=True), strict=True)
with torch.inference_mode():
    actual = model(torch.zeros(1, 3, 640, 640))
for key, ref in torch.load(p / 'ref.pth', weights_only=True).items():
    torch.testing.assert_close(actual[key], ref, rtol=0, atol=0)
assert 'thesis_kd' not in sys.modules
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("THESIS_KD_")}
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=ROOT, env=env, check=True)
