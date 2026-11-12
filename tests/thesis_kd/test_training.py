"""Synthetic fixtures exercise mechanics; native forward tests live in test_models."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from thesis_kd import checkpoint
from thesis_kd.config import kd_enabled, resolve_config
from thesis_kd.losses import effective_lambda, feature_loss
from thesis_kd.rng import isolated_rng, seed_all
from thesis_kd.student_wrapper import StudentWrapper
from thesis_kd.trainer import optimizer_for


class Encoder(nn.Module):
    out_channels = [3, 3, 3]
    out_strides = [8, 16, 32]

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        self.calls += 1
        x = self.conv(x)
        return [F.avg_pool2d(x, s) for s in self.out_strides]


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, samples, targets=None):
        return {"pred_boxes": self.encoder(samples)[-1].flatten(1)}


channels = dict(P3=2, P4=2, P5=2)


def test_active_scale_elementwise_and_gradient():
    tensor = torch.tensor([2., 6., 4., 0.]).reshape(1, 1, 2, 2).requires_grad_()
    out = {"P3": tensor, "P5": torch.ones(1, 1, 1, 1)}
    ref = {k: torch.zeros_like(v, requires_grad=True) for k, v in out.items()}
    mean, scales = feature_loss(out, ref)
    assert scales["P3"].item() == 14
    assert mean.item() == (14 + 1) / 2
    mean.backward()
    torch.testing.assert_close(tensor.grad, tensor / 4)
    assert all(v.grad is None for v in ref.values())
    assert feature_loss({"P3": tensor}, ref)[0].item() == 14
    with pytest.raises(ValueError, match="shape"):
        feature_loss({"P3": tensor}, {"P3": ref["P5"]})


def test_hook_unchanged_output_single_encoder_exception_cleanup():
    detector = Detector()
    x = torch.randn(1, 3, 64, 64)
    expected = detector(x)
    wrapped = StudentWrapper(detector, ["P4"], channels)
    prior = detector.encoder.calls
    out, projected = wrapped(x, targets=[{"labels": torch.empty(0)}])
    torch.testing.assert_close(out["pred_boxes"], expected["pred_boxes"])
    assert detector.encoder.calls == prior + 1
    assert list(projected) == list(wrapped.projectors) == ["P4"]
    assert projected["P4"].shape == (1, 2, 4, 4)
    projected["P4"].square().mean().backward()
    assert detector.encoder.conv.weight.grad.abs().sum() > 0
    assert wrapped._features is None
    with pytest.raises(RuntimeError):
        wrapped(torch.zeros(1, 1, 64, 64))
    assert wrapped._features is None and not wrapped._capturing
    wrapped.close()
    assert not detector.encoder._forward_hooks


def test_initialization_rng_isolation():
    def trial(enable):
        seed_all(42)
        detector = Detector()
        with isolated_rng():
            if enable:
                Detector()
                random.random()
                np.random.random()
                if torch.cuda.is_available():
                    torch.rand(10, device="cuda")
            StudentWrapper(detector, ["P3"] if enable else [], channels)
        seq = (random.random(), np.random.random(), torch.rand(1).item(),
               torch.rand(1, device="cuda").item() if torch.cuda.is_available() else None)
        return detector.state_dict(), seq
    base, base_seq = trial(False)
    kd, kd_seq = trial(True)
    assert kd_seq == base_seq
    for name in base:
        assert torch.equal(base[name], kd[name])


def test_lambda_zero_same_update_and_no_capture():
    cfg = resolve_config("s_kd")
    cfg["kd"]["lambda_kd"] = 0.
    assert not kd_enabled(cfg)
    detector = Detector()
    base = deepcopy(detector)
    wrapper = StudentWrapper(detector)
    x = torch.randn(2, 3, 32, 32)
    out, features = wrapper(x, targets=[{}, {}])
    loss = out["pred_boxes"].square().mean()
    baseline = base(x)["pred_boxes"].square().mean()
    assert not features and not wrapper.projectors and not detector.encoder._forward_hooks
    assert torch.equal(baseline, loss)
    for net, value in [(base, baseline), (wrapper, loss)]:
        opt = torch.optim.AdamW(net.parameters())
        value.backward()
        opt.step()
    for a, b in zip(base.parameters(), detector.parameters()):
        assert torch.equal(a, b)
    assert [effective_lambda(1., 5, i, 2) for i in (0, 5, 10, 20)] == [0., .5, 1., 1.]


def test_resume_export_and_ema(tmp_path):
    from engine.optim.ema import ModelEMA
    cfg = resolve_config("m_kd")
    detector = Detector()
    ema = ModelEMA(detector)
    wrapper = StudentWrapper(detector, ["P3"], channels)
    frozen = Detector().requires_grad_(False)
    opt = optimizer_for(wrapper, cfg["native"]["optimizer"])
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    group_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(group_ids) == len(set(group_ids)) == len(list(wrapper.parameters()))
    assert set(group_ids).isdisjoint(map(id, frozen.parameters()))
    assert set(ema.module.state_dict()) == set(detector.state_dict())
    assert not ema.module.encoder._forward_hooks
    y, z = wrapper(torch.randn(2, 3, 32, 32))
    (y["pred_boxes"].sum() + z["P3"].square().sum()).backward()
    opt.step()
    ema.update(detector)
    sig = {"protocol_id": cfg["protocol_id"], "teacher": "teacher-A", "steps_per_epoch": 2}
    state = checkpoint.training_state(wrapper, ema, opt, scaler, epoch=2, updates=4, signature=sig,
        rank_states=[], history=[], diagnostics=[], selection={}, cfg=cfg)
    path = tmp_path / "last_training.pth"
    checkpoint.atomic_save(state, path)
    state = torch.load(path, weights_only=False)
    student2 = Detector()
    ema2 = ModelEMA(student2)
    wrapper2 = StudentWrapper(student2, ["P3"], channels)
    opt2 = optimizer_for(wrapper2, cfg["native"]["optimizer"])
    for field in ("teacher", "protocol_id"):
        with pytest.raises(ValueError, match=field):
            checkpoint.restore_training(state, {**sig, field: "different"}, wrapper2, ema2, opt2, scaler)
    checkpoint.restore_training(state, sig, wrapper2, ema2, opt2, scaler)
    assert state["kd_progress"] == state["global_step"] == state["scheduler"]["updates"] == 4
    torch.testing.assert_close(wrapper2.projectors["P3"].weight, wrapper.projectors["P3"].weight)
    torch.testing.assert_close(opt2.state[wrapper2.projectors["P3"].weight]["exp_avg"],
                               opt.state[wrapper.projectors["P3"].weight]["exp_avg"])
    assert ema2.updates == ema.updates == 1
    pending = tmp_path / "best_pending.pth"
    checkpoint.atomic_save(ema.module.state_dict(), pending)
    selection = {"sha256": checkpoint.digest(pending), "checkpoint": "best_student.pth"}
    checkpoint.publish_epoch(tmp_path, selection, [{"epoch": 2}], [{"epoch": 2}], recover=True)
    deployed = tmp_path / "exported.pth"
    checkpoint.export_student(tmp_path, deployed)
    clean = Detector().eval()
    clean.load_state_dict(torch.load(deployed, weights_only=True), strict=True)
    sample = torch.randn(1, 3, 32, 32)
    torch.testing.assert_close(ema.module(sample)["pred_boxes"], clean(sample)["pred_boxes"])
    with pytest.raises(FileExistsError):
        checkpoint.export_student(tmp_path, deployed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_fp32_projectors_and_loss():
    net = StudentWrapper(Detector(), ["P3", "P4", "P5"], channels).cuda()
    x = torch.randn(2, 3, 64, 64, device="cuda")
    with torch.autocast("cuda"):
        native, projected = net(x)
        loss, _ = feature_loss(projected, {k: torch.zeros_like(v) for k, v in projected.items()})
    (loss + native["pred_boxes"].float().mean()).backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(v.dtype == torch.float32 for v in projected.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def gloo_worker(rank, path):
    from datetime import timedelta
    dist.init_process_group("gloo", store=dist.FileStore(path, 2), rank=rank, world_size=2,
                            timeout=timedelta(seconds=55))
    try:
        seed_all(12)
        wrapped = StudentWrapper(Detector(), ["P4"], channels)
        model = nn.parallel.DistributedDataParallel(wrapped)
        local_reference = deepcopy(wrapped.projectors["P4"])
        samples = torch.full((1, 3, 32, 32), float(rank + 1))
        native, projected = model(samples)
        (native["pred_boxes"].sum() + projected["P4"].sum()).backward()
        local_reference(wrapped.detector.encoder(samples)[1].detach()).sum().backward()
        # Unique inputs produce different local gradients. Compare DDP to their explicit mean.
        expected = local_reference.weight.grad
        before = expected.clone()
        dist.all_reduce(expected)
        expected /= 2
        assert not torch.allclose(before, expected)
        torch.testing.assert_close(wrapped.projectors["P4"].weight.grad, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo unavailable")
def test_actual_two_rank_gloo_projector_average(tmp_path):
    torch.multiprocessing.spawn(gloo_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)


def test_teacher_pin(tmp_path):
    checkpoint.pin_teacher(tmp_path, {"id": "A", "source": "ema"})
    checkpoint.pin_teacher(tmp_path, {"id": "A", "source": "ema"})
    with pytest.raises(ValueError, match="T0"):
        checkpoint.pin_teacher(tmp_path, {"id": "B", "source": "ema"})


@pytest.mark.skipif(not torch.cuda.is_available() or not dist.is_available(),
                    reason="requires CUDA and a DDP backend")
def test_native_ddp_single_gpu_alternating_empty_gt(tmp_path):
    """Two DDP iterations through the native S denoising branch; synthetic 160px input."""
    from thesis_kd.factories import build_student, component
    seed_all(3407)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, store=dist.FileStore(str(tmp_path / "native_store"), 1),
                            rank=0, world_size=1)
    try:
        cfg = resolve_config("s_kd")
        native = cfg["native"]
        student, _ = build_student(cfg, initialize=False)
        wrapped = StudentWrapper(student, ["P3", "P4", "P5"],
                                 {k: 224 for k in ("P3", "P4", "P5")}).cuda()
        model = nn.parallel.DistributedDataParallel(wrapped, device_ids=[torch.cuda.current_device()],
                                                   find_unused_parameters=native["find_unused_parameters"])
        opt = optimizer_for(wrapped, native["optimizer"])
        criterion = component(native, native["criterion"]).cuda()
        images = torch.randn(1, 3, 160, 160, device="cuda")
        for count in [0, 1]:
            targets = [{"labels": torch.zeros(count, dtype=torch.long, device="cuda"),
                        "boxes": torch.full((count, 4), 0.4, device="cuda")}]
            out, projected = model(images, targets=targets)
            native_loss = sum(criterion(out, targets).values())
            mean, _ = feature_loss(projected, {k: torch.zeros_like(v) for k, v in projected.items()})
            total = native_loss + mean
            assert torch.isfinite(total)
            opt.zero_grad(set_to_none=True)
            total.backward()
            assert all(p.grad is not None and p.grad.norm().item() > 0 for p in wrapped.projectors.parameters())
            opt.step()
        wrapped.close()
    finally:
        dist.destroy_process_group()


def test_fixture_full_kd_epoch_commit_and_resume(tmp_path, monkeypatch):
    """Real S/VQ-L/native COCO components, two tiny epochs with generated weights and images."""
    import gc
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from PIL import Image
    from thesis_kd.config import ROOT
    from thesis_kd.factories import build_student, component
    from thesis_kd.teacher import teacher_config
    from thesis_kd.trainer import run_training

    with TemporaryDirectory(dir=tmp_path) as fixture:
        base = Path(fixture)
        seed_all(42)
        monkeypatch.setenv("TOMATO_DATA_ROOT", str(base))
        cfg = resolve_config("s_kd")
        cfg["protocol_id"] = "synthetic_resume_fixture"
        native = cfg["native"]
        native["epoches"] = 2
        native["warmup_iter"] = 1
        native["no_aug_epoch"] = 0
        native["flat_epoch"] = 1
        native["train_dataloader"]["collate_fn"]["base_size_repeat"] = None
        (base / "annotations").mkdir()
        for split in ("train", "val"):
            settings = native[f"{split}_dataloader"]
            settings.update(total_batch_size=1, num_workers=0)
            if split == "train":
                for op in settings["dataset"]["transforms"]["ops"]:
                    if op["type"] == "Resize":
                        op["size"] = [160, 160]
            images = base / "images" / split
            images.mkdir(parents=True)
            Image.new("RGB", (160, 160), (170, 70, 80)).save(images / "fixture.png")
            data = {"images": [{"id": 1, "file_name": "fixture.png", "height": 160, "width": 160}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [30, 40, 60, 50],
                                     "iscrowd": 0, "area": 3000}],
                    "categories": [{"id": i, "name": name} for i, name in
                                   enumerate(["immature", "half", "ripe"], start=1)]}
            (base / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))
        student, _ = build_student(cfg, initialize=False)
        r = cfg["resources"]
        for role, model in [("BACKBONE_WEIGHTS", student.backbone.dinov3._model),
                            ("DETECTOR_CHECKPOINT", student)]:
            path = base / f"fixture_s_{role}.pth"
            torch.save(model.state_dict(), path)
            r["THESIS_KD_S_" + role] = str(path)
        del model, student
        t0_path = ROOT / "configs/tomato/paper_v2/vq_dense_o2o.yml"
        t0_cfg = teacher_config(t0_path)
        t0 = component(t0_cfg, t0_cfg["model"])
        path = base / "fixture_t0_ema.pth"
        torch.save({"ema": {"module": t0.state_dict()}}, path)
        r.update(THESIS_KD_TEACHER_CONFIG=str(t0_path), THESIS_KD_TEACHER_CHECKPOINT=str(path),
                 THESIS_KD_TEACHER_VALIDATION="Generated random model for I/O integration only")
        del t0
        gc.collect()
        run = base / "fixture_tag" / "s_kd" / "seed_42"
        run.mkdir(parents=True)
        publish = checkpoint.publish_epoch

        def interrupt_after_commit(*args, **kwargs):
            raise InterruptedError("fixture: process interrupted after epoch commit")

        monkeypatch.setattr(checkpoint, "publish_epoch", interrupt_after_commit)
        with pytest.raises(InterruptedError, match="fixture"):
            run_training(cfg, run, 42, device="cpu")
        monkeypatch.setattr(checkpoint, "publish_epoch", publish)
        result = run_training(cfg, run, 42, device="cpu", resume=True)
        assert result["epoch"] == result["optimizer_updates"] == 2
        assert (run / "best_student.pth").is_file()
        record = json.loads((run / "kd_diagnostics.json").read_text())
        assert [row["optimizer_updates"] for row in record] == [1, 2]
        assert record[0]["effective_lambda"] == 0 < record[1]["effective_lambda"]
        assert all(record[1][f"kd_P{i}"] > 0 for i in [3, 4, 5])
        selected = json.loads((run / "selection.json").read_text())
        assert checkpoint.digest(run / "best_student.pth") == selected["sha256"]
        assert len((run / "validation_history.csv").read_text().splitlines()) == 3
