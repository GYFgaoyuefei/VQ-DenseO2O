"""Small generated Train batches and full-state checkpoint publication fixtures."""
import math

import pytest
import torch
from torch.utils.data import DataLoader

from engine.optim.ema import ModelEMA
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from thesis_kd import checkpoint
from thesis_kd.config import ROOT, resolve_config
from thesis_kd.factories import build_student, component
from thesis_kd.safeguards import inspect_tree
from thesis_kd.student_wrapper import StudentWrapper
from thesis_kd.teacher import Teacher, teacher_config
from thesis_kd.trainer import optimizer_for, rank_state, run_training, train_epoch


@pytest.mark.parametrize("experiment", ["m_mse_l5", "m_cwd_l5_t1"])
@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP smoke requires a CUDA device"))])
def test_train_objective_native_smoke(experiment, backend):
    cfg = resolve_config(experiment)
    native = cfg["native"]
    device = torch.device(backend)
    torch.manual_seed(42)
    student, _ = build_student(cfg, initialize=False)
    student.to(device)
    ema = ModelEMA(student)
    teacher_cfg = teacher_config(ROOT / "configs/tomato/paper_v2/vq_dense_o2o.yml")
    teacher = Teacher(component(teacher_cfg, teacher_cfg["model"])).to(device)
    wrapper = StudentWrapper(student, cfg["kd"]["active_scales"], teacher.channels).to(device)
    optimizer = optimizer_for(wrapper, native["optimizer"])
    criterion = component(native, native["criterion"]).to(device)
    # An ordinary and an empty target. Both seeds and images here are fixtures.
    images = torch.randn(2, 3, 160, 160)
    targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[.5, .5, .3, .3]])},
               {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
    loader = DataLoader([(images, targets), (images, targets)], batch_size=None,
                        generator=torch.Generator().manual_seed(42))
    scheduler = FlatCosineLRScheduler(optimizer, native["lr_gamma"], len(loader), native["epoches"],
                                      native["warmup_iter"], native["flat_epoch"], native["no_aug_epoch"])
    scaler = torch.amp.GradScaler(backend, enabled=backend == "cuda", init_scale=32)
    try:
        report, updates = train_epoch(wrapper, wrapper, teacher, criterion, ema, loader, optimizer,
                                      scaler, scheduler, cfg, device, epoch=0, updates=0,
                                      smoke=True, world=1, smoke_steps=2)
        assert report["batches"] == updates == 2
        assert report["amp_skipped"] == 0
        for metric in ("native_loss", "mean_kd", "weighted_kd", "effective_lambda", "total_loss",
                       "kd_P3", "kd_P4", "kd_P5"):
            assert math.isfinite(report[metric]) and report[metric] > 0, metric
        for proj in wrapper.projectors.values():
            grad = proj.weight.grad
            assert grad is not None and grad.isfinite().all() and grad.abs().max() > 0
        grads = [p.grad for p in student.encoder.parameters() if p.grad is not None]
        assert grads and all(g.isfinite().all() for g in grads) and any(g.abs().max() > 0 for g in grads)
        assert all(p.grad is None for p in teacher.parameters())
    finally:
        wrapper.close()


def test_objective_formal_requires_cuda_amp(tmp_path):
    cfg = resolve_config("m_mse_l5")
    with pytest.raises(ValueError, match="requires CUDA AMP"):
        run_training(cfg, tmp_path, 42, device="cpu")
    cfg["native"]["use_amp"] = False
    with pytest.raises(ValueError, match="requires CUDA AMP"):
        run_training(cfg, tmp_path, 42, device="cuda")


def test_immutable_snapshot_recovery_and_restore(tmp_path, monkeypatch):
    from torch import nn

    model = nn.Module()
    model.detector = nn.Linear(2, 1)
    model.projectors = nn.ModuleDict({"P3": nn.Linear(1, 1, bias=False)})
    ema = ModelEMA(model.detector)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    value = model.projectors["P3"](model.detector(torch.ones(1, 2))).sum()
    value.backward()
    optimizer.step()
    ema.update(model.detector)
    loader = DataLoader([1], generator=torch.Generator().manual_seed(42))
    cfg = resolve_config("m_mse_l5")
    schedule = cfg["snapshot_epochs"]
    assert schedule == [5, 20, 60, 120]
    signature = {"steps_per_epoch": 1, "teacher": {"id": "fixture"},
                 "distillation": cfg["kd"], "snapshot_epochs": schedule}
    last = tmp_path / "last_training.pth"
    checkpoint.publish_training_snapshot(tmp_path, 4, schedule)
    assert not list(tmp_path.iterdir())
    for epoch in schedule:
        state = checkpoint.training_state(model, ema, optimizer, scaler, epoch=epoch, updates=epoch,
            signature=signature, rank_states=[rank_state(loader)], history=[{"epoch": epoch}],
            diagnostics=[{"epoch": epoch}], selection={"epoch": 1}, cfg=cfg)
        checkpoint.atomic_save(state, last)
        target = tmp_path / f"epoch_{epoch:03d}_training.pth"
        if epoch == 5:
            def interrupted(src, dst):
                dst.write_bytes(b"partial")
                raise OSError("interrupted copy")

            with monkeypatch.context() as m:
                m.setattr(checkpoint.shutil, "copyfile", interrupted)
                with pytest.raises(OSError, match="interrupted copy"):
                    checkpoint.publish_training_snapshot(tmp_path, epoch, schedule)
            assert not target.exists()
        checkpoint.publish_training_snapshot(tmp_path, epoch, schedule)
        stamp = target.stat().st_mtime_ns
        checkpoint.publish_training_snapshot(tmp_path, epoch, schedule)
        assert target.stat().st_mtime_ns == stamp
        assert checkpoint.digest(target) == checkpoint.digest(last)
        assert not target.samefile(last)
        inspect_tree(tmp_path)
    restored = torch.load(tmp_path / "epoch_005_training.pth", map_location="cpu", weights_only=False)
    assert restored["epoch"] == restored["scheduler"]["updates"] == 5
    assert restored["rank_states"][0].keys() >= {"rng", "generator"}
    assert restored["history"] == restored["diagnostics"] == [{"epoch": 5}]
    assert restored["config"] == cfg and restored["selection"] == {"epoch": 1}
    with torch.no_grad():
        model.projectors["P3"].weight.add_(2)
    checkpoint.restore_training(restored, signature, model, ema, optimizer, scaler)
    torch.testing.assert_close(model.projectors["P3"].weight, restored["projectors"]["P3.weight"])
    assert optimizer.state[model.projectors["P3"].weight]["exp_avg"].abs().max() > 0
    with pytest.raises(ValueError, match="Conflicting immutable"):
        checkpoint.publish_training_snapshot(tmp_path, 5, schedule)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Objective protocol formal entry requires CUDA")
def test_resume_republishes_interrupted_milestone(tmp_path, monkeypatch):
    """Test trainer orchestration with cheap epoch results, including the final publication."""
    from thesis_kd import data, evaluate, factories, objective_inputs, teacher, trainer

    cfg = resolve_config("m_cwd_l5_t1")
    cfg["native"]["epoches"] = 5
    identity = {"id": "fixture", "checkpoint_sha256": "0" * 64,
                "state_path": "ema.module", "config_digest": "1" * 64}
    loader = DataLoader([1], generator=torch.Generator().manual_seed(42))

    def student_fixture(cfg, initialize):
        detector = torch.nn.Module()
        detector.encoder = torch.nn.Conv2d(2, 2, 1)
        detector.encoder.out_strides = [8, 16, 32]
        detector.encoder.out_channels = [2, 2, 2]
        return detector, {}

    teacher_model = torch.nn.Linear(1, 1)
    teacher_model.channels = {scale: 2 for scale in ("P3", "P4", "P5")}
    monkeypatch.setattr(teacher, "build_teacher", lambda cfg: (teacher_model, identity, {}))
    monkeypatch.setattr(factories, "build_student", student_fixture)
    monkeypatch.setattr(factories, "component", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(data, "build_train_loader", lambda cfg, seed: loader)
    monkeypatch.setattr(evaluate, "validate_model", lambda *args: [.5] * 12)
    monkeypatch.setattr(objective_inputs, "verify_study02_inputs", lambda *args, **kwargs: {"fixture": True})
    epochs_seen = []

    def epoch_fixture(model, wrapper, teacher, criterion, ema, loader, optimizer, scaler,
                      scheduler, cfg, device, epoch, updates, smoke, world, smoke_steps):
        epochs_seen.append(epoch + 1)
        return {"epoch": epoch + 1}, updates + 1

    monkeypatch.setattr(trainer, "train_epoch", epoch_fixture)
    run = tmp_path / "tag" / cfg["experiment"] / "seed_42"
    run.mkdir(parents=True)
    publish = checkpoint.publish_training_snapshot

    def interrupt_on_fifth(run_dir, epoch, snapshot_epochs):
        if epoch == 5:
            raise InterruptedError("interrupted at milestone")
        return publish(run_dir, epoch, snapshot_epochs)

    with monkeypatch.context() as m:
        m.setattr(checkpoint, "publish_training_snapshot", interrupt_on_fifth)
        with pytest.raises(InterruptedError, match="at milestone"):
            run_training(cfg, run, 42)
    assert epochs_seen == [1, 2, 3, 4, 5]
    target = run / "epoch_005_training.pth"
    assert not target.exists()
    record = run_training(cfg, run, 42, resume=True)
    assert record["epoch"] == 5 and epochs_seen == [1, 2, 3, 4, 5]
    assert checkpoint.digest(target) == checkpoint.digest(run / "last_training.pth")
