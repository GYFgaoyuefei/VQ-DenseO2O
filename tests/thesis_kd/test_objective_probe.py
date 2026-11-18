"""Train-only smoke of the milestone reader/CLI with private synthetic model resources."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from thesis_kd import objective_probe as probe
from thesis_kd import checkpoint
from thesis_kd.config import resolve_config
from thesis_kd.student_wrapper import StudentWrapper


class TinyEncoder(nn.Module):
    out_strides = (8, 16, 32)
    out_channels = (2, 2, 2)

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        return [F.avg_pool2d(self.conv(x), stride) for stride in self.out_strides]


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2))
        self.encoder = TinyEncoder()

    def forward(self, x, targets=None):
        return {"values": self.encoder(self.backbone(x))[0]}


class TinyCriterion(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return {"loss": outputs["values"].square().mean()}


class TinyTeacher(TinyDetector):
    channels = {"P3": 2, "P4": 2, "P5": 2}

    def forward(self, images):
        return dict(zip(self.channels, self.encoder(self.backbone(images))))


@pytest.mark.parametrize("experiment", ["m_cwd_l5_t1", "m_mse_l5"])
def test_probe_isolated_rng_bn_and_snapshot(tmp_path, monkeypatch, experiment):
    cfg = resolve_config(experiment)
    run = tmp_path / experiment / "seed_42"
    run.mkdir(parents=True)
    detector = TinyDetector()
    teacher = TinyTeacher()
    trained = StudentWrapper(detector, cfg["kd"]["active_scales"], teacher.channels)
    for p in trained.projectors.parameters():
        nn.init.constant_(p, 0.25)
    teacher_bn = deepcopy(teacher.backbone[1].state_dict())
    identity = {"id": "synthetic_T0"}
    info = {"arch": "m", "experiment": experiment, "protocol_id": cfg["protocol_id"], "seed": 42,
            "teacher": identity, "smoke": False, "run_tag": "study02",
            "study02_inputs": {"annotation_sha256": {"train": "synthetic_annotation"}}}
    (run / "run_info.json").write_text(json.dumps(info))
    sig = {**info, "distillation": cfg["kd"], "steps_per_epoch": 1}
    snap = run / "epoch_005_training.pth"
    torch.save({"config": cfg, "signature": sig, "protocol_id": cfg["protocol_id"], "teacher": identity,
                "epoch": 5, "global_step": 5, "optimizer_updates": 5, "kd_progress": 5,
                "scheduler": {"updates": 5, "steps_per_epoch": 1}, "selection": {"run_tag": "study02"},
                "student": trained.detector.state_dict(), "projectors": trained.projectors.state_dict()}, snap)
    trained.close()
    before = snap.read_bytes()
    def loader(cfg, seed):
        class Batch(list):
            dataset = collate_fn = sampler = None
        return Batch([(torch.randn(1, 3, 64, 64), [{"image_id": torch.tensor(17)}])])
    monkeypatch.setattr(probe, "run_path", lambda *args: run)
    monkeypatch.setattr(probe, "safe_output", Path)
    monkeypatch.setattr(probe, "verify_annotation", lambda *args: "synthetic_annotation")
    monkeypatch.setattr(probe, "build_teacher", lambda cfg: (teacher, identity, {}))
    monkeypatch.setattr(probe, "build_student", lambda *args, **kw: (TinyDetector(), {}))
    monkeypatch.setattr(probe, "component", lambda *args: TinyCriterion())
    monkeypatch.setattr(probe, "build_train_loader", loader)
    monkeypatch.setattr(sys, "argv", ["objective_probe", "--experiment", experiment, "--epoch", "5",
                                      "--batches", "1", "--device", "cpu"])
    rng = torch.get_rng_state()
    # Training may hold the run lock while the probe reads a published milestone.
    with probe.writer_lock(run):
        probe.main()
    assert snap.read_bytes() == before and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(x, teacher_bn[k]) for k, x in teacher.backbone[1].state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    dest = run / "objective_probes/epoch_005_1_batches.json"
    report = json.loads(dest.read_text())
    row = report["batches"][0]
    assert report["effective_lambda"] == 5 and report["split"] == "train"
    assert set(row["per_scale"]) == {"P3", "P4", "P5"}
    assert row["native_loss"] + row["weighted_kd"] == pytest.approx(row["total_loss"])
    for group in row["shared_gradients"].values():
        assert group["weighted_kd_norm"] > 0 and group["R"] > 0 and abs(group["cosine"]) <= 1
    with pytest.raises(FileExistsError):
        probe.main()


def test_probe_publication_failure_is_retryable_and_preserves_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "run_path", lambda *args: tmp_path)
    monkeypatch.setattr(probe, "safe_output", Path)
    monkeypatch.setattr(probe, "probe", lambda *args: {"split": "train"})
    monkeypatch.setattr(sys, "argv", ["objective_probe", "--experiment", "m_mse_l5", "--epoch", "5"])
    dest = tmp_path / "objective_probes/epoch_005_8_batches.json"

    def fail_replace(source, target):
        assert json.loads(Path(source).read_text()) == {"split": "train"}
        assert target == dest
        raise OSError("simulated rename failure")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.os, "replace", fail_replace)
        with pytest.raises(OSError, match="rename failure"):
            probe.main()
    assert not dest.exists() and not dest.with_suffix(".json.tmp").exists()
    probe.main()
    winner = dest.read_bytes()

    # Simulate another reporter publishing after preflight but before this writer locks.
    dest.unlink()
    def competing_probe(*args):
        with probe.writer_lock(dest.parent):
            dest.write_bytes(winner)
        return {"would_overwrite_winner": True}
    monkeypatch.setattr(probe, "probe", competing_probe)
    with pytest.raises(FileExistsError):
        probe.main()
    assert dest.read_bytes() == winner and dest.stat().st_nlink == 1
