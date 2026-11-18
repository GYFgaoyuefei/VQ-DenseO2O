"""Measure unscaled, pre-clip Train gradients on a private study02 snapshot copy."""
import argparse
from itertools import islice
import json
from pathlib import Path
import statistics

import torch

from .checkpoint import atomic_text, comparable_teacher
from .config import resolve_config
from .data import build_train_loader
from .factories import build_student, component, sha256_file
from .losses import effective_lambda
from .objective_inputs import native_fingerprint, verify_annotation
from .objective_losses import objective_loss
from .rng import isolated_rng, seed_all
from .safeguards import run_path, safe_output, writer_lock
from .student_wrapper import StudentWrapper
from .teacher import build_teacher
from .trainer import OBJECTIVE_PROTOCOL, set_epoch


def _shared_gradients(detector, native, weighted):
    """Compare gradients on the same trainable parameters, including unused entries as zeros."""
    groups = [[(name, p) for name, p in module.named_parameters() if p.requires_grad]
              for module in (detector.backbone, detector.encoder)]
    params = [p for group in groups for _, p in group]
    det_grad = torch.autograd.grad(native, params, allow_unused=True, retain_graph=True)
    kd_grad = torch.autograd.grad(weighted, params, allow_unused=True)
    result = {}
    start = 0
    for label, group in zip(("backbone", "encoder"), groups):
        end = start + len(group)
        native_none, kd_none = [], []
        # Scalar reductions stay on the device until the whole group is accumulated.
        squares_a = native.new_zeros((), dtype=torch.float64)
        squares_b = torch.zeros_like(squares_a)
        dot = torch.zeros_like(squares_a)
        for (name, _), a, b in zip(group, det_grad[start:end], kd_grad[start:end]):
            if a is None:
                native_none.append(name)
            else:
                squares_a += a.norm().double().square()
            if b is None:
                kd_none.append(name)
            else:
                squares_b += b.norm().double().square()
            if a is not None and b is not None:
                dot += (a * b).sum(dtype=torch.float64)
        an, bn, product = torch.stack([squares_a.sqrt(), squares_b.sqrt(), dot]).tolist()
        result[label] = {"native_norm": an, "weighted_kd_norm": bn,
                         "R": bn / (an + 1e-12), "cosine": product / (an * bn) if an and bn else None,
                         "native_none": native_none, "weighted_kd_none": kd_none}
        start = end
    return result


def _snapshot(cfg, run, tag, epoch):
    path = safe_output(run / f"epoch_{epoch:03d}_training.pth")
    info = json.loads(safe_output(run / "run_info.json").read_text(encoding="utf8"))
    state = torch.load(path, map_location="cpu", weights_only=False)
    sig = state["signature"]
    for key, value in (("protocol_id", cfg["protocol_id"]), ("experiment", cfg["experiment"]),
                       ("seed", 42), ("arch", cfg["arch"])):
        if info[key] != value or sig[key] != value:
            raise ValueError(f"Run and checkpoint identity must match {key}={value!r}")
    if (info["smoke"] or info["run_tag"] != tag or state["selection"]["run_tag"] != tag
            or state["epoch"] != epoch or state["protocol_id"] != cfg["protocol_id"]
            or sig["distillation"] != cfg["kd"] or state["config"]["kd"] != cfg["kd"]
            or native_fingerprint(cfg["native"]) != native_fingerprint(state["config"]["native"])
            or not state["projectors"] or info["teacher"] != sig["teacher"]
            or state["teacher"] != sig["teacher"]):
        raise ValueError("Snapshot differs from the requested run, epoch, objective, recipe or teacher")
    updates, steps = state["optimizer_updates"], sig["steps_per_epoch"]
    if (state["scheduler"] != {"updates": updates, "steps_per_epoch": steps}
            or state["kd_progress"] != updates or state["global_step"] != epoch * steps
            or not 0 < updates <= state["global_step"]):
        raise ValueError("Inconsistent training progress in snapshot")
    return path, info, state["student"], state["projectors"], updates, steps


@isolated_rng()
def probe(cfg, tag, epoch, batches, device):
    from engine.solver.det_engine import fp32_evaluation

    if cfg["protocol_id"] != OBJECTIVE_PROTOCOL or epoch not in cfg.get("snapshot_epochs", []):
        raise ValueError("Select a study02 experiment and a retained snapshot epoch")
    if batches < 1:
        raise ValueError("batches must be positive")
    run = run_path(cfg, tag, 42)
    path, info, weights, projectors, updates, steps = _snapshot(cfg, run, tag, epoch)
    dataset = cfg["native"]["train_dataloader"]["dataset"]
    annotation = verify_annotation("train", dataset["ann_file"])
    if annotation != info["study02_inputs"]["annotation_sha256"]["train"]:
        raise ValueError("Train annotations differ from the recorded run")
    device = torch.device(device)
    teacher, identity, _ = build_teacher(cfg)
    if comparable_teacher(identity) != comparable_teacher(info["teacher"]):
        raise ValueError("Teacher differs from the snapshot")
    teacher.to(device).eval().requires_grad_(False)
    student, _ = build_student(cfg, initialize=False)
    wrapper = StudentWrapper(student, cfg["kd"]["active_scales"], teacher.channels).float()
    try:
        student.load_state_dict(weights, strict=True)
        wrapper.projectors.load_state_dict(projectors, strict=True)
        del weights, projectors
        wrapper.to(device).train()
        criterion = component(cfg["native"], cfg["native"]["criterion"]).to(device).train()
        seed_all(info["seed"])
        loader = build_train_loader(cfg, info["seed"])
        if len(loader) != steps or batches > steps:
            raise ValueError(f"Train loader has {len(loader)} batches, snapshot has {steps}, requested {batches}")
        batch_epoch = min(epoch, cfg["native"]["epoches"] - 1)
        set_epoch(loader, batch_epoch)
        kd = cfg["kd"]
        weight = effective_lambda(kd["lambda_kd"], kd["warmup_epochs"], updates, steps)
        rows = []
        with fp32_evaluation(wrapper, device, enabled=True):
            for i, (images, targets) in enumerate(islice(loader, batches)):
                images = images.to(device).float()
                targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
                outputs, projected = wrapper(images, targets=targets)
                native = sum(criterion(outputs, targets, epoch=batch_epoch, step=i,
                                       global_step=batch_epoch * steps + i, epoch_step=steps).values())
                with torch.no_grad():
                    reference = teacher(images)
                mean, per_scale = objective_loss(projected, reference, kd)
                weighted = weight * mean
                rows.append({"batch": i, "image_ids": [int(t["image_id"].item()) for t in targets],
                             "shape": list(images.shape), "native_loss": native.item(), "kd_loss": mean.item(),
                             "weighted_kd": weighted.item(), "total_loss": (native + weighted).item(),
                             "per_scale": {k: v.item() for k, v in per_scale.items()},
                             "shared_gradients": _shared_gradients(student, native, weighted)})
                del outputs, projected, reference, native, weighted, mean, per_scale
    finally:
        wrapper.close()
    distributions = {}
    for name in ("backbone", "encoder"):
        distributions[name] = {}
        for metric in ("R", "cosine"):
            values = [r["shared_gradients"][name][metric] for r in rows]
            valid = [v for v in values if v is not None]
            distributions[name][metric] = ({"count": len(valid), "min": min(valid), "median": statistics.median(valid),
                                            "max": max(valid)} if valid else {"count": 0})
    return {"experiment": cfg["experiment"], "protocol_id": cfg["protocol_id"], "run_tag": tag, "seed": info["seed"],
            "split": "train", "mode": "fresh single-process FP32 batches at fixed snapshot; no optimizer updates",
            "device": str(device), "snapshot": path.name, "snapshot_sha256": sha256_file(path),
            "completed_epochs": epoch, "probe_epoch_zero_based": batch_epoch, "optimizer_updates": updates,
            "teacher_id": identity["id"], "train_annotation_sha256": annotation,
            "kd": kd, "effective_lambda": weight, "batches": rows, "distributions": distributions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("m_mse_l5", "m_cwd_l5_t1"), required=True)
    parser.add_argument("--run-tag", default="study02")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--paths", type=Path)
    args = parser.parse_args()
    cfg = resolve_config(args.experiment, args.paths)
    dest = safe_output(run_path(cfg, args.run_tag, 42) / "objective_probes"
                       / f"epoch_{args.epoch:03d}_{args.batches}_batches.json")
    if dest.exists():
        raise FileExistsError(dest)
    report = probe(cfg, args.run_tag, args.epoch, args.batches, args.device)
    payload = json.dumps(report, indent=2, allow_nan=False)
    dest.parent.mkdir(exist_ok=True)
    with writer_lock(dest.parent):
        if dest.exists():
            raise FileExistsError(dest)
        try:
            atomic_text(dest, payload + "\n")
        finally:
            dest.with_suffix(dest.suffix + ".tmp").unlink(missing_ok=True)
    print(dest)


if __name__ == "__main__":
    main()
