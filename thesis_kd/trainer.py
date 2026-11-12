"""Continuous 120-epoch student training with one DDP forward per RGB batch."""
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from . import checkpoint
from .losses import effective_lambda, feature_loss
from .rng import capture_rng, isolated_rng, restore_rng, seed_all
from .student_wrapper import StudentWrapper


def optimizer_for(wrapper, config):
    """Apply native regexes to unprefixed names; also identify unnamed normalization layers."""
    lr, decay = config["lr"], config["weight_decay"]
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._NormBase,
                  nn.modules.instancenorm._InstanceNorm)
    norms = {id(p) for layer in wrapper.detector.modules() if (isinstance(layer, norm_types) or type(layer).__name__ == "RMSNorm")
             for p in layer.parameters(recurse=False)}
    grouped = {}
    for name, param in wrapper.detector.named_parameters():
        if not param.requires_grad:
            continue
        rules = [r for r in config.get("params", []) if re.search(r["params"], name)]
        if len(rules) > 1:
            raise ValueError(f"Overlapping optimizer rules for {name}")
        rule = rules[0] if rules else {}
        default_lr = config["backbone_lr"] if name.startswith("backbone.dinov3.") else lr
        group_lr = rule.get("lr", default_lr)
        group_decay = 0. if id(param) in norms or "bias" in name.lower() else rule.get("weight_decay", decay)
        key = ("detector", group_lr, group_decay)
        grouped.setdefault(key, []).append((name, param))
    if wrapper.projectors:
        grouped["projectors", config["projector_lr"], config["projector_weight_decay"]] = list(wrapper.projectors.named_parameters())
    groups = [
        {"params": [p for _, p in pairs], "param_names": [name for name, _ in pairs],
         "role": role, "lr": group_lr, "initial_lr": group_lr, "weight_decay": group_decay}
        for (role, group_lr, group_decay), pairs in grouped.items()
    ]
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in wrapper.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer coverage is not exactly one group per trainable parameter")
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(config["betas"]), weight_decay=decay)


def rank_state(loader):
    return {"rng": capture_rng(), "generator": loader.generator.get_state(),
            "sampler_epoch": getattr(loader.sampler, "epoch", None)}


def gather_state(loader, world):
    current = rank_state(loader)
    if world == 1:
        return [current]
    states = [None] * world
    dist.all_gather_object(states, current)
    return states


def set_epoch(loader, epoch):
    for item in (loader.dataset, loader.collate_fn, loader.sampler):
        if hasattr(item, "set_epoch"):
            item.set_epoch(epoch)


def train_epoch(model, wrapper, teacher, criterion, ema, loader, optimizer, scaler, scheduler,
                cfg, device, epoch, updates, smoke, world, smoke_steps=2):
    model.train()
    criterion.train()
    set_epoch(loader, epoch)
    native, kd = cfg["native"], cfg["kd"]
    steps = len(loader)
    totals = {}
    count = 0
    initial_updates = updates
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        scheduler.step(updates + 1, optimizer)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            output, projected = model(images, targets=targets)
        with torch.autocast(device_type=device.type, enabled=False):
            native_terms = criterion(output, targets, epoch=epoch, step=i,
                                     global_step=epoch * steps + i, epoch_step=steps)
            native_loss = sum(native_terms.values())
            raw = {}
            mean = native_loss.new_zeros(())
            weight = 0.
            if teacher is not None:
                with torch.no_grad():
                    reference = teacher(images.float())
                mean, raw = feature_loss(projected, reference)
                weight = effective_lambda(kd["lambda_kd"], kd["warmup_epochs"], updates, steps)
                del reference
            total = native_loss + weight * mean
        if not torch.isfinite(total) or not torch.isfinite(output["pred_boxes"]).all():
            raise FloatingPointError(f"Nonfinite loss/pred_boxes in epoch {epoch + 1}, batch {i}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is None for p in wrapper.projectors.parameters()):
            raise RuntimeError("Active projector disconnected from loss")
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), native["clip_max_norm"],
                                      error_if_nonfinite=not scaler.is_enabled())
        last_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= last_scale:
            ema.update(wrapper.detector)
            updates += 1
        metrics = {"native_loss": native_loss, "mean_kd": mean, "weighted_kd": weight * mean,
                   "effective_lambda": weight, "total_loss": total}
        metrics.update({f"kd_{key}": value for key, value in raw.items()})
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + (value.detach() if isinstance(value, torch.Tensor) else value)
        count += 1
        del output, projected, native_terms, raw, total, mean, metrics, native_loss, value
        if smoke and count >= smoke_steps:
            break
    values = torch.stack([torch.as_tensor(v, device=device).double() for v in totals.values()])
    if world > 1:
        dist.all_reduce(values)
    report = dict(zip(totals, (values / (count * world)).tolist()))
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    resources = torch.tensor([elapsed, peak], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(resources, op=dist.ReduceOp.MAX)
    elapsed, peak = resources.tolist()
    report.update(epoch=epoch + 1, train_seconds=elapsed, peak_memory_bytes=int(peak),
                  lr=[g["lr"] for g in optimizer.param_groups], optimizer_updates=updates,
                  amp_skipped=count - (updates - initial_updates), batches=count)
    return report, updates


def run_training(cfg, run_dir, seed, device="cuda", *, resume=False, smoke=False, smoke_steps=2):
    from engine.optim.ema import ModelEMA
    from engine.optim.lr_scheduler import FlatCosineLRScheduler
    from .config import kd_enabled
    from .data import build_train_loader
    from .evaluate import validate_model
    from .factories import build_student, component
    from .teacher import build_teacher

    if smoke and smoke_steps < 1:
        raise ValueError("smoke_steps must be positive")
    device, run_dir = torch.device(device), Path(run_dir)
    if cfg["protocol_id"] == "feature_kd_ms_v1" and not smoke and (
            device.type != "cuda" or not cfg["native"]["use_amp"]):
        raise ValueError("Formal feature_kd_ms_v1 training requires CUDA AMP; CPU is supported for smoke/fixtures")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    native = cfg["native"]
    seed_all(seed)
    student, init = build_student(cfg, initialize=not resume)
    student = student.to(device)
    if world > 1 and native.get("sync_bn", False) and device.type == "cuda":
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    ema_args = {k: v for k, v in native["ema"].items() if k != "type"}
    ema = ModelEMA(student, **ema_args)  # deepcopy before attaching any hook
    ema.module.float()
    teacher = identity = teacher_native = None
    with isolated_rng():
        if kd_enabled(cfg):
            teacher, identity, teacher_native = build_teacher(cfg)
            teacher = teacher.to(device).eval().requires_grad_(False)
        wrapper = StudentWrapper(student, cfg["kd"]["active_scales"] if teacher else (),
                                 teacher.channels if teacher else None).to(device)
    try:
        if identity and not smoke:
            checkpoint.pin_teacher(run_dir.parent.parent, identity)
        # Native denoising heads are conditional on GT. Each enabled projector must receive a
        # gradient independently of GT, even at lambda=0 during warm-up (verified in train_epoch).
        model = DistributedDataParallel(wrapper, device_ids=[device.index if device.index is not None
                                        else torch.cuda.current_device()] if device.type == "cuda" else None,
                                        find_unused_parameters=native.get("find_unused_parameters", False)
                                        ) if world > 1 else wrapper
        optimizer = optimizer_for(wrapper, native["optimizer"])
        criterion = component(native, native["criterion"]).to(device)
        seed_all(seed + rank)
        loader = build_train_loader(cfg, seed)
        steps = len(loader)
        if not steps:
            raise ValueError("Training loader contains zero batches at the configured global batch size")
        epochs = native["epoches"]
        scheduler = FlatCosineLRScheduler(optimizer, native["lr_gamma"], steps, epochs,
                                          native["warmup_iter"], native["flat_epoch"], native["no_aug_epoch"])
        scaler = torch.amp.GradScaler(device.type, enabled=native["use_amp"] and device.type == "cuda")
        groups = [{k: g[k] for k in ("role", "initial_lr", "weight_decay", "param_names")}
                  for g in optimizer.param_groups]
        signature = {"arch": cfg["arch"], "experiment": cfg["experiment"], "seed": seed,
                     "teacher": identity, "distillation": cfg["kd"], "protocol_id": cfg["protocol_id"],
                     "optimizer_groups": groups, "world_size": world, "steps_per_epoch": steps,
                     "stage_policy": cfg["stage_policy"], "selection": cfg["selection"], "tie_rule": cfg["tie_rule"],
                     "native_digest": hashlib.sha256(json.dumps(native, sort_keys=True,
                                                 default=str).encode("utf-8")).hexdigest()}
        epoch0 = updates = 0
        selection = None
        history, diagnostics = [], []
        if resume:
            state = torch.load(run_dir / "last_training.pth", map_location="cpu", weights_only=False)
            checkpoint.restore_training(state, signature, wrapper, ema, optimizer, scaler)
            identity = state["teacher"]  # Preserve original T0 provenance when a file is relocated.
            signature["teacher"] = identity
            epoch0, updates = state["epoch"], state["optimizer_updates"]
            selection, history, diagnostics = state["selection"], state["history"], state["diagnostics"]
            local = state["rank_states"][rank]
            restore_rng(local["rng"])
            loader.generator.set_state(local["generator"])
            if rank == 0:
                checkpoint.publish_epoch(run_dir, selection, history, diagnostics, recover=True)
            del state
        elif rank == 0:
            if teacher_native is not None:
                checkpoint.atomic_text(run_dir / "teacher_model_config.yml", yaml.safe_dump(teacher_native))
            checkpoint.write_json(run_dir / "run_info.json", dict(
                experiment=cfg["experiment"], protocol_id=cfg["protocol_id"], arch=cfg["arch"], seed=seed,
                run_tag=run_dir.parent.parent.name, global_batch=native["train_dataloader"]["total_batch_size"],
                per_rank_batch=loader.batch_size, world_size=world, teacher=identity,
                train_init=init, smoke=smoke, pytorch_version=str(torch.__version__), device=str(device),
                argv=sys.argv, invocation=getattr(sys, "orig_argv", sys.argv),
                python_executable=sys.executable, cuda_version=torch.version.cuda,
                hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                cuda_amp=scaler.is_enabled(), validation_amp=False, validation_tf32=False,
                training_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                training_cudnn_tf32=torch.backends.cudnn.allow_tf32,
                sdp_flash=torch.backends.cuda.flash_sdp_enabled(),
                sdp_math=torch.backends.cuda.math_sdp_enabled(),
                sdp_memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                train_workers_per_rank=loader.num_workers,
                validation_global_batch=native["val_dataloader"]["total_batch_size"],
                validation_per_rank_batch=native["val_dataloader"]["total_batch_size"] // world,
                validation_workers_per_rank=native["val_dataloader"]["num_workers"],
                ddp_backend=dist.get_backend() if world > 1 else None,
                stage_policy=cfg["stage_policy"]))
        for epoch in range(epoch0, min(1, epochs) if smoke else epochs):
            report, updates = train_epoch(model, wrapper, teacher, criterion, ema, loader, optimizer, scaler,
                                          scheduler, cfg, device, epoch, updates, smoke, world, smoke_steps)
            diagnostics.append(report)
            if smoke:
                if rank == 0:
                    checkpoint.write_json(run_dir / "kd_diagnostics.json", diagnostics)
                    checkpoint.atomic_text(run_dir / "train.log", json.dumps(report) + "\n")
                if updates == 0:
                    raise FloatingPointError("Smoke completed without a successful optimizer update")
                return report
            with isolated_rng():
                stats = validate_model(cfg, ema.module, device, seed)
            keys = ("AP", "AP50", "AP75", "AP-S", "AP-M", "AP-L", "AR1", "AR10", "AR100", "AR-S", "AR-M", "AR-L")
            if len(stats) != len(keys) or not math.isfinite(stats[0]) or stats[0] < 0:
                raise ValueError("No valid FP32 mixed validation AP for checkpoint selection")
            history.append({"epoch": epoch + 1, **dict(zip(keys, stats))})
            states = gather_state(loader, world)
            if rank == 0:
                if selection is None or stats[0] > selection["mixed_ap"]:
                    pending = run_dir / "best_pending.pth"
                    checkpoint.atomic_save(ema.module.state_dict(), pending)
                    selection = dict(epoch=epoch + 1, mixed_ap=stats[0], sha256=checkpoint.digest(pending),
                                     protocol_id=cfg["protocol_id"], experiment=cfg["experiment"], seed=seed,
                                     run_tag=run_dir.parent.parent.name, teacher=identity,
                                     checkpoint="best_student.pth", tie_break="earliest_epoch",
                                     source="FP32 EMA Mixed Validation AP")
                state = checkpoint.training_state(wrapper, ema, optimizer, scaler, epoch=epoch + 1,
                    updates=updates, signature=signature, rank_states=states, history=history,
                    diagnostics=diagnostics, selection=selection, cfg=cfg)
                checkpoint.atomic_save(state, run_dir / "last_training.pth")
                checkpoint.publish_epoch(run_dir, selection, history, diagnostics)
            if world > 1:
                dist.barrier()
        return {"epoch": epochs, "global_step": epochs * steps, "optimizer_updates": updates, "selection": selection}
    finally:
        wrapper.close()
