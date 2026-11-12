"""Command line entry point for the isolated student experiments."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import yaml

from .safeguards import (ROOT, inspect_tree, integrity, run_path, safe_output,
                         segment, writer_lock)

CORE = ("m_base", "m_kd", "s_base", "s_kd")
SCALES = ("m_kd_p3", "m_kd_p4", "m_kd_p5")
SENSITIVITY = ("m_kd_lambda_025", "m_kd_lambda_050", "m_kd_lambda_200")
NAMES = CORE + SCALES + SENSITIVITY


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    commands = main.add_subparsers(dest="action", required=True)
    for name in ("resolve-config", "preflight", "smoke", "train", "resume",
                 "validate", "export-student", "evaluate-test", "benchmark",
                 "summarize", "export-commands", "check-legacy-integrity"):
        p = commands.add_parser(name)
        if name == "check-legacy-integrity":
            p.add_argument("--snapshot", action="store_true")
            p.add_argument("--source", type=Path)
            continue
        p.add_argument("--run-tag", default="study01")
        if name == "export-commands":
            p.add_argument("--include-sensitivity", action="store_true")
            continue
        p.add_argument("--paths", type=Path, help="Local YAML resource paths")
        if name == "summarize":
            p.add_argument("--teacher-result", type=Path)
            p.add_argument("--split", choices=("val", "test"), default="test")
            continue
        p.add_argument("--experiment", choices=NAMES)
        p.add_argument("--seed", type=int, default=None)
        if name in ("train", "resume", "smoke", "validate", "evaluate-test", "benchmark", "preflight"):
            p.add_argument("--device", default="cuda")
        if name == "preflight":
            group = p.add_mutually_exclusive_group()
            group.add_argument("--static", action="store_true")
            group.add_argument("--with-resources", action="store_true")
        if name == "smoke":
            p.add_argument("--steps", type=int, default=2)
        if name in ("resume", "validate", "evaluate-test", "export-student", "benchmark"):
            p.add_argument("--run-dir", type=Path)
        if name == "resume":
            p.add_argument("--checkpoint", type=Path, help="Bound last_training.pth")
        if name == "export-student":
            p.add_argument("--destination", type=Path)
        if name == "evaluate-test":
            p.add_argument("--confirm-frozen", action="store_true")
        if name in ("validate", "evaluate-test"):
            p.add_argument("--illumination-map", type=Path)
        if name == "benchmark":
            p.add_argument("--exclusive-gpu", action="store_true")
            p.add_argument("--teacher", action="store_true")
    return main


def export_commands(tag, sensitivity=False):
    segment(tag)
    rows = [(name, seed) for seed in (42, 0, 3407) for name in CORE]
    rows.extend((name, 42) for name in SCALES + (SENSITIVITY if sensitivity else ()))
    return "\n".join(f"python -m thesis_kd.cli train --experiment {name} --seed {seed} --run-tag {tag}"
                     for name, seed in rows)


def check_engine_location():
    spec = importlib.util.find_spec("engine")
    if spec is None or Path(spec.origin or "").resolve() != ROOT / "engine/__init__.py":
        raise RuntimeError(f"Expected engine under {ROOT}, found {spec}")


def parse_run(args):
    path = getattr(args, "run_dir", None)
    ckpt = getattr(args, "checkpoint", None)
    if ckpt:
        ckpt = safe_output(ckpt)
        if ckpt.name != "last_training.pth":
            raise ValueError("Full resume requires last_training.pth")
        if path and safe_output(path) != ckpt.parent:
            raise ValueError("Checkpoint and run directory differ")
        path = ckpt.parent
    if path:
        path = safe_output(path)
        seed_name = path.name
        if not seed_name.startswith("seed_"):
            raise ValueError("Expected experiment/seed_N run directory")
        seed = int(seed_name[5:])
        name = path.parent.name
        if (args.seed is not None and seed != args.seed
                or args.experiment and name != args.experiment):
            raise ValueError("Run directory disagrees with experiment or seed")
        args.seed, args.experiment, args.run_tag = seed, name, path.parents[1].name
    return path


def locate(args):
    from .config import resolve_config
    explicit = parse_run(args)
    cfg = resolve_config(args.experiment or "m_base", args.paths)
    if args.seed is None:
        args.seed = 42
    run = run_path(cfg, args.run_tag, args.seed, smoke=args.action == "smoke")
    if explicit and explicit != run:
        raise ValueError(f"Run directory must match the resolved run path {run}")
    return cfg, run


def train(args, cfg, run):
    from .config import check_resources
    from .trainer import run_training
    import torch
    import torch.distributed as dist

    smoke, resume = args.action == "smoke", args.action == "resume"
    steps = getattr(args, "steps", 2)
    if steps < 1:
        raise ValueError("--steps must be positive")
    check_resources(cfg, resume=resume)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device)
    if device.type == "cuda":
        local = int(os.environ["LOCAL_RANK"]) if world > 1 else (device.index or 0)
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    if world > 1:
        backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if world > 1 else 0
    try:
        with contextlib.ExitStack() as stack:
            prepare_error = [None]
            if rank == 0:
                try:
                    if resume:
                        if (run / "frozen_selection.json").exists():
                            raise ValueError("Test selection frozen for this run; resume would change its identity")
                        if not (run / "last_training.pth").is_file():
                            raise FileNotFoundError(run / "last_training.pth")
                    else:
                        run.mkdir(parents=True, exist_ok=False)
                    stack.enter_context(writer_lock(run))
                    if not resume:
                        (run / "config_resolved.yml").write_text(
                            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                except (ValueError, OSError) as exc:
                    prepare_error[0] = str(exc)
            if world > 1:
                dist.broadcast_object_list(prepare_error, src=0)
            if prepare_error[0] is not None:
                raise RuntimeError(prepare_error[0])
            result = run_training(cfg, run, args.seed, str(device), resume=resume,
                                  smoke=smoke, smoke_steps=steps)
            return result if rank == 0 else None
    finally:
        if world > 1:
            dist.destroy_process_group()


def dispatch(args):
    action = args.action
    if action == "check-legacy-integrity":
        report = integrity(args.snapshot, args.source)
        if not report.get("passed", True):
            raise RuntimeError(json.dumps(report, ensure_ascii=False))
        return report
    if action == "export-commands":
        return export_commands(args.run_tag, args.include_sensitivity)

    from .config import check_resources, pair_diff, resolve_config
    if action == "summarize":
        from .summarize import summarize_runs
        cfg = resolve_config("m_base", args.paths)
        tag_dir = run_path(cfg, args.run_tag, 42).parents[1]
        if not tag_dir.is_dir():
            raise FileNotFoundError(tag_dir)
        inspect_tree(tag_dir)
        return summarize_runs(tag_dir, teacher_result=args.teacher_result, split=args.split)
    cfg, run = locate(args)
    if action == "resolve-config":
        return yaml.safe_dump({"resolved": cfg, "base_kd_difference": pair_diff(cfg)},
                              sort_keys=False, allow_unicode=True)
    check_engine_location()
    if action == "preflight":
        result = {"experiment": cfg["experiment"], "protocol": cfg["protocol_id"],
                  "engine": str(ROOT / "engine"), "run": str(run),
                  "base_kd_difference": pair_diff(cfg), "scope": "static"}
        if result["base_kd_difference"]["non_kd_differences"]:
            raise ValueError(f"Unequal Base/KD conditions: {result['base_kd_difference']}")
        if args.with_resources:
            check_resources(cfg)
            import torch
            device = torch.device(args.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            from .factories import build_student
            _, result["student_init"] = build_student(cfg)
            if cfg["kd"]["enabled"] and cfg["kd"]["lambda_kd"] != 0:
                from .teacher import build_teacher
                _, result["teacher"], _ = build_teacher(cfg)
            result["scope"] = "resources and checkpoint loading"
        return result
    if action in ("train", "resume", "smoke"):
        return train(args, cfg, run)
    if action == "benchmark" and args.teacher:
        from .benchmark import benchmark_run
        target = safe_output(run.parents[1] / "T0")
        target.mkdir(parents=True, exist_ok=True)
        with writer_lock(target):
            return benchmark_run(cfg, target, args.device, formal=args.exclusive_gpu, teacher=True)
    if not run.is_dir():
        raise FileNotFoundError(run)
    with writer_lock(run):
        archived = yaml.safe_load((run / "config_resolved.yml").read_text(encoding="utf-8"))
        for key in ("protocol_id", "experiment", "arch"):
            if cfg[key] != archived[key]:
                raise ValueError(f"Selected run disagrees on {key}")
        # Reuse the archived architecture, transforms and postprocessor. Only relocate
        # Train/Val input paths when the caller supplies a resource root on this host.
        if cfg["resources"].get("TOMATO_DATA_ROOT"):
            for split in ("train", "val"):
                ds = split + "_dataloader"
                for field in ("ann_file", "img_folder"):
                    archived["native"][ds]["dataset"][field] = cfg["native"][ds]["dataset"][field]
            archived["resources"]["TOMATO_DATA_ROOT"] = cfg["resources"]["TOMATO_DATA_ROOT"]
        cfg = archived
        if action == "export-student":
            from .checkpoint import export_student
            destination = safe_output(args.destination or run / "student_deploy.pth")
            if destination.exists():
                raise FileExistsError(destination)
            return export_student(run, destination)
        if action == "benchmark":
            from .benchmark import benchmark_run
            return benchmark_run(cfg, run, args.device, formal=args.exclusive_gpu)
        from .evaluate import evaluate_run
        if action == "evaluate-test":
            if not args.confirm_frozen:
                raise ValueError("Frozen selection must be confirmed with --confirm-frozen")
            return evaluate_run(cfg, run, args.device, split="test", confirm_frozen=True,
                                illumination_map=args.illumination_map)
        return evaluate_run(cfg, run, args.device, split="val", illumination_map=args.illumination_map)


def main(argv=None):
    sys.dont_write_bytecode = True
    args = parser().parse_args(argv)
    try:
        value = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if value is not None:
        print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
