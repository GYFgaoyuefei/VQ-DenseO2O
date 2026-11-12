"""Epoch-boundary snapshots, native deployment weights, recoverable best selection."""
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    staging = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, staging)
    os.replace(staging, path)


def atomic_text(path, content):
    path = Path(path)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def training_state(wrapper, ema, optimizer, scaler, *, epoch, updates, signature, rank_states,
                   history, diagnostics, selection, cfg):
    return dict(
        epoch=epoch, global_step=epoch * signature["steps_per_epoch"], optimizer_updates=updates,
        kd_progress=updates, signature=signature,
        protocol_id=cfg["protocol_id"], teacher=signature["teacher"], config=cfg,
        student=wrapper.detector.state_dict(), projectors=wrapper.projectors.state_dict(),
        ema=ema.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        scheduler={"updates": updates, "steps_per_epoch": signature["steps_per_epoch"]},
        rank_states=rank_states, history=history, diagnostics=diagnostics, selection=selection,
    )


def comparable_teacher(identity):
    if isinstance(identity, dict) and "checkpoint_sha256" in identity:
        return tuple(identity[k] for k in ("checkpoint_sha256", "state_path", "config_digest"))
    return identity


def restore_training(state, signature, wrapper, ema, optimizer, scaler):
    differing = [key for key in signature.keys() | state["signature"].keys()
                 if key != "teacher" and state["signature"].get(key) != signature.get(key)]
    if comparable_teacher(state["signature"]["teacher"]) != comparable_teacher(signature["teacher"]):
        differing.append("teacher")
    if differing:
        raise ValueError(f"Incompatible training resume: {', '.join(sorted(differing))}")
    updates = state["optimizer_updates"]
    if state["global_step"] != state["epoch"] * signature["steps_per_epoch"] or updates > state["global_step"]:
        raise ValueError("Inconsistent epoch / batch / update counters")
    expected = {"updates": updates, "steps_per_epoch": signature["steps_per_epoch"]}
    if state["kd_progress"] != updates or state["scheduler"] != expected:
        raise ValueError("Inconsistent KD / scheduler update counter")
    wrapper.detector.load_state_dict(state["student"], strict=True)
    wrapper.projectors.load_state_dict(state["projectors"], strict=True)
    ema.load_state_dict(state["ema"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])


def publish_epoch(run_dir, selection, history, diagnostics, recover=False):
    # last_training is committed first. A crash between this commit and rename is recoverable.
    pending = run_dir / "best_pending.pth"
    final = run_dir / "best_student.pth"
    if pending.is_file():
        if digest(pending) == selection["sha256"]:
            os.replace(pending, final)
        else:
            pending.unlink()  # crash before checkpoint commit
    if not final.is_file() or (recover and digest(final) != selection["sha256"]):
        raise ValueError("Selected deployment checkpoint missing or altered")
    write_json(run_dir / "selection.json", selection)
    write_json(run_dir / "kd_diagnostics.json", diagnostics)
    rows = io.StringIO(newline="")
    writer = csv.DictWriter(rows, fieldnames=history[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    atomic_text(run_dir / "validation_history.csv", rows.getvalue())
    atomic_text(run_dir / "train.log", "".join(json.dumps(row) + "\n" for row in diagnostics))


def export_student(run_dir, destination):
    run_dir, destination = Path(run_dir), Path(destination)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    source = run_dir / "best_student.pth"
    if selection["checkpoint"] != source.name or digest(source) != selection["sha256"]:
        raise ValueError("Selection digest does not match best_student.pth")
    # Only export to a new file. A failed copy does not replace an existing artifact.
    with source.open("rb") as src, destination.open("xb") as dst:
        try:
            shutil.copyfileobj(src, dst)
        except BaseException:
            dst.close()
            destination.unlink(missing_ok=True)
            raise
    return {"path": str(destination), "sha256": selection["sha256"]}

def pin_teacher(tag_dir, identity):
    """Publish one complete T0 manifest per run tag, including simultaneous launches."""
    path = Path(tag_dir) / "teacher_pin.json"
    with tempfile.NamedTemporaryFile(dir=tag_dir, suffix=".pin", delete=False) as temp:
        staging = Path(temp.name)
        temp.write(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    try:
        try:
            os.link(staging, path)
        except FileExistsError:
            if comparable_teacher(json.loads(path.read_text(encoding="utf-8"))) != comparable_teacher(identity):
                raise ValueError(f"Teacher differs from fixed T0 for this run tag: {path}")
    finally:
        staging.unlink(missing_ok=True)
