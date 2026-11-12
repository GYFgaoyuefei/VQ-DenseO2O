"""Contain all run writes in the isolated checkout; audit the legacy sources."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("thesis_kd/", "configs/tomato/thesis_kd/", "tests/thesis_kd/",
           "scripts/thesis_kd/", "docs/thesis_kd/")
OUTPUTS = ("thesis_kd", "thesis_kd_smoke", "thesis_kd_checks")


def segment(value):
    value = str(value)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value)
            or value.upper() in {"AUX", "CON", "NUL", "PRN"}
            or re.fullmatch(r"(?:LPT|COM)[0-9]", value, re.I)):
        raise ValueError(f"Invalid run path component {value!r}")
    return value


def reject_link(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (path.is_symlink() or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or stat.S_ISREG(info.st_mode) and info.st_nlink > 1):
        raise ValueError(f"Unsafe linked output path: {path}")


def safe_output(path, kind="thesis_kd", root=None):
    if kind not in OUTPUTS:
        raise ValueError(kind)
    base = Path(root or ROOT).resolve()
    allowed = base / "outputs" / kind
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden in an output path")
    path = base / path
    if not path.is_relative_to(allowed):
        raise ValueError(f"Output must be inside {allowed}, got {path}")
    current = base
    for part in path.relative_to(base).parts:
        current = current / part
        reject_link(current)
    if not path.resolve().is_relative_to(allowed):
        raise ValueError(f"Output escaped {allowed}")
    return path


def inspect_tree(path):
    reject_link(path)
    if path.is_dir():
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                reject_link(Path(directory) / name)


def run_path(cfg, tag, seed, *, smoke=False):
    kind = "thesis_kd_smoke" if smoke else "thesis_kd"
    location = ROOT / "outputs" / kind
    custom = cfg["resources"].get("THESIS_KD_OUTPUT_ROOT")
    if custom and not smoke:
        location = Path(custom)
    return safe_output(location / segment(cfg["protocol_id"]) / segment(tag)
                       / segment(cfg["experiment"]) / f"seed_{int(seed)}", kind)


@contextlib.contextmanager
def writer_lock(run):
    # Advisory lock is released by the OS if the training process dies.
    inspect_tree(run)
    with (run / ".writer.lock").open("a+b") as fd:
        if os.name == "nt":
            import msvcrt
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)


def git(root, *args):
    return subprocess.check_output(["git", "-c", "core.quotePath=false", *args],
                                   cwd=root).decode("utf-8").rstrip("\r\n")


def tracked_hashes(root):
    paths = git(root, "ls-files", "-z").split("\0")
    def protected(name):
        return not name.startswith(ALLOWED) and (
            name.startswith(("engine/", "tools/", "configs/", "scripts/", "tests/"))
            or name == "train.py" or Path(name).name.startswith(
                ("requirements", "environment", "poetry.lock", "conda-lock", "uv.lock", "Pipfile")))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths if protected(name) and (root / name).is_file()}


def integrity(create=False, source=None):
    directory = safe_output(ROOT / "outputs/thesis_kd_checks", "thesis_kd_checks")
    directory.mkdir(parents=True, exist_ok=True)
    baseline = safe_output(directory / "baseline.json", "thesis_kd_checks")
    status = lambda root: git(root, "status", "--short", "-uall")
    if create:
        src = Path(source or ROOT).resolve()
        data = dict(source=str(src), development=str(ROOT), head=git(src, "rev-parse", "HEAD"),
                    source_status=status(src), development_status=status(ROOT), sha256=tracked_hashes(src),
                    development_sha256=tracked_hashes(ROOT))
        with baseline.open("x", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        return {"baseline": str(baseline), "protected_files": len(data["sha256"])}
    data = json.loads(baseline.read_text(encoding="utf-8"))
    expected = data["sha256"]
    def changed(after, reference):
        return sorted(key for key in reference.keys() | after.keys() if reference.get(key) != after.get(key))
    src = Path(data["source"])
    checkout_reference = directory / "checkout_reference.json"
    dev_expected = (json.loads(checkout_reference.read_text(encoding="utf-8"))["sha256"]
                    if checkout_reference.is_file() else data.get("development_sha256", expected))
    source_delta = sorted(set(data["source_status"].splitlines()) ^ set(status(src).splitlines()))
    outside = [line for line in status(ROOT).splitlines() if not line[3:].startswith(ALLOWED)]
    prior = [line for line in data["development_status"].splitlines() if not line[3:].startswith(ALLOWED)]
    unexpected = sorted(set(prior).symmetric_difference(outside))
    unexpected += [str(p.relative_to(ROOT)) for p in directory.parent.iterdir() if p.name not in OUTPUTS]
    final_hashes = {"source": tracked_hashes(src), "development": tracked_hashes(ROOT)}
    safe_output(directory / "protected_after.json", "thesis_kd_checks").write_text(
        json.dumps(final_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"head": data["head"], "source": str(src), "development": str(ROOT),
              "protected_files": len(expected), "changed_source": changed(final_hashes["source"], expected),
              "changed_development": changed(final_hashes["development"], dev_expected), "unexpected_outside_allowed": unexpected,
              "source_status_preserved": not source_delta,
              "source_status_before": data["source_status"], "source_status_after": status(src),
              "source_status_difference": source_delta,
              "baseline_head_preserved": git(src, "rev-parse", "HEAD") == data["head"]}
    report["protected_integrity_passed"] = not (report["changed_source"] or report["changed_development"])
    report["passed"] = not (report["changed_source"] or report["changed_development"] or unexpected
                            or not report["source_status_preserved"] or not report["baseline_head_preserved"])
    target = safe_output(directory / "legacy_integrity_report.json", "thesis_kd_checks")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
