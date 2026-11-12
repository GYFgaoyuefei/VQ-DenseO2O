import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from thesis_kd.cli import export_commands
from thesis_kd.safeguards import ROOT, inspect_tree, safe_output, segment


def test_static_commands_and_schedule():
    rows = export_commands("seed_study").splitlines()
    assert len(rows) == len(set(rows)) == 15
    assert len(export_commands("study", True).splitlines()) == 18
    assert sum("--experiment m_kd --" in line for line in rows) == 3
    code = """from thesis_kd.cli import main
import sys
assert main(['preflight', '--experiment', 'm_base', '--static']) == 0
assert 'torch' not in sys.modules and 'engine' not in sys.modules"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("name", ["../x", "x/y", "C:run", "CON", "a.", "COM1", " "])
def test_unsafe_tag(name):
    with pytest.raises(ValueError):
        segment(name)


def test_output_guard(tmp_path):
    out = tmp_path / "outputs/thesis_kd"
    out.mkdir(parents=True)
    assert safe_output(out / "new", root=tmp_path) == out / "new"
    with pytest.raises(ValueError):
        safe_output(out / "../historical/seed_42", root=tmp_path)
    with pytest.raises(ValueError):
        safe_output(tmp_path / "outputs/paper_v2", root=tmp_path)
    legacy = tmp_path / "old.pth"
    legacy.write_bytes(b"old")
    os.link(legacy, out / "best_student.pth")
    with pytest.raises(ValueError):
        inspect_tree(out)
    assert legacy.read_bytes() == b"old"


def test_output_link_escape(tmp_path):
    out = tmp_path / "outputs/thesis_kd"
    out.mkdir(parents=True)
    history = tmp_path / "history"
    history.mkdir()
    if os.name == "nt":
        # Create the junction inside the throwaway test tree.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(out / "alias"), str(history)], check=True)
    else:
        (out / "alias").symlink_to(history, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_output(out / "alias/file", root=tmp_path)


def test_legacy_import_isolation():
    code = """import runpy, sys, pathlib
runpy.run_module('train', run_name='legacy_entry_import_test')
import engine
assert pathlib.Path(engine.__file__).resolve().parent == pathlib.Path('engine').resolve()
assert not any(name == 'thesis_kd' or name.startswith('thesis_kd.') for name in sys.modules)"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)
