from __future__ import annotations

from typing import Sequence

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def run_pytest(args: Sequence[str] | None = None) -> int:
    """
    Run pytest from a notebook-friendly function.

    Example:
        from scripts.run_pytest import run_pytest
        run_pytest(["-q", "tests/test_decompose_physics.py"])
    """
    root_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(root_dir / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    if importlib.util.find_spec("pytest") is None:
        raise RuntimeError("pytest is not installed. Install it with: python3 -m pip install pytest")

    cmd = [sys.executable, "-m", "pytest"]
    if args is None:
        cmd.append("-q")
    else:
        cmd.extend(list(args))

    proc = subprocess.run(cmd, cwd=str(root_dir), env=env, check=False)
    return int(proc.returncode)


__all__ = ["run_pytest"]
