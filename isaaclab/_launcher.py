from __future__ import annotations

import runpy
import sys
from pathlib import Path

import isaaclab  # type: ignore
import triple_pendulum_isaac  # noqa: F401


def _isaaclab_root() -> Path:
    current = Path(isaaclab.__file__).resolve()
    for parent in current.parents:
        if (parent / "scripts").exists() and (parent / "apps").exists():
            return parent
    raise RuntimeError(f"Could not locate Isaac Lab root from {current}")


def run_relative(script_rel_path: str) -> None:
    script_path = _isaaclab_root() / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab script: {script_path}")
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")

