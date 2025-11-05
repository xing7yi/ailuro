"""Common filesystem locations used by the responsefit toolkit."""

from __future__ import annotations

from pathlib import Path

RUNNER_DIR = Path("results_runner")
STATISTICS_DIR = Path("results_statistics")
FIT_PLOTS_DIR = Path("results_fit_plots")


def ensure_dir(path: Path) -> Path:
    """Create *path* (including parents) if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["RUNNER_DIR", "STATISTICS_DIR", "FIT_PLOTS_DIR", "ensure_dir"]
