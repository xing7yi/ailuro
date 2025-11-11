"""Data loading helpers for response curve fitting."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .paths import RUNNER_DIR, STATISTICS_DIR

@dataclass
class SampleData:
    rows: List[Dict[str, float]]
    keys: List[str]


def extract_runner_number(filename: str) -> int:
    """Extract the runner index from a file name."""
    match = re.search(r"runner(\d+)", filename)
    return int(match.group(1)) if match else -1


def find_runner_files(file_base: Optional[str] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Return a sorted list of runner CSV files and the detected base name."""
    base_dir = RUNNER_DIR
    if not base_dir.exists():
        print(f"Error: directory '{base_dir}' does not exist")
        return None, None

    if file_base:
        pattern = f"{file_base}_out_runner*.csv"
        candidates: Sequence[Path] = sorted(base_dir.glob(pattern), key=lambda p: extract_runner_number(str(p)))
    else:
        candidates = sorted(base_dir.glob("*_out_runner*.csv"), key=lambda p: extract_runner_number(str(p)))
        if not candidates:
            print("Error: no *_out_runner*.csv files found")
            return None, None
        first_file = candidates[0]
        match = re.match(r"(.+?)_out_runner\d+\.csv", first_file.name)
        if not match:
            print(f"Error: unable to infer base name from {first_file.name}")
            return None, None
        file_base = match.group(1)

    if not candidates:
        print(f"Error: no files matching pattern {file_base}_out_runner*.csv in {base_dir}")
        return None, None

    files = [str(path) for path in candidates]
    return files, file_base


def read_runner_data_for_plot(csv_file: str, disp_col: str, force_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load displacement and force columns for plotting."""
    displacement: List[float] = []
    force: List[float] = []

    with open(csv_file, "r", encoding="utf-8") as handle:
        reader: Iterable[Dict[str, str]] = csv.DictReader(handle)
        for row in reader:
            try:
                disp = float(row.get(disp_col, 0.0))
                frc = float(row.get(force_col, 0.0))
            except (TypeError, ValueError):
                continue
            if abs(disp) <= 1e-6 or frc <= 0:
                continue
            displacement.append(abs(disp))
            force.append(frc)

    return np.asarray(displacement), np.asarray(force)


def load_sample_parameters(file_base: str) -> Optional[SampleData]:
    """Load material parameter samples associated with *file_base*."""

    sample_file = STATISTICS_DIR / f"{file_base}_samples_sample_data_0000.csv"
    if not sample_file.exists():
        return None

    with open(sample_file, "r", encoding="utf-8") as handle:
        reader: Iterable[Dict[str, str]] = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None

        num_sample_params = len(reader.fieldnames)
        material_keys: List[str] = [f"p_{i}" for i in range(num_sample_params)]

        rows: List[Dict[str, float]] = []
        for row in reader:
            values: Dict[str, float] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue

                # Assign new material key based on order
                new_key = material_keys[len(values)]
                if new_key not in values:
                    values[new_key] = numeric
            rows.append(values)

    if not rows:
        return None

    return SampleData(rows=rows, keys=material_keys)


__all__ = [
    "extract_runner_number",
    "find_runner_files",
    "load_sample_parameters",
    "read_runner_data_for_plot",
    "SampleData",
    "SAMPLE_COLUMN_RENAMES",
]
