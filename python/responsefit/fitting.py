"""Fitting routines for response curve data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .models import FITTING_MODELS


@dataclass
class Geometry:
    """Geometric parameters used to compute strain and stress."""

    initial_height: float
    contact_area: float


@dataclass
class CurveConfig:
    """Configuration used for data extraction and curve preparation."""

    disp_col: str
    force_col: str
    min_displacement: float
    curve_type: str
    fitting_model: str

def prepare_curve(
    displacement: np.ndarray, force: np.ndarray, geometry: Geometry, curve_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw displacement/force into the selected response curve."""
    epsilon = displacement / geometry.initial_height
    stress = force / geometry.contact_area

    if curve_type == "raw":
        return displacement, force
    if curve_type == "nominal":
        return epsilon, stress
    if curve_type == "true":
        return -np.log(1 - epsilon), (1 - epsilon) * stress
    raise ValueError(f"Unsupported response curve type: {curve_type}")


def fit_time_series(csv_file: str, geometry: Geometry, config: CurveConfig) -> Dict[str, float]:
    """Fit a single runner CSV file and return metrics/parameters."""
    model_key = config.fitting_model
    if model_key not in FITTING_MODELS:
        raise ValueError(
            f"Unknown fitting model: {model_key}. Available: {list(FITTING_MODELS.keys())}"
        )

    model_info = FITTING_MODELS[model_key]
    fitting_func = model_info["function"]
    param_names = model_info["param_safe_names"]
    bounds = model_info["bounds"]

    displacement: List[float] = []
    force: List[float] = []

    try:
        with open(csv_file, "r", encoding="utf-8") as handle:
            reader: Iterable[Dict[str, str]] = csv.DictReader(handle)
            for row in reader:
                try:
                    disp = float(row.get(config.disp_col, 0.0))
                    frc = float(row.get(config.force_col, 0.0))
                except (TypeError, ValueError):
                    continue

                if abs(disp) <= 1e-6 or frc <= 0:
                    continue

                displacement.append(abs(disp))
                force.append(frc)
    except Exception as exc:
        result = {name: np.nan for name in param_names}
        result.update(
            {
                "r_squared": np.nan,
                "rmse": np.nan,
                "rel_rmse": np.nan,
                "n_points": 0,
                "max_force": np.nan,
                "max_displacement": np.nan,
                "fit_success": False,
                "message": f"Error reading file: {exc}",
                "model": model_key,
            }
        )
        return result

    displacement_arr = np.asarray(displacement)
    force_arr = np.asarray(force)

    if displacement_arr.size == 0:
        result = {name: np.nan for name in param_names}
        result.update(
            {
                "r_squared": np.nan,
                "rmse": np.nan,
                "rel_rmse": np.nan,
                "n_points": 0,
                "max_force": np.nan,
                "max_displacement": np.nan,
                "fit_success": False,
                "message": "No valid data rows detected",
                "model": model_key,
            }
        )
        return result

    max_disp = np.max(displacement_arr)
    if max_disp < config.min_displacement:
        result = {name: np.nan for name in param_names}
        result.update(
            {
                "r_squared": np.nan,
                "rmse": np.nan,
                "rel_rmse": np.nan,
                "n_points": displacement_arr.size,
                "max_force": np.max(force_arr),
                "max_displacement": max_disp,
                "fit_success": False,
                "message": (
                    f"Insufficient displacement: max={max_disp:.4f} mm <"
                    f" required {config.min_displacement:.4f} mm"
                ),
                "model": model_key,
            }
        )
        return result

    x_data, y_data = prepare_curve(displacement_arr, force_arr, geometry, config.curve_type)

    try:
        popt, _ = curve_fit(
            fitting_func,
            x_data,
            y_data,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception as exc:
        result = {name: np.nan for name in param_names}
        result.update(
            {
                "r_squared": np.nan,
                "rmse": np.nan,
                "rel_rmse": np.nan,
                "n_points": displacement_arr.size,
                "max_force": np.max(force_arr),
                "max_displacement": max_disp,
                "fit_success": False,
                "message": f"Fitting failed: {exc}",
                "model": model_key,
            }
        )
        return result

    y_pred = fitting_func(x_data, *popt)
    residuals = y_data - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals ** 2))
    rel_rmse = rmse / np.mean(y_data) if np.mean(y_data) > 0 else np.inf

    result = {name: value for name, value in zip(param_names, popt)}
    result.update(
        {
            "r_squared": r_squared,
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "n_points": displacement_arr.size,
            "max_force": float(np.max(force_arr)),
            "max_displacement": float(max_disp),
            "fit_success": True,
            "message": "Success",
            "model": model_key,
        }
    )
    return result


__all__ = ["CurveConfig", "Geometry", "fit_time_series", "prepare_curve"]
