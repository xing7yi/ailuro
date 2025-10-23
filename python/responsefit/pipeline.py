"""High-level orchestration logic for response fitting workflows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import SampleData, extract_runner_number, find_runner_files, load_sample_parameters
from .fitting import CurveConfig, Geometry, fit_time_series
from .models import FITTING_MODELS
from .paths import STATISTICS_DIR, ensure_dir
from .plotting import plot_fitting_curves, plot_statistics


def process_all_runners(
    runner_files: Sequence[str],
    file_base: str,
    geometry: Geometry,
    config: CurveConfig,
    plot_args: Optional[Dict[str, object]] = None,
) -> None:
    """Fit all runner files, write the summary CSV, and trigger plotting."""
    model_name = config.fitting_model
    model_info = FITTING_MODELS[model_name]
    statistics_dir = ensure_dir(STATISTICS_DIR)
    output_csv = statistics_dir / f"{file_base}_fit_params_{model_name}.csv"

    sample_data: Optional[SampleData] = load_sample_parameters(file_base)
    material_keys = sample_data.keys if sample_data else []
    if sample_data and len(sample_data.rows) != len(runner_files):
        print(
            f"Warning: sample data count ({len(sample_data.rows)}) does not match runner files ({len(runner_files)})."
        )

    print("=" * 70)
    print(f"Fitting model: {model_info['name']}")
    print(f"Formula: {model_info['description']}")
    print("=" * 70)
    print(f"Found {len(runner_files)} runner files")
    print("Starting fit...")

    results: List[Dict[str, float]] = []
    success_count = 0

    for idx, csv_file in enumerate(runner_files):
        runner_num = extract_runner_number(csv_file)
        fit_result = fit_time_series(csv_file, geometry, config)
        result_row: Dict[str, float] = {
            "runner": runner_num,
            "filename": Path(csv_file).name,
        }

        if sample_data and idx < len(sample_data.rows):
            sample_row = sample_data.rows[idx]
            for key in material_keys:
                result_row[key] = sample_row.get(key, float("nan"))

        result_row.update(fit_result)
        results.append(result_row)
        if fit_result.get("fit_success"):
            success_count += 1
        if (idx + 1) % 10 == 0 or (idx + 1) == len(runner_files):
            print(f"  Processed {idx + 1}/{len(runner_files)} files (success: {success_count})")

    if not results:
        print("Warning: no results generated")
        return

    fieldnames = list(results[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 70)
    print("Fitting complete!")
    print("=" * 70)
    print(f"Total samples: {len(results)}")
    print(f"Successful fits: {success_count} ({success_count / len(results) * 100:.1f}%)")
    print(f"Failed fits: {len(results) - success_count}")
    print(f"Results saved to: {output_csv}")

    if success_count:
        successful = [r for r in results if r.get("fit_success")]
        param_names = model_info["param_safe_names"]
        r2_values = [r["r_squared"] for r in successful]

        print("\nParameter statistics:")
        for pname in param_names:
            values = np.array([r[pname] for r in successful])
            print(f"  {pname}: range=[{values.min():.4e}, {values.max():.4e}], mean={values.mean():.4e}, std={values.std():.4e}")

        print("\nFit quality (R²):")
        print(f"  range=[{np.min(r2_values):.4f}, {np.max(r2_values):.4f}], mean={np.mean(r2_values):.4f}")
        best_idx = int(np.argmax(r2_values))
        worst_idx = int(np.argmin(r2_values))
        print(f"  best runner: runner{successful[best_idx]['runner']} (R²={successful[best_idx]['r_squared']:.4f})")
        print(f"  worst runner: runner{successful[worst_idx]['runner']} (R²={successful[worst_idx]['r_squared']:.4f})")

    failed = [r for r in results if not r.get("fit_success")]
    if failed:
        print("\nFailed samples:")
        for row in failed[:10]:
            print(f"  runner{row['runner']}: {row['message']}")
        if len(failed) > 10:
            print(f"  ... {len(failed) - 10} more failures omitted")

    if plot_args and not plot_args.get("no_plot", False):
        if not plot_args.get("no_stats", False):
            plot_statistics(results, model_info, statistics_dir)
        if not plot_args.get("curves_only", False):
            plot_fitting_curves(
                runner_files,
                results,
                geometry,
                config,
                model_info,
                material_keys=material_keys,
                num_plots=plot_args.get("num_plots"),
                save_individual=plot_args.get("save_individual", False),
                grid_size=plot_args.get("grid_size"),
                plot_worse=plot_args.get("plot_worse", False),
                r2_threshold=plot_args.get("r2_threshold"),
            )


__all__ = ["process_all_runners"]
