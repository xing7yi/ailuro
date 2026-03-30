"""Plotting helpers for response curve fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import extract_runner_number, read_runner_data_for_plot
from .fitting import CurveConfig, Geometry, prepare_curve
from .paths import FIT_PLOTS_DIR


def plot_single_fit(
    runner_num: int,
    displacement: np.ndarray,
    force: np.ndarray,
    mat_and_fit_params: Dict[str, float],
    geometry: Geometry,
    config: CurveConfig,
    model_info: Dict[str, object],
    material_keys: Sequence[str] = (),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the original curve and the fitted result for one runner."""
    x_data, y_data = prepare_curve(displacement, force, geometry, config.curve_type)

    x_fit = np.linspace(x_data.min(), x_data.max(), 200)
    fitting_func = model_info["function"]
    safe_param_names = model_info["param_safe_names"]
    params = [mat_and_fit_params[pname] for pname in safe_param_names]
    y_fit = fitting_func(x_fit, *params)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True

    ax.scatter(x_data, y_data, s=20, alpha=0.6, label="Measured", color="tab:blue")
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label=f"{model_info['name']} Fit")

    axis_label_map = {
        "raw": ("Displacement", "Force"),
        "nominal": ("Strain", "Stress"),
        "true": ("True Strain", "True Stress"),
    }
    xlabel, ylabel = axis_label_map.get(config.curve_type, ("Strain", "Stress"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    info_text_lines = [f"Runner {runner_num}"]

    material_lines = [
        f"{key}={mat_and_fit_params[key]:.1f}"
        for key in material_keys
        if key in mat_and_fit_params
    ]
    if material_lines:
        info_text_lines.append(", ".join(material_lines))

    display_names = model_info["param_names"]
    fit_param_lines = [
        f"{display}={mat_and_fit_params[safe]:.3f}"
        for display, safe in zip(display_names, safe_param_names)
    ]

    if len(fit_param_lines) > 3 :
        info_text_lines.extend(
            [
                ", ".join(fit_param_lines[i : i + 2]) for i in range(0, len(fit_param_lines), 2)
            ]
        )
    else:
        info_text_lines.append(", ".join(fit_param_lines))

    info_text_lines.append(f"R²={mat_and_fit_params['r_squared']:.5f}")

    info_text = "\n".join(info_text_lines)
    ax.set_title(info_text, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if standalone:
        ax.figure.tight_layout()
        return ax
    return ax


def plot_statistics(results: Sequence[Dict[str, float]], model_info: Dict[str, object], output_dir: Path) -> None:
    """Plot histograms for fitted parameters and the R² distribution."""
    successful = [r for r in results if r.get("fit_success")]
    if not successful:
        print("No successful fits to visualize")
        return

    param_names = model_info["param_safe_names"]
    n_params = len(param_names)

    ncols = 2
    nrows = (n_params + 2) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.asarray(axes).flatten()

    for idx, pname in enumerate(param_names):
        values = [r[pname] for r in successful]
        axes[idx].hist(values, bins=30, alpha=0.7, edgecolor="black")
        axes[idx].set_xlabel(f"Parameter {pname}")
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{pname} Distribution\nMean={np.mean(values):.2e}, Std={np.std(values):.2e}")
        axes[idx].grid(True, alpha=0.3)

    r2_values = [r["r_squared"] for r in successful]
    axes[n_params].hist(r2_values, bins=30, alpha=0.7, color="purple", edgecolor="black")
    axes[n_params].set_xlabel("R² (Goodness of Fit)")
    axes[n_params].set_ylabel("Frequency")
    axes[n_params].set_title(
        f"Fit Quality\nMean={np.mean(r2_values):.4f}, Min={np.min(r2_values):.4f}"
    )
    axes[n_params].grid(True, alpha=0.3)

    for extra in range(n_params + 1, len(axes)):
        axes[extra].axis("off")

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "parameters_distribution.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved statistics plot to {output_file}")
    plt.close(fig)


def plot_fitting_curves(
    runner_files: Sequence[str],
    results: Sequence[Dict[str, float]],
    geometry: Geometry,
    config: CurveConfig,
    model_info: Dict[str, object],
    num_plots: Optional[int] = None,
    save_individual: bool = False,
    grid_size: Optional[str] = None,
    material_keys: Sequence[str] = (),
    plot_worse: bool = False,
    r2_threshold: Optional[float] = None,
) -> None:
    """Generate plots for fitted curves."""
    output_dir = FIT_PLOTS_DIR
    output_dir.mkdir(exist_ok=True)

    result_lookup = {r["runner"]: r for r in results if r.get("fit_success")}
    file_lookup = {extract_runner_number(path): path for path in runner_files}

    threshold = r2_threshold if r2_threshold is not None else 0.95

    total_samples = len(runner_files)
    num_to_plot = min(num_plots, total_samples) if num_plots else min(250, total_samples)
    print(f"Total samples: {total_samples}")
    print(f"Plotting {num_to_plot} samples")

    if save_individual:
        saved = 0
        for csv_file in runner_files[:num_to_plot]:
            runner_num = extract_runner_number(csv_file)
            runner_result = result_lookup.get(runner_num)
            if runner_result is None:
                continue

            displacement, force = read_runner_data_for_plot(
                csv_file, config.disp_col, config.force_col
            )
            if displacement.size == 0:
                continue

            ax = plot_single_fit(
                runner_num,
                displacement,
                force,
                runner_result,
                geometry,
                config,
                model_info,
                material_keys=material_keys,
            )
            output_file = output_dir / f"runner{runner_num:03d}_{config.fitting_model}_fit.pdf"
            ax.figure.savefig(output_file, dpi=100, bbox_inches="tight")
            plt.close(ax.figure)
            saved += 1
        print(f"Saved {saved} individual figures to {output_dir}")

    if grid_size:
        nrows, ncols = map(int, grid_size.split("x"))
    else:
        nrows, ncols = 5, 5

    plots_per_grid = nrows * ncols
    num_grids = (num_to_plot + plots_per_grid - 1) // plots_per_grid

    def render_grids(
        csv_paths: Sequence[str], suffix: str, label: str, show_threshold: bool = False
    ) -> None:
        total = len(csv_paths)
        if total == 0:
            return

        num_grids_local = (total + plots_per_grid - 1) // plots_per_grid
        if show_threshold:
            print(
                f"{label}: plotting {total} samples across {num_grids_local} page(s) (threshold={threshold:.3f})"
            )
        else:
            print(f"{label}: plotting {total} samples across {num_grids_local} page(s)")

        for grid_idx in range(num_grids_local):
            start_idx = grid_idx * plots_per_grid
            end_idx = min(start_idx + plots_per_grid, total)

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3))
            axes = np.atleast_1d(axes).ravel()

            plot_count = 0
            for csv_file in csv_paths[start_idx:end_idx]:
                runner_num = extract_runner_number(csv_file)
                runner_result = result_lookup.get(runner_num)
                if runner_result is None:
                    continue

                displacement, force = read_runner_data_for_plot(
                    csv_file, config.disp_col, config.force_col
                )
                if displacement.size == 0:
                    continue

                if plot_count >= len(axes):
                    break

                plot_single_fit(
                    runner_num,
                    displacement,
                    force,
                    runner_result,
                    geometry,
                    config,
                    model_info,
                    material_keys=material_keys,
                    ax=axes[plot_count],
                )
                plot_count += 1

            for idx in range(plot_count, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            output_file = output_dir / f"{config.fitting_model}_{suffix}_grid_{grid_idx + 1:02d}.pdf"
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
            print(f"Saved {suffix} grid {grid_idx + 1}/{num_grids_local} to {output_file}")
            plt.close(fig)


    if plot_worse:
        worse_runners = [
            (r["runner"], r["r_squared"])
            for r in results
            if r.get("fit_success") and np.isfinite(r.get("r_squared", np.nan)) and r["r_squared"] <= threshold
        ]
        worse_runners.sort(key=lambda item: item[1])
        worse_files = [file_lookup.get(runner) for runner, _ in worse_runners]
        worse_files = [path for path in worse_files if path is not None]
        render_grids(worse_files, "fit_worse", "Low R² grid plots", show_threshold=True)
    else:
        render_grids(runner_files[:num_to_plot], "fit", "Standard grid plots")


__all__ = ["plot_fitting_curves", "plot_single_fit", "plot_statistics"]
