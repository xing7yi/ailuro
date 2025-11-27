"""Command-line interface for the response fit toolkit."""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from .data import find_runner_files
from .fitting import CurveConfig, Geometry
from .models import FITTING_MODELS
from .pipeline import process_all_runners


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Response curve fitting tool with multiple models and plotting options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models:\n" + "\n".join(
            f"  {key:12s}: {info['name']}" for key, info in FITTING_MODELS.items()
        ),
    )
    parser.add_argument("--model", type=str, default="bilinear", choices=list(FITTING_MODELS.keys()), help="Model key from the registry")
    parser.add_argument(
        "--curve-type",
        type=str,
        default="nominal",
        choices=["raw", "nominal", "true"],
        help="Type of response curve to fit",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip all plotting steps")
    parser.add_argument("--num-plots", type=int, default=None, help="Limit the number of samples to plot")
    parser.add_argument("--grid-size", type=str, default=None, help="Grid layout, e.g. 5x5")
    parser.add_argument("--save-individual", action="store_true", help="Save individual PDF plots per sample")
    parser.add_argument("--stats-only", action="store_true", help="Only create statistics plots")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistics plots")
    parser.add_argument("--plot-worse", action="store_true", help="Plot samples whose R² falls below the threshold")
    parser.add_argument( "--r2-threshold",type=float,default=0.997,help="R² threshold used together with --plot-worse (default: 0.95)",
    )
    parser.add_argument("--file-base", type=str, default=None, help="Base name of CSV files (auto-detected if omitted)")
    parser.add_argument("--initial-height", type=float, default=1.0, help="Initial half height of specimen in mm")
    parser.add_argument("--contact-radius", type=float, default=1.0, help="Contact radius in mm")
    parser.add_argument("--min-displacement", type=float, default=0.4, help="Minimum displacement required for fitting")
    parser.add_argument("--disp-col", type=str, default="disp", help="Displacement column name")
    parser.add_argument("--force-col", type=str, default="force", help="Force column name")
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    geometry = Geometry(
        initial_height=args.initial_height,
        contact_area=float(np.pi * (args.contact_radius ** 2)),
    )
    config = CurveConfig(
        disp_col=args.disp_col,
        force_col=args.force_col,
        min_displacement=args.min_displacement,
        curve_type=args.curve_type,
        fitting_model=args.model,
    )
    plot_args: Dict[str, object] = {
        "no_plot": args.no_plot,
        "num_plots": args.num_plots,
        "grid_size": args.grid_size,
        "save_individual": args.save_individual,
        "curves_only": args.stats_only,
        "no_stats": args.no_stats,
        "plot_worse": args.plot_worse,
        "r2_threshold": args.r2_threshold,
    }

    print("=" * 70)
    print("Response Curve Fitting Tool")
    print("=" * 70)
    for key, info in FITTING_MODELS.items():
        marker = " <- selected" if key == args.model else ""
        print(f"  {key:12s}: {info['name']}{marker}")
        print(f"               {info['description']}")

    runner_files, detected_base = find_runner_files(args.file_base)
    if not runner_files or not detected_base:
        raise SystemExit(1)

    print(f"\nDetected file base: {detected_base}")
    process_all_runners(runner_files, detected_base, geometry, config, plot_args)
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
