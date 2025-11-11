#!/usr/bin/env python3
"""Analyse relationships between material parameters and fitted parameters."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def _detect_material_columns(df: pd.DataFrame) -> List[str]:
	"""Guess the material parameter columns from the CSV header."""
	candidates: List[str] = []
	for col in df.columns:
		if col.startswith("p_") or col.startswith("mat_p_"):
			candidates.append(col)
	return candidates


def _detect_fit_columns(df: pd.DataFrame) -> List[str]:
	"""Guess fitting parameter columns from the CSV header."""
	preferred = ["E", "sigma_0", "R_0", "R_infty", "R_inf"]
	return [col for col in preferred if col in df.columns]


def _ensure_columns_exist(df: pd.DataFrame, cols: Sequence[str], label: str) -> List[str]:
	missing = [col for col in cols if col not in df.columns]
	if missing:
		raise ValueError(
			f"The following {label} columns were not found in the CSV: {', '.join(missing)}"
		)
	return list(cols)


def _prepare_dataframe(csv_path: Path) -> pd.DataFrame:
	try:
		return pd.read_csv(csv_path)
	except Exception as exc:  # pragma: no cover - just in case reading fails
		raise RuntimeError(f"Failed to read CSV file '{csv_path}': {exc}") from exc


def _compute_correlations(
	df: pd.DataFrame,
	material_cols: Sequence[str],
	fit_cols: Sequence[str],
) -> pd.DataFrame:
	"""Return a tidy DataFrame containing pairwise Pearson correlation stats."""
	rows = []
	for mat_col in material_cols:
		for fit_col in fit_cols:
			x = pd.to_numeric(df[mat_col], errors="coerce")
			y = pd.to_numeric(df[fit_col], errors="coerce")
			mask = (~x.isna()) & (~y.isna())
			if mask.sum() < 3:
				continue
			r_val, p_val = pearsonr(x[mask], y[mask])
			rows.append(
				{
					"material_param": mat_col,
					"fit_param": fit_col,
					"pearson_r": r_val,
					"p_value": p_val,
					"samples": int(mask.sum()),
					"abs_pearson_r": abs(r_val),
				}
			)
	result = pd.DataFrame(rows)
	if not result.empty:
		result.sort_values("abs_pearson_r", ascending=False, inplace=True)
	return result


def _compute_multivariate_regressions(
	df: pd.DataFrame,
	material_cols: Sequence[str],
	fit_cols: Sequence[str],
) -> pd.DataFrame:
	"""Fit linear models y = b0 + Σ bi * material_i for each fit column."""

	rows: List[dict] = []
	material_data = df[list(material_cols)].apply(pd.to_numeric, errors="coerce")

	for fit_col in fit_cols:
		y = pd.to_numeric(df[fit_col], errors="coerce")
		combined = pd.concat([material_data, y], axis=1).dropna()

		if combined.shape[0] <= len(material_cols):
			continue

		X = combined[material_cols].to_numpy(dtype=float)
		y_clean = combined[fit_col].to_numpy(dtype=float)

		# prepend bias term for intercept
		X_design = np.column_stack([np.ones(X.shape[0]), X])
		coeffs, *_ = np.linalg.lstsq(X_design, y_clean, rcond=None)
		intercept = coeffs[0]
		weights = coeffs[1:]

		y_pred = X_design @ coeffs
		residuals = y_clean - y_pred
		ss_res = float(np.sum(residuals ** 2))
		ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
		r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
		rmse = float(np.sqrt(np.mean(residuals ** 2)))

		row = {
			"fit_param": fit_col,
			"samples": int(X.shape[0]),
			"intercept": intercept,
			"r_squared": r_squared,
			"rmse": rmse,
		}
		for name, weight in zip(material_cols, weights):
			row[f"coeff_{name}"] = weight
		rows.append(row)

	return pd.DataFrame(rows)


def _format_text_table(df: pd.DataFrame, limit: int = 10) -> str:
	if df.empty:
		return "No valid correlations were computed."
	head = df.head(limit).copy()
	head["pearson_r"] = head["pearson_r"].map(lambda v: f"{v:+.4f}")
	head["p_value"] = head["p_value"].map(lambda v: f"{v:.3e}")
	head["abs_pearson_r"] = head["abs_pearson_r"].map(lambda v: f"{v:.4f}")
	return head.to_string(index=False)


def _format_regression_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "No multivariate regressions could be fitted."

	view = df.copy()
	view["intercept"] = view["intercept"].map(lambda v: f"{v:+.4e}")
	view["r_squared"] = view["r_squared"].map(lambda v: f"{v:.4f}")
	view["rmse"] = view["rmse"].map(lambda v: f"{v:.4e}")

	coeff_cols = [col for col in view.columns if col.startswith("coeff_")]
	for col in coeff_cols:
		view[col] = view[col].map(lambda v: f"{v:+.4e}")

	display_cols = ["fit_param", "samples", "r_squared", "rmse", "intercept", *coeff_cols]
	return view[display_cols].to_string(index=False)


def _make_plot(
	x: pd.Series,
	y: pd.Series,
	mat_label: str,
	fit_label: str,
	r_val: float,
	samples: int,
	output_file: Path,
) -> None:
	"""Create a scatter plot with a least-squares fit line."""
	fig, ax = plt.subplots(figsize=(5, 4))
	ax.scatter(x, y, alpha=0.7, edgecolor="none")

	if samples >= 3:
		coeffs = np.polyfit(x, y, deg=1)
		x_grid = np.linspace(x.min(), x.max(), 200)
		y_fit = np.polyval(coeffs, x_grid)
		ax.plot(x_grid, y_fit, color="tab:red", linewidth=2, label="Linear fit")
		slope, intercept = coeffs
		ax.legend(title=f"y = {slope:.3f}x + {intercept:.3f}")

	ax.set_xlabel(mat_label)
	ax.set_ylabel(fit_label)
	ax.set_title(f"r = {r_val:+.3f} (n = {samples})")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	output_file.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_file, dpi=200)
	plt.close(fig)


def _generate_plots(
	df: pd.DataFrame,
	correlations: pd.DataFrame,
	output_dir: Path,
	top_n: int,
) -> None:
	if correlations.empty:
		return
	plot_dir = output_dir / "correlation_plots"
	plot_dir.mkdir(parents=True, exist_ok=True)

	for _, row in correlations.head(top_n).iterrows():
		mat_col = row["material_param"]
		fit_col = row["fit_param"]
		mask = (~df[mat_col].isna()) & (~df[fit_col].isna())
		if mask.sum() < 3:
			continue
		output_file = plot_dir / f"{mat_col}_vs_{fit_col}.png"
		_make_plot(df[mat_col][mask], df[fit_col][mask], mat_col, fit_col, row["pearson_r"], row["samples"], output_file)

def _plot_regression_heatmap(
	regressions: pd.DataFrame,
	material_cols: Sequence[str],
	output_dir: Path,
) -> None:
	if regressions.empty:
		return

	coeff_cols = [f"coeff_{col}" for col in material_cols if f"coeff_{col}" in regressions.columns]
	if not coeff_cols:
		return

	heatmap_dir = output_dir / "regression_plots"
	heatmap_dir.mkdir(parents=True, exist_ok=True)

	matrix = regressions.set_index("fit_param")[coeff_cols]
	fig, ax = plt.subplots(figsize=(max(6, len(coeff_cols) * 1.5), max(4, len(matrix) * 0.6)))
	sns.heatmap(matrix, annot=True, fmt="+.2e", cmap="coolwarm", center=0.0, ax=ax)
	ax.set_title("Regression coefficients (fit parameter ~ material parameters)")
	ax.set_xlabel("Material parameter coefficients")
	ax.set_ylabel("Fit parameter")
	fig.tight_layout()
	heatmap_path = heatmap_dir / "regression_coefficients.pdf"
	fig.savefig(heatmap_path)
	plt.close(fig)


def _plot_regression_scatter(
	df: pd.DataFrame,
	regressions: pd.DataFrame,
	material_cols: Sequence[str],
	output_dir: Path,
	top_n: int,
) -> None:
	if regressions.empty:
		return

	scatter_dir = output_dir / "regression_plots"
	scatter_dir.mkdir(parents=True, exist_ok=True)

	ranked = regressions.copy()
	ranked.sort_values("r_squared", ascending=False, inplace=True)

	for _, row in ranked.head(top_n).iterrows():
		fit_param = row["fit_param"]
		coeffs = [row.get(f"coeff_{col}", 0.0) for col in material_cols]
		intercept = row["intercept"]

		X = df[list(material_cols)].apply(pd.to_numeric, errors="coerce")
		y = pd.to_numeric(df[fit_param], errors="coerce")
		combined = pd.concat([X, y], axis=1).dropna()
		if combined.empty:
			continue

		X_clean = combined[material_cols].to_numpy(dtype=float)
		y_clean = combined[fit_param].to_numpy(dtype=float)
		y_pred = intercept + np.dot(X_clean, np.array(coeffs))

		fig, ax = plt.subplots(figsize=(5, 4))
		ax.scatter(y_clean, y_pred, alpha=0.7, edgecolor="none")
		lims = [min(y_clean.min(), y_pred.min()), max(y_clean.max(), y_pred.max())]
		ax.plot(lims, lims, "k--", linewidth=1.0, label="Ideal")
		ax.set_xlabel("Observed")
		ax.set_ylabel("Predicted")
		ax.set_title(f"{fit_param} fit (R²={row['r_squared']:.3f}, n={row['samples']})")
		ax.legend()
		ax.grid(alpha=0.3)
		fig.tight_layout()
		scatter_file = scatter_dir / f"pred_vs_obs_{fit_param}.pdf"
		fig.savefig(scatter_file)
		plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyse correlations between material parameters (p_*) and fit parameters",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=textwrap.dedent(
			"""
			Examples
			--------
			python analyse_params.py --csv results_statistics/main_lh_sampler_fit_params_voce.csv
			python analyse_params.py --csv data.csv --material-cols p_0 p_1 p_2 --fit-cols E sigma_0
			python analyse_params.py --csv data.csv --output analysis --top-n 12
			"""
		),
	)
	parser.add_argument("--csv", required=True, help="Path to the fit-parameter CSV file")
	parser.add_argument(
		"--material-cols",
		nargs="*",
		help="Explicit list of material parameter columns; defaults to columns starting with p_ or mat_p_",
	)
	parser.add_argument(
		"--fit-cols",
		nargs="*",
		help="Explicit list of fitted parameter columns; defaults to E, sigma_0, R_0, R_infty (if present)",
	)
	parser.add_argument(
		"--output",
		default="results_analysis",
		help="Directory where correlation tables and plots are written (created if missing)",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=9,
		help="Number of strongest correlations for which scatter plots are generated",
	)
	parser.add_argument(
		"--no-plots",
		action="store_true",
		help="Only compute statistics; do not generate scatter plots",
	)
	parser.add_argument(
		"--regression-plots",
		action="store_true",
		default=True,
		help="Generate additional visualizations for multivariate regression models",
	)
	parser.add_argument(
		"--export-csv",
		action="store_true",
		help="Export the correlation table as CSV inside the output directory",
	)
	return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
	args = parse_args(argv)

	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise SystemExit(f"CSV file not found: {csv_path}")

	df = _prepare_dataframe(csv_path)

	if args.material_cols:
		material_cols = _ensure_columns_exist(df, args.material_cols, "material")
	else:
		material_cols = _detect_material_columns(df)
		if not material_cols:
			raise SystemExit("Unable to auto-detect material parameter columns. Please pass --material-cols.")

	if args.fit_cols:
		fit_cols = _ensure_columns_exist(df, args.fit_cols, "fit")
	else:
		fit_cols = _detect_fit_columns(df)
		if not fit_cols:
			raise SystemExit("Unable to auto-detect fit parameter columns. Please pass --fit-cols.")

	print("Detected material parameters:", ", ".join(material_cols))
	print("Detected fit parameters:", ", ".join(fit_cols))

	correlations = _compute_correlations(df, material_cols, fit_cols)
	print("\nTop correlations (sorted by |r|):")
	print(_format_text_table(correlations))

	regressions = _compute_multivariate_regressions(df, material_cols, fit_cols)
	print("\nMultivariate linear models (fit parameter ~ material parameters):")
	print(_format_regression_table(regressions))

	output_dir = Path(args.output)
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.export_csv and not correlations.empty:
		csv_output = output_dir / "material_fit_correlations.csv"
		correlations.drop(columns=["abs_pearson_r"]).to_csv(csv_output, index=False)
		print(f"\nCorrelation table exported to {csv_output}")

	if args.export_csv and not regressions.empty:
		regression_csv = output_dir / "material_to_fit_regressions.csv"
		regressions.to_csv(regression_csv, index=False)
		print(f"Regression summary exported to {regression_csv}")

	if not args.no_plots:
		_generate_plots(df, correlations, output_dir, max(args.top_n, 1))
		if not correlations.empty:
			print(f"Scatter plots saved under {output_dir / 'correlation_plots'}")

	if args.regression_plots:
		_plot_regression_heatmap(regressions, material_cols, output_dir)
		_plot_regression_scatter(df, regressions, material_cols, output_dir, max(3, args.top_n // 2))
		if not regressions.empty:
			print(f"Regression plots saved under {output_dir / 'regression_plots'}")


if __name__ == "__main__":
	main()


