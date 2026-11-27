"""Statistical analysis of RQ1 experiment results.

Loads the CSV files produced by run_rq1.py, validates experimental coverage,
calculates mean and 95% Student-t confidence intervals, and evaluates paired
differences between each ML selector and its historical selector baseline.

Usage:
    python -m experiments.analyze_rq1
    python -m experiments.analyze_rq1 --models linear gbm
    python -m experiments.analyze_rq1 --results-dir results
"""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import pandas as pd
from experiments.common import EXPOSURES, HISTORICAL, REGIMES, mean_ci
from experiments.run_rq1 import (
    BASELINE,
    METRICS,
    MODEL_KINDS,
    selector_name,
)

KEY_COLUMNS = (
    "exposure_weight",
    "historical_selector",
    "seed",
    "selector",
)

SUMMARY_COLUMNS = (
    "model_kind",
    "exposure_weight",
    "historical_selector",
    "selector",
    "metric",
    "n",
    "mean",
    "ci95_half_width",
)

PAIRED_COLUMNS = (
    "model_kind",
    "exposure_weight",
    "historical_selector",
    "selector",
    "comparison",
    "metric",
    "n_pairs",
    "mean_difference",
    "ci95_half_width",
)

ML_SELECTORS = tuple(
    selector_name(regime, use_birth_info)
    for regime, use_birth_info in product(REGIMES, (False, True))
)

EXPECTED_SELECTORS = {"oracle", BASELINE, *ML_SELECTORS}


def validate_runs(
    runs: pd.DataFrame,
    model_kind: str,
) -> None:
    """Validate the completeness and consistency of one model's experiment data."""
    if not isinstance(runs, pd.DataFrame):
        raise TypeError("runs must be a pandas DataFrame.")

    if runs.empty:
        raise ValueError("experiment results must not be empty.")

    required_columns = set(KEY_COLUMNS) | {
        "model_kind",
        "evaluation_stage",
        *METRICS,
    }

    missing = required_columns - set(runs.columns)

    if missing:
        raise ValueError(f"experiment results are missing columns: {sorted(missing)}.")

    if model_kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind: {model_kind!r}.")

    if set(runs["model_kind"].unique()) != {model_kind}:
        raise ValueError(f"the input file must contain only model_kind={model_kind!r}.")

    if runs[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("experiment grouping columns must not contain missing values.")

    if runs.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate experiment observations were found.")

    if set(runs["exposure_weight"].unique()) != set(EXPOSURES):
        raise ValueError("experiment results do not contain all expected exposure settings.")

    if set(runs["historical_selector"].unique()) != set(HISTORICAL):
        raise ValueError("experiment results do not contain all historical selectors.")

    seeds = sorted(runs["seed"].unique())

    if seeds != list(range(len(seeds))):
        raise ValueError("seed values must be consecutive integers starting from zero.")

    expected_groups = {
        (exposure, historical_name, seed)
        for exposure, historical_name, seed in product(
            EXPOSURES,
            HISTORICAL,
            seeds,
        )
    }

    actual_groups = set(
        runs[["exposure_weight", "historical_selector", "seed"]].itertuples(index=False, name=None)
    )

    if actual_groups != expected_groups:
        raise ValueError("some exposure, historical-selector or seed combinations are missing.")

    for key, block in runs.groupby(
        ["exposure_weight", "historical_selector", "seed"],
        sort=False,
    ):
        actual_selectors = set(block["selector"])

        if actual_selectors != EXPECTED_SELECTORS:
            missing_selectors = EXPECTED_SELECTORS - actual_selectors
            unexpected_selectors = actual_selectors - EXPECTED_SELECTORS

            raise ValueError(
                f"incomplete selector coverage for {key}: "
                f"missing={sorted(missing_selectors)}, "
                f"unexpected={sorted(unexpected_selectors)}."
            )

    for metric in METRICS:
        if not pd.api.types.is_numeric_dtype(runs[metric]):
            raise TypeError(f"metric {metric!r} must contain numeric values.")


def load_runs(
    model_kinds: tuple[str, ...] | list[str],
    results_dir: Path,
) -> pd.DataFrame:
    """Load and validate the requested model result files."""
    if not model_kinds:
        raise ValueError("at least one model kind must be requested.")

    if len(set(model_kinds)) != len(model_kinds):
        raise ValueError("model kinds must not contain duplicates.")

    frames = []

    for model_kind in model_kinds:
        if model_kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind: {model_kind!r}.")

        path = results_dir / f"rq1_runs_{model_kind}.csv"

        if not path.is_file():
            raise FileNotFoundError(f"experiment results not found: {path}")

        runs = pd.read_csv(path)
        validate_runs(runs, model_kind)
        frames.append(runs)

    return pd.concat(frames, ignore_index=True)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and 95% Student-t confidence intervals by experiment group."""
    rows = []

    grouping_columns = [
        "model_kind",
        "exposure_weight",
        "historical_selector",
        "selector",
    ]

    for keys, block in runs.groupby(grouping_columns, sort=False):
        model_kind, exposure, historical_name, selector = keys

        for metric in METRICS:
            mean, half_width = mean_ci(block[metric].to_numpy(dtype=float))

            rows.append(
                {
                    "model_kind": model_kind,
                    "exposure_weight": exposure,
                    "historical_selector": historical_name,
                    "selector": selector,
                    "metric": metric,
                    "n": len(block),
                    "mean": mean,
                    "ci95_half_width": half_width,
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """Calculate ML-minus-historical differences paired by random seed.

    Positive differences indicate higher values for the ML selector.
    The interpretation of whether this is desirable depends on the metric.
    """
    rows = []

    grouping_columns = [
        "model_kind",
        "exposure_weight",
        "historical_selector",
    ]

    for keys, block in runs.groupby(grouping_columns, sort=False):
        model_kind, exposure, historical_name = keys

        for selector in ML_SELECTORS:
            selector_rows = block.loc[block["selector"].isin([selector, BASELINE])]

            for metric in METRICS:
                paired = selector_rows.pivot(
                    index="seed",
                    columns="selector",
                    values=metric,
                )

                differences = (paired[selector] - paired[BASELINE]).dropna()

                mean, half_width = mean_ci(differences.to_numpy(dtype=float))

                rows.append(
                    {
                        "model_kind": model_kind,
                        "exposure_weight": exposure,
                        "historical_selector": historical_name,
                        "selector": selector,
                        "comparison": "ML minus historical selector",
                        "metric": metric,
                        "n_pairs": len(differences),
                        "mean_difference": mean,
                        "ci95_half_width": half_width,
                    }
                )

    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def save_analysis(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Save the aggregate and paired-difference tables."""
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "rq1_summary.csv"
    paired_path = results_dir / "rq1_paired_differences.csv"

    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)

    return summary_path, paired_path


def print_report(paired: pd.DataFrame) -> None:
    """Print paired ML-versus-historical differences for Q1/Q4 disparity."""
    focus = paired.loc[paired["metric"] == "selection_rate_ratio_q1_q4"].copy()

    if focus.empty:
        print("No paired Q1/Q4 comparisons are available.")
        return

    print("\nRQ1 — Paired Q1/Q4 differences (ML minus historical selector)")
    print("Positive values indicate a higher Q1/Q4 ratio for the ML selector.\n")

    columns = [
        "model_kind",
        "exposure_weight",
        "historical_selector",
        "selector",
        "n_pairs",
        "mean_difference",
        "ci95_half_width",
    ]

    print(focus[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Load completed RQ1 runs, analyze them and save statistical summaries."""
    parser = ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KINDS,
        default=list(MODEL_KINDS),
        help="models to analyze (default: all three)",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory containing the RQ1 CSV files",
    )

    args = parser.parse_args()

    runs = load_runs(args.models, args.results_dir)

    summary = summarize_runs(runs)
    paired = paired_differences(runs)

    summary_path, paired_path = save_analysis(
        summary,
        paired,
        args.results_dir,
    )

    print(f"Loaded {len(runs)} experimental observations.")
    print(f"Summary saved to: {summary_path}")
    print(f"Paired differences saved to: {paired_path}")

    print_report(paired)


if __name__ == "__main__":
    main()
