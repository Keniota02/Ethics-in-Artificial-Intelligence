"""Tests for the statistical analysis of RQ1 results."""

import pandas as pd
import pytest
from experiments.analyze_rq1 import (
    BASELINE,
    METRICS,
    ML_SELECTORS,
    load_runs,
    paired_differences,
    save_analysis,
    summarize_runs,
    validate_runs,
)
from experiments.common import EXPOSURES, HISTORICAL
from experiments.run_rq1 import MODEL_KINDS


@pytest.fixture
def synthetic_runs():
    """Create a complete and deterministic two-seed RQ1 dataset."""
    rows = []

    for model_kind in MODEL_KINDS:
        for exposure in EXPOSURES:
            for historical_name in HISTORICAL:
                for seed in range(2):
                    selectors = [
                        "oracle",
                        BASELINE,
                        *ML_SELECTORS,
                    ]

                    for selector_index, selector in enumerate(selectors):
                        row = {
                            "model_kind": model_kind,
                            "exposure_weight": exposure,
                            "historical_selector": historical_name,
                            "seed": seed,
                            "selector": selector,
                            "evaluation_stage": 2,
                        }

                        for metric_index, metric in enumerate(METRICS):
                            row[metric] = float(seed + selector_index + metric_index) + exposure

                        rows.append(row)

    return pd.DataFrame(rows)


def test_validate_runs_accepts_complete_results(synthetic_runs):
    """A complete dataset should pass validation."""
    for model_kind in MODEL_KINDS:
        subset = synthetic_runs.loc[synthetic_runs["model_kind"] == model_kind]

        validate_runs(subset, model_kind)


def test_validate_runs_rejects_missing_selector(synthetic_runs):
    """Incomplete selector coverage should be detected."""
    incomplete = synthetic_runs.drop(synthetic_runs.index[0])

    linear = incomplete.loc[incomplete["model_kind"] == "linear"]

    with pytest.raises(ValueError, match="incomplete selector coverage"):
        validate_runs(linear, "linear")


def test_validate_runs_rejects_duplicate_experiment_rows(synthetic_runs):
    """Every seed and selector combination must appear exactly once."""
    linear = synthetic_runs.loc[synthetic_runs["model_kind"] == "linear"]

    duplicated = pd.concat(
        [linear, linear.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_runs(duplicated, "linear")


def test_summarize_runs_calculates_one_row_per_metric_and_group(synthetic_runs):
    """Summary output should contain all metric-level aggregates."""
    linear = synthetic_runs.loc[synthetic_runs["model_kind"] == "linear"]

    summary = summarize_runs(linear)

    expected_groups = 2 * 2 * 6

    assert len(summary) == expected_groups * len(METRICS)
    assert set(summary["metric"]) == set(METRICS)
    assert (summary["n"] == 2).all()


def test_paired_differences_use_the_historical_selector_baseline(synthetic_runs):
    """Paired differences should use identical seeds for each comparison."""
    linear = synthetic_runs.loc[synthetic_runs["model_kind"] == "linear"]

    paired = paired_differences(linear)

    expected_rows = 2 * 2 * len(ML_SELECTORS) * len(METRICS)

    assert len(paired) == expected_rows
    assert (paired["n_pairs"] == 2).all()
    assert set(paired["comparison"]) == {"ML minus historical selector"}


def test_paired_difference_matches_seed_level_calculation(synthetic_runs):
    """The paired mean should equal the mean of within-seed differences."""
    linear = synthetic_runs.loc[
        (synthetic_runs["model_kind"] == "linear")
        & (synthetic_runs["exposure_weight"] == 0.0)
        & (synthetic_runs["historical_selector"] == "scout")
    ]

    selector = ML_SELECTORS[0]
    paired = paired_differences(linear)

    result = paired.loc[
        (paired["selector"] == selector) & (paired["metric"] == "selection_rate_ratio_q1_q4")
    ].iloc[0]

    pivot = linear.loc[linear["selector"].isin([selector, BASELINE])].pivot(
        index="seed",
        columns="selector",
        values="selection_rate_ratio_q1_q4",
    )

    expected_difference = (pivot[selector] - pivot[BASELINE]).mean()

    assert result["mean_difference"] == pytest.approx(expected_difference)


def test_save_analysis_writes_both_csv_files(tmp_path, synthetic_runs):
    """Both statistical output tables should be saved."""
    linear = synthetic_runs.loc[synthetic_runs["model_kind"] == "linear"]

    summary = summarize_runs(linear)
    paired = paired_differences(linear)

    summary_path, paired_path = save_analysis(
        summary,
        paired,
        tmp_path,
    )

    assert summary_path.exists()
    assert paired_path.exists()

    assert len(pd.read_csv(summary_path)) == len(summary)
    assert len(pd.read_csv(paired_path)) == len(paired)


def test_load_runs_rejects_missing_result_file(tmp_path):
    """A missing model result should produce an explicit error."""
    with pytest.raises(FileNotFoundError):
        load_runs(["linear"], tmp_path)
