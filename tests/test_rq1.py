"""Tests for the RQ1 experiment runner."""

import pandas as pd
import pytest
from experiments.run_rq1 import (
    BASELINE,
    METRICS,
    _final_stage_metrics,
    report,
    run,
)

from rae.config import SimConfig
from rae.pipeline import simulate
from rae.selectors import PerformanceSelector


def test_rq1_runs_all_experimental_combinations():
    """One seed should produce all exposure, history and selector combinations."""
    results = run(
        n_seeds=1,
        model_kind="linear",
        n_athletes=400,
    )

    # 2 exposure settings x 2 historical selectors x 6 evaluated selectors.
    assert len(results) == 24

    assert results["exposure_weight"].nunique() == 2
    assert results["historical_selector"].nunique() == 2
    assert results["seed"].nunique() == 1

    assert results["selector"].nunique() == 6
    assert set(results["model_kind"]) == {"linear"}

    assert set(METRICS).issubset(results.columns)


def test_rq1_contains_oracle_historical_and_all_ml_regimes():
    """The experiment should compare every required selector configuration."""
    results = run(
        n_seeds=1,
        model_kind="linear",
        n_athletes=400,
    )

    expected_selectors = {
        "oracle",
        BASELINE,
        "ML decisions, birth info no",
        "ML decisions, birth info yes",
        "ML later_performance, birth info no",
        "ML later_performance, birth info yes",
    }

    assert set(results["selector"]) == expected_selectors


def test_rq1_metrics_refer_to_the_final_stage():
    """The runner should extract the final-stage metrics from the simulation."""
    config = SimConfig(n_athletes=400, seed=12)
    simulation = simulate(config, PerformanceSelector())

    metrics = _final_stage_metrics(simulation, config)

    assert set(metrics) == set(METRICS)
    assert all(isinstance(value, float) for value in metrics.values())


def test_rq1_rejects_invalid_seed_count():
    """At least one seed is required."""
    with pytest.raises(ValueError, match="n_seeds"):
        run(n_seeds=0, n_athletes=400)


def test_rq1_rejects_non_integer_seed_count():
    """Seed count must be an integer."""
    with pytest.raises(TypeError, match="n_seeds"):
        run(n_seeds=1.5, n_athletes=400)


def test_rq1_rejects_invalid_model_kind():
    """Only the three implemented model architectures are accepted."""
    with pytest.raises(ValueError, match="model_kind"):
        run(n_seeds=1, model_kind="unknown", n_athletes=400)


def test_rq1_report_prints_experiment_groups(capsys):
    """The report should display the historical selector and exposure groups."""
    results = pd.DataFrame(
        [
            {
                "exposure_weight": 0.0,
                "historical_selector": "scout",
                "seed": 0,
                "selector": BASELINE,
                **{metric: 0.5 for metric in METRICS},
            },
            {
                "exposure_weight": 0.0,
                "historical_selector": "scout",
                "seed": 0,
                "selector": "oracle",
                **{metric: 0.0 for metric in METRICS},
            },
        ]
    )

    report(results)

    output = capsys.readouterr().out

    assert "historical: scout" in output
    assert "exposure_weight = 0" in output
    assert BASELINE in output
    assert "oracle" in output
