"""Tests for machine-learning training data construction."""

import numpy as np
import pandas as pd
import pytest

from rae.learning import (
    PerStageModel,
    build_training_data,
    train_selector,
)
from rae.pipeline import simulate_pipeline
from rae.population import generate_population
from rae.selectors import StageContext, choose


@pytest.fixture
def simulation_log():
    """Generate a small reproducible simulation log."""
    rng = np.random.default_rng(42)
    population = generate_population(n=200, rho=0.3, rng=rng)

    return simulate_pipeline(
        population,
        stage_fractions=(0.50, 0.25),
        dropout_unselected=0.30,
        rng=rng,
    )


def test_decisions_regime_uses_all_historical_rows(simulation_log):
    """Decision labels should be available for every observed candidate."""
    X, y = build_training_data([simulation_log], regime="decisions")

    assert len(X) == len(simulation_log)
    assert len(y) == len(simulation_log)
    assert set(y.unique()).issubset({0, 1})

    assert list(X.columns) == [
        "observed",
        "history_mean",
        "times_selected",
        "stage",
    ]


def test_birth_position_is_optional(simulation_log):
    """Birth position should only be included when requested."""
    X_without, _ = build_training_data(
        [simulation_log],
        regime="decisions",
    )
    X_with, _ = build_training_data(
        [simulation_log],
        regime="decisions",
        use_birth_info=True,
    )

    assert "birth_position" not in X_without.columns
    assert "birth_position" in X_with.columns


def test_later_performance_uses_next_stage_observations(simulation_log):
    """Targets should come from the same athlete at the following stage."""
    X, y = build_training_data(
        [simulation_log],
        regime="later_performance",
        features=[
            "observed",
            "history_mean",
            "times_selected",
            "stage",
        ],
    )

    stage_zero = simulation_log[simulation_log["stage"] == 0]
    stage_one = simulation_log[simulation_log["stage"] == 1]

    expected_ids = set(stage_zero["athlete_id"]) & set(stage_one["athlete_id"])

    assert len(X) == len(expected_ids)
    assert len(y) == len(expected_ids)

    expected = (
        stage_zero[["athlete_id"]]
        .merge(
            stage_one[["athlete_id", "performance"]],
            on="athlete_id",
            how="inner",
        )
        .sort_values("athlete_id")
        .reset_index(drop=True)
    )

    assert np.allclose(
        y.to_numpy(),
        expected["performance"].to_numpy(),
    )

    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert np.isfinite(y.to_numpy(dtype=float)).all()


def test_invalid_regime_raises_error(simulation_log):
    """Unknown label regimes should be rejected."""
    with pytest.raises(ValueError, match="regime"):
        build_training_data([simulation_log], regime="unknown")


def test_later_performance_without_next_stage_raises_error():
    """A single-stage log cannot provide next-stage performance labels."""
    log = pd.DataFrame(
        {
            "athlete_id": [1, 2],
            "stage": [0, 0],
            "performance": [1.0, 2.0],
            "times_selected_before": [0, 0],
            "selected": [True, False],
        }
    )

    with pytest.raises(ValueError, match="no training rows"):
        build_training_data([log], regime="later_performance")


def test_trained_decision_selector_returns_valid_scores(simulation_log):
    """A trained decision selector should score every candidate."""
    selector = train_selector(
        [simulation_log],
        regime="decisions",
    )

    first_stage = simulation_log[simulation_log["stage"] == 0].copy()
    context = StageContext(
        stage=0,
        n_stages=2,
        candidates=first_stage.reset_index(drop=True),
        history=first_stage["performance"].to_numpy().reshape(1, -1),
        judgement_noise=np.zeros(len(first_stage)),
    )

    scores = selector.scores(context)

    assert scores.shape == (len(first_stage),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()

    selected = choose(selector, context, k=10)
    assert len(selected) == 10


def test_trained_performance_selector_returns_finite_scores(simulation_log):
    """A trained performance selector should predict next-stage performance."""
    selector = train_selector(
        [simulation_log],
        regime="later_performance",
    )

    first_stage = simulation_log[simulation_log["stage"] == 0].copy()
    context = StageContext(
        stage=0,
        n_stages=2,
        candidates=first_stage.reset_index(drop=True),
        history=first_stage["performance"].to_numpy().reshape(1, -1),
        judgement_noise=np.zeros(len(first_stage)),
    )

    scores = selector.scores(context)

    assert scores.shape == (len(first_stage),)
    assert np.isfinite(scores).all()


def test_linear_by_stage_trains_independent_models(simulation_log):
    """The stage-specific model should train a model for each stage."""
    selector = train_selector(
        [simulation_log],
        regime="decisions",
        model_kind="linear_by_stage",
    )

    assert set(selector.model.models) == {0, 1}


def test_linear_by_stage_predicts_with_previous_stage_fallback(
    simulation_log,
):
    """The last stage should use the closest earlier trained model."""
    selector = train_selector(
        [simulation_log],
        regime="later_performance",
        model_kind="linear_by_stage",
    )

    assert set(selector.model.models) == {0}

    last_stage = simulation_log[simulation_log["stage"] == 1].copy().reset_index(drop=True)

    stage_zero = simulation_log[simulation_log["stage"] == 0].set_index("athlete_id")["performance"]

    previous_performance = stage_zero.reindex(last_stage["athlete_id"]).to_numpy(dtype=float)

    current_performance = last_stage["performance"].to_numpy(dtype=float)

    history = np.vstack([previous_performance, current_performance])

    assert np.isfinite(history).all()

    context = StageContext(
        stage=1,
        n_stages=2,
        candidates=last_stage,
        history=history,
        judgement_noise=np.zeros(len(last_stage)),
    )

    scores = selector.scores(context)

    assert scores.shape == (len(last_stage),)
    assert np.isfinite(scores).all()


def test_linear_by_stage_requires_stage_feature(simulation_log):
    """Stage-specific models must receive the stage feature."""
    with pytest.raises(ValueError, match="stage"):
        train_selector(
            [simulation_log],
            regime="decisions",
            features=["observed", "history_mean"],
            model_kind="linear_by_stage",
        )


@pytest.mark.parametrize(
    "regime",
    ["decisions", "later_performance"],
)
def test_gbm_selector_returns_finite_scores(simulation_log, regime):
    """Gradient boosting should support both training regimes."""
    selector = train_selector(
        [simulation_log],
        regime=regime,
        model_kind="gbm",
        seed=42,
    )

    first_stage = simulation_log[simulation_log["stage"] == 0].copy().reset_index(drop=True)

    context = StageContext(
        stage=0,
        n_stages=2,
        candidates=first_stage,
        history=first_stage["performance"].to_numpy().reshape(1, -1),
        judgement_noise=np.zeros(len(first_stage)),
    )

    scores = selector.scores(context)

    assert scores.shape == (len(first_stage),)
    assert np.isfinite(scores).all()

    if regime == "decisions":
        assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_gbm_selector_is_reproducible(simulation_log):
    """Gradient boosting with the same seed should be reproducible."""
    first = train_selector(
        [simulation_log],
        regime="decisions",
        model_kind="gbm",
        seed=42,
    )
    second = train_selector(
        [simulation_log],
        regime="decisions",
        model_kind="gbm",
        seed=42,
    )

    X = simulation_log[simulation_log["stage"] == 0].copy().reset_index(drop=True)

    context = StageContext(
        stage=0,
        n_stages=2,
        candidates=X,
        history=X["performance"].to_numpy().reshape(1, -1),
        judgement_noise=np.zeros(len(X)),
    )

    assert np.allclose(
        first.scores(context),
        second.scores(context),
    )


def test_linear_by_stage_skips_stages_with_one_decision_class():
    X = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 3.0, 4.0],
            "history_mean": [1.0, 2.0, 3.0, 4.0],
            "times_selected": [0, 0, 1, 1],
            "stage": [0, 0, 1, 1],
        }
    )
    y = pd.Series([0, 0, 0, 1])

    model = PerStageModel("decisions")
    model.fit(X, y)

    assert set(model.models) == {1}


def test_linear_by_stage_fails_if_no_stage_has_two_classes():
    X = pd.DataFrame(
        {
            "observed": [1.0, 2.0],
            "history_mean": [1.0, 2.0],
            "times_selected": [0, 0],
            "stage": [0, 1],
        }
    )
    y = pd.Series([1, 1])

    model = PerStageModel("decisions")

    with pytest.raises(ValueError, match="sufficient data"):
        model.fit(X, y)
