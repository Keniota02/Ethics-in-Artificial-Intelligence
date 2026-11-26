"""Tests for athlete selection strategies."""

import numpy as np
import pandas as pd
import pytest

from rae.selectors import (
    OracleSelector,
    PerformanceSelector,
    ScoutSelector,
    StageContext,
    choose,
)


def create_context(
    history,
    maturity=None,
    judgement_noise=None,
    talent=None,
    adult_maturation_weight=0.0,
):
    """Create a reproducible stage context for selector tests."""
    history = np.asarray(history, dtype=float)
    n_candidates = history.shape[1]
    stage = history.shape[0] - 1

    if maturity is None:
        maturity = np.zeros(n_candidates)

    if judgement_noise is None:
        judgement_noise = np.zeros(n_candidates)

    if talent is None:
        talent = np.zeros(n_candidates)

    candidates = pd.DataFrame(
        {
            "athlete_id": np.arange(n_candidates),
            "talent": talent,
            "maturity": maturity,
        }
    )

    return StageContext(
        stage=stage,
        n_stages=3,
        candidates=candidates,
        history=history,
        judgement_noise=np.asarray(judgement_noise, dtype=float),
        adult_maturation_weight=adult_maturation_weight,
    )


def test_performance_selector_uses_current_observations():
    """PerformanceSelector must use only current-stage performance."""
    context = create_context(
        history=[
            [1.0, 4.0],
            [3.0, 2.0],
        ]
    )

    scores = PerformanceSelector().scores(context)

    np.testing.assert_allclose(scores, [3.0, 2.0])


def test_scout_without_memory_matches_performance_selector():
    """Memory equal to zero must ignore all previous observations."""
    context = create_context(
        history=[
            [10.0, 0.0],
            [0.0, 5.0],
            [2.0, 3.0],
        ]
    )

    scout_scores = ScoutSelector(memory=0.0).scores(context)
    performance_scores = PerformanceSelector().scores(context)

    np.testing.assert_allclose(scout_scores, performance_scores)


def test_scout_with_full_memory_uses_equal_weights():
    """Memory equal to one must average all historical observations."""
    context = create_context(
        history=[
            [4.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ]
    )

    scores = ScoutSelector(memory=1.0).scores(context)

    expected = np.array([2.0, 4.0 / 3.0])

    np.testing.assert_allclose(scores, expected)


def test_scout_weights_recent_observations_more_heavily():
    """Intermediate memory values must apply geometric weights."""
    context = create_context(
        history=[
            [4.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ]
    )

    scores = ScoutSelector(memory=0.5).scores(context)

    # Weights for the oldest, intermediate and current observations:
    # 0.25, 0.5 and 1.0.
    expected = np.array(
        [
            (0.25 * 4.0 + 0.5 * 0.0 + 2.0) / 1.75,
            (0.25 * 0.0 + 0.5 * 2.0 + 2.0) / 1.75,
        ]
    )

    np.testing.assert_allclose(scores, expected)


def test_scout_adds_maturation_bias():
    """Maturation bias must add a weighted maturity contribution."""
    context = create_context(
        history=[[0.0, 0.0, 0.0]],
        maturity=[0.0, 1.0, 2.0],
    )

    scores = ScoutSelector(maturation_bias=2.0).scores(context)

    np.testing.assert_allclose(scores, [0.0, 2.0, 4.0])


def test_scout_adds_subjective_judgement_noise():
    """Subjective noise must be scaled by noise_sd."""
    context = create_context(
        history=[[0.0, 0.0]],
        judgement_noise=[1.0, -2.0],
    )

    scores = ScoutSelector(noise_sd=0.5).scores(context)

    np.testing.assert_allclose(scores, [0.5, -1.0])


def test_scout_combines_memory_maturity_and_noise():
    """All three contributions must be combined correctly."""
    context = create_context(
        history=[
            [4.0, 0.0],
            [0.0, 2.0],
        ],
        maturity=[1.0, -1.0],
        judgement_noise=[2.0, -2.0],
    )

    scout = ScoutSelector(
        memory=0.5,
        maturation_bias=0.3,
        noise_sd=0.5,
    )

    # Historical average:
    # Athlete 0: (0.5 * 4 + 0) / 1.5
    # Athlete 1: (0.5 * 0 + 2) / 1.5
    historical = np.array([4.0 / 3.0, 4.0 / 3.0])

    expected = historical + 0.3 * np.array([1.0, -1.0]) + 0.5 * np.array([2.0, -2.0])

    np.testing.assert_allclose(scout.scores(context), expected)


def test_scout_without_maturation_column_rejects_maturity_bias():
    """A maturity bias requires a maturity column."""
    context = create_context(history=[[1.0, 2.0]])

    context = StageContext(
        stage=context.stage,
        n_stages=context.n_stages,
        candidates=context.candidates.drop(columns="maturity"),
        history=context.history,
        judgement_noise=context.judgement_noise,
    )

    with pytest.raises(ValueError):
        ScoutSelector(maturation_bias=0.5).scores(context)


def test_choose_returns_positions_of_highest_scores():
    """The choose function must return positions in descending score order."""
    context = create_context(history=[[0.8, 1.7, -0.2, 1.2, 0.5]])

    selected = choose(
        PerformanceSelector(),
        context,
        k=2,
    )

    np.testing.assert_array_equal(selected, [1, 3])


def test_choose_resolves_ties_stably():
    """Equal scores must preserve the original candidate order."""
    context = create_context(history=[[1.0, 2.0, 2.0, 3.0]])

    selected = choose(
        PerformanceSelector(),
        context,
        k=2,
    )

    np.testing.assert_array_equal(selected, [3, 1])


def test_choose_rejects_wrong_number_of_scores():
    """A selector must return exactly one score per candidate."""

    class InvalidSelector:
        def scores(self, context):
            return np.array([1.0])

    context = create_context(history=[[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        choose(InvalidSelector(), context, k=2)


def test_choose_rejects_non_finite_scores():

    class InvalidSelector:
        def scores(self, context):
            return np.array([1.0, np.nan])

    context = create_context(history=[[1.0, 2.0]])

    with pytest.raises(ValueError):
        choose(InvalidSelector(), context, k=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory": -0.1},
        {"memory": 1.1},
        {"noise_sd": -0.1},
        {"memory": np.nan},
        {"noise_sd": np.inf},
        {"maturation_bias": np.nan},
    ],
)
def test_invalid_scout_parameters_raise_value_error(kwargs):
    """Invalid scout parameters must be rejected."""
    with pytest.raises(ValueError):
        ScoutSelector(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory": "1.0"},
        {"noise_sd": None},
        {"maturation_bias": True},
    ],
)
def test_non_numeric_scout_parameters_raise_type_error(kwargs):
    """Non-numeric scout parameters must be rejected."""
    with pytest.raises(TypeError):
        ScoutSelector(**kwargs)


def test_context_rejects_inconsistent_history_shape():
    """History must have one row per stage up to the current stage."""
    with pytest.raises(ValueError):
        StageContext(
            stage=1,
            n_stages=3,
            candidates=pd.DataFrame({"athlete_id": [0, 1], "maturity": [0.0, 0.0]}),
            history=np.array([[1.0, 2.0]]),
            judgement_noise=np.zeros(2),
        )


def test_oracle_uses_latent_talent_when_adult_maturation_weight_is_zero():
    """With kappa equal to zero, Oracle scores must equal latent talent."""
    context = create_context(
        history=[[0.0, 0.0, 0.0]],
        talent=[0.8, 0.2, 0.5],
        maturity=[1.0, 3.0, 2.0],
    )

    scores = OracleSelector().scores(context)

    np.testing.assert_allclose(scores, [0.8, 0.2, 0.5])


def test_oracle_includes_adult_maturation_component():
    """Oracle scores must include the configured adult maturity contribution."""
    context = create_context(
        history=[[0.0, 0.0, 0.0]],
        talent=[0.8, 0.2, 0.5],
        maturity=[1.0, 3.0, 2.0],
        adult_maturation_weight=0.5,
    )

    scores = OracleSelector().scores(context)

    expected = np.array([1.3, 1.7, 1.5])

    np.testing.assert_allclose(scores, expected)


def test_oracle_requires_talent_and_maturity_columns():
    """Oracle must reject candidates without latent target information."""
    context = create_context(
        history=[[0.0, 0.0]],
        talent=[0.5, 0.8],
        maturity=[0.2, 0.4],
    )

    context = StageContext(
        stage=context.stage,
        n_stages=context.n_stages,
        candidates=context.candidates.drop(columns="talent"),
        history=context.history,
        judgement_noise=context.judgement_noise,
    )

    with pytest.raises(ValueError):
        OracleSelector().scores(context)


@pytest.mark.parametrize(
    "value",
    [-0.1, np.nan, np.inf, -np.inf],
)
def test_context_rejects_invalid_adult_maturation_weight(value):
    """Adult maturation weight must be finite and non-negative."""
    with pytest.raises(ValueError):
        create_context(
            history=[[0.0, 0.0]],
            adult_maturation_weight=value,
        )


def test_context_rejects_non_numeric_adult_maturation_weight():
    """Adult maturation weight cannot be a string or boolean."""
    with pytest.raises(TypeError):
        create_context(
            history=[[0.0, 0.0]],
            adult_maturation_weight=True,
        )
