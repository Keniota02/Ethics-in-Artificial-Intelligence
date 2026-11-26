"""Tests for selection metrics by birth quarter."""

import numpy as np
import pandas as pd
import pytest

from rae.metrics import summarize_selection_by_quarter


@pytest.fixture
def simulation_log():
    """Create a deterministic three-stage simulation log.

    The initial population contains:
    - 3 athletes in Q1
    - 2 athletes in Q2
    - 2 athletes in Q3
    - 1 athlete in Q4

    Some athletes are selected and subsequently drop out, allowing
    cumulative selection and dropout to overlap.
    """
    rows = [
        # Stage 0: initial population
        (1, 0, 1, True, False),
        (2, 0, 1, False, True),
        (3, 0, 1, False, False),
        (4, 0, 2, True, False),
        (5, 0, 2, False, False),
        (6, 0, 3, False, True),
        (7, 0, 3, True, False),
        (8, 0, 4, False, False),
        # Stage 1: athletes who remain in the pool
        (1, 1, 1, True, False),
        (3, 1, 1, False, True),
        (4, 1, 2, False, True),
        (5, 1, 2, True, False),
        (7, 1, 3, False, False),
        (8, 1, 4, True, False),
        # Stage 2: final stage
        (1, 2, 1, False, True),
        (5, 2, 2, False, True),
        (7, 2, 3, True, False),
        (8, 2, 4, False, True),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "athlete_id",
            "stage",
            "birth_quarter",
            "selected",
            "dropped_out",
        ],
    )


def test_summary_contains_all_stages_and_birth_quarters(simulation_log):
    """The output must contain one row per stage and birth quarter."""
    result = summarize_selection_by_quarter(simulation_log)

    assert len(result) == 12
    assert result["stage"].nunique() == 3
    assert set(result["birth_quarter"]) == {1, 2, 3, 4}

    expected_columns = {
        "stage",
        "birth_quarter",
        "n_initial",
        "n_candidates",
        "n_selected",
        "selection_rate",
        "retention_rate",
        "n_cumulative_selected",
        "cumulative_selection_rate",
        "n_cumulative_dropped_out",
        "cumulative_dropout_rate",
    }

    assert expected_columns.issubset(result.columns)


def test_initial_stage_counts_and_rates_are_correct(simulation_log):
    """Stage zero rates must use the initial quarter populations."""
    result = summarize_selection_by_quarter(simulation_log)

    stage_zero = result.loc[result["stage"] == 0].set_index("birth_quarter")

    assert stage_zero["n_initial"].to_dict() == {
        1: 3,
        2: 2,
        3: 2,
        4: 1,
    }

    assert stage_zero["n_candidates"].to_dict() == {
        1: 3,
        2: 2,
        3: 2,
        4: 1,
    }

    np.testing.assert_allclose(
        stage_zero["selection_rate"].to_numpy(),
        [1 / 3, 1 / 2, 1 / 2, 0.0],
    )

    np.testing.assert_allclose(
        stage_zero["retention_rate"].to_numpy(),
        [1.0, 1.0, 1.0, 1.0],
    )


def test_stage_one_uses_current_candidates_as_denominator(simulation_log):
    """Selection rates must use athletes present at the current stage."""
    result = summarize_selection_by_quarter(simulation_log)

    stage_one = result.loc[result["stage"] == 1].set_index("birth_quarter")

    assert stage_one["n_candidates"].to_dict() == {
        1: 2,
        2: 2,
        3: 1,
        4: 1,
    }

    assert stage_one["n_selected"].to_dict() == {
        1: 1,
        2: 1,
        3: 0,
        4: 1,
    }

    np.testing.assert_allclose(
        stage_one["selection_rate"].to_numpy(),
        [0.5, 0.5, 0.0, 1.0],
    )

    np.testing.assert_allclose(
        stage_one["retention_rate"].to_numpy(),
        [2 / 3, 1.0, 1 / 2, 1.0],
    )


def test_cumulative_selection_and_dropout_are_correct(simulation_log):
    """Cumulative metrics must count distinct athletes from the initial pool."""
    result = summarize_selection_by_quarter(simulation_log)

    stage_two = result.loc[result["stage"] == 2].set_index("birth_quarter")

    assert stage_two["n_cumulative_selected"].to_dict() == {
        1: 1,
        2: 2,
        3: 1,
        4: 1,
    }

    assert stage_two["n_cumulative_dropped_out"].to_dict() == {
        1: 3,
        2: 2,
        3: 1,
        4: 1,
    }

    np.testing.assert_allclose(
        stage_two["cumulative_selection_rate"].to_numpy(),
        [1 / 3, 1.0, 1 / 2, 1.0],
    )

    np.testing.assert_allclose(
        stage_two["cumulative_dropout_rate"].to_numpy(),
        [1.0, 1.0, 1 / 2, 1.0],
    )


def test_selected_athletes_can_later_drop_out(simulation_log):
    """Cumulative selection and dropout rates are not complementary."""
    result = summarize_selection_by_quarter(simulation_log)

    q1_stage_two = result.loc[(result["stage"] == 2) & (result["birth_quarter"] == 1)].iloc[0]

    assert q1_stage_two["n_cumulative_selected"] == 1
    assert q1_stage_two["n_cumulative_dropped_out"] == 3

    assert q1_stage_two["cumulative_selection_rate"] > 0.0
    assert q1_stage_two["cumulative_dropout_rate"] == 1.0


def test_missing_birth_quarter_has_zero_counts_and_nan_rates(simulation_log):
    """An absent quarter must still appear with undefined rates."""
    without_q4 = simulation_log.loc[simulation_log["birth_quarter"] != 4].copy()

    result = summarize_selection_by_quarter(without_q4)

    q4 = result.loc[result["birth_quarter"] == 4]

    assert len(q4) == 3
    assert (q4["n_initial"] == 0).all()
    assert (q4["n_candidates"] == 0).all()
    assert (q4["n_selected"] == 0).all()

    assert q4["selection_rate"].isna().all()
    assert q4["retention_rate"].isna().all()
    assert q4["cumulative_selection_rate"].isna().all()
    assert q4["cumulative_dropout_rate"].isna().all()


def test_non_dataframe_input_raises_type_error():
    """The function must require a pandas DataFrame."""
    with pytest.raises(TypeError):
        summarize_selection_by_quarter([1, 2, 3])


def test_empty_dataframe_raises_value_error():
    """An empty simulation log must be rejected."""
    with pytest.raises(ValueError):
        summarize_selection_by_quarter(pd.DataFrame())


def test_missing_required_column_raises_value_error(simulation_log):
    """Every required input column must be present."""
    incomplete = simulation_log.drop(columns="selected")

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(incomplete)


def test_duplicate_athlete_stage_raises_value_error(simulation_log):
    """An athlete cannot appear twice in the same stage."""
    duplicated = pd.concat(
        [simulation_log, simulation_log.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(duplicated)


def test_inconsistent_birth_quarter_raises_value_error(simulation_log):
    """An athlete must retain the same birth quarter across stages."""
    inconsistent = simulation_log.copy()
    inconsistent.loc[
        (inconsistent["athlete_id"] == 1) & (inconsistent["stage"] == 1),
        "birth_quarter",
    ] = 2

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(inconsistent)


def test_invalid_birth_quarter_raises_value_error(simulation_log):
    """Birth quarters must be integers between one and four."""
    invalid = simulation_log.copy()
    invalid.loc[0, "birth_quarter"] = 5

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(invalid)


def test_non_boolean_selection_flag_raises_type_error(simulation_log):
    """Selection and dropout indicators must be boolean."""
    invalid = simulation_log.copy()
    invalid["selected"] = invalid["selected"].astype(int)

    with pytest.raises(TypeError):
        summarize_selection_by_quarter(invalid)


def test_selected_and_dropped_out_in_same_stage_raises_value_error(
    simulation_log,
):
    """An athlete cannot be selected and drop out in the same stage."""
    invalid = simulation_log.copy()
    invalid.loc[0, "dropped_out"] = True

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(invalid)


def test_non_consecutive_stages_raise_value_error(simulation_log):
    """The simulation must contain consecutive stages starting from zero."""
    invalid = simulation_log.loc[simulation_log["stage"] != 1].copy()

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(invalid)


def test_athlete_cannot_be_observed_after_dropout(simulation_log):
    """An athlete cannot have later observations after dropping out."""
    invalid = simulation_log.copy()
    invalid.loc[
        (invalid["athlete_id"] == 3) & (invalid["stage"] == 0),
        "dropped_out",
    ] = True

    with pytest.raises(ValueError):
        summarize_selection_by_quarter(invalid)
