"""Tests for selection metrics by birth quarter."""

import numpy as np
import pandas as pd
import pytest

from rae.metrics import (
    summarize_selection_by_quarter,
    summarize_talent_loss_by_quarter,
    summarize_talent_metrics,
)


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


@pytest.fixture
def talent_simulation():
    """Create a deterministic two-stage simulation for talent metrics."""
    rows = []

    talent = [10.0, 0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0]
    quarters = [1, 1, 2, 2, 3, 3, 4, 4]

    selected_stage_0 = {0, 2, 4, 7}
    selected_stage_1 = {0, 2, 4, 6}

    for athlete_id in range(8):
        selected_0 = athlete_id in selected_stage_0
        selected_1 = athlete_id in selected_stage_1

        for stage, selected in [
            (0, selected_0),
            (1, selected_1),
        ]:
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "stage": stage,
                    "birth_quarter": quarters[athlete_id],
                    "talent": talent[athlete_id],
                    "maturity": 0.0,
                    "selected": selected,
                    "dropped_out": False,
                }
            )

    return pd.DataFrame(rows)


def test_talent_loss_by_quarter_has_expected_structure(talent_simulation):
    """The output must contain one row per stage and birth quarter."""
    result = summarize_talent_loss_by_quarter(
        talent_simulation,
        top_talent_fraction=0.5,
    )

    assert len(result) == 8
    assert set(result["stage"]) == {0, 1}
    assert set(result["birth_quarter"]) == {1, 2, 3, 4}

    expected_columns = {
        "stage",
        "birth_quarter",
        "n_initial",
        "n_high_talent",
        "n_selected_high_talent",
        "n_lost_high_talent",
        "talent_loss",
    }

    assert expected_columns.issubset(result.columns)


def test_talent_loss_by_quarter_is_calculated_correctly(talent_simulation):
    """The initial high-talent athletes must be the denominator."""
    result = summarize_talent_loss_by_quarter(
        talent_simulation,
        top_talent_fraction=0.5,
    )

    stage_zero = result.loc[result["stage"] == 0].set_index("birth_quarter")

    assert stage_zero["n_high_talent"].to_dict() == {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }

    assert stage_zero["n_selected_high_talent"].to_dict() == {
        1: 1,
        2: 1,
        3: 1,
        4: 0,
    }

    np.testing.assert_allclose(
        stage_zero["talent_loss"].to_numpy(),
        [0.0, 0.0, 0.0, 1.0],
    )


def test_later_stage_talent_loss_uses_initial_high_talent_cohort(talent_simulation):
    """Later stages must retain the initial cohort as denominator."""
    result = summarize_talent_loss_by_quarter(
        talent_simulation,
        top_talent_fraction=0.5,
    )

    stage_one = result.loc[result["stage"] == 1]

    np.testing.assert_allclose(
        stage_one["talent_loss"].to_numpy(),
        [0.0, 0.0, 0.0, 0.0],
    )


def test_summary_calculates_fairness_and_efficiency_metrics(talent_simulation):
    """Overall loss, opportunity gap, efficiency and regret must be correct."""
    result = summarize_talent_metrics(
        talent_simulation,
        top_talent_fraction=0.5,
    ).set_index("stage")

    stage_zero = result.loc[0]
    stage_one = result.loc[1]

    assert stage_zero["overall_talent_loss"] == pytest.approx(0.25)
    assert stage_zero["equal_opportunity_gap"] == pytest.approx(1.0)
    assert stage_zero["mean_selected_talent"] == pytest.approx(7.5)
    assert stage_zero["regret"] == pytest.approx(1.0)

    assert stage_one["overall_talent_loss"] == pytest.approx(0.0)
    assert stage_one["equal_opportunity_gap"] == pytest.approx(0.0)
    assert stage_one["mean_selected_talent"] == pytest.approx(8.5)
    assert stage_one["regret"] == pytest.approx(0.0)


def test_oracle_has_zero_regret(talent_simulation):
    """Selecting the highest target values must produce zero regret."""
    result = summarize_talent_metrics(
        talent_simulation,
        top_talent_fraction=0.5,
    ).set_index("stage")

    assert result.loc[1, "regret"] == pytest.approx(0.0)


def test_adult_maturation_weight_changes_talent_target(talent_simulation):
    """Adult maturation must affect high-talent identification and metrics."""
    modified = talent_simulation.copy()

    modified.loc[
        modified["athlete_id"] == 7,
        "maturity",
    ] = 10.0

    result_talent = summarize_talent_loss_by_quarter(
        modified,
        top_talent_fraction=0.5,
        adult_maturation_weight=0.0,
    )

    result_adult = summarize_talent_loss_by_quarter(
        modified,
        top_talent_fraction=0.5,
        adult_maturation_weight=1.0,
    )

    q4_talent = result_talent.loc[
        (result_talent["stage"] == 0) & (result_talent["birth_quarter"] == 4),
        "talent_loss",
    ].iloc[0]

    q4_adult = result_adult.loc[
        (result_adult["stage"] == 0) & (result_adult["birth_quarter"] == 4),
        "talent_loss",
    ].iloc[0]

    assert q4_talent == pytest.approx(1.0)
    assert q4_adult == pytest.approx(0.0)


@pytest.mark.parametrize(
    "fraction",
    [0.0, 1.0, -0.1, 1.1, np.nan, np.inf],
)
def test_invalid_top_talent_fraction_is_rejected(talent_simulation, fraction):
    """The high-talent fraction must be finite and between zero and one."""
    with pytest.raises(ValueError):
        summarize_talent_metrics(
            talent_simulation,
            top_talent_fraction=fraction,
        )


@pytest.mark.parametrize(
    "weight",
    [-0.1, np.nan, np.inf, -np.inf],
)
def test_invalid_adult_maturation_weight_is_rejected(talent_simulation, weight):
    """Adult maturation weight must be finite and non-negative."""
    with pytest.raises(ValueError):
        summarize_talent_metrics(
            talent_simulation,
            adult_maturation_weight=weight,
        )


def test_non_numeric_talent_metric_parameters_are_rejected(talent_simulation):
    """Metric parameters must be numeric and cannot be booleans."""
    with pytest.raises(TypeError):
        summarize_talent_metrics(
            talent_simulation,
            top_talent_fraction=True,
        )

    with pytest.raises(TypeError):
        summarize_talent_metrics(
            talent_simulation,
            adult_maturation_weight="0.5",
        )


def test_non_finite_talent_values_are_rejected(talent_simulation):
    """Latent talent must contain finite values."""
    invalid = talent_simulation.copy()
    invalid.loc[0, "talent"] = np.nan

    with pytest.raises(ValueError):
        summarize_talent_metrics(invalid)


def test_inconsistent_talent_values_across_stages_are_rejected(talent_simulation):
    """An athlete must retain the same latent talent across all stages."""
    invalid = talent_simulation.copy()

    invalid.loc[
        (invalid["athlete_id"] == 0) & (invalid["stage"] == 1),
        "talent",
    ] = 999.0

    with pytest.raises(ValueError):
        summarize_talent_metrics(invalid)
