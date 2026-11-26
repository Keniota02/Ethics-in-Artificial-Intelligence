"""Tests for the multi-stage athlete selection pipeline."""

import numpy as np
import pandas as pd
import pytest

from rae.pipeline import simulate_pipeline


def create_test_population(n=20, seed=42):
    """Create a reproducible synthetic population."""
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "athlete_id": np.arange(n),
            "talent": rng.normal(size=n),
            "maturity": rng.normal(size=n),
        }
    )


def test_pipeline_returns_expected_structure():
    """Check the output columns and number of stages."""
    population = create_test_population()

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25),
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    expected_columns = [
        "athlete_id",
        "talent",
        "maturity",
        "stage",
        "times_selected_before",
        "exposure",
        "performance_noise",
        "performance",
        "selected",
        "dropped_out",
    ]

    assert result.columns.tolist() == expected_columns
    assert set(result["stage"]) == {0, 1}


def test_selection_quotas_are_based_on_initial_population():
    """Check that selection quotas use the initial population size."""
    population = create_test_population(n=20)

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25, 0.1),
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    selected_counts = result.groupby("stage")["selected"].sum().tolist()

    assert selected_counts == [10, 5, 2]


def test_candidates_remain_when_dropout_is_zero():
    """When dropout is zero, everyone remains in every stage."""
    population = create_test_population(n=20)

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25, 0.1),
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    candidate_counts = result.groupby("stage").size().tolist()

    assert candidate_counts == [20, 20, 20]
    assert not result["dropped_out"].any()


def test_unselected_athletes_leave_when_dropout_is_one():
    """When dropout is one, all unselected athletes leave."""
    population = create_test_population(n=20)

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25, 0.1),
        dropout_unselected=1.0,
        rng=np.random.default_rng(43),
    )

    candidate_counts = result.groupby("stage").size().tolist()

    assert candidate_counts == [20, 10, 5]

    selected_counts = result.groupby("stage")["selected"].sum().tolist()

    assert selected_counts == [10, 5, 2]


def test_dropout_only_affects_unselected_athletes():
    """Selected athletes must never drop out after a stage."""
    population = create_test_population(n=100)

    result = simulate_pipeline(
        population,
        dropout_unselected=0.5,
        rng=np.random.default_rng(43),
    )

    assert not result.loc[result["selected"], "dropped_out"].any()


def test_dropped_out_athletes_do_not_return_in_later_stages():
    """An athlete who drops out must not appear in subsequent stages."""
    population = create_test_population(n=100)

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25, 0.1),
        dropout_unselected=1.0,
        rng=np.random.default_rng(43),
    )

    first_stage = result[result["stage"] == 0]
    second_stage = result[result["stage"] == 1]

    dropped_ids = set(first_stage.loc[first_stage["dropped_out"], "athlete_id"])

    second_stage_ids = set(second_stage["athlete_id"])

    assert dropped_ids.isdisjoint(second_stage_ids)


def test_exposure_increases_after_each_selection():
    """Exposure must reflect selections made in earlier stages."""
    population = create_test_population(n=20)
    exposure_gain = 2.0

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25, 0.1),
        exposure_gain=exposure_gain,
        exposure_weight=1.0,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    stage_0 = result[result["stage"] == 0].set_index("athlete_id")
    stage_1 = result[result["stage"] == 1].set_index("athlete_id")
    stage_2 = result[result["stage"] == 2].set_index("athlete_id")

    expected_exposure_stage_1 = exposure_gain * stage_0["selected"].astype(int)

    expected_exposure_stage_2 = exposure_gain * (
        stage_0["selected"].astype(int) + stage_1["selected"].astype(int)
    )

    pd.testing.assert_series_equal(
        stage_1["exposure"].sort_index(),
        expected_exposure_stage_1.sort_index().rename("exposure"),
    )

    pd.testing.assert_series_equal(
        stage_2["exposure"].sort_index(),
        expected_exposure_stage_2.sort_index().rename("exposure"),
    )


def test_performance_follows_multistage_formula():
    """Check the performance equation at every stage."""
    population = create_test_population(n=20)

    beta_m = 2.0
    maturation_decay = 0.7
    sigma_p = 1.0
    exposure_weight = 1.0

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25),
        beta_m=beta_m,
        maturation_decay=maturation_decay,
        sigma_p=sigma_p,
        exposure_weight=exposure_weight,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    expected_performance = (
        result["talent"]
        + beta_m * maturation_decay ** result["stage"] * result["maturity"]
        + exposure_weight * result["exposure"]
        + result["performance_noise"]
    )

    np.testing.assert_allclose(
        result["performance"],
        expected_performance,
    )


def test_exposure_feedback_does_not_affect_first_stage():
    """The initial stage must have zero accumulated exposure."""
    population = create_test_population(n=20)

    result = simulate_pipeline(
        population,
        exposure_weight=1.0,
        exposure_gain=2.0,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    first_stage = result[result["stage"] == 0]

    assert (first_stage["exposure"] == 0.0).all()


def test_feedback_off_and_on_use_same_random_performance_noise():
    """Changing exposure weight must not change the performance noise."""
    population = create_test_population(n=20)

    result_off = simulate_pipeline(
        population,
        exposure_weight=0.0,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    result_on = simulate_pipeline(
        population,
        exposure_weight=1.0,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    pd.testing.assert_series_equal(
        result_off["performance_noise"],
        result_on["performance_noise"],
    )


def test_feedback_contribution_matches_accumulated_exposure():
    """With the same seed, ON and OFF differ only by exposure contribution."""
    population = create_test_population(n=20)

    result_off = simulate_pipeline(
        population,
        exposure_weight=0.0,
        exposure_gain=2.0,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    result_on = simulate_pipeline(
        population,
        exposure_weight=1.0,
        exposure_gain=2.0,
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    stage_1_off = result_off[result_off["stage"] == 1].set_index("athlete_id")

    stage_1_on = result_on[result_on["stage"] == 1].set_index("athlete_id")

    expected_difference = stage_1_on["exposure"]

    actual_difference = stage_1_on["performance"] - stage_1_off["performance"]

    pd.testing.assert_series_equal(
        actual_difference.sort_index(),
        expected_difference.sort_index(),
        check_names=False,
    )


def test_pipeline_is_reproducible_with_same_seed():
    """Equal random seeds must generate identical simulation results."""
    population = create_test_population(n=100)

    result_1 = simulate_pipeline(
        population,
        rng=np.random.default_rng(43),
    )

    result_2 = simulate_pipeline(
        population,
        rng=np.random.default_rng(43),
    )

    pd.testing.assert_frame_equal(result_1, result_2)


def test_pipeline_does_not_modify_original_population():
    """The original population must remain unchanged."""
    population = create_test_population(n=100)
    original_population = population.copy(deep=True)

    simulate_pipeline(
        population,
        rng=np.random.default_rng(43),
    )

    pd.testing.assert_frame_equal(
        population,
        original_population,
    )


def test_an_athlete_appears_at_most_once_per_stage():
    """An athlete must have at most one record in each stage."""
    population = create_test_population(n=100)

    result = simulate_pipeline(
        population,
        rng=np.random.default_rng(43),
    )

    assert not result.duplicated(subset=["athlete_id", "stage"]).any()


def test_selected_counts_are_integer_and_boolean():
    """Selection and dropout indicators must be Boolean."""
    population = create_test_population()

    result = simulate_pipeline(
        population,
        stage_fractions=(0.5, 0.25),
        dropout_unselected=0.0,
        rng=np.random.default_rng(43),
    )

    assert result["selected"].dtype == bool
    assert result["dropped_out"].dtype == bool


def test_non_dataframe_population_raises_type_error():
    """A non-DataFrame input should raise TypeError."""
    with pytest.raises(TypeError):
        simulate_pipeline([1, 2, 3])


def test_empty_population_raises_value_error():
    """An empty population should raise ValueError."""
    population = pd.DataFrame(columns=["athlete_id", "talent", "maturity"])

    with pytest.raises(ValueError):
        simulate_pipeline(population)


def test_missing_required_columns_raise_value_error():
    """Required population columns must be present."""
    population = pd.DataFrame(
        {
            "athlete_id": [0, 1],
            "talent": [0.5, 0.8],
        }
    )

    with pytest.raises(ValueError):
        simulate_pipeline(population)


def test_duplicate_athlete_ids_raise_value_error():
    """Athlete identifiers must be unique."""
    population = pd.DataFrame(
        {
            "athlete_id": [0, 0],
            "talent": [0.5, 0.8],
            "maturity": [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError):
        simulate_pipeline(population)


@pytest.mark.parametrize(
    "stage_fractions",
    [
        (),
        (0.0,),
        (1.1,),
        (0.5, 0.6),
        (0.5, 0.0),
    ],
)
def test_invalid_stage_fractions_raise_value_error(stage_fractions):
    """Invalid stage fractions should raise ValueError."""
    population = create_test_population()

    with pytest.raises(ValueError):
        simulate_pipeline(
            population,
            stage_fractions=stage_fractions,
        )


def test_non_sequence_stage_fractions_raise_type_error():
    """Stage fractions must be supplied as a tuple or list."""
    population = create_test_population()

    with pytest.raises(TypeError):
        simulate_pipeline(population, stage_fractions=0.5)


@pytest.mark.parametrize(
    "parameter",
    [
        "beta_m",
        "sigma_p",
        "exposure_weight",
        "exposure_gain",
    ],
)
def test_negative_model_parameters_raise_value_error(parameter):
    """Model weights and noise standard deviation cannot be negative."""
    population = create_test_population()

    with pytest.raises(ValueError):
        simulate_pipeline(
            population,
            **{parameter: -0.1},
        )


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, np.nan, np.inf, -np.inf],
)
def test_invalid_dropout_probability_raises_value_error(value):
    """Dropout probability must be finite and between zero and one."""
    population = create_test_population()

    with pytest.raises(ValueError):
        simulate_pipeline(
            population,
            dropout_unselected=value,
        )


def test_invalid_random_generator_raises_type_error():
    """The random generator must be a NumPy Generator."""
    population = create_test_population()

    with pytest.raises(TypeError):
        simulate_pipeline(population, rng=43)
