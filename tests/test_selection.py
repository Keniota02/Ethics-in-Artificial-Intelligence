"""Tests for baseline athlete selection."""

import numpy as np
import pandas as pd
import pytest

from rae.selection import select_top_k


@pytest.fixture
def population():
    """Create a small population with known performance scores."""
    return pd.DataFrame(
        {
            "athlete_id": [0, 1, 2, 3, 4],
            "performance": [0.8, 1.7, -0.2, 1.2, 0.5],
            "birth_quarter": [1, 4, 2, 3, 1],
        }
    )


def test_selection_returns_expected_number_of_athletes(population):
    """Check that exactly k athletes are selected."""
    result = select_top_k(population, k=2)

    assert len(result) == 2


def test_selection_returns_highest_performances(population):
    """Check that the athletes with the highest scores are selected."""
    result = select_top_k(population, k=2)

    assert result["athlete_id"].tolist() == [1, 3]
    assert result["performance"].tolist() == [1.7, 1.2]


def test_selection_is_ordered_by_descending_performance(population):
    """Check that selected athletes are returned in ranking order."""
    result = select_top_k(population, k=4)

    assert result["performance"].is_monotonic_decreasing


def test_selection_of_one_athlete(population):
    """Check that k=1 selects only the highest-performing athlete."""
    result = select_top_k(population, k=1)

    assert result["athlete_id"].tolist() == [1]


def test_selection_of_entire_population(population):
    """Check that selecting everyone still returns the ranking order."""
    result = select_top_k(population, k=len(population))

    assert result["athlete_id"].tolist() == [1, 3, 0, 4, 2]


def test_ties_preserve_original_row_order():
    """Check that tied performances are resolved deterministically."""
    tied_population = pd.DataFrame(
        {
            "athlete_id": [10, 11, 12, 13],
            "performance": [1.0, 2.0, 2.0, 3.0],
        }
    )

    result = select_top_k(tied_population, k=2)

    assert result["athlete_id"].tolist() == [13, 11]


def test_selection_preserves_original_dataframe_index():
    """Check that the selected rows retain their original indices."""
    indexed_population = pd.DataFrame(
        {
            "athlete_id": [0, 1, 2],
            "performance": [1.0, 3.0, 2.0],
        },
        index=[10, 20, 30],
    )

    result = select_top_k(indexed_population, k=2)

    assert result.index.tolist() == [20, 30]


def test_selection_does_not_modify_original_population(population):
    """Check that the original DataFrame remains unchanged."""
    original_population = population.copy(deep=True)

    select_top_k(population, k=2)

    pd.testing.assert_frame_equal(population, original_population)


def test_selection_returns_independent_copy(population):
    """Check that modifying the result does not modify the input."""
    result = select_top_k(population, k=2)

    result.loc[result.index[0], "performance"] = -100.0

    assert population.loc[1, "performance"] == 1.7


def test_selection_can_use_custom_score_column():
    """Check that selection can use a column other than performance."""
    custom_population = pd.DataFrame(
        {
            "athlete_id": [0, 1, 2],
            "talent_score": [0.3, 0.9, 0.6],
        }
    )

    result = select_top_k(
        custom_population,
        k=2,
        score_column="talent_score",
    )

    assert result["athlete_id"].tolist() == [1, 2]


def test_selection_accepts_numpy_integer(population):
    """Check that NumPy integer values are accepted for k."""
    result = select_top_k(population, k=np.int64(2))

    assert len(result) == 2


def test_non_dataframe_input_raises_type_error():
    """A non-DataFrame input should raise TypeError."""
    with pytest.raises(TypeError):
        select_top_k([1.0, 2.0, 3.0], k=2)


def test_empty_population_raises_value_error():
    """An empty population should raise ValueError."""
    empty_population = pd.DataFrame(columns=["performance"])

    with pytest.raises(ValueError):
        select_top_k(empty_population, k=1)


@pytest.mark.parametrize("k", [0, -1, 6])
def test_invalid_selection_size_raises_value_error(population, k):
    """Invalid selection sizes should raise ValueError."""
    with pytest.raises(ValueError):
        select_top_k(population, k=k)


@pytest.mark.parametrize("k", [1.5, "2", True])
def test_non_integer_selection_size_raises_type_error(population, k):
    """A non-integer selection size should raise TypeError."""
    with pytest.raises(TypeError):
        select_top_k(population, k=k)


@pytest.mark.parametrize("score_column", [None, 1, True])
def test_non_string_score_column_raises_type_error(population, score_column):
    """A non-string score column should raise TypeError."""
    with pytest.raises(TypeError):
        select_top_k(population, k=2, score_column=score_column)


def test_empty_score_column_name_raises_value_error(population):
    """An empty score column name should raise ValueError."""
    with pytest.raises(ValueError):
        select_top_k(population, k=2, score_column="")


def test_missing_score_column_raises_value_error(population):
    """A missing score column should raise ValueError."""
    with pytest.raises(ValueError):
        select_top_k(population, k=2, score_column="unknown_score")


def test_non_numeric_scores_raise_type_error():
    """Non-numeric performance scores should raise TypeError."""
    invalid_population = pd.DataFrame(
        {
            "athlete_id": [0, 1],
            "performance": ["high", "low"],
        }
    )

    with pytest.raises(TypeError):
        select_top_k(invalid_population, k=1)


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, np.nan, 3.0],
        [1.0, np.inf, 3.0],
        [1.0, -np.inf, 3.0],
    ],
)
def test_non_finite_scores_raise_value_error(scores):
    """Non-finite performance scores should raise ValueError."""
    invalid_population = pd.DataFrame(
        {
            "athlete_id": range(len(scores)),
            "performance": scores,
        }
    )

    with pytest.raises(ValueError):
        select_top_k(invalid_population, k=1)
