"""Tests for synthetic athlete population generation."""

import numpy as np
import pandas as pd
import pytest

from rae.population import generate_population

EXPECTED_COLUMNS = [
    "athlete_id",
    "talent",
    "birth_position",
    "birth_quarter",
    "relative_age",
    "epsilon",
    "maturity",
]


def test_population_has_expected_structure():
    """Check the shape, columns and athlete identifiers."""
    population = generate_population(
        n=100,
        rho=0.3,
        rng=np.random.default_rng(42),
    )

    assert isinstance(population, pd.DataFrame)
    assert population.shape == (100, 7)
    assert population.columns.tolist() == EXPECTED_COLUMNS
    assert population["athlete_id"].tolist() == list(range(100))


def test_birth_positions_and_quarters_are_valid():
    """Check birth positions and their corresponding quarters."""
    population = generate_population(
        n=1000,
        rho=0.3,
        rng=np.random.default_rng(42),
    )

    assert population["birth_position"].between(0.0, 1.0, inclusive="left").all()

    expected_quarters = np.floor(4.0 * population["birth_position"]).astype(int) + 1

    pd.testing.assert_series_equal(
        population["birth_quarter"],
        expected_quarters.rename("birth_quarter"),
    )

    assert population["birth_quarter"].between(1, 4).all()


def test_population_is_reproducible_with_same_seed():
    """Check that equal random seeds generate equal populations."""
    population_1 = generate_population(
        n=100,
        rho=0.3,
        rng=np.random.default_rng(42),
    )

    population_2 = generate_population(
        n=100,
        rho=0.3,
        rng=np.random.default_rng(42),
    )

    pd.testing.assert_frame_equal(population_1, population_2)


def test_latent_variables_follow_expected_distributions():
    """Check approximate statistical properties of the generated variables."""
    rho = 0.3

    population = generate_population(
        n=20_000,
        rho=rho,
        rng=np.random.default_rng(42),
    )

    # Talent should approximately follow a standard normal distribution.
    assert abs(population["talent"].mean()) < 0.03
    assert abs(population["talent"].std(ddof=0) - 1.0) < 0.03

    # Birth position should approximately follow a uniform distribution.
    assert abs(population["birth_position"].mean() - 0.5) < 0.01

    # Relative age and epsilon should be approximately standardized.
    assert abs(population["relative_age"].mean()) < 0.03
    assert abs(population["relative_age"].std(ddof=0) - 1.0) < 0.03

    assert abs(population["epsilon"].mean()) < 0.03
    assert abs(population["epsilon"].std(ddof=0) - 1.0) < 0.03

    # Maturity should be approximately standardized.
    assert abs(population["maturity"].mean()) < 0.03
    assert abs(population["maturity"].std(ddof=0) - 1.0) < 0.03

    # Relative age and maturity should have approximately correlation rho.
    correlation = population["relative_age"].corr(population["maturity"])

    assert correlation == pytest.approx(rho, abs=0.02)


def test_maturity_when_rho_is_zero():
    """When rho is zero, maturity should equal the individual residual."""
    population = generate_population(
        n=1000,
        rho=0.0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(
        population["maturity"],
        population["epsilon"],
    )


def test_maturity_when_rho_is_one():
    """When rho is one, maturity should equal relative age."""
    population = generate_population(
        n=1000,
        rho=1.0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(
        population["maturity"],
        population["relative_age"],
    )


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_population_size_raises_value_error(n):
    """A non-positive population size should raise ValueError."""
    with pytest.raises(ValueError):
        generate_population(n=n, rho=0.3)


@pytest.mark.parametrize("n", [1.5, "100", True])
def test_non_integer_population_size_raises_type_error(n):
    """A non-integer population size should raise TypeError."""
    with pytest.raises(TypeError):
        generate_population(n=n, rho=0.3)


@pytest.mark.parametrize("rho", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_rho_raises_value_error(rho):
    """An invalid correlation parameter should raise ValueError."""
    with pytest.raises(ValueError):
        generate_population(n=100, rho=rho)


@pytest.mark.parametrize("rho", ["0.3", None, True])
def test_non_numeric_rho_raises_type_error(rho):
    """A non-numeric correlation parameter should raise TypeError."""
    with pytest.raises(TypeError):
        generate_population(n=100, rho=rho)


def test_invalid_random_generator_raises_type_error():
    """An object other than a NumPy Generator should raise TypeError."""
    with pytest.raises(TypeError):
        generate_population(
            n=100,
            rho=0.3,
            rng=42,
        )
