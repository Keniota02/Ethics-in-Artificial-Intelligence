"""Tests for observed athlete performance generation."""

import numpy as np
import pandas as pd
import pytest

from rae.performance import generate_performance
from rae.population import generate_population


def create_test_population(n=100, seed=42):
    """Create a reproducible synthetic population for testing."""
    return generate_population(
        n=n,
        rho=0.3,
        rng=np.random.default_rng(seed),
    )


def test_performance_has_expected_structure():
    """Check the output columns and number of athletes."""
    population = create_test_population(n=100)

    result = generate_performance(
        population,
        rng=np.random.default_rng(43),
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (100, 9)

    assert "performance_noise" in result.columns
    assert "performance" in result.columns

    assert result["athlete_id"].tolist() == population["athlete_id"].tolist()


def test_performance_is_calculated_correctly():
    """Check that observed performance follows the specified formula."""
    population = create_test_population(n=100)

    beta_m = 0.5
    sigma_p = 1.0

    result = generate_performance(
        population,
        beta_m=beta_m,
        sigma_p=sigma_p,
        rng=np.random.default_rng(43),
    )

    expected_performance = (
        population["talent"] + beta_m * population["maturity"] + result["performance_noise"]
    )

    np.testing.assert_allclose(
        result["performance"],
        expected_performance,
    )


def test_performance_does_not_modify_original_population():
    """Check that the input population remains unchanged."""
    population = create_test_population(n=100)
    original_population = population.copy(deep=True)

    generate_performance(
        population,
        rng=np.random.default_rng(43),
    )

    pd.testing.assert_frame_equal(population, original_population)


def test_performance_is_reproducible_with_same_seed():
    """Check that equal random seeds generate equal performances."""
    population = create_test_population(n=100)

    result_1 = generate_performance(
        population,
        rng=np.random.default_rng(43),
    )

    result_2 = generate_performance(
        population,
        rng=np.random.default_rng(43),
    )

    pd.testing.assert_frame_equal(result_1, result_2)


def test_performance_noise_has_expected_distribution():
    """Check that performance noise is approximately standardized."""
    population = create_test_population(n=20_000)

    result = generate_performance(
        population,
        beta_m=0.5,
        sigma_p=1.0,
        rng=np.random.default_rng(43),
    )

    noise = result["performance_noise"]

    assert abs(noise.mean()) < 0.03
    assert abs(noise.std(ddof=0) - 1.0) < 0.03


def test_performance_has_expected_mean_and_variance():
    """Check the theoretical mean and variance of observed performance."""
    population = create_test_population(n=20_000)

    result = generate_performance(
        population,
        beta_m=0.5,
        sigma_p=1.0,
        rng=np.random.default_rng(43),
    )

    performance = result["performance"]

    assert abs(performance.mean()) < 0.03
    assert abs(performance.var(ddof=0) - 2.25) < 0.08


def test_performance_correlation_with_relative_age():
    """Check the expected correlation between relative age and performance."""
    population = create_test_population(n=20_000)

    result = generate_performance(
        population,
        beta_m=0.5,
        sigma_p=1.0,
        rng=np.random.default_rng(43),
    )

    correlation = result["relative_age"].corr(result["performance"])

    # The theoretical correlation is 0.15 / 1.5 = 0.10.
    assert correlation == pytest.approx(0.10, abs=0.02)


def test_zero_maturity_weight_removes_maturity_contribution():
    """When beta_m is zero, maturity does not directly affect performance."""
    population = create_test_population(n=100)

    result = generate_performance(
        population,
        beta_m=0.0,
        sigma_p=1.0,
        rng=np.random.default_rng(43),
    )

    expected_performance = population["talent"] + result["performance_noise"]

    np.testing.assert_allclose(
        result["performance"],
        expected_performance,
    )


def test_zero_noise_produces_deterministic_performance():
    """When sigma_p is zero, performance contains no random noise."""
    population = create_test_population(n=100)

    result = generate_performance(
        population,
        beta_m=0.5,
        sigma_p=0.0,
        rng=np.random.default_rng(43),
    )

    np.testing.assert_allclose(
        result["performance_noise"],
        0.0,
    )

    expected_performance = population["talent"] + 0.5 * population["maturity"]

    np.testing.assert_allclose(
        result["performance"],
        expected_performance,
    )


def test_non_dataframe_population_raises_type_error():
    """A non-DataFrame input should raise TypeError."""
    with pytest.raises(TypeError):
        generate_performance([1, 2, 3])


@pytest.mark.parametrize(
    "population",
    [
        pd.DataFrame({"talent": [0.1, 0.2]}),
        pd.DataFrame({"maturity": [0.3, 0.4]}),
    ],
)
def test_missing_required_columns_raise_value_error(population):
    """A population missing required columns should raise ValueError."""
    with pytest.raises(ValueError):
        generate_performance(population)


def test_empty_population_raises_value_error():
    """An empty population should raise ValueError."""
    population = pd.DataFrame(columns=["talent", "maturity"])

    with pytest.raises(ValueError):
        generate_performance(population)


def test_non_finite_population_values_raise_value_error():
    """Non-finite talent or maturity values should raise ValueError."""
    population = pd.DataFrame(
        {
            "talent": [0.5, np.nan],
            "maturity": [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError):
        generate_performance(population)


@pytest.mark.parametrize("beta_m", [-0.1, -1.0])
def test_negative_beta_m_raises_value_error(beta_m):
    """A negative maturity weight should raise ValueError."""
    population = create_test_population()

    with pytest.raises(ValueError):
        generate_performance(population, beta_m=beta_m)


@pytest.mark.parametrize("sigma_p", [-0.1, -1.0])
def test_negative_sigma_p_raises_value_error(sigma_p):
    """A negative noise standard deviation should raise ValueError."""
    population = create_test_population()

    with pytest.raises(ValueError):
        generate_performance(population, sigma_p=sigma_p)


@pytest.mark.parametrize(
    "parameter",
    ["beta_m", "sigma_p"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_parameters_raise_value_error(parameter, value):
    """Non-finite model parameters should raise ValueError."""
    population = create_test_population()

    with pytest.raises(ValueError):
        generate_performance(population, **{parameter: value})


@pytest.mark.parametrize(
    "parameter",
    ["beta_m", "sigma_p"],
)
@pytest.mark.parametrize("value", ["invalid", None, True])
def test_non_numeric_parameters_raise_type_error(parameter, value):
    """Non-numeric model parameters should raise TypeError."""
    population = create_test_population()

    with pytest.raises(TypeError):
        generate_performance(population, **{parameter: value})


def test_invalid_random_generator_raises_type_error():
    """An object other than a NumPy Generator should raise TypeError."""
    population = create_test_population()

    with pytest.raises(TypeError):
        generate_performance(
            population,
            rng=43,
        )
