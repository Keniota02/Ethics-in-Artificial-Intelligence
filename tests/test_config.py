"""Tests for the centralized simulation configuration."""

from dataclasses import FrozenInstanceError, replace

import pytest

from rae.config import SimConfig


def test_default_configuration_matches_reference_model():
    """Default values must reproduce the original model configuration."""
    config = SimConfig()

    assert config.n_athletes == 20_000
    assert config.stage_fractions == (0.30, 0.20, 0.10)
    assert config.talent_weight == 1.0
    assert config.maturation_weight == 2.0
    assert config.maturation_decay == 0.7
    assert config.maturation_corr == 0.3
    assert config.adult_maturation_weight == 0.0
    assert config.exposure_weight == 0.0
    assert config.exposure_gain == 1.0
    assert config.noise_sd == 1.0
    assert config.dropout_unselected == 0.2
    assert config.top_talent_fraction == 0.10
    assert config.seed == 0
    assert config.n_stages == 3


def test_custom_configuration_accepts_valid_parameters():
    """A configuration must accept valid non-default parameters."""
    config = SimConfig(
        n_athletes=5_000,
        stage_fractions=(0.4, 0.2),
        talent_weight=1.2,
        maturation_weight=1.5,
        maturation_decay=0.6,
        maturation_corr=0.4,
        adult_maturation_weight=0.25,
        exposure_weight=1.0,
        exposure_gain=2.0,
        noise_sd=0.8,
        dropout_unselected=0.1,
        top_talent_fraction=0.15,
        seed=123,
    )

    assert config.n_athletes == 5_000
    assert config.stage_fractions == (0.4, 0.2)
    assert config.n_stages == 2
    assert config.adult_maturation_weight == 0.25
    assert config.seed == 123


def test_stage_fractions_list_is_normalized_to_tuple():
    """A list of stage fractions must be stored as an immutable tuple."""
    config = SimConfig(stage_fractions=[0.4, 0.2])

    assert config.stage_fractions == (0.4, 0.2)
    assert isinstance(config.stage_fractions, tuple)


def test_configuration_supports_dataclasses_replace():
    """Configuration variants must be created with dataclasses.replace."""
    original = SimConfig(seed=10)

    modified = replace(
        original,
        exposure_weight=1.0,
        seed=20,
    )

    assert original.exposure_weight == 0.0
    assert original.seed == 10
    assert modified.exposure_weight == 1.0
    assert modified.seed == 20
    assert modified.n_athletes == original.n_athletes


def test_configuration_is_immutable():
    """A frozen configuration must reject direct field assignment."""
    config = SimConfig()

    with pytest.raises(FrozenInstanceError):
        config.seed = 10


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_population_size_raises_value_error(value):
    """The initial population must contain at least one athlete."""
    with pytest.raises(ValueError):
        SimConfig(n_athletes=value)


@pytest.mark.parametrize("value", [1.5, True, "100"])
def test_non_integer_population_size_raises_type_error(value):
    """Population size must be an integer, not a boolean or string."""
    with pytest.raises(TypeError):
        SimConfig(n_athletes=value)


@pytest.mark.parametrize("value", [-1, -10])
def test_negative_seed_raises_value_error(value):
    """Random seeds must be non-negative."""
    with pytest.raises(ValueError):
        SimConfig(seed=value)


@pytest.mark.parametrize("value", [1.5, True, "42"])
def test_non_integer_seed_raises_type_error(value):
    """Random seeds must be integers."""
    with pytest.raises(TypeError):
        SimConfig(seed=value)


def test_empty_stage_fractions_raise_value_error():
    """At least one selection stage must be configured."""
    with pytest.raises(ValueError):
        SimConfig(stage_fractions=())


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_stage_fractions_outside_valid_interval_raise_value_error(value):
    """Every stage fraction must be in the interval (0, 1]."""
    with pytest.raises(ValueError):
        SimConfig(stage_fractions=(value,))


@pytest.mark.parametrize("value", ["0.3", True])
def test_non_numeric_stage_fraction_raises_type_error(value):
    """Stage fractions must be numeric and cannot be booleans."""
    with pytest.raises(TypeError):
        SimConfig(stage_fractions=(value,))


def test_increasing_stage_fractions_raise_value_error():
    """Selection fractions must be non-increasing across stages."""
    with pytest.raises(ValueError):
        SimConfig(stage_fractions=(0.2, 0.3))


def test_stage_fraction_that_rounds_to_zero_raises_value_error():
    """Every stage must select at least one athlete from the initial cohort."""
    with pytest.raises(ValueError):
        SimConfig(
            n_athletes=2,
            stage_fractions=(0.2, 0.1),
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "talent_weight",
        "maturation_weight",
        "adult_maturation_weight",
        "exposure_weight",
        "exposure_gain",
        "noise_sd",
    ],
)
def test_negative_non_negative_parameters_raise_value_error(parameter):
    """Model weights and noise standard deviation must be non-negative."""
    with pytest.raises(ValueError):
        SimConfig(**{parameter: -0.1})


@pytest.mark.parametrize(
    "parameter",
    [
        "talent_weight",
        "maturation_weight",
        "adult_maturation_weight",
        "exposure_weight",
        "exposure_gain",
        "noise_sd",
    ],
)
def test_non_numeric_model_parameters_raise_type_error(parameter):
    """Model parameters must be numeric and cannot be booleans."""
    with pytest.raises(TypeError):
        SimConfig(**{parameter: True})


@pytest.mark.parametrize(
    "parameter",
    [
        "talent_weight",
        "maturation_weight",
        "adult_maturation_weight",
        "exposure_weight",
        "exposure_gain",
        "noise_sd",
    ],
)
def test_non_finite_model_parameters_raise_value_error(parameter):
    """Model parameters must be finite."""
    with pytest.raises(ValueError):
        SimConfig(**{parameter: float("nan")})


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_maturation_decay_outside_valid_interval_raises_value_error(value):
    """Maturation decay must be in the interval [0, 1]."""
    with pytest.raises(ValueError):
        SimConfig(maturation_decay=value)


@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_maturation_correlation_outside_valid_interval_raises_value_error(value):
    """Maturation correlation must be in the interval [0, 1)."""
    with pytest.raises(ValueError):
        SimConfig(maturation_corr=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_dropout_probability_outside_valid_interval_raises_value_error(value):
    """Dropout probability must be in the interval [0, 1]."""
    with pytest.raises(ValueError):
        SimConfig(dropout_unselected=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_top_talent_fraction_outside_valid_interval_raises_value_error(value):
    """The high-talent fraction must be strictly between zero and one."""
    with pytest.raises(ValueError):
        SimConfig(top_talent_fraction=value)


@pytest.mark.parametrize(
    "parameter",
    [
        "maturation_decay",
        "maturation_corr",
        "dropout_unselected",
        "top_talent_fraction",
    ],
)
def test_non_finite_bounded_parameters_raise_value_error(parameter):
    """Parameters constrained to intervals must also be finite."""
    with pytest.raises(ValueError):
        SimConfig(**{parameter: float("inf")})


def test_integer_zero_and_one_are_valid_boundary_values():
    """Valid boundary values must not be rejected."""
    config = SimConfig(
        n_athletes=1,
        stage_fractions=(1.0,),
        maturation_decay=0.0,
        maturation_corr=0.0,
        dropout_unselected=0.0,
    )

    assert config.n_athletes == 1
    assert config.n_stages == 1
    assert config.maturation_decay == 0.0
    assert config.maturation_corr == 0.0
    assert config.dropout_unselected == 0.0
