"""Configuration of the multi-stage athlete selection model."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _validate_integer(name: str, value: int, minimum: int = 0) -> int:
    """Validate an integer parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")

    return value


def _validate_float(name: str, value: float) -> float:
    """Validate a finite numeric parameter and normalize it to float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric value.")

    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the multi-stage youth talent selection model.

    The notation follows the model specification in ``report/report.md``.

    Parameters
    ----------
    n_athletes:
        Number of athletes in the initial cohort.
    stage_fractions:
        Fractions of the initial cohort selected at each stage.
    talent_weight:
        Weight assigned to latent talent in observed performance.
    maturation_weight:
        Initial weight assigned to biological maturation.
    maturation_decay:
        Multiplicative decay of the maturation weight across stages.
    maturation_corr:
        Correlation between relative age and biological maturation.
    adult_maturation_weight:
        Share of biological maturation that persists into adult potential.
    exposure_weight:
        Weight assigned to accumulated training exposure.
    exposure_gain:
        Exposure units gained for every previous selection.
    noise_sd:
        Standard deviation of observed-performance noise.
    dropout_unselected:
        Probability that an unselected athlete leaves the system.
    top_talent_fraction:
        Fraction of athletes considered high-talent for fairness metrics.
    seed:
        Random seed for reproducible simulation runs.
    """

    n_athletes: int = 20_000
    stage_fractions: tuple[float, ...] = (0.30, 0.20, 0.10)
    talent_weight: float = 1.0
    maturation_weight: float = 2.0
    maturation_decay: float = 0.7
    maturation_corr: float = 0.3
    adult_maturation_weight: float = 0.0
    exposure_weight: float = 0.0
    exposure_gain: float = 1.0
    noise_sd: float = 1.0
    dropout_unselected: float = 0.2
    top_talent_fraction: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate and normalize all configuration parameters."""
        object.__setattr__(
            self,
            "n_athletes",
            _validate_integer("n_athletes", self.n_athletes, minimum=1),
        )
        object.__setattr__(
            self,
            "seed",
            _validate_integer("seed", self.seed, minimum=0),
        )

        if isinstance(self.stage_fractions, (str, bytes)) or not isinstance(
            self.stage_fractions, Sequence
        ):
            raise TypeError("stage_fractions must be a sequence of numeric values.")

        if not self.stage_fractions:
            raise ValueError("stage_fractions must not be empty.")

        fractions = tuple(
            _validate_float("stage fraction", fraction) for fraction in self.stage_fractions
        )

        if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
            raise ValueError("stage fractions must be in the interval (0, 1].")

        if any(later > earlier for earlier, later in zip(fractions, fractions[1:], strict=False)):
            raise ValueError("stage_fractions must be non-increasing.")

        if any(round(fraction * self.n_athletes) < 1 for fraction in fractions):
            raise ValueError("every stage must select at least one athlete.")

        object.__setattr__(self, "stage_fractions", fractions)

        non_negative_parameters = (
            "talent_weight",
            "maturation_weight",
            "adult_maturation_weight",
            "exposure_weight",
            "exposure_gain",
            "noise_sd",
        )

        for name in non_negative_parameters:
            value = _validate_float(name, getattr(self, name))

            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

            object.__setattr__(self, name, value)

        maturation_decay = _validate_float(
            "maturation_decay",
            self.maturation_decay,
        )

        if not 0.0 <= maturation_decay <= 1.0:
            raise ValueError("maturation_decay must be in the interval [0, 1].")

        object.__setattr__(self, "maturation_decay", maturation_decay)

        maturation_corr = _validate_float(
            "maturation_corr",
            self.maturation_corr,
        )

        if not 0.0 <= maturation_corr < 1.0:
            raise ValueError("maturation_corr must be in the interval [0, 1).")

        object.__setattr__(self, "maturation_corr", maturation_corr)

        dropout = _validate_float(
            "dropout_unselected",
            self.dropout_unselected,
        )

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout_unselected must be in the interval [0, 1].")

        object.__setattr__(self, "dropout_unselected", dropout)

        top_fraction = _validate_float(
            "top_talent_fraction",
            self.top_talent_fraction,
        )

        if not 0.0 < top_fraction < 1.0:
            raise ValueError("top_talent_fraction must be in the interval (0, 1).")

        object.__setattr__(self, "top_talent_fraction", top_fraction)

    @property
    def n_stages(self) -> int:
        """Return the number of selection stages."""
        return len(self.stage_fractions)
