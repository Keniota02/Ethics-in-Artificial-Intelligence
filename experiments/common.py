"""Shared settings and utilities for reproducible simulation experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rae.selectors import ScoutSelector

# Historical selectors used to generate training cohorts.
SCOUT = ScoutSelector(
    memory=1.0,
    noise_sd=0.5,
)

BIASED_SCOUT = ScoutSelector(
    memory=1.0,
    noise_sd=0.5,
    maturation_bias=0.5,
)

HISTORICAL = {
    "scout": SCOUT,
    "biased scout": BIASED_SCOUT,
}

# Feedback loop off / on.
EXPOSURES = (0.0, 1.0)

# Label regimes used in RQ1.
REGIMES = ("decisions", "later_performance")

# Number of independent historical cohorts used for model training.
N_HISTORICAL_COHORTS = 3

# Directory for generated experiment outputs.
RESULTS_DIR = Path("results")


def mean_ci(
    values,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the sample mean and Student-t confidence interval half-width.

    A half-width of NaN is returned when fewer than two finite observations
    are available, or when the input contains non-finite values.
    """
    observations = np.asarray(values, dtype=float)

    if observations.ndim != 1:
        raise ValueError("values must be a one-dimensional collection.")

    if observations.size == 0:
        raise ValueError("values must contain at least one observation.")

    if isinstance(confidence, (bool, np.bool_)) or not isinstance(
        confidence,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("confidence must be a numeric value.")

    confidence = float(confidence)

    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    mean = float(np.mean(observations))

    if observations.size < 2 or not np.isfinite(observations).all():
        return mean, float("nan")

    critical_value = stats.t.ppf(
        0.5 + confidence / 2.0,
        df=observations.size - 1,
    )

    half_width = critical_value * observations.std(ddof=1) / np.sqrt(observations.size)

    return mean, float(half_width)


def fmt(values) -> str:
    """Format a mean and 95% Student-t confidence interval."""
    mean, half_width = mean_ci(values)
    return f"{mean:7.3f} +/-{half_width:6.3f}"


def save(
    dataframe: pd.DataFrame,
    name: str,
) -> Path:
    """Save experiment results as a CSV file inside the results directory."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty filename.")

    if Path(name).name != name:
        raise ValueError("name must be a filename, not a path.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / name
    dataframe.to_csv(output_path, index=False)

    return output_path
