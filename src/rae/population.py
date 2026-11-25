"""Generate synthetic athlete populations for the RAE simulation."""

import numpy as np
import pandas as pd


def generate_population(
    n: int,
    rho: float,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a synthetic population of athletes.

    Parameters
    ----------
    n:
        Number of athletes to generate.
    rho:
        Correlation parameter between relative age and biological maturity.
        Must be in the interval [0, 1].
    rng:
        NumPy random number generator. If None, a new generator is created.

    Returns
    -------
    pd.DataFrame
        One row per athlete, containing talent, birth position,
        birth quarter, relative age, individual residual and maturity.
    """
    # Validate the population size.
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be a positive integer.")

    if n < 1:
        raise ValueError("n must be greater than or equal to 1.")

    # Validate the correlation parameter.
    if isinstance(rho, (bool, np.bool_)) or not isinstance(
        rho, (int, float, np.integer, np.floating)
    ):
        raise TypeError("rho must be a numeric value.")

    rho = float(rho)

    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be a finite value between 0 and 1.")

    # Create or validate the random number generator.
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator or None.")

    # Generate independent latent variables.
    talent = rng.normal(loc=0.0, scale=1.0, size=n)

    birth_position = rng.uniform(
        low=0.0,
        high=1.0,
        size=n,
    )

    epsilon = rng.normal(loc=0.0, scale=1.0, size=n)

    # Determine the birth quarter.
    birth_quarter = np.floor(4.0 * birth_position).astype(int) + 1
    birth_quarter = np.minimum(birth_quarter, 4)

    # Calculate standardized relative age.
    relative_age = (0.5 - birth_position) / np.sqrt(1.0 / 12.0)

    # Calculate standardized biological maturity.
    maturity = rho * relative_age + np.sqrt(1.0 - rho**2) * epsilon

    # Assemble the population into a DataFrame.
    population = pd.DataFrame(
        {
            "athlete_id": np.arange(n),
            "talent": talent,
            "birth_position": birth_position,
            "birth_quarter": birth_quarter,
            "relative_age": relative_age,
            "epsilon": epsilon,
            "maturity": maturity,
        }
    )

    return population
