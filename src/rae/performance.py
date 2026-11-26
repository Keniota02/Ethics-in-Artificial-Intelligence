"""Generate observed athlete performances."""

import numpy as np
import pandas as pd


def generate_performance(
    population: pd.DataFrame,
    beta_m: float = 0.5,
    sigma_p: float = 1.0,
    rng: np.random.Generator | None = None,
    *,
    talent_weight: float = 1.0,
) -> pd.DataFrame:
    """Generate observed performances for a synthetic athlete population.

    Parameters
    ----------
    population:
        DataFrame containing the latent talent and maturity of each athlete.
    beta_m:
        Weight assigned to biological maturity.
    sigma_p:
        Standard deviation of the individual performance noise.
    rng:
        NumPy random number generator. If None, a new generator is created.
    talent_weight:
        Weight assigned to latent talent.

    Returns
    -------
    pd.DataFrame
        Copy of the input population with two additional columns:
        performance_noise and performance.
    """
    # Validate the input population.
    if not isinstance(population, pd.DataFrame):
        raise TypeError("population must be a pandas DataFrame.")

    required_columns = {"talent", "maturity"}

    if not required_columns.issubset(population.columns):
        raise ValueError("population must contain 'talent' and 'maturity' columns.")

    if population.empty:
        raise ValueError("population must not be empty.")

    talent = population["talent"].to_numpy(dtype=float)
    maturity = population["maturity"].to_numpy(dtype=float)

    if not np.isfinite(talent).all() or not np.isfinite(maturity).all():
        raise ValueError("talent and maturity must contain finite values.")

    # Validate the model parameters.
    parameters = {
        "talent_weight": talent_weight,
        "beta_m": beta_m,
        "sigma_p": sigma_p,
    }

    for name, value in parameters.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a numeric value.")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    talent_weight = float(talent_weight)
    beta_m = float(beta_m)
    sigma_p = float(sigma_p)

    if talent_weight < 0.0:
        raise ValueError("talent_weight must be greater than or equal to 0.")

    if beta_m < 0.0:
        raise ValueError("beta_m must be greater than or equal to 0.")

    if sigma_p < 0.0:
        raise ValueError("sigma_p must be greater than or equal to 0.")

    # Create or validate the random number generator.
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator or None.")

    # Generate independent performance noise.
    performance_noise = rng.normal(
        loc=0.0,
        scale=sigma_p,
        size=len(population),
    )

    # Calculate observed performance.
    performance = talent_weight * talent + beta_m * maturity + performance_noise

    # Return a copy without modifying the original population.
    result = population.copy()
    result["performance_noise"] = performance_noise
    result["performance"] = performance

    return result
