"""Multi-stage athlete selection pipeline."""

import numpy as np
import pandas as pd

from rae.performance import generate_performance
from rae.selection import select_top_k


def _validate_non_negative_parameter(name: str, value: float) -> float:
    """Validate a finite, non-negative numeric parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a numeric value.")

    value = float(value)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return value


def simulate_pipeline(
    population: pd.DataFrame,
    stage_fractions: tuple[float, ...] = (0.30, 0.20, 0.10),
    beta_m: float = 2.0,
    maturation_decay: float = 0.7,
    sigma_p: float = 1.0,
    exposure_weight: float = 0.0,
    exposure_gain: float = 1.0,
    dropout_unselected: float = 0.2,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate a multi-stage athlete selection process.

    Parameters
    ----------
    population:
        Initial synthetic population containing athlete_id, talent and maturity.
    stage_fractions:
        Fractions of the initial population selected at each stage.
        Must be positive and non-increasing.
    beta_m:
        Initial weight assigned to biological maturity.
    maturation_decay:
        Multiplicative factor by which the maturity weight decreases at each stage.
    sigma_p:
        Standard deviation of performance noise.
    exposure_weight:
        Weight assigned to accumulated training exposure.
        A value of zero disables the performance feedback effect.
    exposure_gain:
        Exposure units gained for every previous selection.
    dropout_unselected:
        Probability that an unselected athlete leaves the system after a stage.
    rng:
        NumPy random number generator. If None, a new generator is created.

    Returns
    -------
    pd.DataFrame
        Long-format simulation log. Each row represents one athlete
        observed at one stage.
    """
    # Validate the input population.
    if not isinstance(population, pd.DataFrame):
        raise TypeError("population must be a pandas DataFrame.")

    if population.empty:
        raise ValueError("population must not be empty.")

    required_columns = {"athlete_id", "talent", "maturity"}

    if not required_columns.issubset(population.columns):
        raise ValueError("population must contain 'athlete_id', 'talent' and 'maturity'.")

    if population["athlete_id"].isna().any():
        raise ValueError("athlete_id must not contain missing values.")

    if not population["athlete_id"].is_unique:
        raise ValueError("athlete_id values must be unique.")

    reserved_columns = {
        "stage",
        "times_selected_before",
        "exposure",
        "performance_noise",
        "performance",
        "selected",
        "dropped_out",
    }

    conflicts = reserved_columns.intersection(population.columns)

    if conflicts:
        raise ValueError(f"population contains reserved output columns: {sorted(conflicts)}.")

    # Validate the stage fractions.
    if not isinstance(stage_fractions, (tuple, list)):
        raise TypeError("stage_fractions must be a tuple or list.")

    if not stage_fractions:
        raise ValueError("stage_fractions must not be empty.")

    fractions = tuple(
        _validate_non_negative_parameter("stage fraction", fraction) for fraction in stage_fractions
    )

    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("stage fractions must be in the interval (0, 1].")

    if any(later > earlier for earlier, later in zip(fractions, fractions[1:], strict=False)):
        raise ValueError("stage fractions must be non-increasing.")

    n = len(population)
    n_stages = len(fractions)

    if any(round(fraction * n) < 1 for fraction in fractions):
        raise ValueError("every stage must select at least one athlete.")

    # Validate model parameters.
    beta_m = _validate_non_negative_parameter("beta_m", beta_m)
    sigma_p = _validate_non_negative_parameter("sigma_p", sigma_p)
    exposure_weight = _validate_non_negative_parameter("exposure_weight", exposure_weight)
    exposure_gain = _validate_non_negative_parameter("exposure_gain", exposure_gain)

    maturation_decay = _validate_non_negative_parameter("maturation_decay", maturation_decay)

    if maturation_decay > 1.0:
        raise ValueError("maturation_decay must be in the interval [0, 1].")

    if isinstance(dropout_unselected, (bool, np.bool_)) or not isinstance(
        dropout_unselected, (int, float, np.integer, np.floating)
    ):
        raise TypeError("dropout_unselected must be a numeric value.")

    dropout_unselected = float(dropout_unselected)

    if not np.isfinite(dropout_unselected) or not 0.0 <= dropout_unselected <= 1.0:
        raise ValueError("dropout_unselected must be in the interval [0, 1].")

    # Initialize the random number generator.
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator or None.")

    # Create independent random streams for each stage and for dropout.
    child_seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_stages + 1,
        dtype=np.uint32,
    )

    stage_rngs = [np.random.default_rng(int(seed)) for seed in child_seeds[:n_stages]]

    dropout_rng = np.random.default_rng(int(child_seeds[-1]))

    # Pre-generate performance and dropout randomness for the entire population.
    # This keeps random draws independent of which athletes remain in the pool.
    stage_performances = []

    for stage, stage_rng in enumerate(stage_rngs):
        stage_beta_m = beta_m * maturation_decay**stage

        stage_result = generate_performance(
            population,
            beta_m=stage_beta_m,
            sigma_p=sigma_p,
            rng=stage_rng,
        )

        stage_performances.append(
            (
                stage_result["performance"].to_numpy(),
                stage_result["performance_noise"].to_numpy(),
            )
        )

    dropout_draws = dropout_rng.random((n_stages, n))

    # Initialize the simulation state.
    in_pool = np.ones(n, dtype=bool)
    times_selected = np.zeros(n, dtype=int)
    stage_records = []

    # Simulate each selection stage.
    for stage, fraction in enumerate(fractions):
        pool_positions = np.flatnonzero(in_pool)

        # The selection quota is always calculated from the initial population.
        k = round(fraction * n)

        if k > len(pool_positions):
            raise ValueError(
                f"stage {stage}: cannot select {k} athletes from a pool of {len(pool_positions)}."
            )

        # Build the DataFrame for the current candidate pool.
        stage_data = population.iloc[pool_positions].copy().reset_index(drop=True)

        previous_selections = times_selected[pool_positions]
        exposure = exposure_gain * previous_selections

        base_performance, performance_noise = stage_performances[stage]

        # Add stage-specific variables.
        stage_data["stage"] = stage
        stage_data["times_selected_before"] = previous_selections
        stage_data["exposure"] = exposure
        stage_data["performance_noise"] = performance_noise[pool_positions]

        # Calculate performance including the exposure feedback effect.
        stage_data["performance"] = base_performance[pool_positions] + exposure_weight * exposure

        # Select the k highest-performing athletes.
        selected_data = select_top_k(
            stage_data,
            k=k,
            score_column="performance",
        )

        selected_positions = selected_data.index.to_numpy(dtype=int)

        selected_mask = np.zeros(len(pool_positions), dtype=bool)
        selected_mask[selected_positions] = True

        # Unselected athletes may leave the system.
        dropout_mask = ~selected_mask & (dropout_draws[stage, pool_positions] < dropout_unselected)

        stage_data["selected"] = selected_mask
        stage_data["dropped_out"] = dropout_mask

        stage_records.append(stage_data)

        # Update the number of previous selections.
        selected_global_positions = pool_positions[selected_mask]
        times_selected[selected_global_positions] += 1

        # Remove athletes who drop out after this stage.
        dropped_global_positions = pool_positions[dropout_mask]
        in_pool[dropped_global_positions] = False

    # Combine all stage records into one long-format DataFrame.
    return pd.concat(stage_records, ignore_index=True)
