"""Baseline athlete selection based on observed performance."""

import numpy as np
import pandas as pd


def select_top_k(
    population: pd.DataFrame,
    k: int,
    score_column: str = "performance",
) -> pd.DataFrame:
    """Select the k athletes with the highest observed performance.

    Parameters
    ----------
    population:
        DataFrame containing the candidate athletes and their scores.
    k:
        Number of athletes to select. Must be between 1 and the population size.
    score_column:
        Name of the column used to rank athletes.

    Returns
    -------
    pd.DataFrame
        Copy of the selected athletes, ordered by decreasing score.
        Ties are resolved by preserving the original row order.
    """
    # Validate the input population.
    if not isinstance(population, pd.DataFrame):
        raise TypeError("population must be a pandas DataFrame.")

    if population.empty:
        raise ValueError("population must not be empty.")

    # Validate the score column.
    if not isinstance(score_column, str):
        raise TypeError("score_column must be a string.")

    if not score_column:
        raise ValueError("score_column must not be empty.")

    if score_column not in population.columns:
        raise ValueError(f"population must contain a '{score_column}' column.")

    # Validate the number of selected athletes.
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")

    if not 1 <= k <= len(population):
        raise ValueError("k must be between 1 and the population size.")

    # Convert scores to a numeric NumPy array.
    try:
        scores = population[score_column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("performance scores must be numeric.") from exc

    if not np.isfinite(scores).all():
        raise ValueError("performance scores must contain only finite values.")

    # Select the positions of the k highest scores.
    selected_positions = np.argsort(-scores, kind="stable")[:k]

    # Return a copy of the selected athletes in ranking order.
    return population.iloc[selected_positions].copy()
