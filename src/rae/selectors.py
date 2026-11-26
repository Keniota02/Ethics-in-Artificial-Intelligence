"""Selection strategies for the multi-stage athlete selection pipeline."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from rae.selection import select_top_k


@dataclass(frozen=True)
class StageContext:
    """Information available to a selector at a given stage.

    Attributes
    ----------
    stage:
        Current stage, starting from zero.
    n_stages:
        Total number of stages in the simulation.
    candidates:
        DataFrame containing the athletes currently in the selection pool.
    history:
        Observed performance history, with shape (stage + 1, n_candidates).
        The last row contains current-stage performance.
    judgement_noise:
        Standard normal random values for subjective selector noise,
        aligned with the candidate rows.
    """

    stage: int
    n_stages: int
    candidates: pd.DataFrame
    history: np.ndarray
    judgement_noise: np.ndarray
    adult_maturation_weight: float = 0.0

    def __post_init__(self):
        """Validate the consistency of the stage context."""
        if isinstance(self.stage, bool) or not isinstance(self.stage, (int, np.integer)):
            raise TypeError("stage must be an integer.")

        if isinstance(self.n_stages, bool) or not isinstance(self.n_stages, (int, np.integer)):
            raise TypeError("n_stages must be an integer.")

        if self.n_stages < 1 or not 0 <= self.stage < self.n_stages:
            raise ValueError("stage must be between 0 and n_stages - 1.")

        if not isinstance(self.candidates, pd.DataFrame):
            raise TypeError("candidates must be a pandas DataFrame.")

        if self.candidates.empty:
            raise ValueError("candidates must not be empty.")

        history = np.asarray(self.history, dtype=float)
        judgement_noise = np.asarray(self.judgement_noise, dtype=float)

        expected_history_shape = (self.stage + 1, len(self.candidates))

        if history.shape != expected_history_shape:
            raise ValueError(f"history must have shape {expected_history_shape}.")

        if judgement_noise.shape != (len(self.candidates),):
            raise ValueError("judgement_noise must contain one value per candidate.")

        if not np.isfinite(history).all():
            raise ValueError("history must contain only finite values.")

        if not np.isfinite(judgement_noise).all():
            raise ValueError("judgement_noise must contain only finite values.")

        if isinstance(self.adult_maturation_weight, (bool, np.bool_)) or not isinstance(
            self.adult_maturation_weight,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("adult_maturation_weight must be a numeric value.")

        adult_maturation_weight = float(self.adult_maturation_weight)

        if not np.isfinite(adult_maturation_weight) or adult_maturation_weight < 0.0:
            raise ValueError("adult_maturation_weight must be finite and non-negative.")

        object.__setattr__(
            self,
            "adult_maturation_weight",
            adult_maturation_weight,
        )

    @property
    def observed(self) -> np.ndarray:
        """Return the performance observed at the current stage."""
        return np.asarray(self.history, dtype=float)[-1]


class Selector(Protocol):
    """Interface shared by all selection strategies."""

    def scores(self, context: StageContext) -> np.ndarray:
        """Return one selection score per candidate."""
        ...


@dataclass(frozen=True)
class PerformanceSelector:
    """Memoryless selector that ranks athletes by current observed performance."""

    def scores(self, context: StageContext) -> np.ndarray:
        return context.observed


@dataclass(frozen=True)
class OracleSelector:
    """Ideal benchmark selector based on adult talent potential.

    The Oracle has access to latent talent and biological maturity,
    which are not directly observable by real-world selectors.

    Its target is:

        adult_potential = talent + adult_maturation_weight * maturity
    """

    def scores(self, context: StageContext) -> np.ndarray:
        """Return each candidate's latent adult potential."""
        required_columns = {"talent", "maturity"}

        if not required_columns.issubset(context.candidates.columns):
            raise ValueError("candidates must contain 'talent' and 'maturity'.")

        talent = context.candidates["talent"].to_numpy(dtype=float)
        maturity = context.candidates["maturity"].to_numpy(dtype=float)

        if not np.isfinite(talent).all() or not np.isfinite(maturity).all():
            raise ValueError("talent and maturity must contain only finite values.")

        return talent + context.adult_maturation_weight * maturity


@dataclass(frozen=True)
class ScoutSelector:
    """Human-like scout with memory, subjective judgement noise and maturity bias.

    Parameters
    ----------
    memory:
        Weight assigned to past observations.
        Zero ignores history; one assigns equal weight to all observations.
    noise_sd:
        Standard deviation of subjective judgement noise.
    maturation_bias:
        Additional weight assigned directly to biological maturity.
        Positive values favour more mature athletes.
    """

    memory: float = 1.0
    noise_sd: float = 0.0
    maturation_bias: float = 0.0

    def __post_init__(self):
        """Validate the scout parameters."""
        for name, value in {
            "memory": self.memory,
            "noise_sd": self.noise_sd,
            "maturation_bias": self.maturation_bias,
        }.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name} must be a numeric value.")

            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if not 0.0 <= self.memory <= 1.0:
            raise ValueError("memory must be between 0 and 1.")

        if self.noise_sd < 0.0:
            raise ValueError("noise_sd must be non-negative.")

    def scores(self, context: StageContext) -> np.ndarray:
        """Calculate scout scores for the current candidate pool."""
        history = np.asarray(context.history, dtype=float)

        # Older observations receive geometrically decreasing weights.
        lags = np.arange(context.stage, -1, -1, dtype=float)
        weights = self.memory**lags

        historical_score = weights @ history / weights.sum()

        score = historical_score.copy()

        # Add an explicit preference for biological maturity.
        if self.maturation_bias != 0.0:
            if "maturity" not in context.candidates.columns:
                raise ValueError("candidates must contain 'maturity' when maturation_bias is used.")

            maturity = context.candidates["maturity"].to_numpy(dtype=float)

            if not np.isfinite(maturity).all():
                raise ValueError("maturity must contain only finite values.")

            score += self.maturation_bias * maturity

        # Add subjective judgement noise.
        if self.noise_sd > 0.0:
            noise = np.asarray(context.judgement_noise, dtype=float)
            score += self.noise_sd * noise

        return score


def choose(
    selector: Selector,
    context: StageContext,
    k: int,
) -> np.ndarray:
    """Return the positions of the k selected candidates.

    Scores are ranked in descending order. Ties preserve the original
    candidate order through the stable top-k selection function.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")

    if not 1 <= k <= len(context.candidates):
        raise ValueError("k must be between 1 and the candidate count.")

    scores = np.asarray(selector.scores(context), dtype=float)

    if scores.shape != (len(context.candidates),):
        raise ValueError("selector must return exactly one score per candidate.")

    if not np.isfinite(scores).all():
        raise ValueError("selector scores must contain only finite values.")

    # Use the existing top-k implementation instead of duplicating ranking logic.
    score_frame = pd.DataFrame({"score": scores})

    selected = select_top_k(
        score_frame,
        k=k,
        score_column="score",
    )

    return selected.index.to_numpy(dtype=int)
