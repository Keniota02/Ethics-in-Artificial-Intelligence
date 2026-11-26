"""Machine-learning selectors trained on historical simulation outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rae.selectors import StageContext

Regime = Literal["decisions", "later_performance"]
ModelKind = Literal["linear", "linear_by_stage", "gbm"]

# Keep the feature definitions aligned with the reference implementation.
FEATURES = (
    "observed",
    "history_mean",
    "times_selected",
    "stage",
)
DEFAULT_FEATURES = FEATURES
BIRTH_FEATURE = "birth_position"
VALID_FEATURES = frozenset((*FEATURES, BIRTH_FEATURE))

REQUIRED_COLUMNS = {
    "athlete_id",
    "stage",
    "performance",
    "times_selected_before",
    "selected",
}


def feature_names(
    use_birth_info: bool,
    features: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return ordered model features, optionally appending birth position.

    ``features`` selects a subset of the four standard observable features.
    When ``use_birth_info`` is true, ``birth_position`` is appended, matching
    the feature-selection convention in the reference repository.
    """
    if not isinstance(use_birth_info, (bool, np.bool_)):
        raise TypeError("use_birth_info must be a boolean value.")

    if features is None:
        selected = FEATURES
    else:
        if isinstance(features, str) or not isinstance(features, Sequence):
            raise TypeError("features must be a sequence of feature names or None.")
        selected = tuple(features)

    if any(not isinstance(name, str) for name in selected):
        raise TypeError("every feature name must be a string.")

    if len(set(selected)) != len(selected):
        raise ValueError("features must not contain duplicates.")

    requested_birth_feature = BIRTH_FEATURE in selected
    if requested_birth_feature and not use_birth_info:
        raise ValueError("set use_birth_info=True to include birth_position.")

    base_features = tuple(name for name in selected if name != BIRTH_FEATURE)
    unknown = set(base_features) - VALID_FEATURES
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}.")

    if not base_features and not use_birth_info:
        raise ValueError("a model needs at least one feature.")

    names = (*base_features, BIRTH_FEATURE) if use_birth_info else base_features
    return tuple(names)


def _validate_regime(regime: str) -> None:
    if regime not in ("decisions", "later_performance"):
        raise ValueError("regime must be 'decisions' or 'later_performance'.")


def _numeric_values(
    values: pd.Series,
    column_name: str,
    *,
    non_negative: bool = False,
) -> np.ndarray:
    """Convert a dataframe column to finite numeric values with clear errors."""
    try:
        numeric = values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{column_name} must contain numeric values.") from exc

    if not np.isfinite(numeric).all():
        raise ValueError(f"{column_name} must contain only finite values.")

    if non_negative and (numeric < 0.0).any():
        raise ValueError(f"{column_name} must contain non-negative values.")

    return numeric


def _validate_birth_position(values: pd.Series) -> np.ndarray:
    birth_position = _numeric_values(values, BIRTH_FEATURE)
    if ((birth_position < 0.0) | (birth_position > 1.0)).any():
        raise ValueError("birth_position must be within the interval [0, 1].")
    return birth_position


def build_training_data(
    results: Sequence[pd.DataFrame],
    regime: Regime = "decisions",
    use_birth_info: bool = False,
    features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training features and labels from historical simulation logs.

    Parameters
    ----------
    results:
        Historical simulation logs, one DataFrame per independent cohort.
    regime:
        ``"decisions"`` learns historical selection decisions.
        ``"later_performance"`` learns next-stage observed performance.
    use_birth_info:
        Whether to append continuous ``birth_position`` to the features.
    features:
        Optional subset of the standard observable features. Birth position
        is appended separately when ``use_birth_info`` is true.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target labels, in matching row order.
    """
    _validate_regime(regime)
    names = feature_names(use_birth_info, features)

    if isinstance(results, pd.DataFrame) or not isinstance(results, Sequence):
        raise TypeError("results must be a sequence of simulation DataFrames.")

    if not results:
        raise ValueError("results must contain at least one simulation log.")

    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.Series] = []

    for cohort_index, result in enumerate(results):
        if not isinstance(result, pd.DataFrame):
            raise TypeError("each result must be a pandas DataFrame.")

        missing = REQUIRED_COLUMNS - set(result.columns)
        if missing:
            raise ValueError(
                f"simulation log {cohort_index} is missing columns: {sorted(missing)}."
            )

        if result.empty:
            continue

        log = result.copy()

        if log["athlete_id"].isna().any():
            raise ValueError("athlete_id must not contain missing values.")

        if log["stage"].isna().any() or not pd.api.types.is_integer_dtype(log["stage"]):
            raise TypeError("stage values must be non-missing integers.")

        if (log["stage"] < 0).any():
            raise ValueError("stage values must be non-negative.")
        log["stage"] = log["stage"].astype(np.int64)

        if log.duplicated(["athlete_id", "stage"]).any():
            raise ValueError("each athlete must have at most one row per stage.")

        performance = _numeric_values(log["performance"], "performance")
        previous_selections = _numeric_values(
            log["times_selected_before"],
            "times_selected_before",
            non_negative=True,
        )

        if log["selected"].isna().any() or not log["selected"].isin([True, False, 0, 1]).all():
            raise ValueError("selected must contain only boolean or binary values.")

        log["performance"] = performance
        log["times_selected_before"] = previous_selections
        log = log.sort_values(["athlete_id", "stage"], kind="stable").reset_index(drop=True)

        # Features available to the selector at the current stage only.
        log["observed"] = log["performance"]
        log["history_mean"] = log.groupby("athlete_id")["performance"].transform(
            lambda values: values.expanding().mean()
        )
        log["times_selected"] = log["times_selected_before"]

        if use_birth_info:
            if BIRTH_FEATURE not in log.columns:
                raise ValueError(
                    "simulation logs must contain birth_position when birth information is enabled."
                )
            log[BIRTH_FEATURE] = _validate_birth_position(log[BIRTH_FEATURE])

        if regime == "decisions":
            cohort_features = log.loc[:, list(names)].copy()
            cohort_targets = log["selected"].astype(int).copy()
        else:
            # Pair each stage-s observation with the same athlete's stage-(s+1)
            # performance. Athletes absent from the next stage have no label.
            next_stage = log[["athlete_id", "stage", "performance"]].copy()
            next_stage["stage"] -= 1
            next_stage = next_stage.rename(columns={"performance": "target"})

            matched = log.merge(
                next_stage,
                on=["athlete_id", "stage"],
                how="inner",
                validate="one_to_one",
                sort=False,
            )
            matched = matched.sort_values(["athlete_id", "stage"], kind="stable").reset_index(
                drop=True
            )

            cohort_features = matched.loc[:, list(names)].copy()
            cohort_targets = matched["target"].copy()

        if not cohort_features.empty:
            feature_frames.append(cohort_features)
            target_frames.append(cohort_targets)

    if not feature_frames:
        raise ValueError("no training rows available for the requested label regime.")

    X = pd.concat(feature_frames, ignore_index=True)
    y = pd.concat(target_frames, ignore_index=True)

    try:
        feature_values = X.to_numpy(dtype=float)
        target_values = y.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("training features and labels must be numeric.") from exc

    if not np.isfinite(feature_values).all():
        raise ValueError("training features must contain only finite values.")

    if not np.isfinite(target_values).all():
        raise ValueError("training labels must contain only finite values.")

    return X, y.reset_index(drop=True)


def _make_linear_model(regime: Regime):
    """Create a scaled linear model for the requested label regime."""
    if regime == "decisions":
        estimator = LogisticRegression(max_iter=1000)
    elif regime == "later_performance":
        estimator = LinearRegression()
    else:
        raise ValueError("regime must be 'decisions' or 'later_performance'.")

    return make_pipeline(StandardScaler(), estimator)


def _make_gbm_model(regime: Regime, seed: int = 0):
    """Create a histogram gradient boosting model."""
    if regime == "decisions":
        return HistGradientBoostingClassifier(random_state=seed)

    if regime == "later_performance":
        return HistGradientBoostingRegressor(random_state=seed)

    raise ValueError("regime must be 'decisions' or 'later_performance'.")


class PerStageModel:
    """Linear models fitted independently by stage, with stage fallback.

    A stage with only one observed decision class is skipped because logistic
    regression cannot learn a binary decision boundary from one class. At
    prediction time, the closest fitted earlier stage is used. If the requested
    stage precedes every fitted stage, the earliest fitted model is used, as in
    the reference implementation.
    """

    def __init__(self, regime: Regime):
        _validate_regime(regime)
        self.regime = regime
        self.models: dict[int, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PerStageModel:
        """Fit one independent linear model for each usable training stage."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if "stage" not in X.columns:
            raise ValueError("stage must be included in the training features.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must be aligned.")

        if X.empty:
            raise ValueError("training data must not be empty.")

        stages = _numeric_values(X["stage"], "stage")
        if (stages < 0.0).any() or not np.equal(stages, np.floor(stages)).all():
            raise ValueError("stage values must be non-negative integers.")

        self.models = {}
        stage_values = stages.astype(int)
        for stage in np.unique(stage_values):
            positions = np.flatnonzero(stage_values == stage)
            stage_y = y.iloc[positions]

            # A one-class subset cannot fit LogisticRegression. Skip it and
            # let prediction use the closest available stage model instead.
            if self.regime == "decisions" and set(stage_y.unique()) != {0, 1}:
                continue

            model = _make_linear_model(self.regime)
            model.fit(X.iloc[positions], stage_y)
            self.models[int(stage)] = model

        if not self.models:
            raise ValueError("no stage has sufficient data to train a model.")

        return self

    @property
    def classes_(self) -> np.ndarray:
        """Expose the binary class order used by the fitted stage models."""
        if self.regime != "decisions":
            raise AttributeError("regression stage models do not expose classes_.")
        return np.array([0, 1], dtype=int)

    def _get_model(self, stage: int):
        """Return the closest earlier model, or the earliest available model."""
        if not self.models:
            raise ValueError("the stage-specific model has not been fitted.")

        fitted_stages = sorted(self.models)
        earlier_stages = [fitted_stage for fitted_stage in fitted_stages if fitted_stage <= stage]
        selected_stage = earlier_stages[-1] if earlier_stages else fitted_stages[0]
        return self.models[selected_stage]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict values using the model associated with each requested stage."""
        if "stage" not in X.columns:
            raise ValueError("stage must be included in prediction features.")

        stages = _numeric_values(X["stage"], "stage")
        if (stages < 0.0).any() or not np.equal(stages, np.floor(stages)).all():
            raise ValueError("stage values must be non-negative integers.")

        predictions = np.empty(len(X), dtype=float)
        stage_values = stages.astype(int)
        for stage in np.unique(stage_values):
            positions = np.flatnonzero(stage_values == stage)
            model = self._get_model(int(stage))
            predictions[positions] = model.predict(X.iloc[positions])

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class probabilities using stage-specific models."""
        if self.regime != "decisions":
            raise ValueError("predict_proba is only available for the decisions regime.")

        if "stage" not in X.columns:
            raise ValueError("stage must be included in prediction features.")

        stages = _numeric_values(X["stage"], "stage")
        if (stages < 0.0).any() or not np.equal(stages, np.floor(stages)).all():
            raise ValueError("stage values must be non-negative integers.")

        probabilities = np.empty((len(X), 2), dtype=float)
        stage_values = stages.astype(int)
        for stage in np.unique(stage_values):
            positions = np.flatnonzero(stage_values == stage)
            model = self._get_model(int(stage))
            probabilities[positions] = model.predict_proba(X.iloc[positions])

        return probabilities


@dataclass(frozen=True)
class MLSelector:
    """Rank candidates using a model trained on historical simulation logs."""

    model: Any
    regime: Regime
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_regime(self.regime)
        if not self.feature_names:
            raise ValueError("an MLSelector must have at least one feature.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates.")

        base_names = tuple(name for name in self.feature_names if name != BIRTH_FEATURE)
        includes_birth = BIRTH_FEATURE in self.feature_names
        validated_names = feature_names(includes_birth, base_names)
        if validated_names != self.feature_names:
            raise ValueError("feature_names contain unsupported or misordered features.")

        required_method = "predict_proba" if self.regime == "decisions" else "predict"
        if not callable(getattr(self.model, required_method, None)):
            raise TypeError(f"model must implement {required_method}().")

        if isinstance(self.model, PerStageModel) and "stage" not in self.feature_names:
            raise ValueError("linear_by_stage requires the stage feature.")

    def scores(self, context: StageContext) -> np.ndarray:
        """Return one predicted selection score for each candidate."""
        candidates = context.candidates
        required_columns = set()
        if "times_selected" in self.feature_names:
            required_columns.add("times_selected_before")
        if BIRTH_FEATURE in self.feature_names:
            required_columns.add(BIRTH_FEATURE)

        missing = required_columns - set(candidates.columns)
        if missing:
            raise ValueError(f"candidate data is missing required columns: {sorted(missing)}.")

        feature_data = pd.DataFrame(index=candidates.index)
        if "observed" in self.feature_names:
            feature_data["observed"] = context.observed
        if "history_mean" in self.feature_names:
            feature_data["history_mean"] = np.asarray(context.history, dtype=float).mean(axis=0)
        if "times_selected" in self.feature_names:
            feature_data["times_selected"] = _numeric_values(
                candidates["times_selected_before"],
                "times_selected_before",
                non_negative=True,
            )
        if "stage" in self.feature_names:
            feature_data["stage"] = float(context.stage)
        if BIRTH_FEATURE in self.feature_names:
            feature_data[BIRTH_FEATURE] = _validate_birth_position(candidates[BIRTH_FEATURE])

        X = feature_data.loc[:, list(self.feature_names)]
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("candidate features must contain only finite values.")

        if self.regime == "decisions":
            probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
            if probabilities.shape != (len(candidates), 2):
                raise ValueError(
                    "decision model must return two class probabilities per candidate."
                )

            classes = np.asarray(self.model.classes_)
            positive_class = np.flatnonzero(classes == 1)
            if positive_class.size != 1:
                raise ValueError("decision model must be fitted with the binary class 1.")
            scores = probabilities[:, int(positive_class[0])]
            if ((scores < 0.0) | (scores > 1.0)).any():
                raise ValueError("predicted selection probabilities must be within [0, 1].")
        else:
            scores = np.asarray(self.model.predict(X), dtype=float)
            if scores.shape != (len(candidates),):
                raise ValueError("regression model must return one score per candidate.")

        if not np.isfinite(scores).all():
            raise ValueError("model scores must contain only finite values.")

        return scores


def train_selector(
    results: Sequence[pd.DataFrame],
    regime: Regime = "decisions",
    use_birth_info: bool = False,
    features: Sequence[str] | None = None,
    model_kind: ModelKind = "linear",
    seed: int = 0,
) -> MLSelector:
    """Train a pooled, stage-specific, or gradient-boosted ML selector.

    ``decisions`` imitates historical selection labels; ``later_performance``
    predicts the next-stage observed performance for athletes who remain in the
    pool. ``linear_by_stage`` fits independent linear models per stage and
    falls back to the closest available stage model. ``gbm`` uses histogram
    gradient boosting. Set ``use_birth_info`` to append ``birth_position``.
    """
    _validate_regime(regime)

    if model_kind not in ("linear", "linear_by_stage", "gbm"):
        raise ValueError("model_kind must be 'linear', 'linear_by_stage' or 'gbm'.")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer.")
    seed = int(seed)
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer.")

    names = feature_names(use_birth_info, features)
    if model_kind == "linear_by_stage" and "stage" not in names:
        raise ValueError("linear_by_stage requires the stage feature.")

    X, y = build_training_data(
        results,
        regime=regime,
        use_birth_info=use_birth_info,
        features=features,
    )

    if regime == "decisions" and set(y.unique()) != {0, 1}:
        raise ValueError("decision training requires examples from both selected classes.")

    if model_kind == "linear":
        model = _make_linear_model(regime)
    elif model_kind == "linear_by_stage":
        model = PerStageModel(regime)
    else:
        model = _make_gbm_model(regime, seed=seed)

    model.fit(X, y)

    return MLSelector(
        model=model,
        regime=regime,
        feature_names=names,
    )
