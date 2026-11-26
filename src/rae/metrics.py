"""Metrics for evaluating selection outcomes by birth quarter."""

import numpy as np
import pandas as pd


def summarize_selection_by_quarter(simulation: pd.DataFrame) -> pd.DataFrame:
    """Summarize selection and retention outcomes by stage and birth quarter.

    The function expects the long-format DataFrame returned by
    ``simulate_pipeline``.

    Definitions
    -----------
    n_candidates:
        Athletes from a birth quarter present at the beginning of a stage.

    n_selected:
        Athletes selected during that stage.

    selection_rate:
        n_selected / n_candidates.

    retention_rate:
        n_candidates / initial population of that birth quarter.

    cumulative_selection_rate:
        Number of distinct athletes selected at least once up to and
        including the current stage, divided by the initial population
        of that birth quarter.

    cumulative_dropout_rate:
        Number of distinct athletes who dropped out up to and including
        the current stage, divided by the initial population of that
        birth quarter.

    Cumulative selection and dropout rates are not complementary:
    an athlete may be selected in an earlier stage and drop out later.

    Parameters
    ----------
    simulation:
        Long-format simulation log containing athlete_id, stage,
        birth_quarter, selected and dropped_out.

    Returns
    -------
    pd.DataFrame
        One row per stage and birth quarter, with counts and rates.
        Birth quarters 1 to 4 are included even when a quarter has
        no athletes in the initial population. Undefined rates are NaN.
    """
    if not isinstance(simulation, pd.DataFrame):
        raise TypeError("simulation must be a pandas DataFrame.")

    if simulation.empty:
        raise ValueError("simulation must not be empty.")

    required_columns = {
        "athlete_id",
        "stage",
        "birth_quarter",
        "selected",
        "dropped_out",
    }

    missing_columns = required_columns.difference(simulation.columns)

    if missing_columns:
        raise ValueError(f"simulation is missing required columns: {sorted(missing_columns)}.")

    data = simulation[list(required_columns)].copy()

    if data["athlete_id"].isna().any():
        raise ValueError("athlete_id must not contain missing values.")

    if data.duplicated(["athlete_id", "stage"]).any():
        raise ValueError("each athlete can appear only once per stage.")

    if data.groupby("athlete_id", sort=False)["birth_quarter"].nunique().gt(1).any():
        raise ValueError("each athlete must have a consistent birth_quarter.")

    # Validate stage values.
    if pd.api.types.is_bool_dtype(data["stage"]) or not pd.api.types.is_numeric_dtype(
        data["stage"]
    ):
        raise TypeError("stage must contain numeric integer values.")

    stage_values = data["stage"].to_numpy(dtype=float)

    if (
        not np.isfinite(stage_values).all()
        or not np.equal(stage_values, np.floor(stage_values)).all()
        or (stage_values < 0).any()
    ):
        raise ValueError("stage must contain non-negative integer values.")

    data["stage"] = stage_values.astype(int)

    # Validate birth quarter values.
    if pd.api.types.is_bool_dtype(data["birth_quarter"]) or not pd.api.types.is_numeric_dtype(
        data["birth_quarter"]
    ):
        raise TypeError("birth_quarter must contain numeric integer values.")

    quarter_values = data["birth_quarter"].to_numpy(dtype=float)

    if (
        not np.isfinite(quarter_values).all()
        or not np.equal(quarter_values, np.floor(quarter_values)).all()
        or ((quarter_values < 1) | (quarter_values > 4)).any()
    ):
        raise ValueError("birth_quarter values must be integers between 1 and 4.")

    data["birth_quarter"] = quarter_values.astype(int)

    # Validate selection and dropout indicators.
    for column in ("selected", "dropped_out"):
        if not pd.api.types.is_bool_dtype(data[column]):
            raise TypeError(f"{column} must contain boolean values.")

        if data[column].isna().any():
            raise ValueError(f"{column} must not contain missing values.")

    if (data["selected"] & data["dropped_out"]).any():
        raise ValueError("an athlete cannot be selected and dropped out in the same stage.")

    # Validate that stage observations form a consistent simulation history.
    stages = np.sort(data["stage"].unique())
    max_stage = int(stages[-1])

    if not np.array_equal(stages, np.arange(max_stage + 1)):
        raise ValueError("stages must be consecutive and start from zero.")

    ordered = data.sort_values(["athlete_id", "stage"])

    for _, athlete_rows in ordered.groupby("athlete_id", sort=False):
        athlete_stages = athlete_rows["stage"].to_numpy()
        last_stage = int(athlete_stages[-1])

        if not np.array_equal(athlete_stages, np.arange(last_stage + 1)):
            raise ValueError(
                "each athlete's observations must start at stage zero and be consecutive."
            )

        if athlete_rows["dropped_out"].iloc[:-1].any():
            raise ValueError("an athlete cannot have observations after dropping out.")

        if last_stage < max_stage and not athlete_rows["dropped_out"].iloc[-1]:
            raise ValueError("an athlete missing from later stages must have dropped out.")

    # Initial population by birth quarter.
    initial_counts = data.loc[data["stage"] == 0].groupby("birth_quarter")["athlete_id"].nunique()

    rows = []

    for stage in stages:
        current = data.loc[data["stage"] == stage]
        cumulative = data.loc[data["stage"] <= stage]

        candidate_counts = current.groupby("birth_quarter")["athlete_id"].nunique()

        selected_counts = (
            current.loc[current["selected"]].groupby("birth_quarter")["athlete_id"].nunique()
        )

        cumulative_selected_counts = (
            cumulative.loc[cumulative["selected"]].groupby("birth_quarter")["athlete_id"].nunique()
        )

        cumulative_dropout_counts = (
            cumulative.loc[cumulative["dropped_out"]]
            .groupby("birth_quarter")["athlete_id"]
            .nunique()
        )

        for quarter in range(1, 5):
            n_initial = int(initial_counts.get(quarter, 0))
            n_candidates = int(candidate_counts.get(quarter, 0))
            n_selected = int(selected_counts.get(quarter, 0))
            n_cumulative_selected = int(cumulative_selected_counts.get(quarter, 0))
            n_cumulative_dropped_out = int(cumulative_dropout_counts.get(quarter, 0))

            rows.append(
                {
                    "stage": int(stage),
                    "birth_quarter": quarter,
                    "n_initial": n_initial,
                    "n_candidates": n_candidates,
                    "n_selected": n_selected,
                    "selection_rate": (n_selected / n_candidates if n_candidates > 0 else np.nan),
                    "retention_rate": (n_candidates / n_initial if n_initial > 0 else np.nan),
                    "n_cumulative_selected": n_cumulative_selected,
                    "cumulative_selection_rate": (
                        n_cumulative_selected / n_initial if n_initial > 0 else np.nan
                    ),
                    "n_cumulative_dropped_out": n_cumulative_dropped_out,
                    "cumulative_dropout_rate": (
                        n_cumulative_dropped_out / n_initial if n_initial > 0 else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)
