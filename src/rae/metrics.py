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


def _prepare_talent_metric_data(
    simulation: pd.DataFrame,
    top_talent_fraction: float,
    adult_maturation_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate simulation data and identify the initial high-talent cohort."""
    if not isinstance(simulation, pd.DataFrame):
        raise TypeError("simulation must be a pandas DataFrame.")

    required_columns = {
        "athlete_id",
        "stage",
        "birth_quarter",
        "selected",
        "dropped_out",
        "talent",
        "maturity",
    }

    missing_columns = required_columns.difference(simulation.columns)

    if missing_columns:
        raise ValueError(f"simulation is missing required columns: {sorted(missing_columns)}.")

    # Reuse the structural validation already implemented for selection metrics.
    summarize_selection_by_quarter(simulation)

    for name, value in {
        "top_talent_fraction": top_talent_fraction,
        "adult_maturation_weight": adult_maturation_weight,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a numeric value.")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    top_talent_fraction = float(top_talent_fraction)
    adult_maturation_weight = float(adult_maturation_weight)

    if not 0.0 < top_talent_fraction < 1.0:
        raise ValueError("top_talent_fraction must be in the interval (0, 1).")

    if adult_maturation_weight < 0.0:
        raise ValueError("adult_maturation_weight must be non-negative.")

    data = simulation[
        [
            "athlete_id",
            "stage",
            "birth_quarter",
            "selected",
            "talent",
            "maturity",
        ]
    ].copy()

    # Validate latent target variables.
    for column in ("talent", "maturity"):
        if pd.api.types.is_bool_dtype(data[column]) or not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            raise TypeError(f"{column} must contain numeric values.")

        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite values.")

        data[column] = values

        if data.groupby("athlete_id", sort=False)[column].nunique().gt(1).any():
            raise ValueError(f"each athlete must have a consistent {column} value.")

    # Build the initial cohort, with one row per athlete.
    cohort = data.loc[
        data["stage"] == 0,
        ["athlete_id", "birth_quarter", "talent", "maturity"],
    ].copy()

    cohort["target"] = cohort["talent"] + adult_maturation_weight * cohort["maturity"]

    if not np.isfinite(cohort["target"].to_numpy()).all():
        raise ValueError("adult target must contain only finite values.")

    threshold = np.quantile(
        cohort["target"].to_numpy(),
        1.0 - top_talent_fraction,
    )

    # Athletes tied at the threshold are all included.
    cohort["high_talent"] = cohort["target"] >= threshold

    # Attach the same target and high-talent classification to every stage record.
    data = data.merge(
        cohort[["athlete_id", "target", "high_talent"]],
        on="athlete_id",
        how="left",
        validate="many_to_one",
    )

    return data, cohort


def _build_talent_loss_by_quarter(
    data: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate stage-specific high-talent loss by birth quarter."""
    stages = np.sort(data["stage"].unique())

    initial_counts = cohort.groupby("birth_quarter")["athlete_id"].nunique()

    high_counts = cohort.loc[cohort["high_talent"]].groupby("birth_quarter")["athlete_id"].nunique()

    rows = []

    for stage in stages:
        current = data.loc[data["stage"] == stage]
        selected = current.loc[current["selected"] & current["high_talent"]]

        selected_high_counts = selected.groupby("birth_quarter")["athlete_id"].nunique()

        for quarter in range(1, 5):
            n_initial = int(initial_counts.get(quarter, 0))
            n_high_talent = int(high_counts.get(quarter, 0))
            n_selected_high = int(selected_high_counts.get(quarter, 0))

            n_lost_high = n_high_talent - n_selected_high

            rows.append(
                {
                    "stage": int(stage),
                    "birth_quarter": quarter,
                    "n_initial": n_initial,
                    "n_high_talent": n_high_talent,
                    "n_selected_high_talent": n_selected_high,
                    "n_lost_high_talent": n_lost_high,
                    "talent_loss": (n_lost_high / n_high_talent if n_high_talent > 0 else np.nan),
                }
            )

    return pd.DataFrame(rows)


def summarize_talent_loss_by_quarter(
    simulation: pd.DataFrame,
    top_talent_fraction: float = 0.10,
    adult_maturation_weight: float = 0.0,
) -> pd.DataFrame:
    """Summarize high-talent loss by stage and birth quarter.

    High talent is defined using the top fraction of the initial cohort,
    ranked by latent talent or adult potential.

    Talent loss is the share of high-talent athletes not selected
    at the current stage, using the initial high-talent cohort as denominator.
    """
    data, cohort = _prepare_talent_metric_data(
        simulation,
        top_talent_fraction,
        adult_maturation_weight,
    )

    return _build_talent_loss_by_quarter(data, cohort)


def summarize_talent_metrics(
    simulation: pd.DataFrame,
    top_talent_fraction: float = 0.10,
    adult_maturation_weight: float = 0.0,
) -> pd.DataFrame:
    """Summarize fairness and efficiency metrics at every selection stage.

    Returns one row per stage, including overall talent loss,
    loss by birth quarter, equal-opportunity gap, mean selected target
    and regret compared with an Oracle cohort of the same size.
    """
    data, cohort = _prepare_talent_metric_data(
        simulation,
        top_talent_fraction,
        adult_maturation_weight,
    )

    quarter_metrics = _build_talent_loss_by_quarter(data, cohort)

    initial_counts = cohort.groupby("birth_quarter")["athlete_id"].nunique()
    n_high_talent = int(cohort["high_talent"].sum())
    oracle_targets = np.sort(cohort["target"].to_numpy())[::-1]

    rows = []

    for stage in np.sort(data["stage"].unique()):
        current = data.loc[data["stage"] == stage]
        selected = current.loc[current["selected"]]

        n_selected = len(selected)
        n_selected_high = int(selected["high_talent"].sum())

        overall_loss = 1.0 - n_selected_high / n_high_talent if n_high_talent > 0 else np.nan

        stage_quarters = quarter_metrics.loc[quarter_metrics["stage"] == stage]

        quarter_losses = {
            int(row.birth_quarter): row.talent_loss for row in stage_quarters.itertuples()
        }

        finite_losses = [value for value in quarter_losses.values() if np.isfinite(value)]

        opportunity_gap = (
            max(finite_losses) - min(finite_losses) if len(finite_losses) >= 2 else np.nan
        )

        mean_selected = float(selected["target"].mean()) if n_selected > 0 else np.nan

        oracle_mean = float(oracle_targets[:n_selected].mean()) if n_selected > 0 else np.nan

        regret_value = oracle_mean - mean_selected if n_selected > 0 else np.nan

        # Q1/Q4 selection-rate ratio uses initial quarter populations.
        selected_counts = selected.groupby("birth_quarter")["athlete_id"].nunique()

        q1_initial = int(initial_counts.get(1, 0))
        q4_initial = int(initial_counts.get(4, 0))

        if q1_initial == 0 or q4_initial == 0:
            selection_ratio = np.nan
        else:
            q1_rate = int(selected_counts.get(1, 0)) / q1_initial
            q4_rate = int(selected_counts.get(4, 0)) / q4_initial

            selection_ratio = q1_rate / q4_rate if q4_rate > 0 else np.inf

        rows.append(
            {
                "stage": int(stage),
                "n_selected": n_selected,
                "n_high_talent": n_high_talent,
                "n_selected_high_talent": n_selected_high,
                "overall_talent_loss": overall_loss,
                "talent_loss_q1": quarter_losses[1],
                "talent_loss_q2": quarter_losses[2],
                "talent_loss_q3": quarter_losses[3],
                "talent_loss_q4": quarter_losses[4],
                "equal_opportunity_gap": opportunity_gap,
                "mean_selected_talent": mean_selected,
                "regret": regret_value,
                "selection_rate_ratio_q1_q4": selection_ratio,
            }
        )

    return pd.DataFrame(rows)
