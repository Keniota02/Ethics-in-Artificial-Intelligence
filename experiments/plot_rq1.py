"""Generate RQ1 figures from paired experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("report") / "figures"

MODELS = ("linear", "linear_by_stage", "gbm")

SELECTORS = (
    "ML decisions, birth info no",
    "ML decisions, birth info yes",
    "ML later_performance, birth info no",
    "ML later_performance, birth info yes",
)

SELECTOR_LABELS = {
    "ML decisions, birth info no": "Decisions / no birth info",
    "ML decisions, birth info yes": "Decisions / birth info",
    "ML later_performance, birth info no": "Later performance / no birth info",
    "ML later_performance, birth info yes": "Later performance / birth info",
}

MODEL_LABELS = {
    "linear": "Pooled linear",
    "linear_by_stage": "Linear by stage",
    "gbm": "Gradient boosting",
}


def load_paired_results() -> pd.DataFrame:
    """Load and validate the paired RQ1 analysis file."""
    path = RESULTS_DIR / "rq1_paired_differences.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run 'python -m experiments.analyze_rq1' first."
        )

    data = pd.read_csv(path)

    required = {
        "model_kind",
        "exposure_weight",
        "historical_selector",
        "selector",
        "metric",
        "n_pairs",
        "mean_difference",
        "ci95_half_width",
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in paired results: {sorted(missing)}")

    if data[list(required)].isna().any().any():
        raise ValueError("Paired results contain missing values.")

    if not set(data["model_kind"]).issubset(MODELS):
        raise ValueError("Unexpected model kind in paired results.")

    if not (data["n_pairs"] == 20).all():
        raise ValueError("Expected exactly 20 paired seeds per comparison.")

    return data


def plot_q1_q4_by_history(data: pd.DataFrame, historical: str) -> None:
    """Plot paired Q1/Q4 differences for exposure disabled."""
    subset = data[
        (data["exposure_weight"] == 0)
        & (data["historical_selector"] == historical)
        & (data["metric"] == "selection_rate_ratio_q1_q4")
    ]

    records = []

    for model in MODELS:
        for selector in SELECTORS:
            row = subset[(subset["model_kind"] == model) & (subset["selector"] == selector)]

            if len(row) != 1:
                raise ValueError(f"Expected one row for {historical}, {model}, {selector}.")

            result = row.iloc[0]
            records.append(
                {
                    "model": model,
                    "selector": selector,
                    "label": (f"{MODEL_LABELS[model]} — {SELECTOR_LABELS[selector]}"),
                    "mean": result["mean_difference"],
                    "ci": result["ci95_half_width"],
                }
            )

    plot_data = pd.DataFrame(records)
    positions = list(range(len(plot_data)))

    fig, ax = plt.subplots(figsize=(11, 7))

    for model in MODELS:
        model_data = plot_data[plot_data["model"] == model]
        model_positions = [plot_data.index.get_loc(index) for index in model_data.index]

        ax.errorbar(
            model_data["mean"],
            model_positions,
            xerr=model_data["ci"],
            fmt="o",
            capsize=3,
            linestyle="none",
            label=MODEL_LABELS[model],
        )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_data["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference in Q1/Q4 ratio (ML − historical)")
    ax.set_title(f"RQ1 — Relative-age disparity: historical {historical}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / f"rq1_q1q4_{historical.replace(' ', '_')}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")


def plot_fairness_efficiency_tradeoff(data: pd.DataFrame) -> None:
    """Plot fairness-efficiency differences in four experimental conditions."""
    keys = [
        "model_kind",
        "exposure_weight",
        "historical_selector",
        "selector",
    ]

    loss = data[data["metric"] == "overall_talent_loss"][
        keys + ["mean_difference", "ci95_half_width"]
    ].rename(
        columns={
            "mean_difference": "loss_difference",
            "ci95_half_width": "loss_ci",
        }
    )

    gap = data[data["metric"] == "equal_opportunity_gap"][
        keys + ["mean_difference", "ci95_half_width"]
    ].rename(
        columns={
            "mean_difference": "gap_difference",
            "ci95_half_width": "gap_ci",
        }
    )

    tradeoff = loss.merge(gap, on=keys, validate="one_to_one")

    panels = [
        ("scout", 0.0),
        ("scout", 1.0),
        ("biased scout", 0.0),
        ("biased scout", 1.0),
    ]

    markers = {
        SELECTORS[0]: "o",
        SELECTORS[1]: "s",
        SELECTORS[2]: "^",
        SELECTORS[3]: "D",
    }

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {model: colors[index % len(colors)] for index, model in enumerate(MODELS)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

    for ax, (historical, exposure) in zip(axes.flat, panels, strict=True):
        panel = tradeoff[
            (tradeoff["historical_selector"] == historical)
            & (tradeoff["exposure_weight"] == exposure)
        ]

        for model in MODELS:
            for selector in SELECTORS:
                row = panel[(panel["model_kind"] == model) & (panel["selector"] == selector)]

                if len(row) != 1:
                    raise ValueError(
                        f"Expected one row for {historical}, exposure={exposure}, "
                        f"{model}, {selector}."
                    )

                result = row.iloc[0]

                ax.errorbar(
                    result["loss_difference"],
                    result["gap_difference"],
                    xerr=result["loss_ci"],
                    yerr=result["gap_ci"],
                    fmt=markers[selector],
                    color=model_colors[model],
                    markersize=6,
                    capsize=2,
                    linestyle="none",
                )

        ax.axvline(0, linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{historical.title()}, exposure = {exposure:g}")
        ax.grid(alpha=0.25)

    axes[1, 0].set_xlabel("Change in overall talent loss (ML − historical)")
    axes[1, 1].set_xlabel("Change in overall talent loss (ML − historical)")
    fig.supylabel(
        "Change in Equal Opportunity Gap (ML − historical)",
        x=0.015,
    )

    model_handles = [
        Line2D(
            [],
            [],
            color=model_colors[model],
            marker="o",
            linestyle="none",
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]

    selector_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=markers[selector],
            linestyle="none",
            label=SELECTOR_LABELS[selector],
        )
        for selector in SELECTORS
    ]

    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        title="Model",
    )

    fig.legend(
        handles=selector_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        title="Training configuration",
    )

    fig.suptitle(
        "RQ1 — Talent retention and fairness trade-off",
        y=0.995,
    )
    fig.tight_layout(rect=(0.06, 0.13, 1, 0.90))

    output = FIGURES_DIR / "rq1_fairness_efficiency_tradeoff.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")


def main() -> None:
    """Generate all RQ1 figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_paired_results()

    plot_q1_q4_by_history(data, "scout")
    plot_q1_q4_by_history(data, "biased scout")
    plot_fairness_efficiency_tradeoff(data)


if __name__ == "__main__":
    main()
