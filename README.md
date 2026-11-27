# Relative Age Effect and Feedback Loops in Youth Talent Selection

Simulation study for the *Ethics in Artificial Intelligence* course at the University of Bologna.

The project investigates how annual age-grouping in youth sport can affect multi-stage talent selection, and how machine-learning selectors trained on historical decisions or outcomes may reproduce or amplify the resulting Relative Age Effect (RAE).

The study uses synthetic athletes whose latent talent is known by construction and independent of birth date. This makes it possible to measure the loss of highly talented athletes directly. The simulator explores mechanisms and compares selection policies; it is **not a predictor of real-world athlete outcomes**. No personal data are used.

## Research questions

- **RQ1 — Historical bias and machine learning.** How much does a selector trained on historical outcomes amplify the RAE, compared with the bias already present in the data?
- **RQ2 — Feedback loops.** Does the training-exposure feedback loop increase talent loss across successive selection stages and cohorts?
- **RQ3 — Mitigation.** Which mitigation strategies reduce talent loss, and what is their cost in terms of selection efficiency?

The proposed RQ3 mitigation strategies are relative-age-adjusted scores, bio-banding, birth-quarter quotas and rotating selection slots.

## Current implementation status

| Component | Status |
|---|---|
| Synthetic population and birth-quarter generation | Implemented and tested |
| Observed-performance model and multi-stage selection pipeline | Implemented and tested |
| Performance-based, Scout and Oracle selectors | Implemented and tested |
| Talent-loss, fairness and selection-efficiency metrics | Implemented and tested |
| ML selectors and historical training-data construction | Implemented and tested |
| RQ1 experimental runner, paired analysis and figures | Completed and documented |
| RQ2 experiments across stages and cohorts | Not yet implemented |
| RQ3 mitigation experiments | Not yet implemented |

The completed RQ1 experiment and its interpretation are documented in [`report/rq1.md`](report/rq1.md).

## Model overview

The simulator represents a cohort of synthetic athletes. Each athlete has latent talent, a birth position within the selection year, a birth quarter and biological maturation. Latent talent is independent of birth date by construction, while biological maturation is correlated with relative age.

At each selection stage, observed performance combines latent talent, biological maturation, accumulated exposure from earlier selections and random measurement noise. A selector ranks the athletes in the current pool, and the selected share is reduced across successive stages. Unselected athletes may leave the system.

The model provides a common interface for different selectors. The ML selectors can be trained on either historical selection decisions or later observed performance. Birth-position information can be included or withheld from their feature set. Latent talent and true biological maturation are not supplied as ML input features.

The default simulation configuration is centralized in `src/rae/config.py`.

## Evaluation metrics

The project distinguishes selection disparity from harm to high-talent athletes.

- **Q1/Q4 selection-rate ratio:** selection rate among athletes born in Q1 divided by the rate among athletes born in Q4.
- **Talent loss:** proportion of high-talent athletes who are not selected, measured overall and by birth quarter.
- **Equal Opportunity Gap:** largest difference in talent loss across birth quarters.
- **Mean selected talent:** average latent talent of the selected cohort.
- **Regret:** difference between the mean talent of an Oracle-selected cohort of the same size and the mean talent of the actual selected cohort.

The Q1/Q4 ratio measures disparity between birth quarters. Talent loss and the Equal Opportunity Gap measure whether high-talent athletes are excluded and whether that exclusion differs across quarters. These metrics are related, but they are not interchangeable.

## Repository structure

```text
src/rae/
    __init__.py       Package exports
    config.py         Central simulation configuration
    population.py     Synthetic athlete population
    performance.py    Observed-performance generation
    selection.py      Selection utilities
    selectors.py      Performance, Scout and Oracle selectors
    pipeline.py       Multi-stage simulation pipeline
    learning.py       Training data and ML selectors
    metrics.py        Fairness and selection-efficiency metrics

experiments/
    common.py         Shared experiment settings and statistics
    run_rq1.py        RQ1 simulation runner
    analyze_rq1.py    RQ1 summary statistics and paired comparisons
    plot_rq1.py       RQ1 figure generation

tests/
    Unit tests for the simulator, selectors, metrics and RQ1 workflow

report/
    rq1.md            RQ1 methodology, results and interpretation
    figures/          Figures generated from the RQ1 analysis

.github/workflows/
    ci.yml            GitHub Actions continuous integration
```

Generated experiment outputs are written to `results/`, which is ignored by Git. The scripts and configuration needed to regenerate the RQ1 outputs are versioned in the repository.

## Environment setup

The project requires Python 3.11. With Conda installed, create and activate the environment from the repository root:

```powershell
conda env create -f environment.yml
conda activate rae-rebuild
python -m pip install -e ".[dev]"
```

The editable installation installs the package and its development dependencies, including pytest and Ruff.

## Running checks

The repository includes a `Makefile` with the following aggregate check:

```bash
make check
```

On Windows PowerShell, or on systems without GNU Make, run the underlying commands directly:

```powershell
python -m ruff check src tests experiments
python -m ruff format --check src tests experiments
python -m pytest
```

To format the code rather than only check it:

```powershell
python -m ruff format src tests experiments
```

Continuous integration runs the repository checks on pushes to `main` and on pull requests targeting `main`.

## Reproducing RQ1

RQ1 evaluates three ML model architectures (`linear`, `linear_by_stage` and `gbm`), two historical selectors (Scout and Biased Scout), two training targets (historical decisions and later performance), two birth-information settings and two exposure weights.

The reported experiment uses 20 independent seeds and 20,000 athletes per cohort. Run each model separately from the repository root:

```powershell
python -m experiments.run_rq1 --seeds 20 --model linear --athletes 20000
python -m experiments.run_rq1 --seeds 20 --model linear_by_stage --athletes 20000
python -m experiments.run_rq1 --seeds 20 --model gbm --athletes 20000
```

The runner writes one file per model to `results/`:

- `rq1_runs_linear.csv`
- `rq1_runs_linear_by_stage.csv`
- `rq1_runs_gbm.csv`

Once all three runs are complete, calculate the summary statistics and paired ML-versus-historical differences:

```powershell
python -m experiments.analyze_rq1
```

This produces `results/rq1_summary.csv` and `results/rq1_paired_differences.csv`.

Generate the report figures from the paired results with:

```powershell
python -m experiments.plot_rq1
```

The figures are written to `report/figures/`. They are tracked by Git so that the RQ1 report renders with its visual results. The CSV files are not tracked; retain the generated files locally when reproducing or extending the analysis.

For command-line options, run the relevant module with `--help`.

## Previous prototype

An earlier prototype of the project is available at [Keniota02/rae-feedback-loops](https://github.com/Keniota02/rae-feedback-loops). The current repository is an independent reimplementation. The comparison with the previous prototype is documented in the RQ1 report and is based on the aggregate results published there; the previous per-seed RQ1 CSV files are not tracked in that repository.

## Development workflow

Develop implementation changes on a short-lived feature branch and integrate them into `main` through a pull request after the checks pass. Keep experiment outputs under `results/` out of version control; commit source code, tests, methodology and report figures needed to reproduce and interpret the study.
