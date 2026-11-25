# Relative Age Effect and Feedback Loops in Youth Talent Selection

Simulation study for the Ethics in Artificial Intelligence course
at the University of Bologna.

## Overview

This project investigates how annual age-grouping in youth sports
can influence talent selection and how machine-learning selectors
may inherit or amplify the resulting bias through feedback loops.

The study uses a multi-stage simulation with synthetic athletes.
Latent talent is known by construction and is independent of
birth date, allowing selection-related talent loss to be measured.

The simulator explores possible mechanisms and mitigation strategies.
It is not intended to predict real-world outcomes.

No personal data are used.

## Research Questions

The project addresses three research questions:

- **RQ1:** How much does a selector trained on historical outcomes
  amplify the Relative Age Effect compared with the bias already
  present in the data?

- **RQ2:** Does the training-exposure feedback loop increase talent
  loss across successive selection stages and cohorts?

- **RQ3:** Which mitigation strategies reduce talent loss, and
  what is their cost in terms of selection efficiency?

The mitigation strategies considered are:

- Relative-age-adjusted scores
- Bio-banding
- Birth-quarter quotas
- Rotating selection slots

## Methodology

The simulation models:

- Synthetic athletes with latent talent and birth dates
- Biological maturation associated with relative age
- Observed performance affected by talent, maturation,
  training exposure and random noise
- Multi-stage selection and athlete dropout
- Human-like and machine-learning selectors
- Feedback loops within and across cohorts

Fairness and efficiency are evaluated using quantitative metrics,
including talent loss, equal-opportunity gaps, mean selected
talent and regret against an oracle selector.

## Development Status

The repository is currently in the initial development stage.

The mathematical model, software architecture, implementation
and experimental methodology will be documented progressively.

## Repository Structure

The source code, tests, experiments and project report will be
organized into separate directories as development progresses.

## Reproducibility

The project will use a dedicated Python environment and automated
tests to support reproducible experiments.

Detailed installation and execution instructions will be added
as the implementation develops.

## Previous Prototype

An earlier project prototype is available at:

https://github.com/Keniota02/rae-feedback-loops