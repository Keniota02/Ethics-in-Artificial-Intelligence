"""Tests for shared experiment utilities."""

import numpy as np
import pandas as pd
import pytest
from experiments.common import fmt, mean_ci, save


def test_mean_ci_returns_mean_and_positive_half_width():
    """Student-t confidence intervals should be calculated correctly."""
    mean, half_width = mean_ci([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert half_width > 0.0
    assert np.isfinite(half_width)


def test_mean_ci_with_one_observation_has_undefined_half_width():
    """A confidence interval cannot be estimated from one observation."""
    mean, half_width = mean_ci([4.0])

    assert mean == pytest.approx(4.0)
    assert np.isnan(half_width)


def test_mean_ci_rejects_empty_input():
    """At least one observation is required."""
    with pytest.raises(ValueError, match="at least one"):
        mean_ci([])


def test_mean_ci_rejects_invalid_confidence():
    """Confidence must be strictly between zero and one."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        mean_ci([1.0, 2.0], confidence=1.0)


def test_mean_ci_handles_non_finite_values():
    """Non-finite values should not produce a misleading confidence interval."""
    mean, half_width = mean_ci([1.0, np.inf])

    assert np.isinf(mean)
    assert np.isnan(half_width)


def test_fmt_returns_a_formatted_string():
    """The formatter should return the mean and confidence half-width."""
    result = fmt([1.0, 2.0, 3.0])

    assert "+/-" in result
    assert "2.000" in result


def test_save_creates_results_csv(tmp_path, monkeypatch):
    """Experiment results should be saved as CSV files."""
    import experiments.common as common

    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)

    dataframe = pd.DataFrame(
        {
            "seed": [0, 1],
            "score": [0.5, 0.7],
        }
    )

    output_path = save(dataframe, "test_results.csv")

    assert output_path.exists()

    loaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(loaded, dataframe)


def test_save_rejects_invalid_filename():
    """The save helper should accept filenames, not arbitrary paths."""
    with pytest.raises(ValueError, match="filename"):
        save(pd.DataFrame({"score": [1.0]}), "../results.csv")
