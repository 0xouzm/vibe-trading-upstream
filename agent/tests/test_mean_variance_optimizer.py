"""Tests for the mean-variance (max Sharpe) optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.optimizers.mean_variance import MeanVarianceOptimizer


class TestMeanVarianceOptimize:
    """Integration tests for the module-level optimize function."""

    def test_optimize_preserves_sign(self) -> None:
        """Optimizer should preserve signal direction (long/short)."""
        dates = pd.bdate_range("2025-01-01", periods=100)
        codes = ["A", "B"]
        rng = np.random.default_rng(42)
        ret = pd.DataFrame(rng.normal(0, 0.02, (100, 2)), index=dates, columns=codes)
        pos = pd.DataFrame(0.0, index=dates, columns=codes)
        pos.iloc[60:, 0] = 1.0
        pos.iloc[60:, 1] = -1.0

        opt = MeanVarianceOptimizer(lookback=60)
        result = opt.optimize(ret, pos, dates)

        assert (result.iloc[61:, 0] >= 0).all(), "A should remain long"
        assert (result.iloc[61:, 1] <= 0).all(), "B should remain short"

    def test_strong_short_sized_above_weak_short(self) -> None:
        """A short with a strongly negative drift is a better short than one
        with near-zero drift, so it must receive more capital, not less.

        Regression: mu was the raw unsigned asset drift, so a strong short
        (very negative raw mu) scored as a bad "long" in the Sharpe
        objective and was starved of capital relative to a weak short
        (near-zero raw mu) -- sizing was inverted for the short book.
        """
        dates = pd.bdate_range("2025-01-01", periods=140)
        codes = ["WEAK", "STRONG"]
        rng = np.random.default_rng(7)
        weak_ret = rng.normal(-0.0005, 0.01, 140)
        strong_ret = rng.normal(-0.02, 0.01, 140)
        ret = pd.DataFrame({"WEAK": weak_ret, "STRONG": strong_ret}, index=dates)

        pos = pd.DataFrame(0.0, index=dates, columns=codes)
        pos.iloc[120:, 0] = -1.0
        pos.iloc[120:, 1] = -1.0

        opt = MeanVarianceOptimizer(lookback=120)
        result = opt.optimize(ret, pos, dates)
        last = result.iloc[-1]

        assert abs(last["STRONG"]) > abs(last["WEAK"])

    def test_single_asset_unchanged(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=100)
        ret = pd.DataFrame(
            np.random.default_rng(1).normal(0, 0.02, (100, 1)),
            index=dates,
            columns=["A"],
        )
        pos = pd.DataFrame(1.0, index=dates, columns=["A"])

        opt = MeanVarianceOptimizer(lookback=60)
        result = opt.optimize(ret, pos, dates)
        pd.testing.assert_frame_equal(result, pos)

    def test_result_weights_on_simplex(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=100)
        codes = ["A", "B", "C"]
        rng = np.random.default_rng(3)
        ret = pd.DataFrame(rng.normal(0, 0.02, (100, 3)), index=dates, columns=codes)
        pos = pd.DataFrame(1.0, index=dates, columns=codes)

        opt = MeanVarianceOptimizer(lookback=60)
        result = opt.optimize(ret, pos, dates)
        last = result.iloc[-1].values
        assert abs(abs(last).sum() - 1.0) < 1e-6
