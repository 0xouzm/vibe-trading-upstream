"""One annualisation convention for cross-market runs (issue #1237).

A basket spanning markets has no single per-market bar count, so the runner
passes ``bars_per_year=None`` (``runner.py``: *"Cross-market: use calendar-day
annualization"*). Four consumers have to agree on what that means — portfolio
metrics, the risk x-ray, options metrics, and validation — or one run card
reports a Sharpe and an annualised volatility computed on different footings.

These tests pin all four to ``metrics.effective_bars_per_year``. They fail if
any consumer grows its own copy of the span derivation and drifts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.engines.options_portfolio import _calc_options_metrics
from backtest.metrics import calc_bars_per_year, calc_metrics, effective_bars_per_year
from backtest.risk_xray import compute_risk_xray
from backtest.validation import _sharpe, run_validation


def _zigzag(n: int, base: float, step: float, period: int) -> list[float]:
    """Prices with both up and down moves, so downside statistics exist."""
    return [base + (i % period) * step - step * (period - 1) / 2 for i in range(n)]


class TestEffectiveBarsPerYear:
    def test_daily_bars_over_a_full_year(self):
        idx = pd.date_range("2024-01-01", periods=253, freq="B")
        span_years = (idx[-1] - idx[0]).days / 365.25
        assert effective_bars_per_year(idx) == int(253 / span_years)

    def test_span_shorter_than_a_day_counts_as_one_year(self):
        # Two bars on the same calendar day: no measurable span, so the series
        # annualises to itself rather than exploding on a near-zero divisor.
        idx = pd.DatetimeIndex(["2024-01-01T09:30", "2024-01-01T15:00"])
        assert effective_bars_per_year(idx) == 2

    def test_empty_index_falls_back_to_default(self):
        assert effective_bars_per_year(pd.DatetimeIndex([])) == 252
        assert effective_bars_per_year(pd.DatetimeIndex([]), default=365) == 365

    def test_non_datetime_index_has_no_measurable_span(self):
        # An integer index carries no ``days``; the series annualises to its
        # own length rather than raising.
        assert effective_bars_per_year(pd.Index([0, 1, 2, 3])) == 4


class TestConsumersShareTheConvention:
    """Every ``bars_per_year=None`` consumer resolves the same factor."""

    idx = pd.date_range("2024-01-01", periods=120, freq="B")

    @property
    def expected_bpy(self) -> int:
        return effective_bars_per_year(self.idx)

    def test_risk_xray(self):
        closes = pd.DataFrame(
            {
                "AAA": _zigzag(120, 100.0, 2.0, 5),
                "BBB": _zigzag(120, 50.0, 1.0, 7),
            },
            index=self.idx,
        )
        weights = {"AAA": 0.5, "BBB": 0.5}
        result = compute_risk_xray(closes, weights, min_history=10, periods_per_year=None)

        port = (closes.pct_change().dropna() * pd.Series(weights)).sum(axis=1)
        expected = effective_bars_per_year(port.index)
        assert result["volatility"]["annualized_vol"] == pytest.approx(
            port.std(ddof=1) * math.sqrt(expected)
        )

    def test_options_metrics(self):
        equity = pd.Series(_zigzag(120, 100_000.0, 500.0, 5), index=self.idx)
        metrics = _calc_options_metrics(equity, 100_000.0, [], bars_per_year=None)

        returns = equity.pct_change(fill_method=None).iloc[1:]
        # Options metrics round their reported ratios to 4 decimals.
        assert metrics["sharpe"] == pytest.approx(
            returns.mean() / returns.std() * math.sqrt(self.expected_bpy), abs=5e-5
        )

    def test_validation(self):
        equity = pd.Series(_zigzag(120, 100_000.0, 500.0, 5), index=self.idx)
        result = run_validation(
            {"validation": {"bootstrap": {"n_bootstrap": 10}}},
            equity,
            [],
            100_000.0,
            bars_per_year=None,
        )

        returns = equity.pct_change().dropna().to_numpy()
        assert result["bootstrap"]["observed_sharpe"] == pytest.approx(
            round(_sharpe(returns, self.expected_bpy), 4)
        )

    def test_portfolio_metrics(self):
        equity = pd.Series(np.linspace(100_000.0, 130_000.0, 120), index=self.idx)
        metrics = calc_metrics(equity, [], 100_000.0, bars_per_year=None)

        growth = 1.3
        assert metrics["annual_return"] == pytest.approx(
            growth ** (self.expected_bpy / 120) - 1, rel=1e-6
        )


class TestSingleMarketAnnualisationChecksTheServedData:
    """The declared interval is a request, not a fact about what arrived.

    A loader may legitimately serve coarser bars than asked for — the local
    loader cannot upsample a daily file to ``1H`` and only logs a warning —
    and annualising at the declared rate then scales CAGR, Sharpe and the
    annualised volatility by the ratio between the two.
    """

    @staticmethod
    def _daily(n: int = 654) -> dict:
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        return {"600519.SH": pd.DataFrame({"close": [10.0] * n}, index=idx)}

    def test_matching_declaration_keeps_the_per_source_table(self):
        """A correctly served run must keep the trading-day table it always had,
        not drift to the measured count (252 declared vs ~261 measured)."""
        from backtest.runner import _annualisation_bars

        assert _annualisation_bars("1D", "tushare", self._daily(), ["600519.SH"]) == 252

    def test_declared_intraday_against_daily_bars_uses_the_observed_count(self):
        from backtest.runner import _annualisation_bars

        declared = calc_bars_per_year("1H", "tushare")
        resolved = _annualisation_bars("1H", "tushare", self._daily(), ["600519.SH"])

        assert declared > 1000                      # the declaration is intraday
        assert resolved < 400                       # the data is not
        assert resolved == effective_bars_per_year(self._daily()["600519.SH"].index)

    def test_crypto_daily_is_not_tripped_by_the_check(self):
        """365-day markets measure close to their declared count."""
        from backtest.runner import _annualisation_bars

        n = 700
        idx = pd.date_range("2024-01-02", periods=n, freq="D")
        data = {"BTC-USDT": pd.DataFrame({"close": [10.0] * n}, index=idx)}
        assert _annualisation_bars("1D", "okx", data, ["BTC-USDT"]) == 365

    @pytest.mark.parametrize("data", [{}, {"600519.SH": pd.DataFrame({"close": []})}])
    def test_unmeasurable_data_falls_back_to_the_declaration(self, data):
        from backtest.runner import _annualisation_bars

        assert _annualisation_bars("1D", "tushare", data, ["600519.SH"]) == 252

    def test_only_price_frames_are_measured(self):
        """Injected fundamental panels must not decide the annualisation."""
        from backtest.runner import _annualisation_bars

        data = self._daily()
        data["_fundamentals"] = pd.DataFrame(
            {"pe": [1.0] * 5}, index=pd.date_range("2024-01-02", periods=5, freq="YE")
        )
        assert _annualisation_bars("1D", "tushare", data, ["600519.SH"]) == 252

    def test_mismatch_is_logged(self, caplog):
        from backtest.runner import _annualisation_bars

        with caplog.at_level("WARNING", logger="backtest.runner"):
            _annualisation_bars("1H", "tushare", self._daily(), ["600519.SH"])

        assert any("1H" in r.getMessage() for r in caplog.records)
