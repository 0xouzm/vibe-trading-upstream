"""No value may be computed from a window that holds a missing input (#1463).

``Registry.compute`` masks the bar a dependency is missing on, not the bars
after it. An alpha that turns a comparison into 0/1, calls ``.where(cond, 0)``
or takes ``np.fmax`` over a missing side substitutes a constant for the missing
input, and every later bar whose window reaches back to the gap then carries a
value that bar never supported.

Oracle: remove one symbol's dependencies on one bar, then perturb the same bar
up and down. A later cell of that symbol whose value moves with the perturbation
depends on the bar, so it must be NaN when the bar is missing. The panel is
dyadic (prices in sixteenths, integer volume) and differences are compared with a
tight tolerance, so pandas' online rolling sums carry no rounding residue that
would read as a dependence.

The recursive smoothers (``ewm`` / SMA) are deliberately not listed: whether a
smoother may skip a missing observation is a policy decision still open on #1463.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.registry import get_default_registry

FIXED = [
    "alpha101_065",
    "alpha101_071",
    "alpha101_073",
    "alpha101_076",
    "alpha101_082",
    "alpha101_087",
    "alpha101_092",
    "gtja191_043",
    "gtja191_049",
    "gtja191_050",
    "gtja191_051",
    "gtja191_053",
    "gtja191_058",
    "gtja191_084",
    "gtja191_094",
    "gtja191_112",
    "gtja191_128",
    "gtja191_129",
    "gtja191_148",
]

# 24 symbols: with fewer, a one-tick perturbation rarely flips a cross-sectional
# rank comparison, and the oracle would see no dependence at all.
_N, _T, _GAP, _SYMBOL = 24, 330, 250, 7
_PRICES = ("close", "open", "high", "low", "vwap")


def _base() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1463)
    ticks = np.maximum(320 + np.cumsum(rng.integers(-6, 7, (_T, _N)), axis=0), 40)
    close = ticks / 16.0
    open_ = (ticks + rng.integers(-3, 4, (_T, _N))) / 16.0
    high = np.maximum(close, open_) + rng.integers(0, 4, (_T, _N)) / 16.0
    low = np.minimum(close, open_) - rng.integers(0, 4, (_T, _N)) / 16.0
    volume = rng.integers(1000, 50000, (_T, _N)).astype(float)
    vwap = (high + low + 2 * close) / 4.0
    return {"close": close, "open": open_, "high": high, "low": low, "volume": volume, "vwap": vwap}


def _panel(base: dict[str, np.ndarray], deps: set[str], mode: str) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2020-01-01", periods=_T)
    columns = [f"S{i}" for i in range(_N)]
    panel = {}
    for name, values in base.items():
        array = values.copy()
        if name in deps and mode == "missing":
            array[_GAP, _SYMBOL] = np.nan
        elif name in deps and mode in ("up", "down"):
            # Eight ticks, still dyadic: large enough to flip a comparison with the
            # neighbouring bars, so up/down comparisons actually depend on the gap.
            step = 0.5 if name in _PRICES else 4096.0
            array[_GAP, _SYMBOL] += step if mode == "up" else -step
        panel[name] = pd.DataFrame(array, index=index, columns=columns)
    panel["amount"] = panel["vwap"] * panel["volume"]
    panel["sector"] = pd.DataFrame(np.tile(np.arange(_N) % 3, (_T, 1)), index=index, columns=columns)
    return panel


@pytest.mark.parametrize("alpha_id", FIXED)
def test_no_value_after_a_missing_bar_depends_on_it(alpha_id: str) -> None:
    registry = get_default_registry()
    deps = set(registry.get(alpha_id).meta.get("columns_required", []))
    base = _base()
    out = {
        mode: registry.compute(alpha_id, _panel(base, deps, mode)).iloc[_GAP:, _SYMBOL].to_numpy(dtype=float)
        for mode in ("missing", "up", "down")
    }

    moved = np.isfinite(out["up"]) != np.isfinite(out["down"])
    both = np.isfinite(out["up"]) & np.isfinite(out["down"])
    # A tolerance, not !=: a typical price divided by 3 is not dyadic, so a rolling
    # sum still carries a rounding residue far past the window (gtja191_128).
    moved[both] = ~np.isclose(out["up"][both], out["down"][both], rtol=1e-9, atol=1e-12)
    fabricated = np.flatnonzero(moved & np.isfinite(out["missing"]))

    assert moved.any(), "the perturbation reached no later bar; the oracle proves nothing"
    assert fabricated.size == 0, f"bars after the gap computed from it: {fabricated.tolist()}"
