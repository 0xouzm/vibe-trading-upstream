"""Semantic test for ``zoo/qlib158/wvma{5,10,20,30,60}``.

qlib's WVMA is ``ts_std(|ret|*v, n) / ts_mean(|ret|*v, n)`` — both the
numerator and denominator are volume-weighted by the *absolute* return, not
the signed one. This pins that the numerator does use ``|ret|``: with equal-
magnitude, alternating-sign returns and constant volume, ``|ret|*v`` is
constant within the window, so its rolling std is exactly zero. A rolling std
of the signed ``ret*v`` series would instead be large and nonzero.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.factors.zoo.qlib158.wvma5 import compute as wvma5
from src.factors.zoo.qlib158.wvma10 import compute as wvma10
from src.factors.zoo.qlib158.wvma20 import compute as wvma20
from src.factors.zoo.qlib158.wvma30 import compute as wvma30
from src.factors.zoo.qlib158.wvma60 import compute as wvma60


def _alternating_panel(window: int) -> dict[str, pd.DataFrame]:
    """One asset whose daily return alternates +/-2% with constant volume.

    ``|ret|*v`` is therefore constant across the window (2.0 per bar), so a
    correctly-implemented WVMA has a zero numerator and evaluates to 0.
    """
    n_bars = window + 1
    close = [100.0]
    for i in range(n_bars - 1):
        step = 1.02 if i % 2 == 0 else 0.98
        close.append(close[-1] * step)
    close_df = pd.DataFrame({"AAA": close})
    volume_df = pd.DataFrame({"AAA": [100.0] * n_bars})
    return {"close": close_df, "volume": volume_df}


@pytest.mark.parametrize(
    "compute,window",
    [(wvma5, 5), (wvma10, 10), (wvma20, 20), (wvma30, 30), (wvma60, 60)],
)
def test_wvma_numerator_uses_absolute_return(compute, window: int) -> None:
    panel = _alternating_panel(window)
    out = compute(panel)
    last = out["AAA"].iloc[-1]
    assert last == pytest.approx(0.0, abs=1e-9)
