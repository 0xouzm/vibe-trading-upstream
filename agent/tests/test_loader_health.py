"""Weekly live-source health canary (#1623).

For every registered loader that needs no credentials, fetch one known-liquid
symbol over a recent window and assert the normalized frame comes back
non-empty, in the expected columnar shape, with a last bar no older than two
weeks. The point is that a dead or drifted third-party endpoint fails here
before a user files it.

Off the default CI lane: this module only runs when VIBE_TRADING_LOADER_HEALTH=1
(the weekly loader-health workflow sets it). A loader that cannot be reached
from the runner at all (geo-blocked or network-filtered) is reported as
unreachable rather than failed when the fetch raises a network error; an empty,
malformed, or stale answer is a hard failure, because that is what drift
actually looks like. Note loaders swallow request failures into an empty
result, so an empty frame can mean runner connectivity or endpoint drift —
either way it deserves eyes, and the retry plus weekly cadence keep transient
flaps from crying wolf.
"""

from __future__ import annotations

import os
import socket
from datetime import date, timedelta

import pytest
import requests

from backtest.loaders.registry import LOADER_REGISTRY, _ensure_registered

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VIBE_TRADING_LOADER_HEALTH") != "1",
        reason="weekly loader-health lane only (VIBE_TRADING_LOADER_HEALTH=1)",
    ),
]

# One known-liquid symbol per no-auth loader, in that loader's own format.
CANARY_SYMBOLS: dict[str, str] = {
    "akshare": "601398",
    "baostock": "sh.601398",
    "binance": "BTC/USDT",
    "ccxt": "BTC/USDT",
    "eastmoney": "601398.SH",
    "mootdx": "601398",
    "nobitex": "BTC-IRT",
    "okx": "BTC-USDT",
    "pykrx": "005930.KS",
    "sina": "AAPL.US",
    "stooq": "AAPL.US",
    "tencent": "601398.SH",
    "wallex": "BTC-TMN",
    "yahoo": "AAPL",
    "yfinance": "AAPL",
}

_FRESHNESS_DAYS = 14  # equities close on weekends and holidays
_WINDOW_DAYS = 21  # fetch window comfortably covers the freshness window

# Exceptions that mean "the runner cannot reach this source", not "the source
# is broken". Those are reported, not failed.
_UNREACHABLE = (
    ConnectionError,
    TimeoutError,
    socket.timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
)


def _canary_loaders() -> list[tuple[str, type, str]]:
    _ensure_registered()
    out = []
    for name, cls in sorted(LOADER_REGISTRY.items()):
        if name not in CANARY_SYMBOLS or getattr(cls, "requires_auth", True):
            continue
        out.append((name, cls, CANARY_SYMBOLS[name]))
    return out


def _collect() -> list[str]:
    return [name for name, _, _ in _canary_loaders()]


@pytest.mark.parametrize("loader_name", _collect(), ids=_collect())
def test_loader_canary(loader_name: str) -> None:
    _, cls, symbol = next((n, c, s) for n, c, s in _canary_loaders() if n == loader_name)
    loader = cls()
    if not loader.is_available():
        pytest.skip(f"{loader_name} package not installed in this environment")

    end = date.today()
    start = end - timedelta(days=_WINDOW_DAYS)

    last_exc: Exception | None = None
    result = None
    for attempt in (1, 2):  # one retry rides out a flaky first connection
        try:
            result = loader.fetch(
                [symbol],
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                interval="1D",
            )
            last_exc = None
            break
        except _UNREACHABLE as exc:
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 - reported by class below
            last_exc = exc
            break

    if last_exc is not None:
        if isinstance(last_exc, _UNREACHABLE):
            pytest.skip(f"{loader_name} unreachable from this runner: {last_exc}")
        pytest.fail(f"{loader_name} raised {type(last_exc).__name__}: {last_exc}")

    assert result is not None and symbol in result, f"{loader_name} returned no frame for {symbol}: {result!r}"
    frame = result[symbol]
    assert not frame.empty, f"{loader_name} returned an empty frame for {symbol}"
    for col in ("open", "high", "low", "close", "volume"):
        assert col in frame.columns, f"{loader_name} frame missing {col}"

    last_bar = frame.index[-1]
    last_date = last_bar.date() if hasattr(last_bar, "date") else last_bar
    age = (date.today() - last_date).days
    assert age <= _FRESHNESS_DAYS, f"{loader_name} last bar for {symbol} is {last_date} ({age}d old)"
