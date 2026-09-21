"""Channel adapters log through stdlib ``logging``, not loguru (#1531, #1533).

``BaseChannel.logger`` is ``logging.getLogger(...)``. A loguru-style ``"{}"``
message with arguments makes stdlib raise inside handler formatting, so the
message is lost; a loguru-only ``.opt(exception=True)`` raises AttributeError
before anything is logged.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from src.channels.signal import SignalChannel

pytestmark = pytest.mark.unit


def test_signal_safe_handle_logs_and_swallows(caplog) -> None:
    """``_safe_handle`` promises to swallow; ``.opt()`` made it raise AttributeError."""
    host = SimpleNamespace(logger=logging.getLogger("tests.signal.safe_handle"))

    async def run() -> None:
        async with SignalChannel._safe_handle(host, "receive", {"envelope": "100%"}):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="tests.signal.safe_handle"):
        asyncio.run(run())

    record = next(r for r in caplog.records if r.name == "tests.signal.safe_handle")
    assert "Error in receive: boom" in record.getMessage()
    assert "100%" in record.getMessage()
    assert record.exc_info is not None
