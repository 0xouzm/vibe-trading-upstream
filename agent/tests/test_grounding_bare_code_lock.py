"""Spec decision 4: a bare code the user typed is locked by the fetch itself.

A user who writes ``159516`` has named the instrument; only the venue suffix is
the model's choice. ``get_market_data`` may fetch it before any resolver call,
and rows coming back for exactly one venue lock that venue. Rows for two venues
of the same code, the ``000xxx`` range (a Shanghai index and a Shenzhen stock
share those digits), a code the user never typed, any other tool, and a root
something else already locked all keep the resolver requirement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from src.agent.grounding import GroundingLedger

pytestmark = pytest.mark.unit


def _rows() -> list[dict[str, Any]]:
    return [
        {"trade_date": "2026-09-09", "open": 0.670, "high": 0.681, "low": 0.567, "close": 0.666}
    ]


def _fetch(
    ledger: GroundingLedger,
    call_id: str,
    codes: Iterable[str],
    *,
    tool: str = "get_market_data",
    rows_for: set[str] | None = None,
    success: bool = True,
):
    arguments = {"codes": list(codes)}
    decision = ledger.authorize_tool_call(
        tool,
        arguments,
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id=call_id,
        batch_identity_status=ledger.identity_status,
    )
    if decision.allowed:
        payload = {
            code: _rows() if rows_for is None or code in rows_for else []
            for code in arguments["codes"]
        }
        ledger.ingest_tool_result(
            tool_name=tool,
            arguments=arguments,
            result=json.dumps(payload),
            call_id=call_id,
            success=success,
        )
    return decision


def test_a_bare_code_is_locked_by_the_one_venue_that_returned_rows(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516 现在能买吗？给我一个买入价")

    decision = _fetch(ledger, "c1", ["159516.SZ"])

    assert decision.allowed is True
    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"159516.SZ"}
    record = next(r for r in ledger.identity_summary()["records"] if r["symbol"] == "159516.SZ")
    assert record["source_tool_call_id"] == "c1"


def test_two_venues_with_rows_leave_the_code_unlocked(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516 现在能买吗？给我一个买入价")

    _fetch(ledger, "c1", ["159516.SZ", "159516.SH"])

    assert ledger.identity_status == "ambiguous"
    assert ledger.authorized_symbols == set()


def test_a_second_venue_in_a_later_call_retracts_the_lock(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516 现在能买吗？给我一个买入价")

    _fetch(ledger, "c1", ["159516.SZ"])
    _fetch(ledger, "c2", ["159516.SH"])

    assert ledger.identity_status == "ambiguous"
    assert ledger.authorized_symbols == set()


def test_a_venue_that_returned_no_rows_does_not_count(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516 现在能买吗？给我一个买入价")

    _fetch(ledger, "c1", ["159516.SZ", "159516.SH"], rows_for={"159516.SZ"})

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"159516.SZ"}


def test_a_failed_fetch_locks_nothing(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516 现在能买吗？给我一个买入价")

    _fetch(ledger, "c1", ["159516.SZ"], success=False)

    assert ledger.authorized_symbols == set()


def test_the_000_range_is_never_locked_from_a_bare_code(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="000001 现在能买吗？给我一个买入价")

    decision = _fetch(ledger, "c1", ["000001.SZ"])

    assert decision.allowed is False
    assert decision.error_code == "identity_required"
    assert ledger.authorized_symbols == set()


def test_a_code_the_user_did_not_type_still_needs_the_resolver(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="这个机器人ETF现在能买吗？给我一个买入价")

    decision = _fetch(ledger, "c1", ["159516.SZ"])

    assert decision.allowed is False
    assert ledger.authorized_symbols == set()


def test_only_get_market_data_takes_the_bare_code_path(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516 现在能买吗？给我一个买入价")

    decision = _fetch(ledger, "c1", ["159516.SZ"], tool="technical_indicators")

    assert decision.allowed is False
    assert ledger.authorized_symbols == set()


def test_a_root_the_user_locked_with_a_suffix_keeps_its_venue(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="159516.SZ 现在能买吗？给我一个买入价")

    decision = _fetch(ledger, "c1", ["159516.SH"])

    assert decision.allowed is False
    assert decision.error_code == "identity_mismatch"
    assert ledger.authorized_symbols == {"159516.SZ"}
