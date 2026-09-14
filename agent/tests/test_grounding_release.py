"""Release path for a rejected draft: repair, redact, derivation spellings.

A rejected draft used to end one of two ways: corrected by the model inside
the revision cap, or replaced wholesale by the canned safe fallback. The
second threw away every sentence the gate had no issue with, after the user
had waited through every revision — a ten-minute run that ends in three
sentences of refusal (owner-forwarded screenshot, 2026-09-09). These tests pin
the third outcome: the draft is released with the rejected figures cut out
and re-checked by the same gate. They also pin the zero-round provenance
repair, and the derivation spellings a correcting model actually writes,
which the gate rejected with the multiplier read as a quoted price.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from src.agent.grounding import (
    _PRICE_INDICATOR_TOKENS,
    _REDACTION_MARKER_EN,
    _REDACTION_MARKER_ZH,
    GroundingLedger,
    ValidationResult,
    _CLAUSE_SEPARATOR_RE,
    _THOUSANDS_SEPARATOR_RE,
    _clause_spans,
    _is_price_denominated_indicator,
    _lines_with_offsets,
)
from src.agent.loop import AgentLoop
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter
from tests.message_roles_helpers import assert_system_messages_only_lead

pytestmark = pytest.mark.unit

SYMBOL = "562500.SS"
HDR = "562500.SS（Yahoo，CNY）最新收盘价 1.171 元。"


def _market_payload() -> str:
    return json.dumps(
        {
            SYMBOL: [
                {"trade_date": "2026-06-23", "open": 1.141, "high": 1.164, "low": 1.121, "close": 1.137, "volume": 1},
                {"trade_date": "2026-06-24", "open": 1.137, "high": 1.180, "low": 1.110, "close": 1.171, "volume": 1},
            ],
            "_provenance": {
                SYMBOL: {
                    "source": "yahoo",
                    "requested_source": "auto",
                    "detected_source": "yahoo",
                    "fallback_used": False,
                    "currency_conversion": "none",
                }
            },
        }
    )


def _indicator_payload(**extra: Any) -> str:
    indicators: dict[str, Any] = {
        "rsi_14": 55.2,
        "sma_20": 1.150,
        "sma_50": 1.090,
        "bollinger": {"upper": 1.21, "middle": 1.15, "lower": 1.09},
        "macd": {"macd_line": -0.0187, "signal_line": -0.0134, "histogram": -0.0053},
    }
    indicators.update(extra)
    return json.dumps(
        {
            "ok": True,
            "symbol": SYMBOL,
            "interval": "1d",
            "latest_close": 1.171,
            "latest_date": "2026-06-24",
            "indicators": indicators,
        }
    )


def _resolver_payload() -> str:
    return json.dumps(
        {
            "ok": True,
            "source": "symbol_search",
            "data": {
                "query": "机器人ETF",
                "count": 1,
                "candidates": [
                    {
                        "symbol": SYMBOL,
                        "name": "机器人ETF",
                        "market": "cn",
                        "type": "ETF",
                        "source": "yahoo",
                        "also_from": ["eastmoney"],
                    }
                ],
                "sources": {"eastmoney": "ok", "yahoo": "ok"},
            },
        },
        ensure_ascii=False,
    )


def _ledger(
    tmp_path: Path,
    message: str = "请分析 562500.SS 并给出买入价",
    *,
    market: bool = True,
    indicators: bool = True,
    **extra_indicators: Any,
) -> GroundingLedger:
    ledger = GroundingLedger(run_dir=tmp_path, user_message=message)
    if market:
        ledger.ingest_tool_result(
            tool_name="get_market_data",
            arguments={"codes": [SYMBOL]},
            result=_market_payload(),
            call_id="prices",
            success=True,
        )
    if indicators:
        ledger.ingest_tool_result(
            tool_name="technical_indicators",
            arguments={"symbol": SYMBOL},
            result=_indicator_payload(**extra_indicators),
            call_id="indicators",
            success=True,
        )
    return ledger


def _codes(result: ValidationResult) -> list[str]:
    return [str(issue.get("code")) for issue in result.issues]


# ---------------------------------------------------------------------------
# Price-denominated indicator values are observed evidence
# ---------------------------------------------------------------------------


def test_price_denominated_indicator_values_are_observed(tmp_path: Path) -> None:
    """A moving average or band the session fetched is a quoted tool value."""
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(HDR + " SMA20 位于 1.150，布林下轨 1.09 为支撑位。")

    assert result.valid is True, result.issues


def test_non_price_indicator_values_still_do_not_ground_a_price(tmp_path: Path) -> None:
    """RSI and a volume average are not prices; quoting one as a price stays rejected."""
    ledger = _ledger(tmp_path, volume_ma_20=1.30)

    rsi_as_price = ledger.validate_final_answer(HDR + " 现价 55.2 元。")
    volume_ma_as_price = ledger.validate_final_answer(HDR + " 现价 1.30 元。")

    assert "numeric_claim_conflict" in _codes(rsi_as_price)
    assert "numeric_claim_conflict" in _codes(volume_ma_as_price)


def test_indicator_evidence_never_satisfies_a_labelled_ohlc_cell(tmp_path: Path) -> None:
    """A close column must match a close; an SMA equal to the cell does not count."""
    ledger = _ledger(tmp_path, sma_20=1.150)

    table = HDR + "\n\n| 日期 | 收盘价 |\n|---|---|\n| 2026-06-24 | 1.150 |\n"
    result = ledger.validate_final_answer(table)

    assert result.valid is False
    assert any(issue.get("value") == 1.15 for issue in result.issues)


# ---------------------------------------------------------------------------
# Derivation spellings a correcting model writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "separator",
    ["=", "≈", "≒", "约", "约等于"],
)
def test_derivation_accepts_every_result_separator(tmp_path: Path, separator: str) -> None:
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(
        HDR + f" 基于收盘价 1.171 × 0.97 {separator} 1.136 作为买入价。"
    )

    assert result.valid is True, (separator, result.issues)


def test_derivation_with_wrong_result_stays_rejected_whatever_the_separator(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(HDR + " 基于收盘价 1.171 × 0.97 ≈ 1.20 作为买入价。")

    assert result.valid is False
    assert 1.2 in {issue.get("value") for issue in result.issues}


def test_derivation_accepts_an_observed_indicator_as_input(tmp_path: Path) -> None:
    """`基于 SMA20 1.150 × 0.95 = 1.093`: the input is a tool value, the arithmetic holds."""
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(HDR + " 基于 SMA20 1.150 × 0.95 = 1.093 作为买入价。")

    assert result.valid is True, result.issues


def test_derivation_from_a_non_price_indicator_stays_rejected(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(HDR + " 基于 RSI 55.2 × 0.02 = 1.104 作为买入价。")

    assert result.valid is False


def test_bare_arithmetic_equation_counts_as_a_derivation(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    good = ledger.validate_final_answer(HDR + " 买入价 = 1.171 × 0.97 = 1.136。")
    wrong = ledger.validate_final_answer(HDR + " 买入价 = 1.171 × 0.97 = 1.20。")
    unanchored = ledger.validate_final_answer(HDR + " 买入价 = 1.50 × 0.97 = 1.455。")

    assert good.valid is True, good.issues
    assert wrong.valid is False
    assert unanchored.valid is False


def test_an_entry_without_a_formula_is_still_rejected(tmp_path: Path) -> None:
    """The product rule stands: a proposed entry must be observed or derived."""
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(HDR + " 建议买入价 1.10 元。")

    assert _codes(result) == ["numeric_claim_conflict"]


# ---------------------------------------------------------------------------
# Analysis figures that are arithmetic on observed endpoints
# ---------------------------------------------------------------------------


def test_drawdown_against_an_observed_high_is_arithmetic_not_a_backtest_metric(tmp_path: Path) -> None:
    """(1.110 − 1.180) / 1.180 = −5.93%: "约 6%" rounds to it, "约 8%" does not."""
    ledger = _ledger(tmp_path)

    rounded = ledger.validate_final_answer(HDR + " 最低价 1.110 元，较高点 1.180 元已回撤约 6%。")
    # 5.9% matches only the high→low direction by magnitude: the reversed
    # pair (1.110→1.180) is +6.31%, outside a one-decimal claim's half unit.
    precise = ledger.validate_final_answer(HDR + " 最低价 1.110 元，较高点 1.180 元已回撤 5.9%。")
    wrong = ledger.validate_final_answer(HDR + " 最低价 1.110 元，较高点 1.180 元已回撤约 8%。")
    unanchored = ledger.validate_final_answer(HDR + " 策略最大回撤 6%。")

    assert rounded.valid is True, rounded.issues
    assert precise.valid is True, precise.issues
    assert "analysis_claim_unavailable" in _codes(wrong)
    assert "analysis_claim_unavailable" in _codes(unanchored)


def test_return_figure_is_matched_at_its_written_precision(tmp_path: Path) -> None:
    """1.121 → 1.180 is +5.26%: "约 5%" and "5.3%" round to it; "6%" and "5.4%" do not."""
    ledger = _ledger(tmp_path)
    line = "从最低价 1.121 元涨到最高价 1.180 元，区间收益率{}。"

    for claim in ("约 5%", "5.3%", "5.26%"):
        result = ledger.validate_final_answer(HDR + " " + line.format(claim))
        assert result.valid is True, (claim, result.issues)
    for claim in ("约 6%", "5.4%", "5.20%"):
        result = ledger.validate_final_answer(HDR + " " + line.format(claim))
        assert "analysis_claim_unavailable" in _codes(result), claim


# ---------------------------------------------------------------------------
# Provenance repair: a missing word is appended, not regenerated
# ---------------------------------------------------------------------------


def test_repair_provenance_appends_the_missing_words(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    draft = "562500.SS 最新收盘价 1.171 元，处于下行趋势。"
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["data_source_not_surfaced"]

    repaired = ledger.repair_provenance(draft, validation)

    assert repaired is not None
    assert repaired.startswith(draft)
    assert "yahoo" in repaired and "CNY" in repaired
    assert ledger.validate_final_answer(repaired).valid is True


def test_repair_provenance_refuses_a_draft_with_a_numeric_issue(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    draft = "562500.SS 最新收盘价 1.50 元。"
    validation = ledger.validate_final_answer(draft)
    assert "numeric_claim_conflict" in _codes(validation)

    assert ledger.repair_provenance(draft, validation) is None


def test_repair_provenance_refuses_without_price_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, market=False, indicators=False)
    validation = ValidationResult(valid=False, issues=[{"code": "data_source_not_surfaced"}])

    assert ledger.repair_provenance("x", validation) is None


# ---------------------------------------------------------------------------
# Redacted release: the analysis survives, the rejected figure does not
# ---------------------------------------------------------------------------


def test_redacted_release_keeps_the_analysis_and_cuts_the_figure(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    draft = HDR + " 均线空头排列，处于下行趋势。建议买入价 1.10 元，分批建仓。"
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["numeric_claim_conflict"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "1.171" in released and "下行趋势" in released and "分批建仓" in released
    assert "1.10" not in released
    assert _REDACTION_MARKER_ZH in released
    assert "※ 略去 1 处" in released
    assert "（略※）元" not in released and "建议买入价（略※）" in released
    assert "1.11–1.18 CNY" in released
    assert ledger.validate_final_answer(released).valid is True


def test_redacted_release_reports_every_cut(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 1.10 元。目标买入价 1.05 元。"
    validation = ledger.validate_final_answer(draft)
    assert sorted(issue["value"] for issue in validation.issues) == [1.05, 1.1]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "2 处" in released
    assert "1.10" not in released and "1.05" not in released


def test_redacted_release_cuts_what_the_recheck_reveals(tmp_path: Path) -> None:
    """The analysis validator reports one figure per clause; the second shows on recheck."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 策略最大回撤 12% 且夏普 2.5 处于下行趋势。"
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == ["12%"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "12%" not in released and "2.5" not in released
    assert "下行趋势" in released and "2 处" in released


def test_redacted_release_gives_up_after_the_pass_bound(tmp_path: Path) -> None:
    """Four figures in one clause surface one per pass; three passes are the bound."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 策略最大回撤 12% 夏普 2.5 胜率 55% 年化收益 30% 处于下行趋势。"
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == ["12%"]

    assert ledger.redacted_release(draft, validation) is None


def test_redacted_release_adds_provenance_the_cut_draft_still_lacks(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    draft = "562500.SS 最新收盘价 1.171 元。建议买入价 1.10 元。"
    validation = ledger.validate_final_answer(draft)
    assert set(_codes(validation)) == {"numeric_claim_conflict", "data_source_not_surfaced"}

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "yahoo" in released and "1.10" not in released


def test_redacted_release_never_cuts_ticker_digits(tmp_path: Path) -> None:
    """A clause attributing figures to an unhandled symbol loses its figures, not the ticker."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 同业 000001.SZ 收盘价 12.3 元。"
    validation = ledger.validate_final_answer(draft)
    assert "unsourced_symbol_figures" in _codes(validation)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "000001.SZ" in released
    assert "12.3" not in released


def test_redacted_release_writes_english_for_an_english_user(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, message="Analyse 562500.SS and give me an entry price")
    draft = "562500.SS (Yahoo, CNY) last close 1.171. Suggested entry price $1.10."
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["numeric_claim_conflict"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert _REDACTION_MARKER_EN in released and "※ 1 figure(s)" in released
    assert "1.10" not in released and "$" not in released.split("※")[0]


def test_redacted_release_refuses_without_price_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, market=False, indicators=False)
    draft = HDR + " 建议买入价 1.10 元。"
    validation = ledger.validate_final_answer(draft)
    assert "numeric_claim_unavailable" in _codes(validation)

    assert ledger.redacted_release(draft, validation) is None


def test_redacted_release_refuses_an_unlocked_identity(tmp_path: Path) -> None:
    """A failed resolution leaves identity invalidated; no cut-down draft may ship."""
    ledger = _ledger(tmp_path, message="分析机器人ETF并给出买入价", indicators=False)
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "机器人ETF"},
        result=json.dumps({"ok": False, "error": "timeout"}),
        call_id="resolve",
        success=False,
    )
    draft = HDR + " 建议买入价 1.10 元。"
    validation = ledger.validate_final_answer(draft)
    assert "identity_not_locked" in _codes(validation)

    assert ledger.redacted_release(draft, validation) is None


def test_redacted_release_refuses_a_code_it_cannot_cut(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    validation = ValidationResult(
        valid=False,
        issues=[
            {"code": "numeric_claim_conflict", "claim": "建议买入价 1.10 元", "value": 1.1},
            {"code": "listed_identity_relabelled_private", "symbols": [SYMBOL]},
        ],
    )

    assert ledger.redacted_release(HDR + " 建议买入价 1.10 元。", validation) is None


def test_redacted_release_refuses_when_the_clause_cannot_be_located(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    validation = ValidationResult(
        valid=False,
        issues=[{"code": "numeric_claim_conflict", "claim": "not in the draft 1.10", "value": 1.1}],
    )

    assert ledger.redacted_release(HDR + " 建议买入价 1.10 元。", validation) is None


def test_redacted_release_cuts_a_figure_the_recheck_finds_in_a_table(tmp_path: Path) -> None:
    """A figure withheld from the first pass is still a gate finding on recheck, so it is cut."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 1.10 元。\n\n| 日期 | 收盘价 |\n|---|---|\n| 2026-06-24 | 1.10 |\n"
    validation = ledger.validate_final_answer(draft)
    prose_only = ValidationResult(
        valid=False,
        issues=[issue for issue in validation.issues if issue.get("field") is None],
    )
    assert prose_only.issues

    released = ledger.redacted_release(draft, prose_only)

    assert released is not None
    assert "1.10" not in released and "2 处" in released


def test_redacted_release_refuses_when_the_recheck_finds_what_it_cannot_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only text that passes the gate is released; an identity finding on recheck ends it."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 1.10 元。"
    validation = ledger.validate_final_answer(draft)
    # The release path re-checks through ``_validate(..., record=False)`` so
    # its rechecks are not counted as rejected model drafts.
    monkeypatch.setattr(
        ledger,
        "_validate",
        lambda content, *, record: ValidationResult(
            valid=False, issues=[{"code": "identity_not_locked", "status": "conflicting"}]
        ),
    )

    assert ledger.redacted_release(draft, validation) is None


# ---------------------------------------------------------------------------
# Loop integration
# ---------------------------------------------------------------------------


# Both stubs take ``result`` in ``__init__`` on purpose. ``build_registry`` walks
# ``BaseTool.__subclasses__()`` and caches whatever it can instantiate, so a
# no-argument stub named ``get_market_data`` defined at collection time REPLACES
# the real tool for every later test in the session — fifteen MCP market-data
# tests failed that way on 2026-09-09. A required constructor argument makes
# discovery skip the stub (logged as "Failed to register"), which is the
# convention the other grounding test modules already rely on.
class _ResolverTool(BaseTool):
    name = "search_symbol"
    description = "Resolve a company or instrument name."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    repeatable = True

    def __init__(self, result: str) -> None:
        self.result = result

    def execute(self, **kwargs: Any) -> str:
        return self.result


class _MarketTool(BaseTool):
    name = "get_market_data"
    description = "Fetch OHLCV bars."
    parameters = {
        "type": "object",
        "properties": {"codes": {"type": "array", "items": {"type": "string"}}},
        "required": ["codes"],
    }
    repeatable = True

    def __init__(self, result: str) -> None:
        self.result = result

    def execute(self, **kwargs: Any) -> str:
        return self.result


class _Response:
    def __init__(self, *, content: str = "", tool_calls: list[SimpleNamespace] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None
        self.has_tool_calls = bool(self.tool_calls)


class _ScriptedLLM:
    """Plays a script, then keeps returning its last response forever."""

    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.messages_history: list[list[dict[str, Any]]] = []

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        timeout: int | None = None,
        idle_timeout_s: float | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> _Response:
        self.calls += 1
        self.messages_history.append(list(messages))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if response.content and on_text_chunk:
            on_text_chunk(response.content)
        return response

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Response:
        return _Response()


def _tool_call(call_id: str, tool_name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, name=tool_name, arguments=arguments)


def _run(
    tmp_path: Path, llm: _ScriptedLLM, *, max_iterations: int
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]], AgentLoop]:
    registry = ToolRegistry()
    registry.register(_ResolverTool(_resolver_payload()))
    registry.register(_MarketTool(_market_payload()))
    events: list[tuple[str, dict[str, Any]]] = []
    agent = AgentLoop(
        registry=registry,
        llm=llm,
        max_iterations=max_iterations,
        event_callback=lambda event, data: events.append((event, data)),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    return agent.run("请分析机器人ETF并给出买入价"), events, agent


_SCRIPT_HEAD = [
    _Response(tool_calls=[_tool_call("resolve", "search_symbol", query="机器人ETF")]),
    _Response(
        tool_calls=[
            _tool_call(
                "prices",
                "get_market_data",
                codes=[SYMBOL],
                start_date="2026-06-23",
                end_date="2026-06-24",
                source="auto",
            )
        ]
    ),
]


def test_loop_releases_the_redacted_draft_instead_of_the_canned_refusal(tmp_path: Path) -> None:
    """A model that never drops its invented entry still delivers its analysis."""
    stubborn = (
        "562500.SS（Yahoo，CNY）在 2026-06-23 的已观测收盘价为 1.137，均线空头排列。"
        "建议买入价为 0.881。"
    )
    llm = _ScriptedLLM(_SCRIPT_HEAD + [_Response(content=stubborn)])

    result, events, agent = _run(tmp_path, llm, max_iterations=8)

    assert result["status"] == "success"
    assert result.get("degraded") is True
    content = result["content"]
    assert "1.137" in content and "均线空头排列" in content
    assert "0.881" not in content
    assert _REDACTION_MARKER_ZH in content and "※ 略去" in content
    assert "已拒绝上一版答案" not in content
    streamed = "".join(data.get("delta", "") for event, data in events if event == "text_delta")
    assert "0.881" not in streamed and _REDACTION_MARKER_ZH in streamed
    trace = TraceWriter.read(tmp_path / "run")
    assert [e for e in trace if e.get("type") == "answer_released_redacted"]
    end = next(e for e in trace if e.get("type") == "end")
    assert "redacted" in end.get("reason", "")
    # The count in that reason is the one number the user and the offline
    # verifier have for how much of the run was spent being refused, and the
    # release path's own rechecks must not inflate it.
    reported = re.search(r"after (\d+) rejected drafts", end["reason"])
    assert reported is not None, end["reason"]
    assert int(reported.group(1)) == agent._grounding.validation_count
    assert int(reported.group(1)) == 4
    assert llm.calls >= 5
    assert_system_messages_only_lead(llm.messages_history)


def test_loop_repairs_a_missing_source_word_without_another_model_round(tmp_path: Path) -> None:
    missing_source = "562500.SS 在 2026-06-23 的已观测收盘价为 1.137 CNY，均线空头排列。"
    llm = _ScriptedLLM(
        _SCRIPT_HEAD + [_Response(content=missing_source), _Response(content="MUST NOT BE ASKED")]
    )

    result, events, agent = _run(tmp_path, llm, max_iterations=8)

    assert result["status"] == "success"
    assert result.get("degraded") is None
    assert llm.calls == 3
    content = result["content"]
    assert content.startswith(missing_source)
    assert "yahoo" in content and "数据说明" in content
    streamed = "".join(data.get("delta", "") for event, data in events if event == "text_delta")
    assert "yahoo" in streamed
    trace = TraceWriter.read(tmp_path / "run")
    assert [e for e in trace if e.get("type") == "answer_repaired"]
    assert not [e for e in trace if e.get("type") == "answer_rejected"]
    # The repair costs no model round, so it must cost no revision either.
    # ``validation_count`` is both the revision budget and the rejected-draft
    # number in the run reason; recording the recheck spent one of each on a
    # draft the model never wrote.
    assert agent._grounding.validation_count == 1


# ---------------------------------------------------------------------------
# Endpoint arithmetic: direction and units
#
# The four cases below were all live escapes on the first cut of this change
# and every one of them left the rest of this module green, which is the
# mutation signal that the exemption had only positive-side coverage.
# ---------------------------------------------------------------------------

WIDE_BARS = [
    {"trade_date": "2026-05-10", "open": 1.050, "high": 1.053, "low": 1.040, "close": 1.050, "volume": 1},
    {"trade_date": "2026-09-09", "open": 0.670, "high": 0.680, "low": 0.660, "close": 0.666, "volume": 1},
]
WIDE_HDR = "562500.SS（Yahoo，CNY）最新收盘价 0.666 元。"


def _wide_ledger(tmp_path: Path) -> GroundingLedger:
    """A ledger whose two observed endpoints are 1.053 (high) and 0.666 (close).

    The fall is −36.75%; the same pair read the other way is +58.11%. A dense
    real frame cannot probe this — 83 bars × 4 fields put some print within
    0.5% of almost any two-decimal number — so the endpoints are sparse and
    far apart on purpose.
    """
    ledger = GroundingLedger(run_dir=tmp_path, user_message="请分析 562500.SS 的回撤")
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": [SYMBOL]},
        result=json.dumps(
            {
                SYMBOL: WIDE_BARS,
                "_provenance": {
                    SYMBOL: {
                        "source": "yahoo",
                        "requested_source": "auto",
                        "detected_source": "yahoo",
                        "fallback_used": False,
                        "currency_conversion": "none",
                    }
                },
            }
        ),
        call_id="prices",
        success=True,
    )
    return ledger


def test_a_drawdown_may_only_be_grounded_by_the_falling_direction(tmp_path: Path) -> None:
    """A drawdown is a fall; the inverse rise between the same endpoints is not one."""
    line = "最新收盘 0.666 元 较 5 月高点 1.053 元 最大回撤约 {}。"

    for claim in ("36.8%", "37%", "-36.8%"):
        result = _wide_ledger(tmp_path).validate_final_answer(WIDE_HDR + " " + line.format(claim))
        assert result.valid is True, (claim, result.issues)
    # 58.11% is (1.053 − 0.666) / 0.666 — the rise back, not the drawdown.
    for claim in ("58.1%", "58%", "45%"):
        result = _wide_ledger(tmp_path).validate_final_answer(WIDE_HDR + " " + line.format(claim))
        assert "analysis_claim_unavailable" in _codes(result), claim


def test_magnitude_is_pinned_to_drawdown_and_off_for_a_return(tmp_path: Path) -> None:
    """The same fall written as a positive RETURN keeps its sign check."""
    fall = "最新收盘 0.666 元 较 5 月高点 1.053 元 {}约 {}。"

    as_drawdown = _wide_ledger(tmp_path).validate_final_answer(
        WIDE_HDR + " " + fall.format("最大回撤", "37%")
    )
    signed_return = _wide_ledger(tmp_path).validate_final_answer(
        WIDE_HDR + " " + fall.format("区间收益率", "-37%")
    )
    unsigned_return = _wide_ledger(tmp_path).validate_final_answer(
        WIDE_HDR + " " + fall.format("区间收益率", "37%")
    )

    assert as_drawdown.valid is True, as_drawdown.issues
    assert signed_return.valid is True, signed_return.issues
    assert "analysis_claim_unavailable" in _codes(unsigned_return)


@pytest.mark.parametrize("metric", ["年化波动率", "胜率", "夏普比率"])
def test_endpoint_arithmetic_grounds_only_returns_and_drawdowns(
    tmp_path: Path, metric: str
) -> None:
    """A volatility / win rate / Sharpe equal to the endpoint growth is not grounded by it."""
    result = _wide_ledger(tmp_path).validate_final_answer(
        WIDE_HDR + f" 最新收盘 0.666 元 较 5 月高点 1.053 元 {metric} 58.1%。"
    )

    assert "analysis_claim_unavailable" in _codes(result)


def test_a_percent_figure_is_compared_in_percentage_points_only(tmp_path: Path) -> None:
    """"约 0%" is not within half a unit of a 58% move — the half unit is in pp."""
    line = "最新收盘 0.666 元 较 5 月高点 1.053 元 区间收益率约 {}。"

    for claim in ("0%", "1%", "-1%", "2%"):
        result = _wide_ledger(tmp_path).validate_final_answer(WIDE_HDR + " " + line.format(claim))
        assert "analysis_claim_unavailable" in _codes(result), claim
    for claim in ("58%", "58.1%", "58.11%"):
        result = _wide_ledger(tmp_path).validate_final_answer(WIDE_HDR + " " + line.format(claim))
        assert result.valid is True, (claim, result.issues)


def test_the_written_precision_band_is_half_a_unit_not_more(tmp_path: Path) -> None:
    """The band is exactly half a unit of the last written digit — pinned from both sides.

    1.110 → 1.171 is +5.4955%. An integer 5% is 0.4955 away and must be
    accepted; 6% is 0.5045 away and must be rejected. The pair straddles 0.5
    by a twentieth of a point, so inflating the multiplier even to 0.7 —
    "a claim 0.7pp off the arithmetic is grounded" — flips the second
    assertion. The nearest existing fixture is 0.737 away and cannot see it.
    """
    ledger = _ledger(tmp_path)
    line = "从最低价 1.110 元涨到收盘 1.171 元，区间收益率约 {}。"

    inside = ledger.validate_final_answer(HDR + " " + line.format("5%"))
    outside = ledger.validate_final_answer(HDR + " " + line.format("6%"))
    # One decimal narrows the band to 0.05: 5.4% is 0.0955 away.
    one_decimal = ledger.validate_final_answer(HDR + " " + line.format("5.4%"))

    assert inside.valid is True, inside.issues
    assert "analysis_claim_unavailable" in _codes(outside)
    assert "analysis_claim_unavailable" in _codes(one_decimal)


# ---------------------------------------------------------------------------
# What a derivation justifies, and in which language
# ---------------------------------------------------------------------------


def test_a_derivation_exempts_its_own_values_not_its_neighbours(tmp_path: Path) -> None:
    """One valid equation used to exempt the whole clause it sat in.

    The system prompt now asks the model to write exactly this shape, so an
    invented entry price beside a correct formula is the ordinary draft, not a
    corner case.
    """
    ledger = _ledger(tmp_path)

    riding_along = ledger.validate_final_answer(
        HDR + " 建议买入价 0.95 元（收盘 1.171 × 0.97 = 1.136 参考）。"
    )
    keyword_form = ledger.validate_final_answer(
        HDR + " 基于收盘价 1.171 × 0.97 = 1.136 的买入价 1.30 元。"
    )
    formula_only = ledger.validate_final_answer(HDR + " 基于收盘价 1.171 × 0.97 = 1.136 买入。")

    assert 0.95 in {issue.get("value") for issue in riding_along.issues}
    assert 1.3 in {issue.get("value") for issue in keyword_form.issues}
    # The operands and the result themselves stay exempt.
    assert formula_only.valid is True, formula_only.issues


def test_a_derivation_gets_the_same_verdict_in_both_languages(tmp_path: Path) -> None:
    """Every result separator must work in English too, or the fix moves the bug.

    ``test_derived_return_exemption_is_structural_not_phrasal`` is the standing
    rule; the first cut of this change added ≈ / 约 / 约等于 and left
    "approximately" rejected, so a correct English derivation still burned a
    revision round.
    """

    def verdict(answer: str) -> bool:
        return _ledger(tmp_path).validate_final_answer(answer).valid

    english_header = "562500.SS (Yahoo, CNY) last close 1.171."
    for chinese_separator, english_separator in (
        ("=", "="),
        ("≈", "≈"),
        ("约", "approximately"),
        ("约", "about"),
        ("约等于", "around"),
        ("约", "roughly"),
        ("≒", "~"),
    ):
        chinese = verdict(
            HDR + f" 基于收盘价 1.171 × 0.97 {chinese_separator} 1.136 作为买入价。"
        )
        english = verdict(
            english_header
            + f" Based on the close 1.171 * 0.97 {english_separator} 1.136 as the entry."
        )
        assert chinese is english is True, (chinese_separator, english_separator)

    # Wrong arithmetic stays rejected on both sides of the separator list.
    assert verdict(HDR + " 基于收盘价 1.171 × 0.97 约 1.20 作为买入价。") is False
    assert (
        verdict(english_header + " Based on the close 1.171 * 0.97 approximately 1.20.")
        is False
    )


def test_a_restated_value_is_not_a_derivation(tmp_path: Path) -> None:
    """A derivation needs an operation, not "1.171 约 1.171".

    This is what makes it safe for ``_derivation_justified_values`` to carry
    no keyword or regex precondition: a "formula" of one operand is a
    restatement, and ``_evaluate_formula``'s two-operand rule is the thing
    that refuses it.
    """
    ledger = _ledger(tmp_path)
    records = ledger._comparable_price_records()

    # The evidence symbol is canonicalised (.SS → .SH); pass it as the records
    # carry it, not as the answer spells it. Asserted on the values the
    # production path actually consumes, not on a bool no caller reads.
    assert ledger._derivation_justified_values("收盘 1.171 约 1.171", records, "562500.SH") == []
    assert ledger._derivation_justified_values(
        "收盘 1.171 × 0.97 约 1.136", records, "562500.SH"
    ) == [1.171, 0.97, 1.136]


# ---------------------------------------------------------------------------
# Which evidence may back which claim
# ---------------------------------------------------------------------------


def test_an_indicator_never_backs_a_claim_labelled_as_an_ohlc_field(tmp_path: Path) -> None:
    """Same rule in prose as in a table: an SMA is not the close.

    The table path filters evidence on the column's field, so an indicator
    could never satisfy a 收盘价 column; prose passed no field at all, and
    "收盘价为 1.150 元" was grounded by the session's sma_20 while the observed
    close was 1.171. The undated cell is the case the field label actually
    decides — a dated cell is settled by the date match before the label is
    consulted.
    """
    ledger = _ledger(tmp_path, sma_20=1.150)

    prose = ledger.validate_final_answer(HDR + " 收盘价为 1.150 元。")
    english_prose = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171. The closing price was 1.150."
    )
    undated_cell = ledger.validate_final_answer(
        HDR + "\n\n| 代码 | 收盘价 |\n|---|---|\n| 562500.SS | 1.150 |\n"
    )
    # An unlabelled price word keeps the indicator exemption the release path
    # was written for.
    unlabelled = ledger.validate_final_answer(HDR + " SMA20 位于 1.150。")

    assert "numeric_claim_conflict" in _codes(prose)
    assert "numeric_claim_conflict" in _codes(english_prose)
    assert "numeric_claim_conflict" in _codes(undated_cell)
    assert unlabelled.valid is True, unlabelled.issues


def _two_symbol_ledger(tmp_path: Path) -> GroundingLedger:
    """Prices for two symbols, an indicator for one of them only."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="对比 562500.SH 和 600519.SH")
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["562500.SH", "600519.SH"]},
        result=json.dumps(
            {
                "562500.SH": [
                    {"trade_date": "2026-06-24", "open": 1.137, "high": 1.180, "low": 1.110, "close": 1.171}
                ],
                "600519.SH": [
                    {"trade_date": "2026-06-24", "open": 1400.0, "high": 1425.0, "low": 1395.0, "close": 1420.0}
                ],
                "_provenance": {
                    "562500.SH": {"source": "yahoo", "currency_conversion": "none"},
                    "600519.SH": {"source": "yahoo", "currency_conversion": "none"},
                },
            }
        ),
        call_id="prices",
        success=True,
    )
    ledger.ingest_tool_result(
        tool_name="technical_indicators",
        arguments={"symbol": "562500.SH"},
        result=json.dumps({"ok": True, "symbol": "562500.SH", "indicators": {"sma_20": 1.150}}),
        call_id="indicators",
        success=True,
    )
    return ledger


def test_one_symbols_indicator_does_not_ground_an_unattributed_claim(tmp_path: Path) -> None:
    """The cross-symbol union is for OHLC quotes; an indicator is symbol-bound.

    ``technical_indicators`` returns a level for one symbol, so 562500's SMA
    must not satisfy a clause on a line that names two instruments and
    attributes the figure to neither. The claim is a LEVEL claim on both
    sides, because that is the only claim an indicator may answer at all — a
    spot quote is settled by the indicator-admissibility rule before the
    symbol is consulted.
    """
    header = "562500.SH 与 600519.SH（Yahoo，CNY）对比。"

    unattributed_indicator = _two_symbol_ledger(tmp_path).validate_final_answer(
        header + "布林下轨 1.150 元为支撑位。"
    )
    # Both halves of the rule the union was argued for stay intact.
    unattributed_ohlc = _two_symbol_ledger(tmp_path).validate_final_answer(
        header + "现价 1420 元。"
    )
    attributed_indicator = _two_symbol_ledger(tmp_path).validate_final_answer(
        header + "562500.SH 布林下轨 1.150 元为支撑位。"
    )

    assert "numeric_claim_conflict" in _codes(unattributed_indicator)
    assert unattributed_ohlc.valid is True, unattributed_ohlc.issues
    assert attributed_indicator.valid is True, attributed_indicator.issues


@pytest.mark.parametrize(
    "path",
    [
        "indicators.sma_20", "indicators.ema_50", "indicators.ma", "indicators.vwap",
        "indicators.atr", "indicators.bollinger.lower", "indicators.bollinger.upper",
        "indicators.keltner.upper", "indicators.donchian.lower", "indicators.supertrend",
        "indicators.ichimoku.tenkan", "indicators.highest_20", "indicators.support",
        "indicators.resistance", "pivot.pp", "pivot.r1", "indicators.r1", "levels.s2",
    ],
)
def test_price_denominated_leaves_are_evidence(path: str) -> None:
    assert _is_price_denominated_indicator(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # Derived shapes under a price family: a difference, a gap, a crossover,
        # a direction, a strength score, a bandwidth. The old rule admitted all
        # of these because some segment of the path names a price family.
        "indicators.ma_diff", "indicators.ema_gap", "indicators.sma_cross",
        "indicators.supertrend_direction", "indicators.support_strength",
        "indicators.atr_multiple", "indicators.ma_deviation", "indicators.psar_reversals",
        "indicators.boll_bandwidth", "indicators.bollinger.bandwidth",
        "summary.resistance_touches",
        # Parameters, not levels.
        "params.ma_period", "entry_condition.ma_window", "settings.ema_span",
        "config.ma_length",
        # A generic band leaf with no price family behind it.
        "summary.lower",
        # Non-price quantities.
        "indicators.rsi_14", "indicators.macd.histogram", "indicators.volume_ma_20", "rows",
        # A PRICE leaf under a non-price family. Only the token denylist can
        # reject these — the leaf rule alone admits every one — and deleting
        # that guard left all 415 cases in these four suites green.
        "indicators.volume.sma_20", "indicators.turnover.vwap", "indicators.rsi.ma_5",
        "indicators.volume.high", "volume.pivot.r1",
    ],
)
def test_non_price_leaves_are_not_evidence(path: str) -> None:
    assert _is_price_denominated_indicator(path) is False


def test_a_moving_average_of_volume_never_grounds_a_price(tmp_path: Path) -> None:
    """The token denylist, end to end: nesting an SMA under a volume family.

    ``_ingest_generic_numeric`` classifies every non-market tool result, so a
    tool that returns ``{"indicators": {"volume": {"sma_20": 1.30}}}`` is all
    it takes. The leaf ``sma_20`` is a price name; only the family token stops
    it becoming evidence for "现价 1.30 元".
    """
    ledger = GroundingLedger(run_dir=tmp_path, user_message="请分析 562500.SS")
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": [SYMBOL]},
        result=_market_payload(),
        call_id="prices",
        success=True,
    )
    ledger.ingest_tool_result(
        tool_name="custom_indicator_tool",
        arguments={"symbol": SYMBOL},
        result=json.dumps({"symbol": SYMBOL, "indicators": {"volume": {"sma_20": 1.30}}}),
        call_id="custom",
        success=True,
    )

    result = ledger.validate_final_answer(HDR + " 现价 1.30 元。")

    assert "numeric_claim_conflict" in _codes(result)


def test_no_price_indicator_token_is_unreachable() -> None:
    """Every declared family name must actually classify as one.

    ``r1``/``s1`` sat in the family list for a change whose CHANGELOG entry
    claims pivot levels are covered, while the digit strip turned them into
    ``r``/``s`` and no lookup could ever match.
    """
    unreachable = [
        token
        for token in sorted(_PRICE_INDICATOR_TOKENS)
        if not _is_price_denominated_indicator("indicators." + token)
    ]

    assert unreachable == []
    assert all(_is_price_denominated_indicator(f"pivot.{level}") for level in ("r1", "r2", "r3", "s1", "s2", "s3"))


# ---------------------------------------------------------------------------
# Locating what to cut
#
# The claim string a validator records is normalised — thousands separators
# stripped, a table cell rendered as "label: value" — so it is not a substring
# of the draft. Every case here either shipped the canned refusal or rewrote
# the wrong characters before the clause span was carried on the issue.
# ---------------------------------------------------------------------------


def test_clause_spans_segment_exactly_as_the_split_did(tmp_path: Path) -> None:
    """Carrying spans must not change the segmentation itself.

    The clause splitter strips grouping commas BEFORE splitting so a price
    above 999 survives as one number; the span version has to skip those
    commas as separators instead, and the two must agree everywhere.
    """
    for text in (
        "收盘价 ¥1,309.22 元，建议买入价 1,450.50 元。",
        "a,b;c。d、e",
        "1,234,567 与 999,9",
        "no separators at all",
        "",
    ):
        expected = _CLAUSE_SEPARATOR_RE.split(_THOUSANDS_SEPARATOR_RE.sub("", text))
        assert [segment for segment, _, _ in _clause_spans(text)] == expected
        # A span points at the clause it describes, comma grouping included.
        for segment, start, end in _clause_spans(text):
            assert _THOUSANDS_SEPARATOR_RE.sub("", text[start:end]) == segment


def test_redacted_release_locates_a_grouped_number(tmp_path: Path) -> None:
    """A price above 999 written the ordinary way must still reach the release path."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="请分析 562500.SS 并给出买入价")
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": [SYMBOL]},
        result=json.dumps(
            {
                SYMBOL: [
                    {"trade_date": "2026-06-23", "open": 1300.0, "high": 1363.35, "low": 1280.0, "close": 1309.22},
                    {"trade_date": "2026-06-24", "open": 1310.0, "high": 1360.0, "low": 1300.01, "close": 1350.0},
                ],
                "_provenance": {SYMBOL: {"source": "yahoo", "currency_conversion": "none"}},
            }
        ),
        call_id="prices",
        success=True,
    )
    draft = "562500.SS（Yahoo，CNY）最新收盘价 1309.22 元。建议买入价 1,450.50 元。"
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["numeric_claim_conflict"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "1,450.50" not in released and "建议买入价（略※）。" in released
    assert "1309.22" in released
    assert "※ 略去 1 处" in released


def test_redacted_release_cuts_a_metric_cell_in_a_table(tmp_path: Path) -> None:
    """A metrics table is the ordinary shape of the report this path exists to rescue."""
    ledger = _ledger(tmp_path)
    draft = HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 最大回撤 | 12% |\n| 夏普比率 | 2.5 |\n"
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["analysis_claim_unavailable"] * 2

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "12%" not in released and "2.5" not in released
    assert "最大回撤" in released and "夏普比率" in released
    assert "※ 略去 2 处" in released


def test_redacted_release_never_cuts_inside_another_number(tmp_path: Path) -> None:
    """A bare table cell "1.10" must not be located inside "21.10 亿元"."""
    ledger = _ledger(tmp_path)
    draft = (
        "562500.SS（Yahoo，CNY）最新收盘价 1.171 元。成交额约 21.10 亿元。"
        "\n\n| 日期 | 收盘价 |\n|---|---|\n| 2026-06-24 | 1.10 |\n"
    )
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["numeric_claim_conflict"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "成交额约 21.10 亿元" in released
    assert "| 2026-06-24 |" in released and "| 2026-06-24 | 1.10 |" not in released
    assert "※ 略去 1 处" in released


def test_redacted_release_cuts_the_same_figure_everywhere_it_survives(tmp_path: Path) -> None:
    """The footnote says the figure was removed, so it must be removed everywhere.

    Neither a table under a non-OHLC header nor a bullet without a price word
    is scanned by the validators, so cutting only the flagged clause left the
    rejected number standing while the footnote asserted it was gone.
    """
    ledger = _ledger(tmp_path)
    in_a_table = HDR + " 建议买入价 1.30 元。\n\n| 档位 | 建议买入价 |\n|---|---|\n| 第一档 | 1.30 |\n"
    in_a_bullet = HDR + " 建议买入价 1.30 元。\n- 关键位：1.30 / 1.45\n"

    for draft in (in_a_table, in_a_bullet):
        validation = ledger.validate_final_answer(draft)
        released = ledger.redacted_release(draft, validation)

        assert released is not None, draft
        assert "1.30" not in released, draft
        assert "※ 略去 2 处" in released, draft
    # The sweep is scoped to the rejected values: 1.45 was never flagged.
    assert "1.45" in ledger.redacted_release(
        in_a_bullet, ledger.validate_final_answer(in_a_bullet)
    )


def test_a_metric_cell_restating_observed_closes_is_not_an_invented_metric(
    tmp_path: Path,
) -> None:
    """"| 峰值→当前 | 1.180 → 1.110 |" states two observed closes, not a drawdown figure.

    The prose path exempts an observed value from the analysis check; the
    table path did not, so a peak-to-current row was reported as an
    unevidenced drawdown and the release cut the observed prices out of it
    (deepseek 159516.SZ E2E, 2026-09-09). A figure the ledger does NOT hold
    stays rejected in the same table.
    """
    ledger = _ledger(tmp_path)

    observed_pair = ledger.validate_final_answer(
        HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 最大回撤 | 1.180 → 1.110 |\n"
    )
    invented = ledger.validate_final_answer(
        HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 最大回撤 | 1.180 → 0.900 |\n"
    )
    invented_percent = ledger.validate_final_answer(
        HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 最大回撤 | -39.32% |\n"
    )

    assert observed_pair.valid is True, observed_pair.issues
    assert "0.900" in {issue.get("value") for issue in invented.issues}
    assert "analysis_claim_unavailable" in _codes(invented_percent)


def test_the_sweep_never_cuts_a_valid_formulas_operand(tmp_path: Path) -> None:
    """A rejected entry price is often the multiplier of the correct derivation.

    "建议买入价 0.95 元" is cut, but "基于 MA20 1.150 × 0.95 = 1.093" is exactly
    the shape the correction prompt asks for — sweeping 0.95 out of it left
    "× （略※） =" and destroyed the one derivation the answer got right
    (159516.SZ replay, 2026-09-09).
    """
    ledger = _ledger(tmp_path)
    draft = HDR + " 基于 SMA20 1.150 × 0.95 = 1.093 作为第一档。建议买入价 0.95 元。"
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == [0.95]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "基于 SMA20 1.150 × 0.95 = 1.093 作为第一档。" in released
    assert "建议买入价（略※）。" in released
    assert "※ 略去 1 处" in released


def test_redacted_release_never_cuts_an_observed_price(tmp_path: Path) -> None:
    """A correctly quoted close sharing a clause with an ungrounded metric survives.

    ``_validate_analysis_claims`` reports the first unmatched figure in the
    clause, and an observed close was that figure — so the release cut the one
    number the ledger holds and footnoted it as unverifiable.
    """
    ledger = _ledger(tmp_path)
    draft = HDR + " 最新收盘 1.171 元且策略最大回撤 12%。"
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == ["12%"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "最新收盘 1.171 元" in released
    assert "12%" not in released
    assert "※ 略去 1 处" in released


def test_redacted_release_swallows_the_unit_flush_against_the_marker(tmp_path: Path) -> None:
    """The marker replaces the figure AND its unit, in either language."""
    zh = _ledger(tmp_path)
    zh_draft = HDR + " 建议买入价 1.10 元，分批建仓。"
    zh_released = zh.redacted_release(zh_draft, zh.validate_final_answer(zh_draft))

    no_space = _ledger(tmp_path)
    no_space_draft = HDR + " 建议买入价 1.10元，分批建仓。"
    no_space_released = no_space.redacted_release(
        no_space_draft, no_space.validate_final_answer(no_space_draft)
    )

    en = _ledger(tmp_path, message="Analyse 562500.SS and give me an entry price")
    en_draft = "562500.SS (Yahoo, CNY) last close 1.171. Suggested entry price 1.10 USD."
    en_released = en.redacted_release(en_draft, en.validate_final_answer(en_draft))

    assert zh_released is not None and "建议买入价（略※），分批建仓。" in zh_released
    assert no_space_released is not None and "建议买入价（略※），分批建仓。" in no_space_released
    assert en_released is not None and "Suggested entry price (omitted※)." in en_released


def test_a_unit_character_that_starts_a_word_is_not_swallowed(tmp_path: Path) -> None:
    """"0.95 元宵节" must not become "（略※）宵节"."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 0.95 元宵节后关注。"
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "元宵节后关注" in released
    assert "（略※）宵节" not in released
    assert "0.95" not in released


def test_the_marker_follows_the_script_of_the_text_it_cuts(tmp_path: Path) -> None:
    """A Chinese user gets English tables; the marker must match the clause, not the user."""
    ledger = _ledger(tmp_path)  # Chinese user message
    draft = "562500.SS (Yahoo, CNY) last close 1.171. Suggested entry price $0.95 per share."
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert _REDACTION_MARKER_EN in released and _REDACTION_MARKER_ZH not in released.split("※ 略去")[0]
    # The footnote stays in the user's language.
    assert "※ 略去 1 处" in released


def test_redacted_release_releases_three_figures_in_one_clause(tmp_path: Path) -> None:
    """The pass bound is pinned from below as well as above.

    ``test_redacted_release_gives_up_after_the_pass_bound`` only proves the
    bound is under four; without this, lowering it to two would silently push
    more runs back onto the canned refusal — the failure this change exists
    to fix.
    """
    ledger = _ledger(tmp_path)
    draft = HDR + " 策略最大回撤 12% 夏普 2.5 胜率 55% 处于下行趋势。"
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "下行趋势" in released
    assert "12%" not in released and "2.5" not in released and "55%" not in released
    assert "※ 略去 3 处" in released


def test_redacted_release_declines_a_provenance_only_validation(tmp_path: Path) -> None:
    """Nothing to cut means no release; the loop's repair path owns that case."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 均线空头排列。"
    validation = ValidationResult(
        valid=False, issues=[{"code": "data_source_not_surfaced", "claim": draft}]
    )

    assert ledger.redacted_release(draft, validation) is None


# ---------------------------------------------------------------------------
# Provenance repair: what it may and may not fill in
# ---------------------------------------------------------------------------


def test_repair_provenance_refuses_a_misattributed_symbol(tmp_path: Path) -> None:
    """The symbol is the figure's subject, not metadata about it.

    Appending "562500.SH: 行情来源 yahoo" under an answer that calls the
    instrument 贵州茅台 releases Kweichow Moutai's close as 1.171 with a
    footnote naming a different instrument, in zero model rounds.
    """
    ledger = _ledger(tmp_path)
    draft = "贵州茅台（雅虎，CNY）最新收盘价 1.171 元，日内最高 1.180 元。"
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["canonical_symbol_not_surfaced"]

    assert ledger.repair_provenance(draft, validation) is None
    assert ledger.redacted_release(draft, validation) is None


def test_repair_provenance_writes_english_for_an_english_user(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, message="Analyse 562500.SS and give me an entry price")
    draft = "562500.SS last close 1.171 CNY, trending down."
    validation = ledger.validate_final_answer(draft)
    assert _codes(validation) == ["data_source_not_surfaced"]

    repaired = ledger.repair_provenance(draft, validation)

    assert repaired is not None
    # The note spells the symbol the way the answer does, not the canonical form.
    assert "Data note: 562500.SS: price source yahoo, quote currency CNY." in repaired
    assert ledger.validate_final_answer(repaired).valid is True


def test_the_release_note_prints_a_large_price_in_full(tmp_path: Path) -> None:
    """``%g`` turns an index level into "1.23457e+06" six significant digits in."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="请分析 562500.SS 并给出买入价")
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": [SYMBOL]},
        result=json.dumps(
            {
                SYMBOL: [
                    {"trade_date": "2026-06-24", "open": 1240000.0, "high": 1250000.5,
                     "low": 1234567.0, "close": 1245000.0}
                ],
                "_provenance": {SYMBOL: {"source": "yahoo", "currency_conversion": "none"}},
            }
        ),
        call_id="prices",
        success=True,
    )
    draft = "562500.SS（Yahoo，CNY）最新收盘价 1245000.0 元。建议买入价 999.0 元。"
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "1234567–1250000.5" in released
    assert "e+06" not in released


def test_the_released_document_is_validated_with_its_footnote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footnote is answer text, so the gate must see it before release.

    The note carries the observed range and the symbols, and it used to be
    concatenated AFTER the recheck — the one part of a released answer with
    nothing behind it. A note that would not pass must refuse the release,
    which is what the method's fail-closed docstring promises.
    """
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 1.10 元，分批建仓。"
    validation = ledger.validate_final_answer(draft)
    assert ledger.redacted_release(draft, validation) is not None

    monkeypatch.setattr(
        ledger,
        "_release_note",
        lambda removed, content=None: "※ 参考买入价 9.99 元。",
    )

    assert ledger.redacted_release(draft, validation) is None


def test_a_gate_recheck_is_not_counted_as_a_rejected_draft(tmp_path: Path) -> None:
    """``validation_count`` is the revision budget and a user-facing number."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 1.10 元，分批建仓。"
    validation = ledger.validate_final_answer(draft)
    assert ledger.validation_count == 1

    assert ledger.redacted_release(draft, validation) is not None
    assert ledger.validation_count == 1


# ---------------------------------------------------------------------------
# Which evidence may back which claim, stated as an allowlist
#
# The rule used to be a denylist of OHLC field phrases, and a denylist is only
# as complete as the day it was typed. Every case below was released on the
# first cut of this change and rejected on the commit before it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "现价 1.21 元。",
        "最新价 1.21 元。",
        "成交价 1.21 元。",
        "报价 1.21 元。",
        "股价 1.21 元。",
        "价格 1.21 元。",
        "买入价 1.21 元。",
        "入场价 1.21 元。",
    ],
)
def test_an_indicator_never_answers_a_spot_quote(tmp_path: Path, claim: str) -> None:
    """1.21 is the session's bollinger.upper — above every bar in the run.

    A spot quote names the latest print. Ten spellings of it leaked
    identically while the rule was "reject only a named OHLC field", because
    none of them names one.
    """
    ledger = _ledger(tmp_path)

    result = ledger.validate_final_answer(HDR + " " + claim)

    assert "numeric_claim_conflict" in _codes(result), claim


@pytest.mark.parametrize(
    ("chinese", "english"),
    [
        ("收盘价为 1.150 元。", "The closing price was 1.150."),
        ("收盘于 1.150 元。", "It closed at 1.150."),
        ("收盘为 1.150 元。", "The close was 1.150."),
        ("盘中最高 1.150 元。", "The intraday high was 1.150."),
        ("当日最高价 1.150 元。", "The day's high was 1.150."),
        ("最低价 1.150 元。", "The lowest price was 1.150."),
        ("现价 1.150 元。", "The current price is 1.150."),
        ("成交价 1.150 元。", "It last traded at 1.150."),
    ],
)
def test_an_ohlc_or_spot_claim_gets_the_same_verdict_in_both_scripts(
    tmp_path: Path, chinese: str, english: str
) -> None:
    """The session's sma_20 is 1.150 and the observed close is 1.171.

    Every pair here is the same claim in two scripts, and the verdict must be
    the same one — the standing rule of this repo since the 2026-09-08
    grounding-gate asymmetry, where English was the leaking side.
    """
    zh = _ledger(tmp_path).validate_final_answer(HDR + " " + chinese)
    en = _ledger(tmp_path).validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n" + english
    )

    assert "numeric_claim_conflict" in _codes(zh), chinese
    assert "numeric_claim_conflict" in _codes(en), english


def test_a_level_claim_still_takes_its_indicator(tmp_path: Path) -> None:
    """The other side: the claim the indicator exemption was written for."""
    ledger = _ledger(tmp_path)

    band = ledger.validate_final_answer(HDR + " 布林下轨 1.09 元为支撑位。")
    english = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\nThe lower bollinger band is support at 1.09."
    )

    assert band.valid is True, band.issues
    assert english.valid is True, english.issues


# ---------------------------------------------------------------------------
# What a formula justifies: which occurrence, and in which role
# ---------------------------------------------------------------------------


def test_a_formula_may_not_produce_an_observed_print(tmp_path: Path) -> None:
    """A close is observed, never derived.

    "2026-06-23 收盘价 = 1.171 × 0.80 = 0.937" is true arithmetic whose result
    is claimed to be a close the ledger holds as 1.137, and the exemption that
    accepts the derivations the correction prompt asks for released it. The
    operands stay exempt; only the RESULT is compared, and only when an
    observation word labels the equation itself.
    """
    ledger = _ledger(tmp_path)

    zh = ledger.validate_final_answer(HDR + " 2026-06-23 收盘价 = 1.171 × 0.80 = 0.937。")
    en = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\nClose on 2026-06-23 = 1.171 * 0.80 = 0.937."
    )
    # The shape the prompt asks for: the observation word labels an OPERAND
    # and the result is the entry price the answer proposes. The multiplier is
    # 0.80 so the result lands nowhere near an observed print — 1.171 × 0.97 =
    # 1.136 would have passed on the observed 1.137 whatever this rule did.
    proposed = ledger.validate_final_answer(
        HDR + " 基于收盘价 1.171 × 0.80 = 0.937 作为买入价。"
    )
    proposed_en = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "Based on the closing price 1.171 * 0.80 = 0.937 as the entry."
    )

    assert "numeric_claim_conflict" in _codes(zh)
    assert "numeric_claim_conflict" in _codes(en)
    assert proposed.valid is True, proposed.issues
    assert proposed_en.valid is True, proposed_en.issues


def test_a_free_operand_of_a_true_formula_is_not_a_derivation(tmp_path: Path) -> None:
    """Arithmetic that derives nothing must not launder its own multiplier.

    "建议买入价 0.95 元（0.95 × 1.171 = 1.11245 参考）" is true and derives
    nothing: 0.95 is 19% below every observed bar. Under a value-scoped
    exemption the copy outside the bracket inherited what the copy inside it
    earned, so the exemption is scoped to the OCCURRENCE.
    """
    ledger = _ledger(tmp_path)

    self_referential = ledger.validate_final_answer(
        HDR + " 建议买入价 0.95 元（0.95 × 1.171 = 1.11245 参考）。"
    )
    english = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "Suggested entry price 0.5 (0.5 * 1.171 = 0.5855)."
    )
    # The multiplier keeps its exemption where it actually sits.
    real_derivation = ledger.validate_final_answer(
        HDR + " 基于收盘价 1.171 × 0.97 = 1.136 作为第一档。"
    )

    assert "numeric_claim_conflict" in _codes(self_referential)
    assert "numeric_claim_conflict" in _codes(english)
    assert real_derivation.valid is True, real_derivation.issues


def test_english_sentences_on_one_line_are_separate_clauses(tmp_path: Path) -> None:
    """A period ends a clause, exactly as 。 does.

    Two English sentences sharing a line were ONE clause, so a derivation in
    the first exempted a fabricated price in the second — while the Chinese
    translation of the same text was rejected, because 。 always split. The
    per-language verdict is the bug; the paragraph-break spelling of the same
    English text was already rejected.
    """
    english = _ledger(tmp_path).validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "Based on close 1.171 * 0.80 = 0.937. The last traded price is 0.937."
    )
    chinese = _ledger(tmp_path).validate_final_answer(
        HDR + "\n\n基于收盘价 1.171 × 0.80 = 0.937。最新成交价为 0.937 元。"
    )
    # A decimal point, a ticker suffix and an abbreviation are not sentence
    # ends, so the derivation is still read as one clause.
    intact = _ledger(tmp_path).validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "Based on close 1.171 * 0.97 = 1.136 as the entry."
    )

    # The other direction of the same parity: a level claim in its own English
    # sentence must not inherit the header sentence's "last close", which is
    # what merged the two into one clause and cost a correction round.
    zh_level = _ledger(tmp_path).validate_final_answer(HDR + "20 日均线价格为 1.150 元。")
    en_level = _ledger(tmp_path).validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171. "
        "The 20-day moving-average price was 1.150."
    )

    assert "numeric_claim_conflict" in _codes(english)
    assert "numeric_claim_conflict" in _codes(chinese)
    assert intact.valid is True, intact.issues
    assert zh_level.valid is True, zh_level.issues
    assert en_level.valid is True, en_level.issues


def test_a_derivation_anchored_to_another_symbol_grounds_nothing(tmp_path: Path) -> None:
    """The cross-symbol bail, for derivations as well as for indicators.

    In a comparison run the expensive instrument's close is arithmetically
    fine and attached to the wrong symbol; removing the bail left every case
    in these suites green.
    """
    header = "562500.SH 与 600519.SH（Yahoo，CNY）对比。"

    unattributed = _two_symbol_ledger(tmp_path).validate_final_answer(
        header + "买入价 = 1420 × 0.97 = 1377.4。"
    )
    attributed = _two_symbol_ledger(tmp_path).validate_final_answer(
        header + "600519.SH 买入价 = 1420 × 0.97 = 1377.4。"
    )

    assert "numeric_claim_conflict" in _codes(unattributed)
    assert attributed.valid is True, attributed.issues


# ---------------------------------------------------------------------------
# An analysis metric needs analysis evidence
# ---------------------------------------------------------------------------


def test_a_strategy_metric_is_not_grounded_by_endpoint_arithmetic(tmp_path: Path) -> None:
    """A strategy's drawdown is a property of an equity curve, not of two prints.

    The subject word is what flips the verdict: the same two endpoints ground
    a PRICE drawdown, which is the 159516.SZ case the exemption exists for.
    """
    ledger = _ledger(tmp_path)

    backtested = ledger.validate_final_answer(
        HDR + " 回测显示该策略最大回撤 5.9%（从 1.180 跌至 1.110）。"
    )
    strategy = ledger.validate_final_answer(
        HDR + " 策略最大回撤 5.9%，区间高点 1.180 元、低点 1.110 元。"
    )
    english = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "Strategy max drawdown 5.9% (1.180 down to 1.110)."
    )
    price_subject = ledger.validate_final_answer(
        HDR + " 股价较高点 1.180 元已回撤约 5.9%，现报 1.110 元。"
    )

    assert "analysis_claim_unavailable" in _codes(backtested)
    assert "analysis_claim_unavailable" in _codes(strategy)
    assert "analysis_claim_unavailable" in _codes(english)
    assert price_subject.valid is True, price_subject.issues


def test_the_endpoint_exemption_grounds_only_the_figures_it_derives(
    tmp_path: Path,
) -> None:
    """One correct ratio used to carry every other figure in its clause out.

    The English sentence carries no clause separator at all, which is how an
    invented 60% rode along beside a correct 37%.
    """
    ledger = _wide_ledger(tmp_path)

    both = ledger.validate_final_answer(
        WIDE_HDR + " 当前 0.666 元 较 5 月高点 1.053 元已回撤约 37% 最大回撤 60%。"
    )
    english = _wide_ledger(tmp_path).validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 0.666. "
        "At 0.666 down about 37% from the 1.053 May high with a max drawdown of 60%."
    )
    only_the_derived = _wide_ledger(tmp_path).validate_final_answer(
        WIDE_HDR + " 当前 0.666 元 较 5 月高点 1.053 元已回撤约 37%。"
    )

    assert "analysis_claim_unavailable" in _codes(both)
    assert "analysis_claim_unavailable" in _codes(english)
    assert only_the_derived.valid is True, only_the_derived.issues


def test_a_formula_in_another_clause_does_not_ground_a_metric(tmp_path: Path) -> None:
    """The derivation exemption is measured in the clause the claim was measured in.

    Line-scoped, any true arithmetic anywhere on the line grounded the metric:
    "基于 40 × 1.171 = 46.84 计算，策略最大回撤 40%" released a max drawdown no
    backtest produced.
    """
    ledger = _ledger(tmp_path)

    other_clause = ledger.validate_final_answer(
        HDR + " 基于 40 × 1.171 = 46.84 计算，策略最大回撤 40%。"
    )
    same_clause = ledger.validate_final_answer(
        HDR + " 策略最大回撤 40%（基于 40 × 1.171 = 46.84）。"
    )
    # Without a strategy subject the exemption is available, so this pair is
    # what the clause scope alone decides: the same formula, once in the
    # neighbouring clause and once beside the figure it is supposed to derive.
    price_metric_other_clause = ledger.validate_final_answer(
        HDR + " 基于 40 × 1.171 = 46.84 计算，区间收益率 40%。"
    )
    price_metric_same_clause = ledger.validate_final_answer(
        HDR + " 区间收益率 40%（基于 40 × 1.171 = 46.84）。"
    )
    # Both halves of the scope: the clause carries the derivation KEYWORD, so
    # the branch is entered, and the formula it points at is in the next
    # clause. Line-scoped values grounded the 40% from there.
    keyword_here_formula_elsewhere = ledger.validate_final_answer(
        HDR + " 区间收益率 40%（基于历史数据计算），其中 40 × 1.171 = 46.84。"
    )
    # A price drawdown derived in its own clause is still grounded.
    derived_in_place = ledger.validate_final_answer(
        HDR + " 从 1.180 跌至 1.110，股价回撤约 5.9%。"
    )

    assert "analysis_claim_unavailable" in _codes(other_clause)
    assert "analysis_claim_unavailable" in _codes(same_clause)
    assert "analysis_claim_unavailable" in _codes(price_metric_other_clause)
    assert price_metric_same_clause.valid is True, price_metric_same_clause.issues
    assert "analysis_claim_unavailable" in _codes(keyword_here_formula_elsewhere)
    assert derived_in_place.valid is True, derived_in_place.issues


def test_a_metric_is_not_grounded_by_equalling_an_observed_price(tmp_path: Path) -> None:
    """Numeric equality with some observed print is not evidence for a metric.

    A run holds hundreds of observed values spanning the instrument's range,
    so any percent-free metric written in that range collides with one. On a
    $12.50 stock the collision is "| 最大回撤 | 12.5 |".
    """
    ledger = _ledger(tmp_path)

    prose = ledger.validate_final_answer(HDR + " 策略夏普比率 1.171。")
    single_cell = ledger.validate_final_answer(
        HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 夏普比率 | 1.171 |\n"
    )
    # A cell RESTATING two observed prints is not a metric figure.
    endpoint_cell = ledger.validate_final_answer(
        HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 峰值→当前 | 1.180 → 1.110 |\n"
    )
    # A quoted price keeps its own exemption in prose, where it is written as
    # a price rather than as a metric value.
    quoted_price = ledger.validate_final_answer(HDR + " 最新收盘 1.171 元且策略最大回撤 12%。")

    assert "analysis_claim_unavailable" in _codes(prose)
    assert "analysis_claim_unavailable" in _codes(single_cell)
    assert endpoint_cell.valid is True, endpoint_cell.issues
    assert [issue.get("value") for issue in quoted_price.issues] == ["12%"]


# ---------------------------------------------------------------------------
# The release path: what the footnote is allowed to claim
# ---------------------------------------------------------------------------


def test_the_sweep_cuts_a_restatement_that_shares_a_value_with_a_formula(
    tmp_path: Path,
) -> None:
    """Protection follows the formula's SPAN, not its value.

    The rejected entry price is usually also the multiplier of the correct
    derivation, so protecting the value protected every restatement of the
    rejected figure with it: the table row survived under a footnote saying
    one figure had been cut. The derivation must still come out unmangled —
    that is what the protection is for.
    """
    ledger = _ledger(tmp_path)
    draft = (
        HDR
        + "\n\n建议买入价 0.95 元。\n\n基于收盘价 1.171 × 0.95 = 1.112 作为参考。"
        + "\n\n| 档位 | 挂单价 |\n|---|---|\n| 第一档 | 0.95 |\n"
    )
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == [0.95]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "基于收盘价 1.171 × 0.95 = 1.112 作为参考。" in released
    assert "| 第一档 | 0.95 |" not in released
    assert "建议买入价（略※）。" in released
    assert "※ 略去 2 处" in released


def test_the_sweep_never_cuts_an_observed_value(tmp_path: Path) -> None:
    """The other half of the protection: a value the ledger holds.

    A figure rejected in one clause (because that clause labelled it 收盘价)
    can be a value the session genuinely observed elsewhere. Cutting the
    correct mention too makes the footnote false about a number the run
    fetched.
    """
    ledger = _ledger(tmp_path, sma_20=1.150)
    draft = HDR + " 收盘价为 1.150 元。SMA20 位于 1.150 一线，为均线支撑。"
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == [1.15]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "收盘价为（略※）。" in released
    assert "SMA20 位于 1.150 一线" in released
    assert "※ 略去 1 处" in released


def test_the_sweep_never_cuts_a_percent_the_gate_accepted(tmp_path: Path) -> None:
    """A validated derivation and an invented metric can be the same percentage.

    Percent literals were cut everywhere they appeared, so endpoint arithmetic
    this same gate accepted in this same run was removed and footnoted as
    "could not be matched to this session's tool data".
    """
    ledger = _ledger(tmp_path)
    draft = HDR + "\n从 1.110 涨到 1.171，区间收益率约 5.5%。\n策略年化波动率 5.5%。"
    validation = ledger.validate_final_answer(draft)
    assert [issue.get("value") for issue in validation.issues] == ["5.5%"]

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "区间收益率约 5.5%。" in released
    assert "策略年化波动率（略※）。" in released
    assert "※ 略去 1 处" in released


def test_a_model_authored_release_note_is_stripped_before_redacting(
    tmp_path: Path,
) -> None:
    """The marker and the footnote mean only what the gate put there.

    A draft carrying its own "※ 略去 0 处……" shipped two contradictory
    footnotes, and a "（略※）" pasted into the body read as a redaction that
    never happened.
    """
    ledger = _ledger(tmp_path)
    draft = (
        HDR
        + " 建议买入价 0.95 元。第二档（略※）。"
        + "\n\n※ 略去 0 处无法与本会话工具数据对上的数值。"
    )
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert released.count("※ 略去") == 1
    assert "※ 略去 0 处" not in released
    assert "第二档。" in released


def test_the_marker_follows_each_lines_script_not_the_documents(
    tmp_path: Path,
) -> None:
    """The document-wide sweep picked ONE marker for every line it touched.

    A Chinese report containing an English table released "| First |（略※） |",
    the failure the per-clause rule was written to stop. The column padding
    survives too: the Chinese flush-against-the-word rule must not eat the
    space after a table pipe.
    """
    ledger = _ledger(tmp_path)
    draft = HDR + "\n\n建议买入价 0.95 元。\n\n| Entry | Price |\n|---|---|\n| First | 0.95 |\n"
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "建议买入价（略※）。" in released
    assert "| First | (omitted※) |" in released
    assert "※ 略去 2 处" in released


def test_a_compound_unit_is_not_split_by_the_marker(tmp_path: Path) -> None:
    """"0.95 元/股" is one unit; swallowing its first half leaves "/股" dangling."""
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 0.95 元/股，仓位 3 成。"
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "0.95" not in released
    assert "（略※）/股" not in released
    assert "元/股" in released


def test_overlapping_issue_spans_are_cut_once(tmp_path: Path) -> None:
    """Two validators can flag overlapping ranges on one table row.

    A metric row carrying a clause separator and a foreign ticker raises
    ``analysis_claim_unavailable`` on the whole row and
    ``unsourced_symbol_figures`` on the sub-clause. Without the
    already-consumed guard the row is emitted once per overlapping span and
    the released table is a fragment-duplicated mess with an inflated count.
    """
    ledger = _ledger(tmp_path)
    draft = HDR + "\n\n| 指标 | 数值 |\n|---|---|\n| 最大回撤 | 12%，同业 000001.SZ 5.0 |\n"
    validation = ledger.validate_final_answer(draft)
    assert len(validation.issues) == 2

    released = ledger.redacted_release(draft, validation)

    assert released is not None
    assert "| 最大回撤 | （略※），同业 000001.SZ（略※） |" in released
    assert "※ 略去 2 处" in released


def test_the_released_document_is_recorded_in_the_artifact(tmp_path: Path) -> None:
    """Every recheck on the release path is unrecorded, so the release itself is.

    Without this the artifact carried no evidence at all for the fail-closed
    promise that the document that shipped passed the same gate.
    """
    ledger = _ledger(tmp_path)
    draft = HDR + " 建议买入价 0.95 元。"
    validation = ledger.validate_final_answer(draft)

    released = ledger.redacted_release(draft, validation)
    assert released is not None

    artifact = json.loads(
        (tmp_path / "artifacts" / "grounding_evidence.json").read_text(encoding="utf-8")
    )
    assert ledger.validation_count == 1
    assert len(artifact["validations"]) == 1
    assert artifact["released"]["figures_removed"] == 1
    assert artifact["released"]["revalidated"] is True


def test_provenance_repair_declines_a_draft_with_an_unchecked_price_column(
    tmp_path: Path,
) -> None:
    """A zero-round repair must not attest for figures nothing checked.

    "| 档位 | 挂单价 |" is a price column ``_validate_price_tables`` does not
    key on, and the prose scan skips every line with a pipe. Appending "price
    source yahoo" under it turns the model's last chance to drop an invented
    ladder price into a released answer with a provenance note.
    """
    ledger = _ledger(tmp_path)
    with_column = (
        "562500.SS 最新收盘价 1.171 元。\n\n| 档位 | 挂单价 |\n|---|---|\n| 第一档 | 0.95 |\n"
    )
    without_column = "562500.SS 最新收盘价 1.171 元。\n\n| 档位 | 比例 |\n|---|---|\n| 第一档 | 30 |\n"

    declined = ledger.repair_provenance(
        with_column, ledger.validate_final_answer(with_column)
    )
    repaired = ledger.repair_provenance(
        without_column, ledger.validate_final_answer(without_column)
    )

    assert declined is None
    assert repaired is not None and "行情来源 yahoo" in repaired


def test_line_offsets_are_a_monotone_scan() -> None:
    """A line that also occurs inside an earlier line must not borrow its offset.

    ``content.find(line)`` without the cursor anchors line 2 inside line 1;
    the slice still equals the line, so only monotonicity catches it, and
    every issue span in the module is built on these offsets.
    """
    content = "备注：策略最大回撤 12%。\n策略最大回撤 12%\n结束"

    positions = _lines_with_offsets(content)

    cursor = 0
    expected: list[int] = []
    for line in content.splitlines():
        expected.append(cursor)
        cursor += len(line) + 1
    assert [offset for _, offset in positions] == expected
    assert [offset for _, offset in positions] == sorted(
        offset for _, offset in positions
    )
    assert all(
        content[offset : offset + len(line)] == line for line, offset in positions
    )


def test_the_observation_vocabulary_is_load_bearing_in_both_places(
    tmp_path: Path,
) -> None:
    """A word that names a market print decides two things, not one.

    It refuses an indicator reading even beside a level word, and it refuses
    to let a formula's RESULT be the print. Both halves of the vocabulary have
    to be there: the spot spellings (现价 / current price) name no OHLC field,
    and the OHLC spellings are derived from ``_TABLE_FIELD_ALIASES`` rather
    than typed out a second time — dropping either half leaves every other
    case in this module green, because the level allowlist already settles
    them.
    """
    ledger = _ledger(tmp_path)

    # The result of an equation whose subject is a print.
    spot_formula = ledger.validate_final_answer(HDR + " 现价 = 1.171 × 0.80 = 0.937。")
    high_formula = ledger.validate_final_answer(HDR + " 最高价 = 1.171 × 0.80 = 0.937。")
    open_formula = ledger.validate_final_answer(HDR + " 开盘价 = 1.171 × 0.80 = 0.937。")
    english_spot = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "The current price = 1.171 * 0.80 = 0.937."
    )
    english_ohlc = ledger.validate_final_answer(
        "562500.SS (Yahoo, CNY) last close 1.171.\n\n"
        "The closing price = 1.171 * 0.80 = 0.937."
    )
    # A print claim in a clause that also names a level: the level word must
    # not buy the indicator reading a way in. 1.150 is the session's sma_20.
    spot_beside_a_level = ledger.validate_final_answer(HDR + " 现价 1.150 元位于均线附近。")
    high_beside_a_level = ledger.validate_final_answer(HDR + " 最高价 1.150 元位于均线附近。")
    # The level claim alone still takes it.
    level_only = ledger.validate_final_answer(HDR + " 20 日均线价格为 1.150 元。")

    for result in (
        spot_formula,
        high_formula,
        open_formula,
        english_spot,
        english_ohlc,
        spot_beside_a_level,
        high_beside_a_level,
    ):
        assert "numeric_claim_conflict" in _codes(result), result.issues
    assert level_only.valid is True, level_only.issues
