"""Evidence intake: what a tool result is allowed to ground.

Every observed number the gate can validate against enters through this module
— market-data rows, registered indicator leaves, backtest metrics, run-dir CSVs
and generic numeric leaves. The kind map is keyed on TOOL FIELD NAMES, not on
natural language.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from src.agent.grounding.identity import (
    _infer_currency,
    _infer_venue,
    _normalize_symbol,
    _utc_now,
)

_PRICE_FIELDS = {"open", "high", "low", "close", "adj_close", "price"}


_TIMESTAMP_FIELDS = ("trade_date", "date", "datetime", "timestamp", "time", "index")


_MAX_GENERIC_EVIDENCE = 2_000


# CSV columns (case-insensitive) accepted from OHLC files the run wrote via
# bash+yfinance, and their canonical price-field names. Everything else in the
# file (Volume, Adj Close, etc.) is deliberately ignored so the contradiction
# check does not gain values it would be willing to accept.
_CSV_PRICE_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "price": "price",
}


_CSV_DATE_COLUMNS = {"date", "datetime", "trade_date", "timestamp", "index"}


# Filename -> symbol mapping for run-dir CSVs. The bash workaround writes each
# series with a filesystem-safe stem: ``BYN_V.csv`` for ``BYN.V``, ``PDI_TO.csv``
# for ``PDI.TO``, ``GC_F.csv`` for ``GC=F``.
_CSV_FILENAME_SUFFIX_MAP = (
    ("_V", ".V"),
    ("_TO", ".TO"),
    ("_F", "=F"),
    # A US name is written ``INTC_US.csv`` by the same workaround, and
    # ``.US`` is the venue suffix the rest of the project resolves on. Without
    # this row the CSV was ingested as no evidence at all, so every price the
    # run had actually fetched came back "numeric_claim_unavailable".
    ("_US", ".US"),
)


# Only ``get_market_data`` returns bars whose columns are already the canonical
# OHLC field names. Every other market-sensitive tool nests its quote somewhere,
# and ``_ingest_generic_numeric`` stores that JSON path verbatim — "data.last",
# "quote[0].close_price". Without this map those observations never reach the
# final-answer check, so a price the run genuinely retrieved is rejected as
# "no matching observed tool evidence": measured against the live validator, an
# answer quoting a ``get_stock_profile`` price failed with
# ``numeric_claim_unavailable`` while the identical claim backed by
# ``get_market_data`` passed. Only unambiguous quote fields are mapped; ratios,
# volumes, strikes, and analyst targets stay out so the contradiction check does
# not gain a wider set of values it is willing to accept.
_GENERIC_PRICE_FIELD_ALIASES = {
    "open": "open",
    "open_price": "open",
    "openprice": "open",
    "开盘": "open",
    "开盘价": "open",
    "high": "high",
    "high_price": "high",
    "最高": "high",
    "最高价": "high",
    "low": "low",
    "low_price": "low",
    "最低": "low",
    "最低价": "low",
    "close": "close",
    "close_price": "close",
    "closeprice": "close",
    "prev_close": "close",
    "pre_close": "close",
    "preclose": "close",
    "previous_close": "close",
    "收盘": "close",
    "收盘价": "close",
    "昨收": "close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted_close": "adj_close",
    "price": "price",
    "last": "price",
    "last_price": "price",
    "lastprice": "price",
    "latest_price": "price",
    "current_price": "price",
    "market_price": "price",
    "settle": "price",
    "settlement": "price",
    "settle_price": "price",
    "vwap": "price",
    "现价": "price",
    "最新价": "price",
}


# Metric family for a claim/evidence leaf, so a figure is only grounded by
# evidence of its own kind: an observed price must never stand in for an
# invented volatility (#1336).
_ANALYSIS_KIND_ALIASES = {
    "annualized_vol": "vol",
    "annualized_volatility": "vol",
    "volatility": "vol",
    "return_vol": "vol",
    "return_volatility": "vol",
    "vol": "vol",
    "max_drawdown": "drawdown",
    "maxdd": "drawdown",
    "drawdown": "drawdown",
    "sharpe": "sharpe",
    "sharpe_ratio": "sharpe",
    "win_rate": "win_rate",
    "hit_rate": "win_rate",
    "hitrate": "win_rate",
    "probability": "probability",
    "prob": "probability",
    "total_return": "return",
    "annual_return": "return",
    "cumulative_return": "return",
    "benchmark_return": "return",
    "excess_return": "return",
    "annualized_return": "return",
    "return": "return",
    "returns": "return",
    "ic_positive_ratio": "win_rate",
}


# Order matters: 最大回撤 is drawdown before 收益/return, and 年化波动率 is vol
# before the generic return branch.
_ANALYSIS_KIND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:回撤|drawdown|maxdd|最大亏损)", re.IGNORECASE), "drawdown"),
    (
        re.compile(
            r"(?:波动率|波动性|volatility|annualized\s*vol|return\s*vol|年化波动)",
            re.IGNORECASE,
        ),
        "vol",
    ),
    (re.compile(r"(?:夏普|sharpe)", re.IGNORECASE), "sharpe"),
    (re.compile(r"(?:胜率|命中率|win\s*rate|hit\s*rate)", re.IGNORECASE), "win_rate"),
    (re.compile(r"(?:概率|probability|prob)", re.IGNORECASE), "probability"),
    # 年化/annualized alone ("| 年化 | 18.2% |") resolves to return so a
    # generic-header table cannot dodge the gate with the fragment the prose
    # detector (_ANALYSIS_METRIC_RE) already treats as a metric word.
    (re.compile(r"(?:收益|回报|收益率|回报率|年化|\breturns?\b|\bannualized\b)", re.IGNORECASE), "return"),
)


def _symbol_from_csv_filename(stem: str) -> str | None:
    """Map a run-dir CSV stem back to a canonical project symbol.

    The bash workaround writes filesystem-safe stems: ``BYN_V.csv`` -> ``BYN.V``,
    ``PDI_TO.csv`` -> ``PDI.TO``, ``GC_F.csv`` -> ``GC=F``, ``INTC_US.csv`` ->
    ``INTC.US``. A stem without a recognized suffix (e.g. a bare US name
    ``AAPL``) maps to None because the project convention requires an explicit
    venue suffix.

    Args:
        stem: CSV filename without the ``.csv`` extension.

    Returns:
        The canonical symbol, or ``None`` when the stem has no recognizable
        venue suffix.
    """
    upper = (stem or "").strip().upper()
    if not upper:
        return None
    for raw, canonical in _CSV_FILENAME_SUFFIX_MAP:
        if upper.endswith(raw) and len(upper) > len(raw):
            return upper[: -len(raw)] + canonical
    return None


def _json_object(value: Any) -> dict[str, Any] | None:
    """Parse a JSON object from a tool result when possible."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_number(value: Any) -> bool:
    """Return whether a value is a finite JSON-style number, excluding bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _coerce_csv_number(value: Any) -> int | float | None:
    """Coerce a CSV cell to a finite number, or return None.

    CSV readers return every cell as text (``"0.375"``), so a bare
    ``_is_number`` check would discard them all. Values that do not parse as a
    finite number (blank cells, ``-``, ``N/A``) return ``None``.
    """
    if _is_number(value):
        return value
    if isinstance(value, str):
        try:
            parsed = float(value.strip().replace(",", ""))
        except (TypeError, ValueError):
            return None
        if math.isfinite(parsed):
            return parsed
    return None


# "." is deliberately not a separator: a decimal price such as 8.5 would parse
# as month 8 day 5 and match a real trading day.
# A report writes the day as a table cell -- "08-10(一)", "08-10(周一)盘中",
# "08-10盘中" -- and the weekday or session suffix made the cell match no
# evidence row at all, so every price in that row came back
# "numeric_claim_unavailable" even though the run had fetched the bar.
_TRADING_DAY_SUFFIX = (
    r"(?:\s*[(（]\s*(?:周|星期)?[一二三四五六日天]\s*[)）])?"
    r"\s*(?:盘中|盘后|盘前|收盘|开盘|早盘|尾盘)?"
)


_YEARLESS_CLAIM_DATE_RE = re.compile(
    r"^(0?[1-9]|1[0-2])\s*[-/月]\s*(0?[1-9]|[12]\d|3[01])\s*[日号]?"
    + _TRADING_DAY_SUFFIX
    + r"$"
)


# Two-digit day alternatives are tried before a bare digit so an
# unanchored prefix match consumes the full day ("10" of "08-10(一)")
# instead of stopping at "1".
_ISO_CLAIM_DATE_PREFIX_RE = re.compile(
    r"^\s*((?:19|20)\d{2})\s*[-/]\s*(0?[1-9]|1[0-2])\s*[-/]\s*([12]\d|3[01]|0?[1-9])"
)


_YEARLESS_CLAIM_DATE_PREFIX_RE = re.compile(
    r"^\s*(0?[1-9]|1[0-2])\s*[-/月]\s*([12]\d|3[01]|0?[1-9])"
)


_ISO_TIMESTAMP_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def _claim_date_tuple(date_value: str) -> tuple[int, int] | None:
    """Extract the (month, day) named by a report-style date cell.

    Reports routinely annotate a trading day: the date column reads
    ``08-10(一)``, ``08-10(周一)盘中`` or ``08-10盘中`` rather than the bare
    ``08-10`` the strict full-cell matchers accept. Any leading month-day (or
    full ISO date) prefix is therefore accepted so such a claim still compares
    against the matching evidence row instead of being reported as
    unevidenced.

    Args:
        date_value: Date cell as written in the answer.

    Returns:
        The (month, day) tuple, or None when no date prefix is present.
    """
    claim = (date_value or "").strip()
    match = _YEARLESS_CLAIM_DATE_RE.match(claim)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    match = _ISO_CLAIM_DATE_PREFIX_RE.match(claim)
    if match:
        return (int(match.group(2)), int(match.group(3)))
    match = _YEARLESS_CLAIM_DATE_PREFIX_RE.match(claim)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def _timestamp_matches_claim_date(timestamp: str, date_value: str) -> bool:
    """Match an evidence timestamp against the date cell of a claim.

    The comparison used to be ``timestamp.startswith(date_value)``, which can
    only succeed when the answer repeats the year. A table whose date column
    reads ``08-05`` — the ordinary way a report writes a trading day — matched
    nothing, so every cell in the row was reported as having no supporting
    evidence while that evidence sat right there (#983: 79 such rejections in
    one run, every value inside the observed range).

    A year-less date is matched on month and day, and a date cell may carry
    weekday or intraday annotations (``08-10(一)``, ``08-10盘中``) whose
    leading month-day is still recognized. Matching the wrong year is a
    smaller failure than matching nothing, but it is a real one, so the
    caller still compares the value against every record that matched rather
    than trusting the date.

    Args:
        timestamp: Evidence timestamp, normally ISO ``YYYY-MM-DD``.
        date_value: Date cell as written in the answer.

    Returns:
        True when the timestamp denotes the day the claim names.
    """
    stamp = (timestamp or "").strip()
    claim = (date_value or "").strip()
    if not stamp or not claim:
        return False
    if stamp.startswith(claim):
        return True
    claim_tuple = _claim_date_tuple(claim)
    iso = _ISO_TIMESTAMP_RE.match(stamp)
    if claim_tuple is None or not iso:
        return False
    return (int(iso.group(2)), int(iso.group(3))) == claim_tuple


def _price_field_for_path(path: str) -> str | None:
    """Map a generic evidence JSON path to a canonical price field.

    Args:
        path: Recorded evidence field, e.g. ``"data.quote[0].last_price"``.

    Returns:
        The matching member of ``_PRICE_FIELDS``, or ``None`` when the leaf is
        not an unambiguous quote field.
    """
    leaf = str(path or "").rsplit(".", 1)[-1]
    leaf = re.sub(r"\[\d+\]$", "", leaf).strip().casefold()
    return _GENERIC_PRICE_FIELD_ALIASES.get(leaf)


# A price-denominated indicator the session fetched is observed evidence in
# the same sense an OHLC bar is: ``indicators.sma_20: 0.7158`` came out of
# ``technical_indicators`` for this symbol, and an answer that writes "MA20
# 0.7158" or derives an entry from it is quoting a tool, not inventing a
# number. Before this, the value was compared against OHLC bars only and
# rejected as a fabricated price (deriv2 probe, 2026-09-09). The family list is
# closed and price-denominated on purpose — a moving average, band, pivot,
# support/resistance level, VWAP or ATR is in the instrument's currency; RSI,
# MACD, volume, counts and row totals are not. Letting a quote match
# ``rows: 54`` or a volume average would gut the fabrication check, so any
# non-price token anywhere in the path disqualifies the leaf.
_PRICE_INDICATOR_TOKENS = frozenset(
    {
        "sma", "ema", "wma", "dma", "dema", "tema", "hma", "kama", "ma", "vwap", "vwma",
        "bb", "bbands", "boll", "bollinger", "band", "bands", "keltner", "kc", "donchian",
        "pivot", "pp", "support", "resistance",
        "atr", "psar", "sar", "supertrend", "ichimoku", "tenkan", "kijun", "senkou",
        "chikou", "highest", "lowest",
    }
)


# Generic band leaves. "lower" is a price only under a band family, so these
# need a family token elsewhere in the path: ``bollinger.lower`` qualifies,
# a bare ``summary.lower`` does not.
_PRICE_BAND_LEAVES = frozenset({"upper", "lower", "middle", "mid", "top", "bottom", "basis"})


# Standard pivot-level spellings. They are matched before the trailing digits
# are stripped: ``r1`` would otherwise become ``r``, which is in no set, and
# the six literals ``r1 r2 r3 s1 s2 s3`` sat in the family list unreachable.
_PIVOT_LEVEL_RE = re.compile(r"^[rs][123]$")


_NON_PRICE_TOKENS = frozenset(
    {
        "volume", "vol", "turnover", "amount", "count", "rows", "returned", "signal",
        "histogram", "rsi", "kdj", "k", "d", "j", "cci", "adx", "obv", "mfi", "roc",
        "williams", "wr", "stoch", "pct", "percent", "ratio", "change", "chg", "width",
        "position", "score", "z", "zscore", "slope", "angle", "days", "bars",
        # Parameters and derived shapes, not levels: ``params.ma_period: 20``
        # and ``entry_condition.ma_window: 20`` are settings, ``ma_diff`` and
        # ``sma_cross`` are differences and booleans. The leaf test below
        # already rejects those three, but a path that also carries a family
        # token in another segment must not be readmitted by it.
        "window", "period", "length", "span", "lookback", "n", "cross", "flag",
    }
)


def _is_price_denominated_indicator(path: str) -> bool:
    """Return whether a generic evidence path is a price-denominated indicator.

    The decision is made on the LEAF — the last dotted segment, with a
    trailing index stripped — not on "some token anywhere in the path". Under
    the old any-token rule ``indicators.ma_diff``, ``indicators.sma_cross``
    and ``indicators.supertrend_direction`` were all admitted as observed
    prices because they carry ``ma`` / ``sma`` / ``supertrend`` somewhere,
    and a payload with any of them set to 1.30 let the answer print
    "现价 1.30 元" (attack6 probe, 2026-09-09). A difference, a crossover
    flag and a direction are not prices.

    Args:
        path: Recorded evidence field, e.g. ``"indicators.bollinger.lower"``.

    Returns:
        True when the leaf names a price-denominated indicator level and no
        path token names a non-price quantity.
    """
    text = str(path or "")
    tokens = [
        re.sub(r"\d+$", "", token).casefold()
        for token in re.split(r"[._\[\]\-\s]+", text)
        if token
    ]
    tokens = [token for token in tokens if token]
    if not tokens:
        return False
    if any(token in _NON_PRICE_TOKENS for token in tokens):
        return False
    segments = [segment for segment in re.split(r"[.\[\]]+", text) if segment]
    if not segments:
        return False
    leaf = segments[-1].strip().casefold()
    if _PIVOT_LEVEL_RE.match(leaf):
        return True
    leaf = re.sub(r"[_\-]?\d+$", "", leaf) or leaf
    if leaf in _PRICE_INDICATOR_TOKENS:
        return True
    return leaf in _PRICE_BAND_LEAVES and any(
        token in _PRICE_INDICATOR_TOKENS for token in tokens
    )


def _metric_kind_for_path(path: str) -> str | None:
    """Map an evidence JSON path to an analysis metric kind."""
    leaf = re.sub(r"\[\d+\]$", "", str(path or "").rsplit(".", 1)[-1])
    leaf = leaf.strip().casefold()
    kind = _ANALYSIS_KIND_ALIASES.get(leaf)
    if kind is not None:
        return kind
    # Compound leaves name the kind as a token ("reported_annualized_return",
    # "strategy_max_drawdown"). #1338 review: matching only the verbatim alias
    # table makes every other spelling silently ungroundable. Scan from the
    # right — English compounds put the head noun last, so "return_vol"
    # resolves to vol, never to return.
    tokens = [token for token in re.split(r"[_.]", leaf) if token]
    for size in (2, 1):
        for start in range(len(tokens) - size, -1, -1):
            kind = _ANALYSIS_KIND_ALIASES.get("_".join(tokens[start : start + size]))
            if kind is not None:
                return kind
    return _metric_kind_for_text(path)


def _metric_kind_for_text(text: str) -> str | None:
    """Return the analysis metric kind named in a claim or header."""
    for pattern, kind in _ANALYSIS_KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return None


@dataclass(frozen=True)
class EvidenceRecord:
    """One observed, unavailable, or derived numeric evidence item."""

    call_id: str
    tool: str
    symbol: str | None
    source: str
    timestamp: str | None
    field: str
    value: int | float | None
    status: str
    currency: str | None = None
    venue: str | None = None
    currency_conversion: str | None = None


class _EvidenceMixin:
    """Evidence behaviour of :class:`GroundingLedger`."""

    def _ingest_analysis_result(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        payload: dict[str, Any] | None,
        call_id: str,
    ) -> None:
        """Record metric numbers a completed analysis result actually produced.

        Success of the call envelope is not enough: ``backtest`` reports ok for
        any runner exit, and a deduplicated ("skipped") call carries no new
        result at all (#1336). Only results that yielded at least one
        recognisable metric figure count as completed analysis.
        """
        if payload is None or payload.get("skipped"):
            return
        if tool_name == "backtest":
            if str(payload.get("status") or "").casefold() != "ok" and payload.get(
                "exit_code"
            ) not in (0, "0"):
                return
            recorded = self._record_backtest_metrics(arguments, payload, call_id)
        elif tool_name == "factor_analysis":
            if str(payload.get("status") or "").casefold() != "ok":
                return
            recorded = self._record_leaf_metrics(payload, call_id, tool_name, "")
        elif tool_name == "run_shadow_backtest":
            if str(payload.get("status") or "").casefold() != "ok":
                return
            combined = payload.get("combined")
            if not isinstance(combined, dict):
                # A combined dict containing only {"error": ...} is no analysis.
                return
            recorded = self._record_leaf_metrics(combined, call_id, tool_name, "combined")
        elif tool_name == "quantlib_call":
            if payload.get("ok") is not True or str(
                arguments.get("action") or ""
            ).casefold() != "call":
                return
            function = str(arguments.get("function") or "")
            recorded = self._record_leaf_metrics(
                payload.get("result"), call_id, tool_name, function
            )
        else:
            return
        if recorded:
            self._analysis_completed.append(
                {"call_id": call_id, "tool": tool_name, "recorded_at": _utc_now()}
            )

    def _record_leaf_metrics(
        self,
        value: Any,
        call_id: str,
        tool_name: str,
        field_prefix: str,
    ) -> int:
        """Record nested numeric leaves whose key names a metric kind."""
        recorded = 0

        def visit(item: Any, path: str) -> None:
            nonlocal recorded
            if _is_number(item):
                kind = _metric_kind_for_path(path)
                if kind is None:
                    return
                self._analysis_metrics.append(
                    {
                        "metric": kind,
                        "value": float(item),
                        "tool": tool_name,
                        "call_id": call_id,
                        "field": path,
                    }
                )
                recorded += 1
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, f"{path}.{key}" if path else str(key))
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, f"{path}[{index}]")

        visit(value, field_prefix or "")
        return recorded

    def _record_backtest_metrics(
        self,
        arguments: Mapping[str, Any],
        payload: dict[str, Any],
        call_id: str,
    ) -> int:
        """Parse metric figures from a successful backtest's run-dir artifacts."""
        root = self.run_dir.resolve()
        candidates: list[Path] = []
        raw_dir = arguments.get("run_dir") or payload.get("run_dir")
        if raw_dir:
            candidate = Path(str(raw_dir))
            if not candidate.is_absolute():
                candidate = self.run_dir / candidate
            try:
                resolved = candidate.resolve()
                if resolved == root or resolved.is_relative_to(root):
                    candidates.append(resolved)
            except OSError:
                pass
        # The loop archives a detached backtest's artifacts into the active run
        # dir right after it succeeds, so that copy is the second candidate.
        candidates.append(root)
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, dict):
            for path_value in artifacts.values():
                if not isinstance(path_value, str):
                    continue
                try:
                    resolved = Path(path_value).resolve()
                    if resolved.is_relative_to(root):
                        candidates.append(resolved)
                except OSError:
                    continue
        files: list[Path] = []
        seen_dirs: set[Path] = set()
        for candidate in candidates:
            if candidate.is_file():
                files.append(candidate)
                continue
            if candidate in seen_dirs:
                continue
            seen_dirs.add(candidate)
            for dir_path in (candidate, candidate / "artifacts"):
                for name in ("metrics.csv", "metrics.json"):
                    target = dir_path / name
                    if target.is_file():
                        files.append(target)
        recorded = 0
        seen_files: set[Path] = set()
        for file_path in files:
            if file_path in seen_files:
                continue
            seen_files.add(file_path)
            recorded += self._record_metrics_file(file_path, call_id)
        return recorded

    def _record_metrics_file(self, path: Path, call_id: str) -> int:
        """Record metric figures from one metrics.csv/metrics.json artifact."""
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 0
        if path.suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return 0
            if not isinstance(data, dict):
                return 0
            return self._record_leaf_metrics(data, call_id, "backtest", "")
        try:
            rows = list(csv.reader(text.splitlines()))
        except csv.Error:
            return 0
        if len(rows) < 2:
            return 0
        header = [cell.strip().casefold() for cell in rows[0]]
        recorded = 0
        for index, raw in enumerate(rows[1]):
            if index >= len(header):
                break
            kind = _ANALYSIS_KIND_ALIASES.get(header[index])
            value = _coerce_csv_number(raw)
            if kind is not None and value is not None:
                self._analysis_metrics.append(
                    {
                        "metric": kind,
                        "value": float(value),
                        "tool": "backtest",
                        "call_id": call_id,
                        "field": header[index],
                    }
                )
                recorded += 1
        return recorded

    def _observed_price_values(self) -> list[float]:
        """Every observed price value the run holds, for redaction protection."""
        return [
            float(record.value)
            for record in self._comparable_price_records()
            if record.value is not None
        ]

    def _record_tool_failure(self, tool_name: str, call_id: str, result: str) -> None:
        """Store structured unavailable evidence for failed business envelopes."""
        payload = _json_object(result) or {}
        self._tool_failures.append(
            {
                "call_id": call_id,
                "tool": tool_name,
                "status": "unavailable",
                "error_code": payload.get("error_code"),
                "message": str(payload.get("error") or payload.get("message") or "tool failed")[:500],
                "recorded_at": _utc_now(),
            }
        )

    def _ingest_market_data(
        self,
        arguments: Mapping[str, Any],
        payload: dict[str, Any] | None,
        call_id: str,
    ) -> None:
        """Convert full OHLCV payloads into source-linked evidence rows."""
        if payload is None:
            self._record_tool_failure("get_market_data", call_id, "malformed JSON result")
            return
        requested_source = str(arguments.get("source") or "auto")
        provenance = payload.get("_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        for raw_symbol, raw_rows in payload.items():
            if str(raw_symbol).startswith("_"):
                continue
            symbol = _normalize_symbol(raw_symbol)
            rows = raw_rows.get("data") if isinstance(raw_rows, dict) else raw_rows
            if not isinstance(rows, list):
                continue
            symbol_provenance = provenance.get(raw_symbol)
            actual_source = (
                str(symbol_provenance.get("source"))
                if isinstance(symbol_provenance, dict) and symbol_provenance.get("source")
                else requested_source
            )
            currency_conversion = (
                str(symbol_provenance.get("currency_conversion"))
                if isinstance(symbol_provenance, dict)
                and symbol_provenance.get("currency_conversion")
                else None
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                timestamp = next(
                    (str(row[key]) for key in _TIMESTAMP_FIELDS if row.get(key) is not None),
                    None,
                )
                for field_name, value in row.items():
                    normalized_field = str(field_name).casefold()
                    if normalized_field in _TIMESTAMP_FIELDS or not _is_number(value):
                        continue
                    self._evidence.append(
                        EvidenceRecord(
                            call_id=call_id,
                            tool="get_market_data",
                            symbol=symbol,
                            source=actual_source,
                            timestamp=timestamp,
                            field=normalized_field,
                            value=value,
                            status="observed",
                            currency=_infer_currency(symbol),
                            venue=_infer_venue(symbol),
                            currency_conversion=currency_conversion,
                        )
                    )
        unresolved = payload.get("_unresolved")
        if isinstance(unresolved, list):
            for raw_symbol in unresolved:
                symbol = _normalize_symbol(raw_symbol)
                self._evidence.append(
                    EvidenceRecord(
                        call_id=call_id,
                        tool="get_market_data",
                        symbol=symbol,
                        source=requested_source,
                        timestamp=None,
                        field="availability",
                        value=None,
                        status="unavailable",
                        currency=_infer_currency(symbol),
                        venue=_infer_venue(symbol),
                    )
                )

    def _ingest_generic_numeric(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        payload: dict[str, Any],
        call_id: str,
    ) -> None:
        """Flatten bounded numeric leaves from other market-sensitive tools."""
        symbols = self._extract_symbol_arguments(arguments)
        symbol = symbols[0] if len(symbols) == 1 else None
        if symbol:
            symbol = (
                self._match_authorized_symbol(symbol, self.authorized_symbols) or symbol
            )
        source = str(payload.get("source") or tool_name)
        remaining = _MAX_GENERIC_EVIDENCE
        timestamp_fields = (*_TIMESTAMP_FIELDS, "as_of")

        def visit(value: Any, path: str, timestamp: str | None = None) -> None:
            nonlocal remaining
            if remaining <= 0:
                return
            if _is_number(value):
                self._evidence.append(
                    EvidenceRecord(
                        call_id=call_id,
                        tool=tool_name,
                        symbol=symbol,
                        source=source,
                        timestamp=timestamp,
                        field=path or "value",
                        value=value,
                        status="observed",
                        currency=_infer_currency(symbol or ""),
                        venue=_infer_venue(symbol or ""),
                    )
                )
                remaining -= 1
                return
            if isinstance(value, dict):
                local_timestamp = next(
                    (
                        str(value[key])
                        for key in timestamp_fields
                        if value.get(key) is not None
                    ),
                    timestamp,
                )
                for key, item in value.items():
                    if str(key).casefold() in timestamp_fields:
                        continue
                    visit(
                        item,
                        f"{path}.{key}" if path else str(key),
                        local_timestamp,
                    )
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]", timestamp)

        visit(payload, "")

    def _ingest_run_dir_ohlc_csvs(self) -> None:
        """Register OHLC rows from CSVs the run wrote via the bash workaround.

        The bash+yfinance escape hatch writes per-symbol OHLC CSVs into the run
        directory (e.g. ``data/raw/BYN_V.csv``) instead of returning them through
        ``get_market_data``. Those prices were genuinely observed tool output,
        but they never entered the ledger, so the final-answer gate rejected
        every one of them as ``numeric_claim_unavailable``. Scan the run dir for
        such CSVs and register their open/high/low/close/price rows as observed
        evidence, keyed to the symbol derived from the filename.

        Only files whose filename maps to a symbol already tracked in this run
        are accepted, so a stray CSV cannot mint new identity. Rows are bounded
        by ``_MAX_GENERIC_EVIDENCE`` and each file is ingested at most once.
        """
        if not self.run_dir.is_dir():
            return
        entitled = self._session_symbols | self.authorized_symbols
        if not entitled:
            return
        room = _MAX_GENERIC_EVIDENCE
        for path in sorted(self.run_dir.rglob("*.csv")):
            if room <= 0:
                return
            try:
                identity_key = f"{path.resolve()}:{path.stat().st_mtime_ns}"
            except (OSError, ValueError):
                continue
            if identity_key in self._ingested_csvs:
                continue
            self._ingested_csvs.add(identity_key)
            symbol = _symbol_from_csv_filename(path.stem)
            if not symbol or symbol not in entitled:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                if room <= 0:
                    return
                if not isinstance(row, dict):
                    continue
                timestamp = next(
                    (
                        str(row[key]).strip()
                        for key in row
                        if str(key).strip().casefold() in _CSV_DATE_COLUMNS
                        and row[key] not in (None, "")
                    ),
                    None,
                )
                for key, value in row.items():
                    field_name = _CSV_PRICE_COLUMNS.get(
                        str(key).strip().casefold().replace(" ", "_")
                    )
                    if field_name is None:
                        continue
                    numeric = _coerce_csv_number(value)
                    if numeric is None:
                        continue
                    self._evidence.append(
                        EvidenceRecord(
                            call_id=f"csv:{path.name}",
                            tool="bash",
                            symbol=symbol,
                            source="yfinance",
                            timestamp=timestamp,
                            field=field_name,
                            value=numeric,
                            status="observed",
                            currency=_infer_currency(symbol),
                            venue=_infer_venue(symbol),
                        )
                    )
                    room -= 1

    def _price_records(self) -> list[EvidenceRecord]:
        """Return observed OHLC/price evidence only."""
        return [
            record
            for record in self._evidence
            if record.status == "observed"
            and record.field in _PRICE_FIELDS
            and record.value is not None
        ]

    def _comparable_price_records(self) -> list[EvidenceRecord]:
        """Return every observed quote a numeric claim may be checked against.

        ``_price_records`` only sees fields already named ``open``/``close``/…,
        which in practice means ``get_market_data``. Quotes returned by the
        other market-sensitive tools are re-keyed onto the same canonical field
        so the contradiction check compares like with like instead of reporting
        the claim as unevidenced.

        Returns:
            Observed price evidence with canonical ``field`` values.
        """
        records = self._price_records()
        already_counted = {id(record) for record in records}
        for record in self._evidence:
            if id(record) in already_counted:
                continue
            if record.status != "observed" or record.value is None:
                continue
            field_name = _price_field_for_path(record.field)
            if field_name is None:
                if _is_price_denominated_indicator(record.field):
                    records.append(replace(record, field="indicator"))
                continue
            records.append(replace(record, field=field_name))
        return records
