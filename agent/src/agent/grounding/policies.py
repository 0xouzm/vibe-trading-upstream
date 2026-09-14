"""Per-role validation: what each declared figure has to survive.

The model declares what every measurement-shaped number in its answer IS
(spec §2); this module decides whether the evidence backs that declaration
(spec §4). It reads no prose word. Its predecessor did — twenty-two masks and
a dozen phrase catalogues — and the catalogues could never close: the same
sentence passed in one language and failed in the other, a level word the list
had not seen cost a revision round, and every new phrasing needed a new entry.
Roles now come from the model, evidence comes from the ledger, and shape comes
from :mod:`figures`.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.agent.grounding.identity import (
    _CANONICAL_SYMBOL_RE,
    _normalize_symbol,
    _scan_symbols,
)
from src.agent.grounding.evidence import (
    EvidenceRecord,
    _is_number,
    _metric_kind_for_path,
    _timestamp_matches_claim_date,
)
from src.agent.grounding.figures import (
    Declaration,
    Figure,
    FiguresBlock,
    _lines_with_offsets,
    segment_bounds,
)

import re

#: An answer that relabels a locked listed identity as private contradicts the
#: resolver, which is an identity finding rather than a figure finding.
_PRIVATE_ASSERTION_RE = re.compile(
    r"(?:\b(?:is|remains|still)\s+(?:an?\s+)?(?:private company|privately held)\b|"
    r"\bnot publicly traded\b|\bunlisted company\b|"
    r"(?:是|仍是|属于)(?:一家)?(?:私人|私营|非上市)公司|未上市|没有上市)",
    re.IGNORECASE,
)


# A loader's id is ASCII, but the answer follows the user's language, so a
# Chinese report names the same provider in Chinese. Demanding the ASCII id
# verbatim rejected correct prose: "数据来源：腾讯财经" was reported as
# ``data_source_not_surfaced`` against evidence sourced from ``tencent``.
_SOURCE_ALIASES = {
    "akshare": ("akshare", "ak share"),
    "baostock": ("baostock",),
    "binance": ("binance", "币安"),
    "ccxt": ("ccxt",),
    "eastmoney": ("eastmoney", "东方财富", "东财"),
    "futu": ("futu", "富途"),
    "mootdx": ("mootdx", "通达信"),
    "okx": ("okx", "欧易"),
    "pykrx": ("pykrx", "krx"),
    "sina": ("sina", "新浪"),
    "stooq": ("stooq",),
    "tencent": ("tencent", "腾讯"),
    "tushare": ("tushare",),
    "yahoo": ("yahoo", "雅虎"),
    "yfinance": ("yfinance", "yahoo", "雅虎"),
}


_CURRENCY_ALIASES = {
    "USD": ("usd", "us$", "美元", "美金"),
    # ¥ is how a model actually writes a CNY quote. It is the yen sign too, but
    # ``_infer_currency`` maps no venue to JPY, so nothing in this system can
    # mean yen by it; adding a JPY venue means revisiting this entry.
    "CNY": ("cny", "cnh", "rmb", "人民币", "¥", "￥"),
    "HKD": ("hkd", "hk$", "港元", "港币"),
    "KRW": ("krw", "韩元", "韩圜"),
    "INR": ("inr", "印度卢比", "卢比"),
    "CAD": ("cad", "c$", "加元", "加拿大元"),
}


# "元" is how a Chinese answer writes a CNY quote, but it is also the tail of
# 港元/美元/日元, so accepting it unguarded would let an answer about a Hong
# Kong listing satisfy a CNY requirement. It counts only when no other
# currency's character owns it.
_OTHER_CURRENCY_PREFIXES = "港美日欧韩台新加澳"


#: Relative band a value must fall in to count as matching evidence.
_TOLERANCE = 0.005


@dataclass(frozen=True)
class ValidationResult:
    """Final-answer grounding decision.

    ``released_text`` is the draft with its declaration block removed: the
    block is a contract between the model and this gate, not part of the
    answer, and it never reaches the user.
    """

    valid: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    released_text: str = ""


def _close(value: float, target: float) -> bool:
    """Whether two values agree inside the evidence tolerance."""
    return abs(value - target) <= max(abs(target) * _TOLERANCE, 1e-9)


def _close_any(value: float, targets: Iterable[float]) -> bool:
    """Whether ``value`` agrees with any of ``targets``."""
    return any(_close(value, target) for target in targets)


def _nearest(value: float, targets: Iterable[float], limit: int = 3) -> list[float]:
    """The observed values closest to a rejected figure.

    A range ("0.567-1.053") tells the model the figure is outside the window;
    the nearest prints tell it what to write instead, which is what spec 6 asks
    the correction prompt to say.

    Args:
        value: The rejected figure's value.
        targets: Every value the relevant evidence pool holds.
        limit: How many to name.

    Returns:
        Up to ``limit`` distinct observed values, closest first.
    """
    unique = sorted({float(target) for target in targets}, key=lambda item: (abs(item - value), item))
    return unique[:limit]


def _written_half_unit(text: str) -> float:
    """Half a unit of the last digit a figure was WRITTEN with.

    "约 37%" for a derived 36.75% asserts that the value rounds to 37, and a
    flat relative band rejects every integer-percent rounding a model makes.
    """
    body = text.strip().rstrip("%％").strip()
    decimals = len(body.split(".", 1)[1]) if "." in body else 0
    return 0.5 * 10.0 ** (-decimals)


def _evaluate_formula(expression: str) -> tuple[float, list[float]] | None:
    """Evaluate a numeric ``+ - * /`` expression without executing code.

    Args:
        expression: An arithmetic run, possibly using ``× ÷ −`` and commas.

    Returns:
        ``(result, operands)``, or None when the run is not a well-formed
        expression over at least two numeric operands.
    """
    normalized = (
        expression.replace("×", "*")
        .replace("✕", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("（", "(")
        .replace("）", ")")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )
    if not normalized:
        return None
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    inputs: list[float] = []

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and _is_number(node.value):
            value = float(node.value)
            inputs.append(value)
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError("unsupported formula")

    try:
        value = visit(tree)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if len(inputs) < 2 or not math.isfinite(value):
        return None
    return value, inputs


def _formula_in_note(note: str) -> tuple[float, list[float]] | None:
    """Find the derivation a note states.

    A note is written either as the expression alone ("0.666 × 0.97") or as
    the expression together with its own result ("0.666 × 0.97 = 0.646"). The
    whole note is tried first, then each segment between result separators.
    Those separators are punctuation, not vocabulary, which is why the English
    and Chinese spellings of the same derivation now get the same verdict.

    Args:
        note: The declaration's free-text note.

    Returns:
        ``(result, operands)`` for the first parseable segment, or None.
    """
    candidates = [note]
    parts = [note]
    for separator in ("≈", "≒", "＝", "=", "→", "->"):
        parts = [piece for part in parts for piece in part.split(separator)]
    candidates.extend(part for part in parts if part.strip())
    for candidate in candidates:
        evaluated = _evaluate_formula(candidate)
        if evaluated is not None:
            return evaluated
    return None


class _PolicyMixin:
    """Policy behaviour of :class:`GroundingLedger`."""

    # ------------------------------------------------------------------ #
    # identity
    # ------------------------------------------------------------------ #

    def _validate_identity(self, content: str) -> list[dict[str, Any]]:
        """Validate aggregate state and listed/private contradictions."""
        issues: list[dict[str, Any]] = []
        status = self.identity_status
        # Two conditions, both load-bearing.
        #
        # ``self._identities`` — a run that never named an instrument has no
        # identity to get wrong. The trigger phrase is matched against the user
        # message, so "什么是市盈率估值法？" set identity_required and then failed
        # every draft it could ever produce, including the honest answer.
        #
        # ``ambiguous`` is deliberately absent. A shortlist is an answer, and
        # consumers stay blocked on it in ``authorize_tool_call``, so such a
        # run still cannot fetch a quote to misattribute.
        if (
            self._identity_required
            and self._identities
            and status in {"unresolved", "conflicting", "invalidated"}
        ):
            issues.append(
                {
                    "code": "identity_not_locked",
                    "status": status,
                    "value": None,
                    "role": None,
                    "span": None,
                    "symbol": None,
                    "reason": "identity_not_locked",
                    "message": (
                        f"Instrument identity is {status}; a final market conclusion "
                        "requires locked identity."
                    ),
                }
            )
        listed = [
            record
            for record in self._identities.values()
            if record.status == "locked"
            and record.instrument_type in {"listed_security", "fund"}
        ]
        if listed and _PRIVATE_ASSERTION_RE.search(content):
            symbols = sorted(record.symbol for record in listed if record.symbol)
            issues.append(
                {
                    "code": "listed_identity_relabelled_private",
                    "symbols": symbols,
                    "value": None,
                    "role": None,
                    "span": None,
                    "symbol": None,
                    "reason": "listed_relabelled_private",
                    "message": (
                        f"Locked listed identity {', '.join(symbols)} was relabelled as "
                        "private/unlisted without a conflicting resolver result."
                    ),
                }
            )
        return issues

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #

    def _validate_figures(
        self,
        content: str,
        block: FiguresBlock,
        figures: Sequence[Figure],
    ) -> list[dict[str, Any]]:
        """Check every measurement-shaped number against its declared role.

        Args:
            content: The candidate answer.
            block: Its parsed declaration block.
            figures: Every number located in its prose.

        Returns:
            One issue per figure that its role does not survive, plus one per
            malformed declaration line and the provenance findings.
        """
        issues: list[dict[str, Any]] = [
            {
                "code": "figures_block_malformed",
                "line": line_no,
                "claim": raw,
                "value": None,
                "role": None,
                "span": None,
                "symbol": None,
                "reason": "unparseable_declaration",
                "message": (
                    f"figures block line {line_no} could not be read as "
                    "`value | role | note | ref`: " + raw
                ),
            }
            for line_no, raw in block.malformed
        ]
        records = self._comparable_price_records()
        document_symbol = self._symbol_for_claim(content, records)
        positions = _lines_with_offsets(content)
        line_symbols = [
            self._symbol_for_claim(line, records) for line, _ in positions
        ]
        declared_observed = {
            declaration.value
            for declaration in block.declarations
            if declaration.role == "observed"
        }
        checked_price = False
        for figure in figures:
            if figure.shape != "measured":
                continue
            declaration = block.match(figure.value, figure.percent)
            if block.present and declaration is None:
                issues.append(
                    self._figure_issue(
                        "figure_undeclared",
                        figure,
                        None,
                        None,
                        "undeclared",
                        "is not declared in the figures block; declare it as "
                        "observed / derived / proposed / cited / count, or remove it",
                    )
                )
                continue
            role = declaration.role if declaration else "observed"
            symbol = self._figure_symbol(
                content, figure, declaration, line_symbols, document_symbol, records
            )
            if role == "count":
                continue
            if role == "cited":
                issues.extend(
                    self._check_cited(figure, declaration, symbol, declared_observed)
                )
                continue
            checked_price = True
            if role == "derived":
                issues.extend(
                    self._check_derived(figure, declaration, symbol, records)
                )
                continue
            if role == "proposed":
                issues.extend(
                    self._check_proposed(figure, declaration, symbol, records)
                )
                continue
            issues.extend(self._check_observed(figure, declaration, symbol, records))
        market_records = self._price_records()
        if checked_price and market_records:
            issues.extend(self._validate_price_provenance(content, market_records))
        return issues

    @staticmethod
    def _figure_issue(
        code: str,
        figure: Figure,
        role: str | None,
        symbol: str | None,
        reason: str,
        message: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build one figure-scoped issue.

        Every figure issue carries the same five fields — value, role, span,
        symbol and reason — so the correction prompt can name the exact number
        and the release path can cut exactly it.
        """
        issue = {
            "code": code,
            "value": figure.text,
            "role": role,
            "span": [figure.start, figure.end],
            "symbol": symbol,
            "reason": reason,
            "claim": figure.text,
            "message": f"{figure.text} {message}.",
        }
        issue.update(extra)
        return issue

    def _figure_symbol(
        self,
        content: str,
        figure: Figure,
        declaration: Declaration | None,
        line_symbols: Sequence[str | None],
        document_symbol: str | None,
        records: Sequence[EvidenceRecord],
    ) -> str | None:
        """Resolve which instrument a figure is about (spec §4 order).

        The declaration's own ``ref``/``note`` wins, then the table row's
        symbol column, then the one canonical evidence symbol on the figure's
        line, then the one canonical evidence symbol in the whole answer.
        """
        if declaration is not None:
            declared = self._symbol_for_claim(
                f"{declaration.ref} {declaration.note}", records
            )
            if declared:
                return declared
        if figure.symbol:
            normalized = _normalize_symbol(figure.symbol)
            if normalized:
                return normalized
        # Inside the figure's own segment first. A comparison report names both
        # instruments in its header and then writes about one of them, so the
        # LINE is too coarse an owner: 562500's moving average grounded a level
        # written under 600519 on the same line.
        left, right = segment_bounds(content, figure.start, figure.end)
        segment_symbol = self._symbol_for_claim(content[left:right], records)
        if segment_symbol:
            return segment_symbol
        if 0 <= figure.line < len(line_symbols) and line_symbols[figure.line]:
            return line_symbols[figure.line]
        return document_symbol

    # ------------------------------------------------------------------ #
    # evidence pools
    # ------------------------------------------------------------------ #

    def _referenced_values(self, ref: str) -> list[float] | None:
        """Every observed value one tool call produced, or None when unknown.

        A declaration whose ``ref`` names a call id says "this number came out
        of THAT call". That is the tightest scoping available and the only one
        that can ground a figure the price pool cannot hold — a filing's
        revenue, a factor's IC, a volume.
        """
        key = (ref or "").strip()
        if not key:
            return None
        values = [
            float(record.value)
            for record in self._evidence
            if record.call_id == key
            and record.status == "observed"
            and record.value is not None
        ]
        values.extend(
            float(entry["value"])
            for entry in self._analysis_metrics
            if entry.get("call_id") == key and entry.get("value") is not None
        )
        return values or None

    def _price_pool(
        self,
        symbol: str | None,
        records: Sequence[EvidenceRecord],
        *,
        column: str | None = None,
        date: str | None = None,
    ) -> list[float]:
        """Observed price values a figure may be compared against.

        Filtered by symbol when one was resolved; further narrowed to the OHLC
        field and trade date when the figure sits under those table headers,
        which is the table semantics §4 preserves.
        """
        candidates = list(records)
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        elif len({record.symbol for record in records if record.symbol}) > 1:
            # An indicator reading is symbol-bound in a way an OHLC bar's union
            # is not. With evidence for two instruments and a figure attributed
            # to neither, 562500's sma_20 grounded a level written beside
            # 600519.SH; the union was argued for observed quotes, not for a
            # level attached to one symbol.
            candidates = [record for record in candidates if record.field != "indicator"]
        if column:
            candidates = [record for record in candidates if record.field == column]
        if date:
            candidates = [
                record
                for record in candidates
                if record.timestamp
                and _timestamp_matches_claim_date(record.timestamp, date)
            ]
        return [float(record.value) for record in candidates if record.value is not None]

    def _row_pool(self, symbol: str | None) -> list[float]:
        """Non-price numbers a market-data ROW carried (volume, turnover, …).

        A row is one observation, and a report quoting its volume beside its
        close is quoting the same tool result. Only ``get_market_data`` rows
        and the run-dir CSVs the bash escape hatch writes are included, so a
        generic tool's hundreds of numeric leaves never widen the price check.
        """
        return [
            float(record.value)
            for record in self._evidence
            if record.status == "observed"
            and record.value is not None
            and record.tool in {"get_market_data", "bash"}
            and (not symbol or record.symbol == symbol)
        ]

    def _metric_pool(self, symbol: str | None) -> list[float]:
        """Analysis/risk metric values the run actually recorded.

        Two sources, both keyed on TOOL FIELD NAMES: metrics parsed from a
        completed analysis result, and observed leaves whose path maps to a
        metric kind (``annualized_vol``, ``max_drawdown``, …).
        """
        values = [
            float(entry["value"])
            for entry in self._analysis_metrics
            if entry.get("value") is not None
        ]
        values.extend(
            float(record.value)
            for record in self._evidence
            if record.status == "observed"
            and record.value is not None
            and _metric_kind_for_path(record.field) is not None
            and (not symbol or not record.symbol or record.symbol == symbol)
        )
        return values

    def _matches_evidence(
        self,
        figure: Figure,
        direct: Sequence[float],
        scaled: Sequence[float],
    ) -> bool:
        """Whether a figure equals evidence, at its own scale or a metric's.

        ``direct`` is compared literally; the caller decides what may go in it,
        and for a percent-written figure that is nothing at all — an OHLC close
        is never a ratio. ``scaled`` absorbs the two conventions tools and
        answers disagree on: fraction vs percent (0.182 vs 18.2%), and the sign
        a fall is written with (a drawdown is recorded as -0.094 and quoted as
        9.4%).
        """
        if _close_any(figure.value, direct):
            return True
        candidates = {abs(figure.value), abs(figure.value) / 100.0}
        magnitudes = [abs(target) for target in scaled]
        return any(_close_any(candidate, magnitudes) for candidate in candidates)

    # ------------------------------------------------------------------ #
    # roles
    # ------------------------------------------------------------------ #

    def _check_observed(
        self,
        figure: Figure,
        declaration: Declaration | None,
        symbol: str | None,
        records: Sequence[EvidenceRecord],
    ) -> list[dict[str, Any]]:
        """An observed figure must appear in the evidence it claims to quote."""
        referenced = (
            self._referenced_values(declaration.ref) if declaration else None
        )
        if referenced is not None:
            if self._matches_evidence(figure, referenced, referenced):
                return []
            return [
                self._figure_issue(
                    "numeric_claim_conflict",
                    figure,
                    "observed",
                    symbol,
                    "not_in_referenced_call",
                    f"is declared observed from {declaration.ref}, whose results do "
                    "not contain it",
                    source_tool_call_ids=[declaration.ref],
                    observed_nearest=_nearest(figure.value, referenced),
                )
            ]
        prices = self._price_pool(
            symbol, records, column=figure.column, date=figure.date
        )
        if figure.percent:
            # A percent-written figure is a ratio, and a price or a volume is
            # not, so no amount of price evidence may answer one.
            direct: list[float] = []
        elif figure.column:
            direct = list(prices)
        else:
            direct = list(prices) + self._row_pool(symbol)
        scaled = [] if figure.column else self._metric_pool(symbol)
        if not direct and not scaled:
            return [
                self._figure_issue(
                    "numeric_claim_unavailable",
                    figure,
                    "observed",
                    symbol,
                    "no_evidence",
                    "is declared observed but this session holds no matching tool "
                    "evidence to check it against",
                    field=figure.column,
                    date=figure.date,
                )
            ]
        if self._matches_evidence(figure, direct, scaled):
            return []
        observed = sorted(direct or scaled)
        return [
            self._figure_issue(
                "numeric_claim_conflict",
                figure,
                "observed",
                symbol,
                "value_mismatch",
                "is declared observed but conflicts with the "
                f"{figure.column or 'observed'} evidence "
                f"{observed[0]:g}–{observed[-1]:g}",
                field=figure.column,
                date=figure.date,
                observed_min=observed[0],
                observed_max=observed[-1],
                observed_nearest=_nearest(figure.value, observed),
            )
        ]

    def _derivation(
        self,
        declaration: Declaration | None,
        symbol: str | None,
        records: Sequence[EvidenceRecord],
    ) -> tuple[float, list[float]] | str | None:
        """Evaluate a declaration's note as an observation-anchored formula.

        Returns the ``(result, operands)`` pair when the note is arithmetic
        over at least two operands, at least one of which the run observed;
        otherwise the reason it is not.
        """
        if declaration is None or not declaration.note.strip():
            return "no_formula"
        evaluated = _formula_in_note(declaration.note)
        if evaluated is None:
            return "formula_not_evaluable"
        result, operands = evaluated
        if not symbol and len({record.symbol for record in records if record.symbol}) > 1:
            # A comparison run holds two instruments' bars, and the expensive
            # one's close makes any arithmetic "anchored" no matter which
            # instrument the answer is about. Without a resolved symbol there
            # is nothing to anchor TO.
            return "no_symbol"
        anchors = (
            self._price_pool(symbol, records)
            + self._row_pool(symbol)
            + self._metric_pool(symbol)
        )
        referenced = self._referenced_values(declaration.ref)
        if referenced:
            anchors.extend(referenced)
        if not anchors:
            return "no_evidence"
        if not any(_close_any(operand, anchors) for operand in operands):
            return "formula_not_anchored"
        return result, operands

    @staticmethod
    def _result_matches(declaration: Declaration, result: float) -> bool:
        """Whether a formula's result is the value the declaration states.

        The band is half a unit of the last digit the value was WRITTEN with,
        because "约 37%" for a derived 36.75% asserts that it rounds to 37 and
        a flat relative band rejects every rounding a model makes.

        A figure carrying "%" is percentage points and is compared ONLY
        against the derivation in those units. Running it against the fraction
        too gave an integer percent a half-unit band of 0.5 in FRACTION units
        — fifty percentage points — and "区间收益率约 0%" validated against a
        real +58% move. A figure written without "%" is genuinely ambiguous
        and is tried both ways.

        Magnitudes are compared because the note for a fall is written both as
        ``(low − high) / high`` and as the drop it produces; the note itself is
        what states which subtraction was taken.
        """
        half_unit = _written_half_unit(declaration.value_text)
        targets = (
            {result * 100.0} if declaration.percent else {result, result * 100.0}
        )
        value = abs(declaration.value)
        return any(
            abs(value - abs(target)) <= max(abs(target) * _TOLERANCE, half_unit, 1e-9)
            for target in targets
        )

    def _check_derived(
        self,
        figure: Figure,
        declaration: Declaration | None,
        symbol: str | None,
        records: Sequence[EvidenceRecord],
    ) -> list[dict[str, Any]]:
        """A derived figure must be the arithmetic its note states."""
        derivation = self._derivation(declaration, symbol, records)
        if isinstance(derivation, str):
            return [
                self._figure_issue(
                    "numeric_claim_conflict"
                    if derivation != "no_evidence"
                    else "numeric_claim_unavailable",
                    figure,
                    "derived",
                    symbol,
                    derivation,
                    "is declared derived, but its note is not arithmetic over at "
                    "least two operands with one of them observed in this session",
                )
            ]
        result, _ = derivation
        if declaration is not None and self._result_matches(declaration, result):
            return []
        # Reported in the figure's OWN units. ``_result_matches`` compares a
        # percent-written figure against ``result * 100``, so telling a model
        # that wrote "12%" that its formula "evaluates to -0.3675" names a
        # number that appears nowhere in the comparison it just failed.
        scaled = result * 100.0 if figure.percent else result
        shown = f"{scaled:.6g}%" if figure.percent else f"{scaled:.6g}"
        return [
            self._figure_issue(
                "numeric_claim_conflict",
                figure,
                "derived",
                symbol,
                "derivation_result_mismatch",
                f"is declared derived, but its own formula evaluates to {shown}",
                derived_result=shown,
            )
        ]

    def _check_proposed(
        self,
        figure: Figure,
        declaration: Declaration | None,
        symbol: str | None,
        records: Sequence[EvidenceRecord],
    ) -> list[dict[str, Any]]:
        """A proposed level is derived, or inside the observed range.

        An entry, a target or a stop is a number the answer proposes rather
        than one it observed, so it cannot be required to equal a print. It
        can be required to be anchored: either the note derives it from what
        the run observed, or it lies between the lowest and highest price this
        session actually saw for the instrument. A level far outside that
        window is the invention this gate exists to stop.
        """
        derivation = self._derivation(declaration, symbol, records)
        if (
            not isinstance(derivation, str)
            and declaration is not None
            and self._result_matches(declaration, derivation[0])
        ):
            return []
        prices = self._price_pool(symbol, records)
        if not prices:
            return [
                self._figure_issue(
                    "numeric_claim_unavailable",
                    figure,
                    "proposed",
                    symbol,
                    "no_evidence",
                    "is a proposed level but this session observed no price for "
                    "the instrument to anchor it to",
                )
            ]
        if not figure.percent and min(prices) <= figure.value <= max(prices):
            return []
        return [
            self._figure_issue(
                "numeric_claim_conflict",
                figure,
                "proposed",
                symbol,
                "outside_observed_range",
                "is a proposed level outside the observed range "
                f"{min(prices):g}–{max(prices):g} and its note derives no value",
                observed_min=min(prices),
                observed_max=max(prices),
                observed_nearest=_nearest(figure.value, prices),
            )
        ]

    def _check_cited(
        self,
        figure: Figure,
        declaration: Declaration | None,
        symbol: str | None,
        declared_observed: set[float],
    ) -> list[dict[str, Any]]:
        """A cited figure names its source and does not pose as a print.

        The value is not checked — the run could never have observed a paper's
        Sharpe — so the only thing to enforce is that the citation is not used
        to launder an observation: the same number may not also be declared
        observed, and it may not sit in a table's OHLC column.
        """
        if declaration is not None and not declaration.note.strip():
            return [
                self._figure_issue(
                    "numeric_claim_conflict",
                    figure,
                    "cited",
                    symbol,
                    "citation_without_source",
                    "is declared cited but names no source in its note",
                )
            ]
        if figure.column or any(
            _close(figure.value, value) for value in declared_observed
        ):
            return [
                self._figure_issue(
                    "numeric_claim_conflict",
                    figure,
                    "cited",
                    symbol,
                    "cited_as_observed",
                    "is declared cited yet presented as an observed value of this "
                    "instrument",
                )
            ]
        return []

    # ------------------------------------------------------------------ #
    # symbols and provenance
    # ------------------------------------------------------------------ #

    def _validate_unsourced_symbols(
        self,
        content: str,
        figures: Sequence[Figure],
        block: FiguresBlock,
    ) -> list[dict[str, Any]]:
        """Reject figures attached to an instrument no tool in this run handled.

        This is the mechanically decidable half of "what the tools did not
        return, you do not supply" (#886/#887). Naming a symbol is left alone —
        prose may legitimately mention an index or a peer — but the moment a
        line pairs an unhandled canonical symbol with a figure, the figure has
        no possible origin other than model memory.

        A figure the model declared ``cited`` is exempt, because a citation is
        an origin. That replaces the phrase catalogue of attribution verbs the
        exemption used to be keyed on, which could not tell "the paper reports"
        from "the backtest reports" without listing every subject by hand.
        """
        issues: list[dict[str, Any]] = []
        reported: set[str] = set()
        for index, (line, offset) in enumerate(_lines_with_offsets(content)):
            unknown = sorted(
                symbol
                for symbol in _scan_symbols(line)
                - self._session_symbols
                - reported
                if symbol.rsplit(".", 1)[0] not in self._session_symbol_roots
            )
            if not unknown:
                continue
            carried = [
                figure
                for figure in figures
                if figure.line == index and figure.shape == "measured"
            ]
            if not carried:
                continue
            if all(
                (block.match(figure.value, figure.percent) or _NO_DECLARATION).role
                == "cited"
                for figure in carried
            ):
                continue
            for symbol in unknown:
                reported.add(symbol)
                issues.append(
                    {
                        "code": "unsourced_symbol_figures",
                        "symbol": symbol,
                        "value": None,
                        "role": None,
                        "reason": "symbol_never_handled",
                        "claim": line.strip()[:200],
                        "span": [offset, offset + len(line)],
                        "message": (
                            f"No tool call in this session passed in or returned {symbol}, "
                            "yet the answer attaches figures to it. Retrieve it, or report "
                            "it as not retrieved."
                        ),
                    }
                )
        return issues

    @staticmethod
    def _symbol_for_claim(
        content: str,
        records: Sequence[EvidenceRecord],
    ) -> str | None:
        """Return one canonical evidence symbol explicitly named in a claim."""
        known = {record.symbol for record in records if record.symbol}
        matches = {
            _normalize_symbol(match.group(0))
            for match in _CANONICAL_SYMBOL_RE.finditer(content)
            if _normalize_symbol(match.group(0)) in known
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def _validate_price_provenance(
        self,
        content: str,
        records: Sequence[EvidenceRecord],
    ) -> list[dict[str, Any]]:
        """Require canonical symbol, actual source, and quote currency in output."""
        issues: list[dict[str, Any]] = []
        folded = content.casefold()
        symbols = sorted({record.symbol for record in records if record.symbol})
        # ``_scan_symbols`` canonicalizes, so an answer that writes Shanghai as
        # ``600519.SS`` still surfaces the ``600519.SH`` identity it names.
        written = _scan_symbols(content)
        mentioned = [
            symbol
            for symbol in symbols
            if symbol in written or symbol.casefold() in folded
        ]
        if not mentioned:
            issues.append(
                {
                    "code": "canonical_symbol_not_surfaced",
                    "symbols": symbols,
                    "value": None,
                    "role": None,
                    "span": None,
                    "symbol": None,
                    "reason": "symbol_not_surfaced",
                    "message": (
                        "A price claim must surface its locked canonical symbol and "
                        "venue suffix."
                    ),
                }
            )
        target_symbols = set(mentioned or (symbols if len(symbols) == 1 else []))
        target_records = [
            record
            for record in records
            if not target_symbols or record.symbol in target_symbols
        ]

        sources = sorted(
            {
                record.source
                for record in target_records
                if record.source and record.source.casefold() not in {"auto", "unknown"}
            }
        )
        missing_sources = [
            source
            for source in sources
            if not any(
                alias in folded
                for alias in _SOURCE_ALIASES.get(source.casefold(), (source.casefold(),))
            )
        ]
        if missing_sources:
            issues.append(
                {
                    "code": "data_source_not_surfaced",
                    "sources": missing_sources,
                    "value": None,
                    "role": None,
                    "span": None,
                    "symbol": None,
                    "reason": "source_not_surfaced",
                    "message": (
                        "Price claims must name the actual data source: "
                        + ", ".join(missing_sources)
                        + "."
                    ),
                }
            )

        currencies = sorted(
            {record.currency for record in target_records if record.currency}
        )
        missing_currencies = [
            currency
            for currency in currencies
            if not self._currency_is_surfaced(currency, content)
        ]
        if missing_currencies:
            issues.append(
                {
                    "code": "currency_not_surfaced",
                    "currencies": missing_currencies,
                    "value": None,
                    "role": None,
                    "span": None,
                    "symbol": None,
                    "reason": "currency_not_surfaced",
                    "message": (
                        "Price claims must name their quote currency: "
                        + ", ".join(missing_currencies)
                        + "."
                    ),
                }
            )
        return issues

    @staticmethod
    def _currency_is_surfaced(currency: str, content: str) -> bool:
        """Return whether a quote currency or an unambiguous alias is visible."""
        folded = content.casefold()
        code = currency.upper()
        tokens = _CURRENCY_ALIASES.get(code, (currency.casefold(),))
        if any(token.casefold() in folded for token in tokens):
            return True
        if code != "CNY":
            return False
        return any(
            char == "元"
            and (index == 0 or content[index - 1] not in _OTHER_CURRENCY_PREFIXES)
            for index, char in enumerate(content)
        )

    @staticmethod
    def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate validator findings while preserving order."""
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for issue in issues:
            key = json.dumps(issue, sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            unique.append(issue)
        return unique


#: The role an undeclared figure is validated under (spec §4, undeclared mode).
_NO_DECLARATION = Declaration(
    index=0, value_text="", value=0.0, percent=False, role="observed", note="", ref=""
)
