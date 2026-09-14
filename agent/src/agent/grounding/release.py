"""What happens after a rejection: correction, recovery, repair, redaction."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

from src.agent.grounding.identity import (
    _CANONICAL_SYMBOL_RE,
    _RESOLVER_TOOL,
    _normalize_symbol,
    _utc_now,
)
from src.agent.grounding.evidence import EvidenceRecord
from src.agent.grounding.figures import _NUMBER_RE, _clause_spans, _lines_with_offsets
from src.agent.grounding.policies import (
    ValidationResult,
    _ANALYSIS_METRIC_RE,
    _CURRENCY_BEFORE_RE,
    _CURRENCY_UNIT_WORDS,
    _MEASURE_NUMBER_RE,
    _PolicyMixin,
    _TABLE_FIELD_ALIASES,
    _matches_any,
)

# Bounded read-only recovery (#1081): a missing instrument identity or price
# evidence is often recoverable deterministically, so the loop should keep
# driving the original task through `search_symbol` and `get_market_data`
# instead of handing the user a terminal "confirm and continue" fallback.
# These budgets are separate from the rejected-draft count so real recovery
# progress is never cut off at the three-draft retry cap.
MAX_GROUNDING_RECOVERY_ROUNDS = 6


# Issue codes whose figure can be cut out of its clause and the draft released
# (see ``redacted_release``), and the provenance codes a data note repairs.
_REDACTABLE_CODES = frozenset(
    {
        "numeric_claim_conflict",
        "numeric_claim_unavailable",
        "unsourced_symbol_figures",
        "analysis_claim_unavailable",
    }
)


# The provenance codes a data note may fill in deterministically. The source
# and the currency are metadata ABOUT a figure. ``canonical_symbol_not_surfaced``
# is deliberately absent: the symbol is the figure's SUBJECT, and appending
# "562500.SH: source yahoo" under a draft that calls the instrument 贵州茅台
# footnotes a misattribution instead of repairing it — so the identity binding
# keeps its model-correction round.
_REPAIRABLE_PROVENANCE_CODES = frozenset(
    {"data_source_not_surfaced", "currency_not_surfaced"}
)


_MAX_REDACTION_PASSES = 3


# The in-text mark for an omitted figure. It replaces the number AND the unit
# glued to it ("0.95 元" → "（略※）", "$1.10" → "(omitted※)") so the sentence
# still reads as prose, and the ※ points at the one footnote that says how
# many figures were omitted and why. A bracketed "[value removed]" left the
# unit dangling and read like a redaction stamp.
_REDACTION_MARKER_ZH = "（略※）"


_REDACTION_MARKER_EN = "(omitted※)"


_UNIT_AFTER_RE = re.compile(
    r"\s*(?:(?:" + _CURRENCY_UNIT_WORDS + r")(?![A-Za-z/／])"
    r"|元(?![A-Za-z/／\u3400-\u9fff]))"
)


MAX_SYMBOL_RESOLUTION_ATTEMPTS = 2


MAX_PRICE_EVIDENCE_ATTEMPTS = 3


# A table column whose header names a price. Wider than the OHLC aliases on
# purpose: 挂单价 / 目标价 / Entry Price are price columns that
# ``_validate_price_tables`` does not key on and the prose scan skips, so
# their cells are figures no validator ever reads.
_PRICE_HEADER_RE = re.compile(r"(?:价格?|价位|price)\s*$", re.IGNORECASE)


def _format_price(value: float) -> str:
    """Render an observed price without scientific notation or lost digits.

    ``%g`` switches to scientific notation past six significant digits, so an
    index level printed in the release footnote read "1.23457e+06".
    """
    return format(value, ".10g")


def _redaction_targets(values: Sequence[Any]) -> tuple[bool, list[float], list[str]]:
    """Split issue values into cut-everything / numeric / percent-literal cuts.

    Args:
        values: The ``value`` field of each issue flagged on one clause.

    Returns:
        ``(cut_all, numeric targets, percent literals)``. ``cut_all`` is set by
        a ``None`` value, which is how ``unsourced_symbol_figures`` says "every
        figure in this clause belongs to an instrument no tool handled".
    """
    cut_all = False
    targets: list[float] = []
    literals: list[str] = []
    for value in values:
        if value is None:
            cut_all = True
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            targets.append(float(value))
        elif isinstance(value, str) and value.strip():
            literal = value.strip().replace(" ", "").replace(",", "")
            if literal.endswith(("%", "％")):
                literals.append(literal)
            else:
                try:
                    targets.append(float(literal))
                except ValueError:
                    continue
    return cut_all, targets, literals


_RELEASE_NOTE_LINE_RE = re.compile(r"^[^\S\n]*※[^\n]*\n?", re.MULTILINE)


def _strip_release_markers(content: str) -> str:
    """Remove any redaction marker or release note the MODEL wrote.

    The marker and the footnote are the gate's own statements about what it
    cut. A draft carrying "※ 略去 0 处……" of its own shipped both notes, one
    contradicting the other, and a "（略※）" pasted into the body read as a
    redaction that never happened.

    Args:
        content: The rejected draft.

    Returns:
        The draft with every marker and note-shaped line removed.
    """
    stripped = content.replace(_REDACTION_MARKER_ZH, "").replace(
        _REDACTION_MARKER_EN, ""
    )
    return _RELEASE_NOTE_LINE_RE.sub("", stripped)


class _ReleaseMixin:
    """Release behaviour of :class:`GroundingLedger`."""

    def correction_prompt(self, validation: ValidationResult) -> str:
        """Build bounded feedback for one rejected model draft."""
        lines = [
            "[GROUNDING GATE] The previous draft was rejected and was not released to the user.",
            "Correct every issue using the existing structured identity and tool evidence:",
        ]
        for issue in validation.issues[:12]:
            lines.append(f"- {issue.get('message', issue.get('code', 'grounding error'))}")
        # Name the exact values that must be REMOVED, not rephrased. The model
        # tends to restate a rejected figure in a new format; the gate then
        # rejects it again and the run burns iterations until the fallback.
        banned: list[str] = []
        for issue in validation.issues:
            code = issue.get("code")
            value = issue.get("value")
            if code in {"numeric_claim_conflict", "numeric_claim_unavailable", "unsourced_symbol_figures", "analysis_claim_unavailable"} and value is not None:
                symbol = issue.get("symbol") or ""
                label = f"{value:g}" if isinstance(value, (int, float)) else str(value)
                banned.append(f"{label} ({symbol})" if symbol else label)
        if banned:
            deduped = list(dict.fromkeys(banned))
            lines.append(
                "REMOVE these rejected value(s) entirely - do NOT restate, rephrase, "
                "or recompute them in any other format: " + ", ".join(deduped) + "."
            )
            repeated: list[str] = []
            for prior in self._validations:
                for prior_issue in prior.get("issues", []):
                    prior_value = prior_issue.get("value")
                    if isinstance(prior_value, (int, float)):
                        mark = f"{prior_value:g}"
                        if any(entry.startswith(mark) for entry in deduped):
                            repeated.append(mark)
            if repeated:
                lines.append(
                    "These value(s) have now been rejected repeatedly across drafts: "
                    + ", ".join(dict.fromkeys(repeated))
                    + ". Repeating them in any form keeps failing; drop them, or show "
                    "the full derivation from the observed inputs."
                )
        lines.extend(
            [
                "If a value is a derived or prospective level (stop, target, entry, etc.), "
                "you must EITHER show the full derivation with the observed inputs and the "
                "formula, OR omit it from the draft.",
                "Reuse the exact locked symbol and venue.",
                "Do not attach figures to a symbol no tool call in this session handled; "
                "report it as not retrieved instead.",
            ]
        )
        recovery = self.recovery_action(validation)
        if recovery == _RESOLVER_TOOL:
            lines.extend(
                [
                    "Instrument identity is unresolved. Call `search_symbol` for the "
                    "candidate name in a separate tool-call turn, lock the exact canonical "
                    "symbol and venue it returns, then call `get_market_data` before finalizing.",
                    "Do NOT ask the user to confirm or continue while this read-only recovery "
                    "remains available.",
                ]
            )
        elif recovery == "get_market_data":
            lines.extend(
                [
                    "Identity is locked but price evidence is missing. Call `get_market_data` "
                    "for the locked canonical symbol and venue in a separate tool-call turn, "
                    "then regenerate and re-validate the final answer.",
                    "Do NOT ask the user to confirm or continue while this read-only recovery "
                    "remains available.",
                ]
            )
        else:
            lines.append(
                "If evidence is genuinely unavailable or conflicting and recovery is "
                "exhausted, say so and ask for clarification; do not guess."
            )
        return "\n".join(lines)

    def recovery_action(self, validation: ValidationResult) -> str | None:
        """Decide the next safe read-only recovery step for a rejected draft.

        Returns ``search_symbol`` when instrument identity is unresolved and
        resolution attempts remain; ``get_market_data`` when identity is locked
        but a price claim has no observed evidence and fetch attempts remain;
        otherwise ``None`` (genuinely ambiguous, conflicting, or exhausted —
        the loop must then ask the user or fail closed).

        This is the deterministic half of #1081: recoverable missing evidence is
        often obtainable through read-only tools, so a rejected draft should
        drive the original task forward instead of stopping.
        """
        if self._recovery_rounds >= MAX_GROUNDING_RECOVERY_ROUNDS:
            return None
        if self._identity_required and self.identity_status == "unresolved":
            if self._symbol_resolution_attempts < MAX_SYMBOL_RESOLUTION_ATTEMPTS:
                return _RESOLVER_TOOL
            return None
        if self.identity_status == "locked" and any(
            issue.get("code") in {"numeric_claim_unavailable", "unsourced_symbol_figures"}
            for issue in validation.issues
        ):
            if self._price_evidence_attempts < MAX_PRICE_EVIDENCE_ATTEMPTS:
                return "get_market_data"
        return None

    def record_recovery(self, action: str) -> None:
        """Account one bounded recovery attempt against its budget."""
        self._recovery_rounds += 1
        if action == _RESOLVER_TOOL:
            self._symbol_resolution_attempts += 1
        elif action == "get_market_data":
            self._price_evidence_attempts += 1

    def recovery_prompt(self, action: str, validation: ValidationResult) -> str:
        """Build an executable next-step message for one bounded recovery turn."""
        if action == _RESOLVER_TOOL:
            return (
                "[GROUNDING RECOVERY] Instrument identity is not yet locked and is "
                "recoverable with read-only tools. Call `search_symbol` for the candidate "
                "name in a separate assistant tool-call turn, lock and reuse the exact "
                "canonical symbol and venue it returns, then call `get_market_data`. "
                "Do NOT ask the user to confirm or continue while this read-only recovery "
                "remains available, and do NOT finalize yet."
            )
        if action == "get_market_data":
            return (
                "[GROUNDING RECOVERY] Identity is locked but price evidence is missing. "
                "Call `get_market_data` for the locked canonical symbol and venue in a "
                "separate tool-call turn and use its existing bounded provider fallback, "
                "then regenerate and re-validate the final answer. Do NOT ask the user to "
                "confirm or continue while this read-only recovery remains available, and "
                "do NOT finalize yet."
            )
        return self.correction_prompt(validation)

    def safe_fallback(self) -> str:
        """Return a deterministic fail-closed answer after repeated rejection."""
        is_zh = self._user_writes_chinese()
        joined = self._observed_range_summary(is_zh)
        if joined is not None:
            if is_zh:
                return (
                    "为避免输出与工具证据冲突的价格，我已拒绝上一版答案。"
                    f"当前可验证的已观测 OHLC 范围是：{joined}。"
                    "在重新核对标的或明确展示推导公式前，我不会生成买入价。"
                )
            return (
                "I rejected the previous draft because its prices conflicted with tool evidence. "
                f"The verified observed OHLC range is: {joined}. "
                "I will not invent an entry price without a visible derivation or refreshed evidence."
            )
        # No observed price evidence: distinguish "identity unresolved" from
        # "the draft cited prices this session never observed". Reporting the
        # identity message for the latter is misleading (the run may not even
        # have touched the market tools).
        issue_codes = {
            code
            for validation in self._validations
            for code in (issue.get("code") for issue in validation.get("issues", []))
        }
        if issue_codes & {
            "numeric_claim_unavailable", "numeric_claim_conflict", "unsourced_symbol_figures"
        }:
            if is_zh:
                return (
                    "我的回答被安全门槛拒绝:草稿引用了本会话未通过工具获取的价格数字,无法核验。"
                    "请重新发起任务,让模型先调用行情工具获取数据,或要求它去掉这些价格引用后重试。"
                )
            return (
                "My previous answer was rejected by the verification gate: it cited price "
                "figures that this session never obtained through a tool, so they could not "
                "be verified. Re-run the task and let the agent fetch the market data first, "
                "or ask it to answer without the unverified prices."
            )
        if is_zh:
            return (
                "当前无法安全确认标的身份或价格证据，因此没有生成交易结论。"
                "请确认候选证券代码和交易所后再继续。"
            )
        return (
            "I could not safely lock the instrument identity or price evidence, so I did not "
            "produce a trading conclusion. Please confirm the candidate symbol and venue."
        )

    def _user_writes_chinese(self) -> bool:
        """Return whether user-facing gate text should be Chinese."""
        return bool(re.search(r"[\u3400-\u9fff]", self.user_message))

    def _observed_range_summary(self, is_zh: bool, content: str | None = None) -> str | None:
        """Summarise the observed OHLC range per symbol, or None without prices.

        Args:
            is_zh: Whether to join the facts with Chinese punctuation.
            content: The answer the summary is attached to, when there is one.
                The symbol is then printed the way that answer spells it — a
                footnote on an answer written throughout in ``562500.SS`` used
                to name ``562500.SH``, because the evidence record is
                canonicalised.

        Returns:
            One fact per symbol, or None when the run observed no price.
        """
        price_records = self._price_records()
        if not price_records:
            return None
        by_symbol: dict[str, list[EvidenceRecord]] = {}
        for record in price_records:
            by_symbol.setdefault(record.symbol or "unknown", []).append(record)
        facts = []
        for symbol, records in sorted(by_symbol.items()):
            values = [float(record.value) for record in records if record.value is not None]
            currency = next((record.currency for record in records if record.currency), None)
            sources = sorted({record.source for record in records if record.source})
            source_label = "/".join(sources) if sources else "unknown"
            unit = f" {currency}" if currency else ""
            facts.append(
                f"{self._answer_symbol_spelling(symbol, content)}: "
                f"{_format_price(min(values))}–{_format_price(max(values))}{unit} "
                f"(source: {source_label}; currency conversion: none)"
            )
        return "；".join(facts) if is_zh else "; ".join(facts)

    @staticmethod
    def _answer_symbol_spelling(canonical: str, content: str | None) -> str:
        """Return the spelling ``content`` uses for a canonical symbol."""
        if not content:
            return canonical
        for match in _CANONICAL_SYMBOL_RE.finditer(content):
            if _normalize_symbol(match.group(0)) == canonical:
                return match.group(0)
        return canonical

    def _provenance_note(self, content: str | None = None) -> str | None:
        """Build the one-line data note that satisfies the provenance checks.

        Args:
            content: The answer the note is appended to, so the symbol is
                printed the way that answer spells it.

        Returns:
            The note, or None when the run observed no price.
        """
        price_records = self._price_records()
        if not price_records:
            return None
        is_zh = self._user_writes_chinese()
        by_symbol: dict[str, list[EvidenceRecord]] = {}
        for record in price_records:
            by_symbol.setdefault(record.symbol or "unknown", []).append(record)
        parts = []
        for symbol, records in sorted(by_symbol.items()):
            sources = sorted(
                {
                    record.source
                    for record in records
                    if record.source and record.source.casefold() not in {"auto", "unknown"}
                }
            )
            currency = next((record.currency for record in records if record.currency), None)
            source_label = "/".join(sources) if sources else ("未知" if is_zh else "unknown")
            currency_label = currency or ("未知" if is_zh else "unknown")
            spelling = self._answer_symbol_spelling(symbol, content)
            parts.append(
                f"{spelling}：行情来源 {source_label}，计价货币 {currency_label}"
                if is_zh
                else f"{spelling}: price source {source_label}, quote currency {currency_label}"
            )
        if is_zh:
            return "数据说明：" + "；".join(parts) + "。"
        return "Data note: " + "; ".join(parts) + "."

    def repair_provenance(
        self,
        content: str,
        validation: ValidationResult,
        *,
        require_checked_figures: bool = True,
    ) -> str | None:
        """Append a data note when the only defects are missing provenance words.

        ``data_source_not_surfaced`` and ``currency_not_surfaced`` mean the
        answer quoted a price without naming the source or the currency. Both
        are known to the ledger, so the omission is deterministic — and
        regenerating a multi-minute report to add the word "tencent" is a full
        model round for one word. Nothing here touches a figure: a draft
        carrying any other issue is returned as None so the numeric checks
        keep their round.

        ``canonical_symbol_not_surfaced`` is deliberately NOT repaired. That
        check's job is that a price claim surfaces the symbol it is about, and
        appending "562500.SH: price source yahoo" under a draft that reads
        "贵州茅台 最新收盘价 1.171 元" turns a misattribution into a released
        answer with a footnote naming a different instrument. It keeps its
        model round (base rate: 1 of 27 rejections in the local traces, so
        nearly all of the saving survives).

        The repair is also declined when the draft carries a price COLUMN the
        table validator does not key on ("| 档位 | 挂单价 |"): those cells are
        outside every validator's reach, and the note says where this run's
        prices came from. Attaching it to a draft holding an unchecked ladder
        price attests to a figure the gate never saw, in zero model rounds —
        the round it used to cost was the last chance to drop that figure.

        Args:
            content: The rejected draft.
            validation: Its validation result.
            require_checked_figures: Decline when an unchecked price column is
                present. ``redacted_release`` passes False: there the choice
                is not "repair or one more model round" but "repair or the
                canned refusal", the revision budget is already spent, and the
                document it repairs carries the redaction footnote.

        Returns:
            The draft with a provenance note appended, or None when the issues
            are not provenance-only, an unchecked price column is present, or
            there is no price evidence to cite.
        """
        codes = {issue.get("code") for issue in validation.issues}
        if not codes or not codes <= _REPAIRABLE_PROVENANCE_CODES:
            return None
        if require_checked_figures and self._has_unchecked_price_column(content):
            return None
        note = self._provenance_note(content)
        if note is None:
            return None
        return content.rstrip() + "\n\n" + note

    @staticmethod
    def _has_unchecked_price_column(content: str) -> bool:
        """Whether a Markdown table holds a price column no validator reads.

        ``_validate_price_tables`` keys on ``_TABLE_FIELD_ALIASES`` — the OHLC
        headers — and the prose scan skips every line containing "|". A column
        headed 挂单价 / 目标价 / Entry price is read by neither.

        Args:
            content: The draft to inspect.

        Returns:
            True when such a column carries at least one numeric cell.
        """
        lines = content.splitlines()
        for header, rows, _ in _PolicyMixin._pipe_tables(lines):
            unchecked = [
                position
                for position, cell in enumerate(header)
                if _PRICE_HEADER_RE.search(cell)
                and cell.strip().casefold() not in _TABLE_FIELD_ALIASES
            ]
            if not unchecked:
                continue
            for row in rows:
                for position in unchecked:
                    if position < len(row) and _NUMBER_RE.search(row[position]):
                        return True
        return False

    def redacted_release(self, content: str, validation: ValidationResult) -> str | None:
        """Release the last rejected draft with its unverified figures cut out.

        Once the revision budget is spent, the draft is still the analysis the
        user waited through every revision for, and the gate objected to
        specific figures in specific clauses — not to the trend read, the
        indicator commentary, or the risk notes around them. Each rejected
        figure is replaced by a visible marker AT THE SPAN the validator
        flagged, every other occurrence of the same figure elsewhere in the
        document is cut with it, the missing provenance words are appended if
        that is all that remains, and the whole released document — footnote
        included — is re-validated by the same gate: only text that passes is
        returned, so nothing the gate rejected reaches the user.

        Fail-closed by construction. None — leave the canned fallback in place —
        whenever the run never observed a price at all (the draft's numbers then
        have no basis to stand next to), carries an issue that is not a
        cut-out-able figure (an identity finding is one), has a flagged clause
        that cannot be located, or still fails after the cut.

        Args:
            content: The rejected draft.
            validation: Its validation result.

        Returns:
            The redacted, re-validated draft followed by a note stating how many
            figures were removed and the observed range, or None.
        """
        if not self._price_records():
            return None
        text = _strip_release_markers(content)
        removed = 0
        # Stripping moves every offset after it, and the issue spans are the
        # only anchor the cuts have, so the verdict is retaken on the text the
        # cuts will actually be made in.
        pending = list(
            validation.issues
            if text == content
            else self._validate(text, record=False).issues
        )
        # A validator reports one figure per clause, so cutting it can reveal
        # the next one on the recheck. Cut, recheck, repeat — bounded, and
        # every cut is a figure the gate itself flagged.
        for _ in range(_MAX_REDACTION_PASSES):
            codes = {issue.get("code") for issue in pending}
            if not codes or not codes <= (_REDACTABLE_CODES | _REPAIRABLE_PROVENANCE_CODES):
                return None
            # Two issues can point at one clause ("unsourced symbol" and the
            # conflict on the same figure), so cuts are grouped per span:
            # each clause is rewritten once with every value flagged for it.
            by_span: dict[tuple[int, int], list[Any]] = {}
            for issue in pending:
                if issue.get("code") not in _REDACTABLE_CODES:
                    continue
                span = self._issue_span(text, issue)
                if span is None:
                    return None
                by_span.setdefault(span, []).append(issue.get("value"))
            text, count = self._redact_spans(text, by_span)
            if count == 0:
                return None
            removed += count
            check = self._validate(text, record=False)
            if not check.valid:
                repaired = self.repair_provenance(
                    text, check, require_checked_figures=False
                )
                if repaired is not None:
                    text = repaired
                    check = self._validate(text, record=False)
            if check.valid:
                break
            pending = list(check.issues)
        else:
            return None
        released = text.rstrip() + "\n\n" + self._release_note(removed, text)
        # The note carries the observed range and the canonical symbols, so it
        # is answer text too. Validating the body and shipping body+note left
        # the released document as a whole unchecked, which contradicted the
        # fail-closed promise in this docstring.
        if not self._validate(released, record=False).valid:
            return None
        # Recorded beside the drafts but NOT as one: the artifact otherwise
        # held no evidence for the fail-closed promise above, because every
        # recheck on this path is deliberately unrecorded.
        self._released = {
            "released_at": _utc_now(),
            "content_sha256": hashlib.sha256(released.encode("utf-8")).hexdigest(),
            "figures_removed": removed,
            "revalidated": True,
        }
        self.persist()
        return released

    @staticmethod
    def _issue_span(text: str, issue: dict[str, Any]) -> tuple[int, int] | None:
        """Locate the flagged clause in ``text``.

        The validators record the clause's character span, which is the only
        reliable anchor: the stored ``claim`` is normalised (thousands
        separators stripped, table cells rendered as "label: value"), so
        ``text.find(claim)`` missed every price above 999 and every metric
        table outright, and for a bare numeric cell it matched INSIDE a longer
        number — "1.10" found in "21.10 亿元", rewriting an untouched turnover
        figure into "2（略※）".

        The substring fallback that used to sit here is gone. Every one of the
        six redactable-issue construction sites carries a span, so it was
        reachable only from a hand-built issue, and it carried a documented
        digit-boundary rationale for a hazard the span had already settled —
        which is worse than no code, because a reader trusts it.

        Args:
            text: The document the issue was raised against.
            issue: One validation issue.

        Returns:
            ``(start, end)`` or None when the issue carries no usable span.
        """
        span = issue.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return None
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None
        if 0 <= start <= end <= len(text):
            return start, end
        return None

    def _release_note(self, removed: int, content: str | None = None) -> str:
        """Explain the redaction to the user, with the observed range.

        ``redacted_release`` returns None before this is reached when the run
        holds no price record, which is the only case in which the summary is
        None — so it is read directly rather than behind an ``or ""`` that
        described a state the caller cannot produce.
        """
        is_zh = self._user_writes_chinese()
        joined = self._observed_range_summary(is_zh, content)
        if is_zh:
            return (
                f"※ 略去 {removed} 处无法与本会话工具数据对上的数值。"
                f"已观测 OHLC 范围：{joined}。"
                "如需买入价，请让我基于已观测的收盘价或均线给出带公式的推导。"
            )
        return (
            f"※ {removed} figure(s) that could not be matched to this session's tool data "
            f"were omitted. Observed OHLC range: {joined}. "
            "For an entry price, ask me to derive one with a visible formula from an "
            "observed close or moving average."
        )

    def _redact_spans(
        self,
        text: str,
        by_span: dict[tuple[int, int], list[Any]],
    ) -> tuple[str, int]:
        """Cut the flagged figures at their spans, then everywhere else.

        Numbers are located on the masked clause (symbols, dates, prospective
        levels blanked) so a ticker's digits are never cut. A numeric value is
        matched numerically, a percent-shaped analysis value textually, and a
        ``None`` value (a clause attributing figures to an unhandled symbol)
        cuts every number in that clause.

        The second stage is why the footnote can be believed. Cutting only the
        flagged clause left the same rejected figure standing in a Markdown
        table under a non-OHLC header or in a bullet with no price word —
        neither surface is scanned by the validators — while the footnote
        asserted it had been removed.

        Two kinds of occurrence are never swept, because the gate has already
        accepted them where they stand: a figure the ledger observed, and a
        figure sitting inside a valid formula elsewhere in the document.
        Without the second, the rejected entry price 0.95 was cut out of
        "基于 MA20 0.7158 × 0.95 = 0.680", mangling the one derivation the
        correction prompt asks the model to write (159516.SZ replay,
        2026-09-09).

        The formula protection is scoped to the formula's own character SPANS.
        Scoped to the VALUE it protected the rejected figure document-wide:
        with "基于收盘价 1.171 × 0.95 = 1.112" anywhere in the draft, the same
        0.95 survived in "| 第一档 | 0.95 |" under a footnote saying one figure
        had been cut — the exact restatement this second stage exists to
        remove, protected by the derivation the model wrote beside it.

        A percent literal is protected the same way. Cutting every occurrence
        of a flagged literal took the accepted one with it: "从 1.110 涨到
        1.171，区间收益率约 5.5%" is endpoint arithmetic this gate validated,
        and it was cut and footnoted as unmatched because an unrelated
        "策略年化波动率 5.5%" was flagged in the next clause.

        Args:
            text: The document to rewrite.
            by_span: ``(start, end)`` → the values flagged inside that span.

        Returns:
            The rewritten text and the number of figures replaced.
        """
        swept: list[float] = []
        swept_literals: list[str] = []
        pieces: list[str] = []
        cursor = 0
        total = 0
        for (start, end), values in sorted(by_span.items()):
            if start < cursor:
                continue
            cut_all, targets, literals = _redaction_targets(values)
            swept.extend(targets)
            swept_literals.extend(literals)
            replaced, count = self._rewrite_segment(
                text[start:end], cut_all, targets, literals
            )
            pieces.append(text[cursor:start])
            pieces.append(replaced)
            total += count
            cursor = end
        pieces.append(text[cursor:])
        text = "".join(pieces)
        if total == 0 or not (swept or swept_literals):
            return text, total
        observed = self._observed_price_values()
        survivors = [
            target for target in swept if not _matches_any(target, observed, rel=1e-9)
        ]
        if not survivors and not swept_literals:
            return text, total
        protected = self._derivation_spans(text)
        if swept_literals:
            protected += self._accepted_metric_spans(text)
        # Line by line, so each replacement takes the marker from the script of
        # its OWN line. Run over the whole document the second stage picked one
        # marker for everything, and a Chinese report containing an English
        # table released "| Entry |（略※） |".
        extra = 0
        out: list[str] = []
        cursor = 0
        for line, start in _lines_with_offsets(text):
            out.append(text[cursor:start])
            local = [
                (span_start - start, span_end - start)
                for span_start, span_end in protected
                if span_start >= start and span_end <= start + len(line)
            ]
            rewritten, count = self._rewrite_segment(
                line, False, survivors, swept_literals, protected=local
            )
            out.append(rewritten)
            extra += count
            cursor = start + len(line)
        out.append(text[cursor:])
        return "".join(out), total + extra

    def _derivation_spans(self, text: str) -> list[tuple[int, int]]:
        """Where a valid derivation sits in ``text``, in document offsets.

        Args:
            text: The document about to be swept.

        Returns:
            The character span of every arithmetically valid,
            observation-anchored formula in it.
        """
        records = self._comparable_price_records()
        document_symbol = self._symbol_for_claim(text, records)
        spans: list[tuple[int, int]] = []
        for line, offset in _lines_with_offsets(text):
            if not _NUMBER_RE.search(line):
                continue
            symbol = self._symbol_for_claim(line, records) or document_symbol
            for start, end, _, _ in self._derivation_formulas(line, records, symbol):
                spans.append((offset + start, offset + end))
        return spans

    def _accepted_metric_spans(self, text: str) -> list[tuple[int, int]]:
        """Where this gate ACCEPTED a metric figure in ``text``.

        A clause the analysis validator examined — it carries a metric word
        and a measurement-shaped number — and did not flag is a clause whose
        figures this gate grounded. Sweeping such a figure out because an
        unrelated clause states the same percentage cut a validated derivation
        and footnoted it as unmatched.

        The verdict comes from the validator itself rather than from a second
        copy of its exemption rules: a clause with no metric word was never
        examined (a bullet, an unclaimed table column) and is deliberately not
        protected — that unscanned surface is what the sweep exists for.

        Args:
            text: The document about to be swept.

        Returns:
            The character span of every examined-and-accepted metric clause.
        """
        flagged = {
            (int(issue["span"][0]), int(issue["span"][1]))
            for issue in self._validate(text, record=False).issues
            if isinstance(issue.get("span"), (list, tuple))
            and len(issue["span"]) == 2
        }
        spans: list[tuple[int, int]] = []
        for line, offset in _lines_with_offsets(text):
            for segment, start, end in _clause_spans(line, offset):
                if not _ANALYSIS_METRIC_RE.search(segment):
                    continue
                if not self._measure_numbers(segment):
                    continue
                if any(
                    span_start <= start and end <= span_end
                    for span_start, span_end in flagged
                ):
                    continue
                spans.append((start, end))
        return spans

    def _rewrite_segment(
        self,
        segment: str,
        cut_all: bool,
        targets: Sequence[float],
        literals: Sequence[str],
        *,
        protected: Sequence[tuple[int, int]] = (),
    ) -> tuple[str, int]:
        """Replace the wanted figures in one stretch of text with the marker.

        Args:
            segment: The stretch of the answer to rewrite.
            cut_all: Cut every number, not only the listed targets.
            targets: Numeric values to cut.
            literals: Percent-shaped figures to cut by their written text.
            protected: Character ranges inside ``segment`` that must survive
                whatever they contain — a valid formula, or a metric clause
                this gate accepted. Protection is by POSITION, not by value:
                the rejected entry price is usually also the multiplier of the
                correct derivation, and protecting the value protected every
                restatement of the rejected figure along with it.

        Returns:
            The rewritten stretch and the number of figures replaced.
        """
        if not cut_all and not targets and not literals:
            return segment, 0
        # The marker follows the SCRIPT OF THE TEXT BEING CUT, not the user's
        # message: a Chinese user asking about a US name gets English tables,
        # and "Suggested entry price（略※） per share." was the result. A
        # stretch with neither script — a numeric table row — falls back to
        # the user's language, which is what the footnote is written in.
        if re.search(r"[\u3400-\u9fff]", segment):
            marker = _REDACTION_MARKER_ZH
        elif re.search(r"[A-Za-z]", segment):
            marker = _REDACTION_MARKER_EN
        else:
            marker = (
                _REDACTION_MARKER_ZH
                if self._user_writes_chinese()
                else _REDACTION_MARKER_EN
            )
        spans: list[tuple[int, int]] = []
        if literals:
            for match in _MEASURE_NUMBER_RE.finditer(segment):
                normalized = match.group(0).replace(" ", "").replace(",", "")
                if normalized in literals:
                    spans.append((match.start(), match.end()))
        masked = self._masked_candidate_text(segment)
        for match in _NUMBER_RE.finditer(masked):
            try:
                number = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            if cut_all or _matches_any(number, targets, rel=1e-9):
                spans.append((match.start(), match.end()))
        pieces: list[str] = []
        cursor = 0
        count = 0
        for start, end in sorted(set(spans)):
            if start < cursor:
                continue
            if any(
                keep_start <= start and end <= keep_end
                for keep_start, keep_end in protected
            ):
                continue
            unit = _UNIT_AFTER_RE.match(segment, end)
            if unit:
                end = unit.end()
            symbol = _CURRENCY_BEFORE_RE.search(segment[cursor:start])
            if symbol:
                start = cursor + symbol.start()
            if marker is _REDACTION_MARKER_ZH:
                # "建议买入价 0.95 元" → "建议买入价（略※）": a full-width
                # bracket sits flush against the preceding word. Not against a
                # table pipe, though — "| 第一档 |（略※） |" loses the column's
                # padding and reads as a broken row.
                while (
                    start > cursor
                    and segment[start - 1] == " "
                    and segment[: start - 1].rstrip(" ")[-1:] != "|"
                ):
                    start -= 1
            pieces.append(segment[cursor:start])
            pieces.append(marker)
            cursor = end
            count += 1
        pieces.append(segment[cursor:])
        return "".join(pieces), count
