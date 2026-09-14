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
from src.agent.grounding.figures import (
    _lines_with_offsets,
    currency_prefix_start,
    currency_suffix_end,
    parse_figures_block,
    scan_figures,
)
from src.agent.grounding.policies import ValidationResult

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
        "figure_undeclared",
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


MAX_SYMBOL_RESOLUTION_ATTEMPTS = 2


MAX_PRICE_EVIDENCE_ATTEMPTS = 3


def _format_price(value: float) -> str:
    """Render an observed price without scientific notation or lost digits.

    ``%g`` switches to scientific notation past six significant digits, so an
    index level printed in the release footnote read "1.23457e+06".
    """
    return format(value, ".10g")


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
    return "\n".join(
        line for line in stripped.splitlines() if not line.lstrip().startswith("※")
    )


class _ReleaseMixin:
    """Release behaviour of :class:`GroundingLedger`."""

    def correction_prompt(self, validation: ValidationResult) -> str:
        """Build bounded feedback for one rejected model draft.

        The feedback is per NUMBER, not per rule: every figure issue carries
        the value as it was written, the role it was declared under and the
        reason that role failed, so the message says "0.95 is a proposed level
        outside the observed range 0.567–1.053 and its note derives no value"
        instead of restating the policy. Each figure then has exactly three
        ways out, and they are spelled out once at the end.
        """
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
            if issue.get("code") not in _REDACTABLE_CODES:
                continue
            value = issue.get("value")
            if value is None:
                continue
            symbol = issue.get("symbol") or ""
            label = f"{value:g}" if isinstance(value, (int, float)) else str(value)
            banned.append(f"{label} ({symbol})" if symbol else label)
        if banned:
            deduped = list(dict.fromkeys(banned))
            lines.append(
                "Every figure above must be either DECLARED with the role it really "
                "has, REWRITTEN to a value the tools returned, or REMOVED. Do not "
                "restate a rejected value in another format: " + ", ".join(deduped) + "."
            )
            repeated: list[str] = []
            for prior in self._validations:
                for prior_issue in prior.get("issues", []):
                    prior_value = prior_issue.get("value")
                    if prior_value is None:
                        continue
                    mark = (
                        f"{prior_value:g}"
                        if isinstance(prior_value, (int, float))
                        else str(prior_value)
                    )
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
                "End the answer with a ```figures``` block declaring every number that "
                "carries a decimal point, a percent sign, a currency mark or a table "
                "cell, one per line as `value | role | note | ref`, where role is one "
                "of observed / derived / proposed / cited / count.",
                "observed must appear in the tool results; derived needs a note that is "
                "the arithmetic itself, with one operand this session observed; "
                "proposed must be derived or lie inside the observed price range; "
                "cited needs a source in its note; count is not checked.",
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
        if issue_codes & _REDACTABLE_CODES:
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
        model round.

        The "unchecked price column" veto this used to carry is gone with the
        surface it guarded. It declined the repair whenever a table held a
        price column no validator read (``| 档位 | 挂单价 |``), because the
        note would attest to figures the gate never saw. Every cell of every
        table is now a measurement-shaped figure the gate checks, so no such
        column exists.

        Args:
            content: The rejected draft.
            validation: Its validation result.

        Returns:
            The draft with a provenance note appended, or None when the issues
            are not provenance-only or there is no price evidence to cite.
        """
        codes = {issue.get("code") for issue in validation.issues}
        if not codes or not codes <= _REPAIRABLE_PROVENANCE_CODES:
            return None
        note = self._provenance_note(content)
        if note is None:
            return None
        return content.rstrip() + "\n\n" + note

    def redacted_release(self, content: str, validation: ValidationResult) -> str | None:
        """Release the last rejected draft with its unverified figures cut out.

        Once the revision budget is spent, the draft is still the analysis the
        user waited through every revision for, and the gate objected to
        specific figures — not to the trend read, the indicator commentary, or
        the risk notes around them. Each rejected figure is replaced by a
        visible marker AT ITS OWN SPAN, the missing provenance words are
        appended if that is all that remains, and the cut document is
        re-validated by the same gate: only text that passes is returned.

        The document-wide "sweep every other copy of the figure" second stage
        is gone. It existed because the old validators only ever looked at
        clauses carrying a price word, so the same rejected number could stand
        untouched in a bullet or under a non-OHLC table header while the
        footnote claimed it had been removed. Every measurement-shaped number
        is now located and checked individually, so an occurrence that was not
        flagged is one this gate grounded, and cutting it would remove a
        figure the evidence supports.

        Fail-closed by construction. None — leave the canned fallback in place
        — whenever the run never observed a price at all, an issue is not a
        cut-out-able figure (an identity finding is one), a flagged figure
        cannot be located, or the document still fails after the cut.

        Args:
            content: The rejected draft.
            validation: Its validation result.

        Returns:
            The redacted, re-validated answer with its declaration block
            stripped and a note stating how many figures were removed, or None.
        """
        if not self._price_records():
            return None
        text = _strip_release_markers(content)
        # Stripping moves every offset after it, and the issue spans are the
        # only anchor the cuts have, so the verdict is retaken on the text the
        # cuts will actually be made in.
        check = validation if text == content else self._validate(text, record=False)
        removed = 0
        for _ in range(_MAX_REDACTION_PASSES):
            if check.valid:
                break
            codes = {issue.get("code") for issue in check.issues}
            if not codes <= (_REDACTABLE_CODES | _REPAIRABLE_PROVENANCE_CODES):
                return None
            cut, count = self._cut_flagged(text, check.issues)
            if cut is None:
                return None
            if count:
                text = cut
                removed += count
                check = self._validate(text, record=False)
                if check.valid:
                    break
            repaired = self.repair_provenance(text, check)
            if repaired is None:
                if count == 0:
                    return None
                continue
            text = repaired
            check = self._validate(text, record=False)
        if not check.valid or removed == 0:
            return None
        body = check.released_text
        note = self._release_note(removed, body)
        # The note carries the observed range and the canonical symbols, so it
        # is answer text too, and it goes through the same gate — in undeclared
        # mode, since it has no block of its own. Shipping it unchecked would
        # contradict the fail-closed promise above.
        if not self._validate(note, record=False).valid:
            return None
        released = body.rstrip() + "\n\n" + note
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
        """Locate the flagged figure in ``text``.

        The validators record the character span of the figure itself, which
        is the only reliable anchor: a stored claim string is normalised and
        a bare numeric cell ("1.10") matches inside a longer number ("21.10").

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

    def _cut_flagged(
        self,
        text: str,
        issues: Sequence[dict[str, Any]],
    ) -> tuple[str | None, int]:
        """Replace every flagged figure with the omission marker.

        An issue naming one figure cuts that figure. An issue with no value —
        ``unsourced_symbol_figures``, which says "every figure on this line
        belongs to an instrument no tool handled" — cuts every
        measurement-shaped figure inside its span.

        Args:
            text: The document to rewrite.
            issues: The issues raised against it.

        Returns:
            ``(rewritten text, figures replaced)``, or ``(None, 0)`` when a
            flagged issue could not be located and the release must fail closed.
        """
        block = parse_figures_block(text)
        figures = scan_figures(text, block)
        cuts: set[tuple[int, int]] = set()
        for issue in issues:
            if issue.get("code") not in _REDACTABLE_CODES:
                continue
            span = self._issue_span(text, issue)
            if span is None:
                return None, 0
            if issue.get("value") is None:
                cuts.update(
                    (figure.start, figure.end)
                    for figure in figures
                    if figure.shape == "measured"
                    and span[0] <= figure.start
                    and figure.end <= span[1]
                )
            else:
                cuts.add(span)
        if not cuts:
            return text, 0
        pieces: list[str] = []
        cursor = 0
        count = 0
        for start, end in sorted(cuts):
            if start < cursor:
                continue
            start = max(cursor, currency_prefix_start(text, start))
            end = currency_suffix_end(text, end)
            marker = self._marker_for(text, start)
            if marker is _REDACTION_MARKER_ZH:
                # "建议买入价 0.95 元" → "建议买入价（略※）": a full-width
                # bracket sits flush against the preceding word. Not against a
                # table pipe, though — "| 第一档 |（略※） |" loses the column's
                # padding and reads as a broken row.
                while (
                    start > cursor
                    and text[start - 1] == " "
                    and text[:start - 1].rstrip(" ")[-1:] not in {"|", ""}
                ):
                    start -= 1
            pieces.append(text[cursor:start])
            pieces.append(marker)
            cursor = end
            count += 1
        pieces.append(text[cursor:])
        return "".join(pieces), count

    def _marker_for(self, text: str, position: int) -> str:
        """Pick the omission marker in the script of the line being cut.

        A Chinese user asking about a US name gets English tables, and running
        one marker over the whole document released
        "Suggested entry price（略※） per share." A stretch with neither script
        — a numeric table row — falls back to the user's language, which is
        what the footnote is written in.
        """
        line = ""
        for candidate, start in _lines_with_offsets(text):
            if start <= position <= start + len(candidate):
                line = candidate
                break
        if any("\u3400" <= char <= "\u9fff" for char in line):
            return _REDACTION_MARKER_ZH
        if any(char.isascii() and char.isalpha() for char in line):
            return _REDACTION_MARKER_EN
        return (
            _REDACTION_MARKER_ZH
            if self._user_writes_chinese()
            else _REDACTION_MARKER_EN
        )

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
