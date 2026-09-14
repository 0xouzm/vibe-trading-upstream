"""Shape primitives for locating numbers, dates and tables in answer text.

This module is deliberately language-agnostic: it answers "what shape is this
run of characters" and nothing about what the surrounding prose calls it.
"""

from __future__ import annotations

import re

# "从 2026-08-03 的 100.0 涨到 2026-09-02 的 112.4" / "rose from 100.0 to
# 112.4": a return figure framed as growth between two endpoints is
# arithmetic on sourced inputs, not an invented backtest metric (#1338
# review). The frame alone never grounds anything — the values must also
# match an observed endpoint pair exactly (see _return_derived_from_observed).
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?![A-Za-z0-9_])"
)


# A line-leading ordered-list marker ("1. **标题**") is prose structure, not a
# number. Without masking it, "1." is parsed as a float and rejected downstream
# as a numeric_claim_conflict against an observed OHLC range (#BUGS-1). The
# pattern only matches a digit run at the start of a line followed by "." or ")"
# and whitespace, so an in-text decimal like "1.5" (digit after the dot) is
# never affected.
_MD_LIST_ITEM_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


# re.ASCII keeps ``\b`` a *byte* word boundary. Without it, ``\w`` is
# Unicode-aware and CJK letters count as word characters, so a date that runs
# straight into Chinese text -- "(2026-07-14最低)" -- has no boundary after
# "14" and is left unmasked, contributing 2026/7/14 as candidate prices that
# reject a correct report (#1122).
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", re.ASCII)


# A year-less "8/5" is how a trading day is written in running prose, and it
# contributed 8 and 5 as candidate prices (#983). The month and day ranges are
# bounded, and both sides are fenced off from a longer slash run, so the window
# enumeration "20/50/200-day" cannot be mistaken for a date. Reports also write
# the same day as "08-10(一)" or "08-10盘中"; that dash form is masked by
# ``_DASH_DATE_RE`` below, where a zero-padded month or a weekday/session
# marker is required so a quoted price range like "8-10 元" stays checkable.
_SHORT_DATE_RE = re.compile(
    r"(?<![\d/])(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?![\d/])"
)


# A report writes a trading day as "08-10(一)" or "08-10盘中", and the dash form
# leaked 8 and 10 as candidate prices exactly as "8/5" once did. The dash is
# NOT symmetric with the slash, though: it also separates a range, and "目标价
# 10-20 元" must stay checkable. So the two halves are split -- a zero-padded
# month (01-09) is a formatting intent no price range imitates, while 10/11/12
# have to carry a weekday or session marker to read as a date.
_DASH_DATE_RE = re.compile(
    r"(?<![\d/-])(?:"
    r"0[1-9]-(?:0[1-9]|[12]\d|3[01])"
    r"|1[0-2]-(?:0[1-9]|[12]\d|3[01])"
    r"(?=\s*(?:[(（]\s*(?:周|星期)?[一二三四五六日天]\s*[)）]"
    r"|盘中|盘后|盘前|收盘|开盘|最低|最高"
    r"|\s*(?:close|open|intraday|low|high)\b))"
    r")(?![\d/-])",
    re.IGNORECASE,
)


# Localized calendar text carries digits that the ISO pattern above leaves
# behind: "8 月 3 日" otherwise contributes 8 and 3 as candidate prices.
_LOCALIZED_DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2}\s*年\s*)?\d{1,2}\s*月(?:\s*\d{1,2}\s*[日号])?|(?:19|20)\d{2}\s*年"
)


# A numbered markdown heading ("### 6. 关键价位") names a section index, not
# a price. The ordered-list mask only covers line-leading "1." and stops at
# the "### " prefix, so the section number was extracted as a claim.
_NUMBERED_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*\d+(?:[.、．])?\s*")


# The clause splitter does not split on the ASCII period, so "close was 210.
# In 2024 the market rallied" stays one clause and 2024 would be misread as
# the asserted price. A period/bang/question followed by whitespace and a
# sentence-start (capital, quote, or bracket) is a boundary here — but only
# in this price-claim scan, never in the global clause splitter, where it
# would tear apart "000543.SZ" and "1.5". The capital/open-bracket lookahead
# deliberately excludes abbreviations: "close approx. 2500" keeps 2500 gated.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+(?=[A-Z(（])")


# Full-width enumeration commas delimit prose clauses. Paired brackets (ASCII
# or full-width ()()[]) are deliberately not separators: an explicit
# derivation such as "(8.5 - 7.9) / 2" must stay in one segment for the
# formula check, and 公司名（代码）价格 must stay in one segment so the
# unsourced-symbol gate can see the symbol with its figure (#1260).
# The English sentence period is a clause separator too. Without it two
# English sentences sharing a line were ONE clause, so the header's "last
# close" labelled the whole line and a derivation in the first sentence
# exempted a fabricated price in the second: "Based on close 1.171 * 0.80 =
# 0.937. The last traded price is 0.937." validated, while the byte-identical
# text with a paragraph break between the sentences was rejected — and the
# Chinese translation was rejected either way, because 。 always split. A
# per-language split is a per-language verdict, which is exactly what
# ``test_grounding_language_parity`` exists to stop.
# The lookahead is what keeps a decimal point, a ticker suffix (562500.SS) and
# an abbreviation ("approx. 1.10") out: only a period followed by whitespace
# and the start of a new sentence separates.
_CLAUSE_SEPARATOR_RE = re.compile(
    r"[,，;；。、\n]|[.!?](?=\s+[\"'(\[“‘]?[A-Z])"
)


# The ASCII comma both separates clauses and groups thousands, and the clause
# split ran first: "收盘价 ¥1,309.22" became a clause ending in "¥1", whose 1 was
# compared against the observed 1300.01–1363.35 range and rejected as a
# conflict. That is every price above 999 written the ordinary way, and it is
# self-contradictory — ``_NUMBER_RE`` and the float conversion below it both
# already understand grouped numbers. Only a real group is removed: a comma
# needs a digit before it and exactly three digits after.
_THOUSANDS_SEPARATOR_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _clause_spans(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Split prose into clauses, keeping each clause's span in the source text.

    A grouping comma is not a separator and is stripped from the returned
    segment text, but the span is into the ORIGINAL, un-stripped text. The
    redaction path needs that: it used to locate a flagged clause with
    ``text.find(claim)`` on the normalised string, which never matched once a
    price carried a thousands separator ("1,450.50"), and which matched inside
    an unrelated longer number when the claim was a bare table cell ("1.10"
    found inside "21.10").

    Args:
        text: One line of candidate answer text.
        offset: Where ``text`` starts inside the document being validated.

    Returns:
        ``(segment, start, end)`` per clause, with ``start``/``end`` measured
        in the document (``offset`` already added).
    """
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for match in _CLAUSE_SEPARATOR_RE.finditer(text):
        if match.group(0) == "," and _THOUSANDS_SEPARATOR_RE.match(text, match.start()):
            continue
        bounds.append((cursor, match.start()))
        cursor = match.end()
    bounds.append((cursor, len(text)))
    return [
        (_THOUSANDS_SEPARATOR_RE.sub("", text[start:end]), start + offset, end + offset)
        for start, end in bounds
    ]


def _lines_with_offsets(content: str) -> list[tuple[str, int]]:
    """Return each line of ``content`` with its character offset in it.

    ``str.splitlines`` discards the terminators, so an issue raised on a line
    cannot say where in the document that line sits. Offsets are recovered by
    scanning forward, which is exact because only line terminators separate
    one line's end from the next line's start.
    """
    positions: list[tuple[str, int]] = []
    cursor = 0
    for line in content.splitlines():
        start = content.find(line, cursor) if line else cursor
        if start < 0:
            start = cursor
        positions.append((line, start))
        cursor = start + len(line)
    return positions


_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
