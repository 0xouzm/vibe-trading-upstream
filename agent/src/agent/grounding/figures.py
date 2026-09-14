"""The figures block, the shape of a number, and how the two are matched.

This is the gate's whole inference surface: it finds the model's ``figures``
declarations, classifies every prose number by SHAPE (date, symbol, list
marker, measurement, bare integer) and matches prose numbers to declarations.
It reads no natural-language word; roles are declared by the model and shape
is language-independent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.agent.grounding.identity import _CANONICAL_SYMBOL_RE

#: The five roles a declaration may carry (spec §2).
ROLES = ("observed", "derived", "proposed", "cited", "count")

#: The info string that marks the declaration block.
BLOCK_LANGUAGE = "figures"

# SHAPE 1 — a number. Grouped thousands are one token and the lookbehind keeps
# an identifier's digits ("SMA20") out. The lookahead fences only digits, so
# "3.6pp" reads as 3.6 rather than backtracking to a bare "3".
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)"
)

# SHAPE 2 — a calendar date or a bare year: structure, never a measurement
# (spec §3). A year-less "08-10" is two bare integers and needs no mask.
_DATE_RE = re.compile(
    r"(?:19|20)\d{2}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*[日号]?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*[日号]"
    r"|(?:19|20)\d{2}\s*年"
    r"|(?:19|20)\d{2}"
)

# SHAPE 3 — a line-leading list marker or numbered heading. The punctuation is
# required, so a line opening with a figure ("1.171 元是收盘价") is untouched.
_ORDINAL_RE = re.compile(r"(?m)^[^\S\n]*(?:#{1,6}[^\S\n]*)?\d{1,3}[.)、．][^\S\n]+")

# SHAPE 4 — a fenced block: code or the figures block, never prose.
_FENCE_RE = re.compile(r"(?m)^[^\S\n]*(?:`{3,}|~{3,})[^\n]*$")

# Currency is a symbol set, not a vocabulary: a money character or ISO code
# touching a bare integer ("$100", "820 CNY") makes it measurement-shaped.
_CURRENCY_CHARS = frozenset("$¥￥€£₩₹元币圆镑")

_CURRENCY_CODES = frozenset(
    {
        "USD", "CNY", "CNH", "RMB", "HKD", "JPY", "EUR", "GBP", "KRW", "INR",
        "CAD", "AUD", "SGD", "TWD", "THB", "IRR", "IRT", "USDT", "USDC",
    }
)

_PERCENT_CHARS = "%％"

# A CJK currency word is at most three characters (人民币); bounding the run
# keeps 元宵/元件, inside longer CJK runs, from reading as money.
_MAX_CURRENCY_WORD = 3


@dataclass(frozen=True)
class Declaration:
    """One parsed line of the model's figures block."""

    index: int
    value_text: str
    value: float
    percent: bool
    role: str
    note: str
    ref: str


@dataclass(frozen=True)
class FiguresBlock:
    """The declaration block, or the absence of one."""

    present: bool
    span: tuple[int, int] | None
    raw: str
    declarations: tuple[Declaration, ...]
    malformed: tuple[tuple[int, str], ...]

    def match(self, value: float, percent: bool) -> Declaration | None:
        """Return the declaration covering ``value``, or None.

        Matching is numeric (tolerance 1e-9) and percent-ness must agree:
        ``37%`` and ``0.37`` are different assertions.

        Args:
            value: The prose figure's numeric value.
            percent: Whether the prose figure carries a percent sign.

        Returns:
            The first matching declaration, or None.
        """
        for declaration in self.declarations:
            if declaration.percent != percent:
                continue
            if abs(declaration.value - value) <= max(abs(value) * 1e-9, 1e-9):
                return declaration
        return None


@dataclass(frozen=True)
class Figure:
    """One number located in the prose, with the shape it was written in."""

    text: str
    value: float
    percent: bool
    start: int
    end: int
    line: int
    shape: str
    column: str | None = None
    date: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class TableRow:
    """One Markdown table row and the column roles of its header."""

    line: int
    cells: tuple[tuple[str, int, int], ...]
    columns: dict[int, str]
    date_column: int | None
    symbol_column: int | None

# Header spellings binding a table column to an OHLC field: the table's own
# schema, like a tool field name, not prose inference (spec §4).
_TABLE_FIELD_ALIASES = {
    "open": "open",
    "opening": "open",
    "opening price": "open",
    "开盘": "open",
    "开盘价": "open",
    "high": "high",
    "highest": "high",
    "最高": "high",
    "最高价": "high",
    "low": "low",
    "lowest": "low",
    "最低": "low",
    "最低价": "low",
    "close": "close",
    "closing": "close",
    "closing price": "close",
    "收盘": "close",
    "收盘价": "close",
}

_DATE_HEADERS = {"date", "datetime", "trade date", "timestamp", "日期", "交易日", "时间"}

_SYMBOL_HEADERS = {"symbol", "ticker", "code", "标的", "代码", "证券代码"}


def _lines_with_offsets(content: str) -> list[tuple[str, int]]:
    """Return each line of ``content`` with its character offset in it."""
    positions: list[tuple[str, int]] = []
    cursor = 0
    for line in content.splitlines():
        start = content.find(line, cursor) if line else cursor
        if start < 0:
            start = cursor
        positions.append((line, start))
        cursor = start + len(line)
    return positions


def _fenced_blocks(content: str) -> list[tuple[int, int, str, tuple[int, int]]]:
    """Return ``(start, end, info, body)`` for every fenced block.

    An unterminated fence runs to the end of the document, which is how a
    truncated answer ends and must not silently un-fence the rest of it.
    """
    fences = list(_FENCE_RE.finditer(content))
    blocks: list[tuple[int, int, str, tuple[int, int]]] = []
    index = 0
    while index < len(fences):
        opener = fences[index]
        info = opener.group(0).strip().lstrip("`~").strip().casefold()
        if index + 1 < len(fences):
            closer = fences[index + 1]
            blocks.append((opener.start(), closer.end(), info, (opener.end(), closer.start())))
            index += 2
        else:
            blocks.append((opener.start(), len(content), info, (opener.end(), len(content))))
            index += 1
    return blocks


def _parse_value(text: str) -> tuple[float, bool] | None:
    """Parse a declared value, tolerating currency marks and separators."""
    raw = text.strip().replace(",", "").replace(" ", "").replace(" ", "")
    percent = raw.endswith(tuple(_PERCENT_CHARS))
    if percent:
        raw = raw[:-1]
    raw = raw.strip("".join(_CURRENCY_CHARS))
    if not raw:
        return None
    try:
        return float(raw), percent
    except ValueError:
        return None


def parse_figures_block(content: str) -> FiguresBlock:
    """Parse the model's ``figures`` block out of a draft.

    Parsing is lenient (full-width pipe, run-on spacing, missing ``ref``), but a
    line that cannot be read as ``value | role | note | ref`` is reported as
    malformed: a skipped declaration is a figure the gate never checked.

    Args:
        content: The candidate answer.

    Returns:
        The parsed block, or an absent one when the draft has no block.
    """
    target: tuple[int, int, str, tuple[int, int]] | None = None
    for block in _fenced_blocks(content):
        if block[2] == BLOCK_LANGUAGE:
            target = block
    if target is None:
        return FiguresBlock(False, None, "", (), ())
    start, end, _, (body_start, body_end) = target
    body = content[body_start:body_end]
    declarations: list[Declaration] = []
    malformed: list[tuple[int, str]] = []
    for offset, line in enumerate(body.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.replace("｜", "|").split("|")]
        parts = [part for part in parts if part != ""] or [""]
        parsed = _parse_value(parts[0]) if parts else None
        role = parts[1].casefold() if len(parts) > 1 else ""
        if parsed is None or role not in ROLES:
            malformed.append((offset + 1, stripped[:120]))
            continue
        value, percent = parsed
        declarations.append(
            Declaration(
                index=offset + 1,
                value_text=parts[0],
                value=value,
                percent=percent,
                role=role,
                note=parts[2] if len(parts) > 2 else "",
                ref=parts[3] if len(parts) > 3 else "",
            )
        )
    return FiguresBlock(True, (start, end), body, tuple(declarations), tuple(malformed))


def strip_figures_block(content: str, block: FiguresBlock) -> str:
    """Return the text to release: the draft without its declaration block."""
    if block.span is None:
        return content
    start, end = block.span
    return (content[:start].rstrip() + "\n" + content[end:].lstrip()).strip()


def _table_cells(line: str, offset: int) -> list[tuple[str, int, int]]:
    """Split one Markdown row into ``(text, start, end)`` cells."""
    if line.count("|") < 2:
        return []
    cells: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in line.split("|"):
        start = cursor
        cursor += len(piece) + 1
        text = piece.strip()
        if not text:
            cells.append(("", offset + start, offset + start + len(piece)))
            continue
        lead = len(piece) - len(piece.lstrip())
        cells.append((text, offset + start + lead, offset + start + lead + len(text)))
    if cells and cells[0][0] == "":
        cells = cells[1:]
    if cells and cells[-1][0] == "":
        cells = cells[:-1]
    return cells


def _is_separator_row(cells: Sequence[tuple[str, int, int]]) -> bool:
    """Whether every cell is a Markdown alignment run (``---``, ``:--:``)."""
    if not cells:
        return False
    return all(
        text and set(text.replace(" ", "")) <= {"-", ":"} and "-" in text
        for text, _, _ in cells
    )


def table_rows(content: str) -> list[TableRow]:
    """Return every Markdown table row with its header's column roles.

    A table is a maximal run of lines with at least two pipes; its first line is
    the header, whose cells bind columns to OHLC fields, trade date and symbol.
    """
    positions = _lines_with_offsets(content)
    rows: list[TableRow] = []
    index = 0
    while index < len(positions):
        if positions[index][0].count("|") < 2:
            index += 1
            continue
        block_start = index
        while index < len(positions) and positions[index][0].count("|") >= 2:
            index += 1
        header = _table_cells(positions[block_start][0], positions[block_start][1])
        headers = [text.casefold() for text, _, _ in header]
        columns = {
            position: _TABLE_FIELD_ALIASES[name]
            for position, name in enumerate(headers)
            if name in _TABLE_FIELD_ALIASES
        }
        date_column = next(
            (position for position, name in enumerate(headers) if name in _DATE_HEADERS),
            None,
        )
        symbol_column = next(
            (position for position, name in enumerate(headers) if name in _SYMBOL_HEADERS),
            None,
        )
        for line_index in range(block_start + 1, index):
            cells = _table_cells(positions[line_index][0], positions[line_index][1])
            if not cells or _is_separator_row(cells):
                continue
            rows.append(
                TableRow(line_index, tuple(cells), columns, date_column, symbol_column)
            )
    return rows


def _currency_before(text: str, start: int) -> bool:
    """Whether a currency symbol touches the number on its left."""
    head = text[:start].rstrip()
    return bool(head) and head[-1] in _CURRENCY_CHARS


def _currency_after(text: str, end: int) -> bool:
    """Whether a currency symbol or ISO code touches the number on its right.

    A CJK run counts only when short enough to be a currency word: "美元" does,
    "元宵节后关注" does not.
    """
    tail = text[end:].lstrip()
    if not tail:
        return False
    if tail[0] in _CURRENCY_CHARS and not _is_cjk(tail[0]):
        return True
    if _is_cjk(tail[0]):
        run = ""
        for char in tail:
            if not _is_cjk(char):
                break
            run += char
        return len(run) <= _MAX_CURRENCY_WORD and any(
            char in _CURRENCY_CHARS for char in run
        )
    code = ""
    for char in tail:
        if not char.isascii() or not char.isalpha():
            break
        code += char
    return code.upper() in _CURRENCY_CODES


def currency_prefix_start(text: str, start: int) -> int:
    """Where a currency symbol attached to the left of a figure ("$1.10") begins."""
    head = text[:start]
    stripped = head.rstrip()
    if stripped and stripped[-1] in _CURRENCY_CHARS and not _is_cjk(stripped[-1]):
        return len(stripped) - 1
    return start


def currency_suffix_end(text: str, end: int) -> int:
    """Where a currency unit attached to the right of a figure ("0.95 元") ends.

    A compound unit (元/股, USD/share) is left whole, or its denominator would be
    left with nothing above it.
    """
    tail = text[end:]
    lead = len(tail) - len(tail.lstrip(" \t"))
    body = tail[lead:]
    if not body:
        return end
    run = ""
    if _is_cjk(body[0]):
        for char in body:
            if not _is_cjk(char):
                break
            run += char
        if len(run) > _MAX_CURRENCY_WORD or not any(
            char in _CURRENCY_CHARS for char in run
        ):
            return end
    else:
        for char in body:
            if not char.isascii() or not char.isalpha():
                break
            run += char
        if run.upper() not in _CURRENCY_CODES:
            return end
    following = body[len(run) : len(run) + 1]
    if following in {"/", "／"}:
        return end
    return end + lead + len(run)


def _is_cjk(char: str) -> bool:
    """Whether a character is in the CJK ideograph range."""
    return "㐀" <= char <= "鿿"


# Statement-ending punctuation, the gate's only segmentation: it decides which
# instrument a figure is about (spec §4), never what it means. "." counts only
# before whitespace, so decimals and ticker suffixes stay in one segment.
_SEGMENT_BREAKS = frozenset("，,；;。、\n！!？?")


def segment_bounds(content: str, start: int, end: int) -> tuple[int, int]:
    """The punctuation-delimited stretch of text a figure sits in.

    Args:
        content: The whole answer.
        start: Where the figure starts.
        end: Where it ends.

    Returns:
        ``(segment start, segment end)`` in document offsets.
    """
    left = start
    while left > 0:
        char = content[left - 1]
        if char in _SEGMENT_BREAKS:
            break
        if char == "." and left < len(content) and content[left].isspace():
            break
        left -= 1
    right = end
    while right < len(content):
        char = content[right]
        if char in _SEGMENT_BREAKS:
            break
        if char == "." and right + 1 < len(content) and content[right + 1].isspace():
            break
        right += 1
    return left, right


def _within(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    """Whether ``span`` sits inside any of ``spans``."""
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def scan_figures(content: str, block: FiguresBlock) -> list[Figure]:
    """Locate and classify every number in the prose of a draft (spec §3).

    * ``exempt`` — a date, year, security-code digits, line-leading ordinal, a
      table's date/symbol column, or anything fenced: structure.
    * ``measured`` — a decimal point, percent sign, touching currency mark or
      table cell: the shape a fabricated price or metric takes; must be declared.
    * ``bare`` — a plain integer (counts, horizons, window lengths): unchecked.

    Args:
        content: The candidate answer.
        block: The parsed figures block (its span is exempt).

    Returns:
        Every number in document order, each with its span and shape.
    """
    positions = _lines_with_offsets(content)
    line_of: list[tuple[int, int, int]] = [
        (start, start + len(line), index) for index, (line, start) in enumerate(positions)
    ]
    # Every fenced block is exempt, the figures block included.
    exempt: list[tuple[int, int]] = [
        (start, end) for start, end, _, _ in _fenced_blocks(content)
    ]
    for pattern in (_DATE_RE, _CANONICAL_SYMBOL_RE, _ORDINAL_RE):
        exempt.extend((match.start(), match.end()) for match in pattern.finditer(content))

    cell_at: list[tuple[int, int, TableRow, int]] = []
    for row in table_rows(content):
        for position, (_, start, end) in enumerate(row.cells):
            if position == row.date_column or position == row.symbol_column:
                exempt.append((start, end))
                continue
            cell_at.append((start, end, row, position))

    figures: list[Figure] = []
    for match in _NUMBER_RE.finditer(content):
        start, end = match.start(), match.end()
        percent = False
        tail = content[end:]
        stripped = len(tail) - len(tail.lstrip(" \t"))
        if tail[stripped : stripped + 1] and tail[stripped] in _PERCENT_CHARS:
            percent = True
            end = end + stripped + 1
        try:
            value = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        line = next(
            (index for low, high, index in line_of if low <= start <= high), 0
        )
        cell = next(
            (entry for entry in cell_at if entry[0] <= start and end <= entry[1]), None
        )
        if _within((start, match.end()), exempt):
            shape = "exempt"
        elif percent or "." in match.group(0) or cell is not None:
            shape = "measured"
        elif _currency_before(content, start) or _currency_after(content, match.end()):
            shape = "measured"
        else:
            shape = "bare"
        column = date_value = symbol_value = None
        if cell is not None:
            row, position = cell[2], cell[3]
            column = row.columns.get(position)
            if row.date_column is not None and row.date_column < len(row.cells):
                date_value = row.cells[row.date_column][0] or None
            if row.symbol_column is not None and row.symbol_column < len(row.cells):
                symbol_value = row.cells[row.symbol_column][0] or None
        figures.append(
            Figure(
                text=content[start:end],
                value=value,
                percent=percent,
                start=start,
                end=end,
                line=line,
                shape=shape,
                column=column,
                date=date_value,
                symbol=symbol_value,
            )
        )
    return figures
