"""Accounting-parentheses negatives in report_audit extraction.

Financial reports write negatives as ``(25.3)`` / ``（25.3）``. The extractor
used to read the bare digits out of the parentheses, so a loss came out
positive and the closing paren swallowed the unit. Issue #1660.
"""

from src.tools.report_audit_tool import extract_data_points


def _by_label(points, label_part):
    return [p for p in points if label_part in p["label"]]


def test_table_parenthesized_negative_keeps_sign_and_unit():
    md = (
        "| 项目 | 2024 |\n"
        "|---|---|\n"
        "| 净利润 | (25.3)亿 |\n"
    )
    pts = extract_data_points(md)
    hits = _by_label(pts, "净利润")
    assert len(hits) == 1
    assert hits[0]["reported_value"] == -25.3
    assert hits[0]["unit"] == "亿"


def test_table_fullwidth_parens_negative():
    md = (
        "| 项目 | 2024 |\n"
        "|---|---|\n"
        "| 归母净利润 | （3.20）亿元 |\n"
    )
    pts = extract_data_points(md)
    hits = _by_label(pts, "归母净利润")
    assert len(hits) == 1
    assert hits[0]["reported_value"] == -3.2
    assert hits[0]["unit"] == "亿元"


def test_table_positive_control_unchanged():
    md = (
        "| 项目 | 2024 |\n"
        "|---|---|\n"
        "| 营收 | 100.5亿 |\n"
    )
    pts = extract_data_points(md)
    hits = _by_label(pts, "营收")
    assert len(hits) == 1
    assert hits[0]["reported_value"] == 100.5
    assert hits[0]["unit"] == "亿"


def test_table_placeholder_cell_still_skipped():
    md = (
        "| 项目 | 2024 |\n"
        "|---|---|\n"
        "| 净利润 | (-) |\n"
    )
    assert extract_data_points(md) == []


def test_kv_line_parenthesized_negative_with_unit():
    pts = extract_data_points("净利润：(25.3)亿元，同比转亏")
    hits = _by_label(pts, "净利润")
    assert len(hits) == 1
    assert hits[0]["reported_value"] == -25.3
    assert hits[0]["unit"] == "亿元"


def test_kv_line_bare_paren_year_stays_dropped():
    # "(2024)" in prose is a year, not a negative; main drops it and the
    # fix must not start extracting it.
    assert extract_data_points("规划：(2024) 年投产") == []


def test_kv_line_positive_control_unchanged():
    pts = extract_data_points("营收：100.5亿元")
    hits = _by_label(pts, "营收")
    assert len(hits) == 1
    assert hits[0]["reported_value"] == 100.5
    assert hits[0]["unit"] == "亿元"
