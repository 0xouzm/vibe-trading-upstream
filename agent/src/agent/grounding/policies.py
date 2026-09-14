"""Per-claim validation: the checks that turn evidence into issues."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.agent.grounding.identity import (
    _CANONICAL_SYMBOL_RE,
    _normalize_symbol,
    _scan_symbols,
)
from src.agent.grounding.evidence import (
    EvidenceRecord,
    _is_number,
    _metric_kind_for_path,
    _metric_kind_for_text,
    _timestamp_matches_claim_date,
)
from src.agent.grounding.figures import (
    _DASH_DATE_RE,
    _DATE_RE,
    _LOCALIZED_DATE_RE,
    _MD_LIST_ITEM_RE,
    _NUMBERED_HEADING_RE,
    _NUMBER_RE,
    _SENTENCE_BOUNDARY_RE,
    _SHORT_DATE_RE,
    _TABLE_SEPARATOR_RE,
    _clause_spans,
    _lines_with_offsets,
)

# Tools whose run produces the equity/regime windows behind a categorical
# "all N-month windows were profitable" statement.
_ANALYSIS_WINDOW_TOOLS = frozenset({"backtest", "run_shadow_backtest"})


# Unit / currency tokens glued to a price, swallowed with the figure.
# The single-character 元 additionally guards against CJK: without it the
# swallow ate the 元 of 元宵/元旦/元件 and left an orphan "宵节后关注" in the
# released text. The multi-character units cannot start a longer word the
# same way, so they keep the Latin-only guard (it is there for USD/USDT).
# Both guards also refuse a following slash: 元/股 and USD/share are ONE
# compound unit, and swallowing only its first half rewrote
# "建议买入价 0.95 元/股" into "建议买入价（略※）/股" — a denominator with
# nothing left above it.
_CURRENCY_UNIT_WORDS = (
    r"人民币|美元|美金|港元|港币|日元|欧元|英镑|"
    r"CNY|RMB|USD|HKD|JPY|EUR|GBP|CAD|AUD|SGD|USDT"
)


# The same tokens without the swallow guards. Those guards decide how much
# text the marker may EAT; this one only asks whether a figure is written as a
# price, and there "1.180 元已回撤" is as much a quote as "1.180 元。" is.
_UNIT_TOKEN_AFTER_RE = re.compile(
    r"\s*(?:" + _CURRENCY_UNIT_WORDS + r"|元)", re.IGNORECASE
)


_CURRENCY_BEFORE_RE = re.compile(r"(?:US\$|HK\$|C\$|A\$|S\$|\$|¥|￥|€|£)\s*$")


_PRIVATE_ASSERTION_RE = re.compile(
    r"(?:\b(?:is|remains|still)\s+(?:an?\s+)?(?:private company|privately held)\b|"
    r"\bnot publicly traded\b|\bunlisted company\b|"
    r"(?:是|仍是|属于)(?:一家)?(?:私人|私营|非上市)公司|未上市|没有上市)",
    re.IGNORECASE,
)


# The bare verbs below are present tense only, which is not how an answer
# actually states an observed price: "closed at 412.35" and "last traded at
# 412.35" are the ordinary spellings and neither matches \bclose\b or
# \btrade\b. Chinese 收盘 / 现价 match, so the gate was strictly leakier in
# English than in Chinese — a fabricated USD price in the most natural
# phrasing walked straight through while its Chinese translation was caught.
# The past-tense forms are required to be followed by "at" so that reporting
# volume ("traded 1.2M shares") or a corporate event ("the deal closed at a
# 30% premium" — a percentage, already masked) is not read as a quote.
_PRICE_VERB_PAST_RE = r"\b(?:closed|opened|traded|quoted|priced|settled|fixed)\s+at\b"


_PRICE_CONTEXT_RE = re.compile(
    r"(?:\b(?:opening|open|high|low|closing|close|price|quote)\b|"
    + _PRICE_VERB_PAST_RE + r"|"
    r"\b(?:entry|buy|target|support|resistance)\s+(?:price|level)\b|"
    r"开盘价?|最高价?|最低价?|收盘价?|买入价|入场价|目标价|支撑位?|阻力位?|"
    # Chinese had the mirror-image gap: these four are as ordinary as 收盘价
    # and none of them matched, so a fabricated 成交价 / 股价 walked through
    # exactly the way "closed at" did in English.
    r"成交价|最新价|股价|收报|"
    r"现价|报价|价格|价位)",
    re.IGNORECASE,
)


# The vocabulary of a price LEVEL — a band, a moving average, a support or
# resistance line. It is the only shape for which an indicator reading is
# admissible evidence, and the rule is stated this way round on purpose.
#
# It used to be stated the other way: an indicator was admissible unless the
# clause named an OHLC field, against a hand-written list of field phrases.
# A hand-written denylist is only ever as complete as the day it was typed,
# and it was not complete for one day: with the session's sma_20 at 1.150 and
# the observed close at 1.171, "收盘价为 1.150 元" was caught while "收盘于"'s
# English twins "closed at 1.150" / "the close was 1.150" / "highest price was
# 1.150" were released, and every spot-quote spelling there is — 现价 / 最新价
# / 成交价 / 报价 / 股价 / current price / the quote is — leaked in both
# scripts, because a spot quote names the latest print and named no OHLC
# field. An allowlist fails the other way: a level word this list misses costs
# a correction round instead of releasing a fabricated print.
_PRICE_LEVEL_WORD_RE = re.compile(
    r"(?:\b(?:support|resistance|pivot|moving[- ]average|"
    r"(?:upper|lower|middle) band|band|channel|bollinger|keltner|donchian|"
    r"ichimoku|supertrend|vwap|atr)\b|"
    r"\b(?:SMA|EMA|WMA|DMA|MA|BOLL)\d{0,3}\b|"
    r"支撑|阻力|压力|均线|上轨|下轨|中轨|轨道|通道|中枢|枢轴|布林)",
    re.IGNORECASE,
)


_ANALYSIS_METRIC_RE = re.compile(
    r"(?:\breturn vol(?:atility)?\b|\bmax(?: |\.)?drawdown\b|\bmaxdd\b|"
    r"\bsharpe(?: ratio)?\b|\bwin rate\b|\bhit rate\b|"
    r"\bprob(?:ability)?\.?\s+of\b|\bvolatility\b|\bdrawdown\b|"
    r"\bannualiz\w*\b|\bwindow(?:s)?\b|\bregime(?:s)?\b|"
    r"\b(?:annual|cumulative|total)\s+return\b|"
    r"夏普|回撤|波动率|胜率|命中率|概率|年化|回测|窗口|收益(?:率)?|回报(?:率)?)",
    re.IGNORECASE,
)


# Phrases that indicate a figure is attributed to an external source rather
# than model memory (paper restatement, #1338 review). Attributing a number
# to a named source is the opposite of an unsourced claim. The subject list
# is deliberately restricted to sources that cannot be this run's own output
# (papers, studies, analysts, filings): "the backtest reports" / "the data
# shows" is exactly how a model dresses up its own numbers (#1336), so
# data/backtest/strategy/report subjects do NOT count.
_ATTRIBUTION_RE = re.compile(
    r"(?:"
    r"\b(?:the\s+(?:papers?|studies?|researchers?|analysts?|surveys?"
    r"|regulators?|authorities|literature|authors?)"
    r"|analysts?|researchers?|literature|sec\s+filings?\b|filings?"
    r"|annual\s+filings?|quarterly\s+filings?|rating\s+agencies?)"
    r"\s+(?:reports?|estimates?|shows?|indicates?|suggests?|states?|claims?"
    r"|notes?|cites?|mentions?|reveals?|discloses?|publishes?|found"
    r"|calculated|computed|derived)\b"
    r"|"
    r"(?:论文|文献|分析师|研究机构|学者)"
    r"[^。，\n]{0,6}?(?:报告|显示|表明|指出|称|估计|发现)"
    r")",
    re.IGNORECASE,
)


# Measurement-shaped numbers for analysis claims: keeps the % sign and sign
# (unlike _numbers_without_dates_or_percent, which drops percentages on
# purpose). A bare integer is NOT a measurement — "252 个交易日年化" is the
# standard convention, not a claim about this run's analysis (#1032-style
# definitional prose must stay untouched). Dates can still look like
# measurements, so date masks run first.
_MEASURE_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)\s*[%％]?"  # decimal: 1.21, -8.1%
    r"|[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)\s*[%％]"  # integer percent: 55%, 70%
)


# A forward-looking frame means the figure is a forecast, not a claim that a
# backtest/analysis measured it — the measurement gate only polices "measured
# facts" (#1336), forecasts stay under the "analysis, not advice" prompt rule.
_FORECAST_FRAME_RE = re.compile(
    r"(?:\b(?:forecast|expected|projected|predicted)\b|预计|预期|预测|估计|展望)",
    re.IGNORECASE,
)


# Definitional prose ("夏普比率大于 1.0 通常被认为较好") states a convention,
# not a measurement this session produced (#1336: preserve explicitly labelled
# definitions). The bare 通常 alone is deliberately not enough — "通常实现
# 18.2% 年化" is a measured claim.
_DEFINITION_FRAME_RE = re.compile(
    r"(?:通常(?:认为|被?认为|说来|指|用于)|一般认为|定义为|是指|惯例|"
    r"conventionally|typically\s+(?:considered|regarded|seen)|"
    r"generally\s+(?:accepted|considered|regarded)|by\s+convention)",
    re.IGNORECASE,
)


# A categorical "all N-month windows were profitable" claim carries only an
# integer window length, which _MEASURE_NUMBER_RE deliberately ignores; it is a
# measured fact about the run and needs a window-producing analysis result.
_CATEGORICAL_WINDOW_RE = re.compile(
    r"(?:所有|全部|每个|任何|each|every|all)[\s\S]{0,40}?"
    r"(?:窗口|区间|windows?|periods?)[\s\S]{0,30}?"
    r"(?:正收益|均为正|都为正|盈利|上涨|positive|profitable)",
    re.IGNORECASE,
)


_DERIVATION_RE = re.compile(
    r"(?:\bderived\b|\bcalculated\b|\bformula\b|\bbased on\b|计算|推导|公式|基于)",
    re.IGNORECASE,
)


# The separator between a formula and its result. "=" is how the gate first
# accepted a derivation; "≈" / "≒" / "约" / "约等于" are how a model writes one
# whose result it rounded, and each was rejected with the multiplier read as a
# quoted price. The evaluation tolerance already absorbs the rounding.
# The English spellings are here for the same reason the Chinese ones are:
# ``test_derived_return_exemption_is_structural_not_phrasal`` is the repo's
# standing rule that identical arithmetic gets an identical verdict in both
# languages, and "1.171 * 0.97 approximately 1.136" was still burning a
# revision round while its ≈ twin passed.
_RESULT_SEPARATOR_RE = re.compile(
    r"约等于|≈|≒|=|约|~|\b(?:approximately|approx\.?|about|around|roughly)\b",
    re.IGNORECASE,
)


# The subject of a performance metric decides which evidence may ground it.
# A strategy's max drawdown is a property of an equity curve a backtest
# produced; two observed prints of the instrument are not that curve, and
# neither is any arithmetic on them. "回测显示该策略最大回撤 5.9%（从 1.180
# 跌至 1.110）" rode out of the session on the endpoint-arithmetic exemption
# with no backtest anywhere in it. A PRICE-subject drawdown ("股价较 5 月高点
# 1.053 元已回撤约 37%") is the arithmetic itself and keeps the exemption.
_STRATEGY_SUBJECT_RE = re.compile(
    r"(?:\bstrateg(?:y|ies)\b|\bbacktest(?:ed|ing)?\b|\bportfolio\b|"
    r"\bequity curve\b|\bNAV\b|\bholdings?\b|"
    r"策略|回测|组合|持仓|净值)",
    re.IGNORECASE,
)


_INDICATOR_IDENTIFIER_RE = re.compile(r"(?<![0-9.])[A-Za-z_]+\d+(?![0-9.])")


# Unitless identity constants in a symbolic rate formula are not quoted prices.
# Without this mask, ``1 - 单边成本率`` in a position-sizing formula is read as
# a one-yuan price merely because the same clause also mentions a close price.
# Keep the relaxation narrow: only 0/1 directly participating in arithmetic
# with a token explicitly labelled as a rate is removed.
_RATE_FORMULA_IDENTITY_RE = re.compile(
    r"\b[01](?=\s*[-+]\s*(?:[A-Za-z_][A-Za-z0-9_]*_?rate\b|[^\d\s()+*/=-]{0,12}(?:成本率|费率|税率|滑点率)))",
    re.IGNORECASE,
)


# A level stated as a RANGE has the same shape: the separator touches the
# second number, so masking "目标价 10" left "-20" behind and a negative price
# matches no OHLC window at all -- a guaranteed rejection of a correct draft.
# The tail is optional, so a single-value level is unaffected.
_RANGE_TAIL = r"(?:\s*[-–—~～至到]\s*[-+]?\d[\d,]*(?:\.\d+)?)?"


# A percentage range masks only its upper bound through the "%" tail check
# below, because the sign touches the second number: "1–2%" left 1 behind
# (#983). Mask the span as a whole.
_PERCENT_RANGE_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*[-–—~至]\s*\d[\d,]*(?:\.\d+)?\s*[%％]"
)


# A percentage-POINT delta is not a quoted price. "~3.6pp below Penumbra"
# describes a margin gap; left unmasked the ".6" is consumed as a decimal
# and the number reaches the OHLC comparator, which then rejects an
# otherwise correct fundamentals answer and demands `get_market_data` to
# substantiate a claim that has nothing to do with price.
# The trailing "%" spellings are already handled above; this covers the
# percentage-point spellings, which the percent masks never matched.
# The Chinese units take an optional measure word ("3.6 个百分点" is the
# ordinary spelling; bare "3.6 百分点" is the rare one) and are matched
# without a trailing \b: after a CJK character \b requires a non-word
# character to follow, so "下降 3.6 个百分点后企稳" would not match. The
# ASCII units keep \b, which is what stops "3.6ppm" being read as pp.
_PERCENTAGE_POINT_RE = re.compile(
    r"[-+~≈]?\s*\d[\d,]*(?:\.\d+)?\s*"
    r"(?:(?:pp|ppt|ppts|bps|bp)\b|个?(?:百分点|基点))",
    re.IGNORECASE,
)


# An aggregate amount is not a quoted price. "100 股成本 820 CNY" states a
# position cost; comparing 820 against a per-share OHLC range is a category
# error. The tradeoff is that a per-share figure written only as "成本 8.20"
# goes unchecked — provenance still requires symbol, source, and currency.
_AGGREGATE_AMOUNT_RE = re.compile(
    r"(?:成本|总额|总价|总市值|市值|合计|金额|cost|total|notional|market value)"
    r"\s*(?:为|是|约)?\s*[:：]?\s*[-+]?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)


# Quantities, horizons, lot sizes, and lookback windows are unit-bearing:
# "100 股", "1–4 周", "3 个月", "52-week", "20/50/200-day". None are prices.
# The hyphenated English compound needs its own branch: the range alternation
# consumes "-4" in "1-4 周" but stalls on "-week", which left 52 behind to be
# compared against an OHLC range (#1001). The slash enumeration shares a single
# trailing unit, so "20/50/200-day" has to be masked as one span or its first
# two window lengths survive. ASCII units carry a trailing word boundary so
# "120 more" is not read as a quantity; the CJK branch cannot, because 周 and
# 内 are both word characters and "1–4 周内" must still mask.
_QUANTITY_WITH_UNIT_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:\s*/\s*\d[\d,]*(?:\.\d+)?)*"
    r"(?:\s*[-–—~至]\s*\d[\d,]*(?:\.\d+)?)?"
    r"\s*[-–—]?\s*"
    r"(?:"
    r"(?:股|手|张|份|口|笔|倍|个月|周|天|日|年|次|个交易日|项|行)"
    r"|(?:shares?|contracts?|lots?|units?|sessions?|bars?|periods?|"
    r"wks?|weeks?|months?|days?|years?|yrs?)\b"
    r")",
    re.IGNORECASE,
)


# A conviction reading is on a labelled scale, not a price scale: the 6 in
# "CONFIDENCE: 6" is bounded by the label that introduces it. Only the value
# bound to the label is masked, so a genuine quote elsewhere in the same
# clause is still checked. The optional denominator covers "6/10" (#1001).
_LABELLED_SCORE_RE = re.compile(
    r"(?:confidence|conviction|score|rating|probability|odds|weighting|"
    r"置信度|信心|评分|得分|概率|胜率)"
    r"\s*(?:is|of|=|为|是)?\s*[:：]?\s*"
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d[\d,]*(?:\.\d+)?)?",
    re.IGNORECASE,
)


# A named indicator reads on its own scale — "RSI of 46.7" is bounded at 100,
# not quoted in the instrument's currency. The name must be adjacent to the
# value, so a bare number elsewhere in the clause stays checked. Only
# unambiguous indicator names are listed: generic words such as "momentum" or
# "volatility" sit too close to price prose to mask safely.
_INDICATOR_VALUE_RE = re.compile(
    r"\b(?:rsi|macd|atr|adx|cci|obv|kdj|boll|dif|dea|vix|iv|"
    r"sharpe|sortino|beta)\b"
    r"(?:\s*\([^)]{0,20}\))?"
    # #1354: "RSI below 30" / "sharpe above 1" — a directional connective is
    # part of the indicator reading, so the reading's number stays masked.
    # Only indicator NAMES get these connectives; a price word with
    # "below/above" ("close above 2500") is an observed-value claim and
    # matches none of these names, so it stays gated.
    r"\s*(?:is|at|of|reads?|=|为|是|below|above|under|over)?\s*[:：]?\s*"
    r"[-+]?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)


# A currency token may sit between a level marker or comparison operator and
# the number: "收盘 <$2.86", "目标位 C$6.80", "trigger at $119.68". Without
# it, the number survives masking and is compared against observed OHLC as a
# price claim even though it is a prospective level, not an observed quote.
_CURRENCY_TOKEN = r"(?:\$|US\$|C\$|HK\$|CAD|USD|CNY|HKD|¥|￥)?"


# An order line is an instruction, not an observation. "100 @ $3.50" states
# where a limit sits and "100" is a share count that was never a price at all,
# yet both went to the OHLC check and rejected a weekly update whose quotes
# were correct. This is the same category as the target/stop levels below -- a
# level the report proposes, not one the data source reported.

# #1354: a signal value trailing an arrow or a signal word ("sinal +1",
# "-> +1", "触发 +1", "Sinal: +1") is the formula's output, not an observed
# price. Signal values are small integers (a +/-1 signal, a 1-10 score); a
# multi-digit price never follows these words, so the digit bound keeps
# "close -> 2500" (a price claim written in arrow notation) gated. The colon
# is optional ("Sinal: +1" — ASCII '.' is not a clause separator, so this can
# sit in the same clause as the price word), the sign may be spaced ("sinal
# - 1"), and the lookahead stops the bound from masking the first two digits
# of a longer number ("signal 2500" keeps the full 2500 gated).
_SIGNAL_VALUE_RE = re.compile(
    r"(?:\b(?:signal|sinal|trigger|triggers|triggered|dispara|信号|触发)\b[:：]?"
    r"|->|→|=>)"
    r"\s*[-+]?\s*\d{1,2}(?:\.\d+)?(?!\d)",
    re.IGNORECASE,
)


_ORDER_LEVEL_RE = re.compile(
    r"(?:"
    # (a) "<qty> [股|shares] @ <price>" -- the whole clause, quantity included
    r"\d[\d,]*(?:\.\d+)?\s*(?:股|shares?)?\s*@\s*"
    + _CURRENCY_TOKEN + r"\s*[-+]?\d[\d,]*(?:\.\d+)?"
    r"|"
    # (b) an order label, optionally carrying its own "<qty> @", then the level.
    # There is deliberately no bare "@ <price>" branch: dates are masked before
    # this runs, so "收盘 2026-08-10 @ 8.20" would arrive here as "@ 8.20" and a
    # genuinely observed close would stop being checked. 买入价 / 卖出价 are
    # absent for the same reason -- in running prose they name a price the
    # report says it observed, not an instruction it proposes.
    r"(?:挂单|限价单|限价|委托价?|订单"
    r"|limit\s+(?:order|price)|\bGTC\b|\bGTD\b|\bIOC\b|\bFOK\b)"
    r"\s*(?:为|是|at|=)?\s*[:：]?\s*"
    r"(?:\d[\d,]*(?:\.\d+)?\s*(?:股|shares?)?\s*@\s*)?"
    + _CURRENCY_TOKEN + r"\s*[-+]?\d[\d,]*(?:\.\d+)?"
    r")" + _RANGE_TAIL,
    re.IGNORECASE,
)


# A historical reference names a price the instrument once traded at — an
# all-time high, a 52-week extreme — and the answer is not claiming it as
# today's observed quote. "8/12 高 149.60 为 6/16 ATH 225.64 以来最高" was
# rejected because 225.64 (the June ATH) fell outside this session's observed
# OHLC range. The marker must be adjacent to the number, so a plain field
# reading such as "8/12 高 149.60" stays checked: its 高 carries no historical
# qualifier. "创历史新高" shares the 历史新高 marker and is masked with it; the
# relaxation follows the same trade-off as prospective levels — the historical
# extreme is a reference, not an assertion about the current bar.
_REFERENCE_LEVEL_RE = re.compile(
    r"(?:"
    r"\bATH\b|all[- ]?time\s+(?:high|low)|"
    r"52\s*[- ]?W(?:EEK)?\s*(?:high|low)|52\s*[- ]?W(?:EEK)?\s*高(?:点|位)?|"
    r"52\s*周(?:高|低)(?:点|位)?|"
    r"历史(?:最高|最低|新高|新低|高|低)(?:点|位|价)?|"
    r"上市以来(?:最高|最低)(?:点|位|价)?"
    r")"
    r"\s*(?:of|为|是|约|at)?\s*[:：]?\s*\(?"
    r"\s*" + _CURRENCY_TOKEN + r"\s*[-+]?\d[\d,]*(?:\.\d+)?" + _RANGE_TAIL,
    re.IGNORECASE,
)


# A date-anchored reference puts the historical extreme after the number:
# "7/10(150.57)以来最高" and "highest since 7/10 (150.57)". The parenthesized
# value is the earlier high/low, not a current quote, and was rejected against
# the session's observed window (which does not reach back to July).
_SINCE_REFERENCE_RE = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?\s*\)?\s*以来(?:最高|最低|高|低)(?:点|位)?"
    # Dates are masked before this runs, so "since 7/10 (150.57)" has lost its
    # date digits by the time the connector is matched.
    r"|(?:highest|lowest)\s+since[^0-9\n]{0,16}"
    r"[-+]?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)


# A validation report cites a plan file by line number — "~line 206",
# "第 206 行" — and the number is a document location, not a price. Before
# this mask, "**文档 ~line 206「…」不成立" contributed 206.0 as a candidate
# price claim and was rejected against the observed OHLC range.
_LINE_REFERENCE_RE = re.compile(
    r"(?:~\s*)?\blines?\b\s*[:#]?\s*\d{1,5}(?:\s*[-–—至~]\s*\d{1,5})?"
    r"|第\s*\d{1,5}(?:\s*[-–—至~]\s*\d{1,5})?\s*行"
    r"|行\s*[:：]?\s*\d{1,5}(?:\s*[-–—至~]\s*\d{1,5})?",
    re.IGNORECASE,
)


# A ratio ("6:1 折算", "10:1") names a conversion basis, not a quote price.
_RATIO_RE = re.compile(r"\d+(?:\.\d+)?\s*[:：]\s*\d+(?:\.\d+)?")


# A currency conversion cited inside a report ("USD/CAD≈1.36") is a basis, not
# an instrument quote. But `EUR/USD` IS this project's canonical forex symbol
# (``backtest/engines/forex.py``, and akshare_loader accepts the slash form), so
# masking every ``AAA/BBB <number>`` would stop checking real FX quotes on a
# first-class market — an invented rate would pass. The pair form therefore
# requires an approximation marker, which a conversion basis carries and a quote
# does not: "USD/CAD≈1.36" is masked, bare "USD/CAD 1.36" stays checked. The
# labelled 汇率 / "exchange rate" form is unambiguous and needs no marker.
_FX_RATE_RE = re.compile(
    r"[A-Z]{3}\s*/\s*[A-Z]{3}\s*(?:≈|~|约)\s*\d+(?:\.\d+)?"
    r"|(?:汇率|FX\s*rate|exchange\s+rate)\s*(?:≈|~|=|为|是)?\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)


# A trading plan quotes levels it does not claim to have observed. In the
# committee report attached to #983, "收盘 ≥6.45 且量 ≥35M手" is an entry
# trigger, "年线 4.63 成目标区" is a target zone, and neither asserts anything
# about what the instrument traded at. Compared against observed OHLC evidence
# they were reported first as numeric_claim_unavailable (before the run fetched
# prices) and then as numeric_claim_conflict (after it did) — the same false
# positive under two codes.
#
# This is a real relaxation, so every branch is span-local and anchored to the
# token that makes the number prospective, never to a word elsewhere in the
# clause: "现价 5.97，目标位 6.45" masks 6.45 and still checks 5.97. An
# assertion carries no such token and stays checked.
#
# Branch (d) accepts a conditional opener. A number inside "若收盘 5.36" is a
# hypothesis, and a hypothesis does not misrepresent observed data the way a
# bare quote does — but it is the widest branch here, so it requires the opener
# to PRECEDE the number with no digits in between, which keeps it from reaching
# back over an assertion that was already made.
_PROSPECTIVE_LEVEL_RE = re.compile(
    r"(?:"
    # (a) comparison operator immediately before the number
    r"(?:>=|<=|≥|≤|>|<|大于|小于|不低于|不高于|高于|低于)"
    r"\s*" + _CURRENCY_TOKEN + r"\s*[-+]?\d[\d,]*(?:\.\d+)?" + _RANGE_TAIL
    + r"|"
    # (b) a level marker introducing the number
    r"(?:目标位|目标区|目标价|均值目标(?:价)?|平均目标(?:价)?|止损位?|止盈位?|触发价|触发位|触发点|上看|下看|"
    r"支撑(?:阶梯|位|线)?|阻力(?:位|线)?|压力位|压力线|support(?:\s+(?:level|line|zone))?|"
    r"resistance(?:\s+(?:level|line|zone))?|target\s+(?:price|level|zone)|trigger|stop[-\s]?loss|take[-\s]?profit)"
    r"\s*(?:为|是|至|到|on|at|of|=)?\s*[:：]?\s*"
    + _CURRENCY_TOKEN
    + r"\s*[-+]?\d[\d,]*(?:\.\d+)?" + _RANGE_TAIL
    + r"|"
    # (c) the number followed by a level marker
    r"" + _CURRENCY_TOKEN + r"[-+]?\d[\d,]*(?:\.\d+)?" + _RANGE_TAIL
    + r"\s*(?:一线|附近)?\s*(?:成为?|作为|是)?\s*"
    r"(?:目标区|目标位|止损位|止盈位)"
    r"|"
    # (d) a conditional opener before the number, digits fencing the reach
    r"(?:若|如果|一旦|倘若|假如|\bif\b|\bwhen\b|\bshould\b)[^0-9\n]{0,12}"
    r"[-+]?\d[\d,]*(?:\.\d+)?"
    r")",
    re.IGNORECASE,
)


# #1354: the closed structural rule for prose price claims. A number that
# follows a price-context word is an observed value only when nothing
# formula-like binds it instead. Two closed token classes decide that:
#
#   * a formula marker sitting between the price word and the number — a
#     comparison/division operator, or an indicator identifier followed by
#     digits (SMA/EMA/…/VWAP + digits; MA for the Chinese MA20 convention) —
#     turns the number into an operand of the formula, not a claimed value;
#   * an observation binder after that marker ("was", "at", 报收/收于/收报/收在)
#     re-attaches the number to the price word, so "close above SMA50 and was
#     2500" stays a claim while "close/SMA50 > 1" claims nothing.
#
# This replaces the per-phrasing denylist (a signal-value mask, an indicator-
# connective mask, …) with one syntactic distinction; the catalogue of
# phrasings can never close, a closed marker set can.
_FORMULA_MARKER_RE = re.compile(
    r"(?:>=|<=|≥|≤|>|<|/)"
    r"|\b(?:SMA|EMA|WMA|DMA|MA|RSI|MACD|ATR|ADX|CCI|OBV|KDJ|BOLL|VWAP)\d+\b",
    re.IGNORECASE | re.ASCII,
)


_OBSERVATION_BINDER_RE = re.compile(
    r"\b(?:was|were|is|are|at)\b|(?:报收|收于|收报|收在)|==|=|≈",
    re.IGNORECASE,
)


def _matches_any(raw: Any, values: Sequence[float], *, rel: float = 0.005) -> bool:
    """Whether a measurement-shaped figure equals one of ``values``.

    Args:
        raw: A figure as ``_measure_numbers`` returns it ("1.136", "5.26%").
        values: Numeric values to match against.
        rel: Relative tolerance, absolute floor 1e-9.

    Returns:
        True when the parsed figure matches any value.
    """
    try:
        number = float(str(raw).strip().rstrip("%％").replace(",", ""))
    except ValueError:
        return False
    return any(abs(number - value) <= max(abs(value) * rel, 1e-9) for value in values)


def _figure_reads_as_a_price(text: str, figure: str) -> bool:
    """Whether ``figure`` is written as a price somewhere in ``text``.

    A price word ahead of it in the clause, a currency symbol in front of it,
    or a currency/unit token glued behind it. Any one of the three is the
    ordinary way an answer quotes a price, and none of them is how a metric
    figure is written.

    Args:
        text: The clause or cell the figure appears in.
        figure: The figure exactly as it was written.

    Returns:
        True when at least one occurrence reads as a price.
    """
    for match in re.finditer(re.escape(figure), text):
        if _UNIT_TOKEN_AFTER_RE.match(text, match.end()):
            return True
        if _CURRENCY_BEFORE_RE.search(text[: match.start()]):
            return True
        if _PRICE_CONTEXT_RE.search(text[: match.start()]):
            return True
    return False


def _labels_its_result_observed(text: str, formula_start: int) -> bool:
    """Whether a market-print word to the left CLAIMS the formula's result.

    The distinguisher is a result separator between the price word and the
    formula's first operand. "收盘价 = 1.171 × 0.80" reads "the close IS this
    expression" — the equation's subject is an observation, so its result is
    a fabricated print. "基于收盘价 1.171 × 0.97 = 1.136" reads "starting from
    the observed close" — the word labels an OPERAND, and the result is the
    entry price the answer proposes, which is the shape the correction prompt
    asks the model to write.

    Args:
        text: The clause the formula was found in.
        formula_start: Where the formula's arithmetic run starts in it.

    Returns:
        True when the formula's result is claimed to be an observed print.
    """
    last: re.Match[str] | None = None
    for match in _OBSERVED_PRICE_WORD_RE.finditer(text[:formula_start]):
        last = match
    if last is None:
        return False
    return bool(_RESULT_SEPARATOR_RE.search(text[last.end() : formula_start]))


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


# A price word that asserts the figure is a MARKET PRINT — an OHLC field or a
# spot quote. Such a figure is an observation, and an observation is never
# something a formula produces: "2026-06-23 收盘价 = 1.171 × 0.80 = 0.937"
# is arithmetic whose result is claimed to be a close the ledger holds as
# 1.137, and the bare-equation derivation exemption released it. Only the
# formula's OPERANDS stay exempt under such a label; see
# ``_labels_its_result_observed``.
#
# The OHLC half is DERIVED from ``_TABLE_FIELD_ALIASES``, which is the table
# path's own source of truth and already carries both scripts — the previous
# hand-written prose list held "closing price" but not "closed at", and
# "收盘价" but not "收盘为". The four bare English words are dropped: "open" is
# a verb, "high"/"low" are adjectives, and "close" only counts when it is not
# "close to". The spot half has no table equivalent and is written out.
_OHLC_PROSE_WORDS = sorted(
    (alias for alias in _TABLE_FIELD_ALIASES if alias not in {"open", "high", "low", "close"}),
    key=len,
    reverse=True,
)


_OBSERVED_PRICE_WORD_RE = re.compile(
    "(?:"
    # ASCII aliases take word boundaries so "closing" does not match inside
    # "disclosing"; the CJK ones are written without spaces and must not.
    + "|".join(
        (r"\b" + re.escape(word) + r"\b") if word.isascii() else re.escape(word)
        for word in _OHLC_PROSE_WORDS
    )
    + r"|\b(?:closed|opened|settled|fixed|traded|quoted|last[- ]traded|changed hands)\s+at\b"
    + r"|\bthe\s+(?:close|open|high|low)\s+(?:was|is)\b"
    + r"|\b(?:last|previous|prior|session|intraday|day's|today's)\s+(?:close|open|high|low)\b"
    + r"|\bclose\b(?!\s+to\b)"
    + r"|\b(?:current|latest|market|spot|share|last(?: traded)?)\s+price\b"
    + r"|\bthe\s+quote\b|\btrad(?:es|ing)\s+at\b|\bchanges?\s+hands\s+at\b"
    + r"|收盘于|收盘为|收报|报收|收于|现价|最新价|最新成交价|成交价|市价|报价|现报|股价"
    + ")",
    re.IGNORECASE,
)


# A loader's id is ASCII, but the answer follows the user's language, so a
# Chinese report names the same provider in Chinese. Demanding the ASCII id
# verbatim rejected correct prose: an answer reading "数据来源：腾讯财经" was
# reported as ``data_source_not_surfaced`` against evidence sourced from
# ``tencent``, and no rewrite short of writing the English word could pass.
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


# "元" is how a Chinese answer writes a CNY quote, but it is also the tail of
# 港元/美元/日元, so accepting it unguarded would let an answer about a Hong Kong
# listing satisfy a CNY requirement. It counts only when no other currency's
# character owns it.
_BARE_YUAN_RE = re.compile(r"(?<![港美日欧韩台新加澳])元")


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


_SYMBOL_HEADERS = {"symbol", "ticker", "code", "标的", "代码", "证券代码"}


@dataclass(frozen=True)
class ValidationResult:
    """Final-answer grounding decision."""

    valid: bool
    issues: list[dict[str, Any]] = field(default_factory=list)


class _PolicyMixin:
    """Policy behaviour of :class:`GroundingLedger`."""

    def _validate_identity(self, content: str) -> list[dict[str, Any]]:
        """Validate aggregate state and listed/private contradictions."""
        issues: list[dict[str, Any]] = []
        status = self.identity_status
        # Two conditions, both load-bearing.
        #
        # ``self._identities`` — a run that never named an instrument has no
        # identity to get wrong. The trigger phrase is matched against the user
        # message, so "什么是市盈率估值法？" set identity_required and then failed
        # every draft it could ever produce, including the honest answer. This
        # relaxation invents no licence to guess: a figure still has to survive
        # ``_validate_price_claims``, and a figure attached to a symbol no tool
        # handled still has to survive ``_validate_unsourced_symbols``.
        #
        # ``ambiguous`` is deliberately absent. A shortlist is an answer, which
        # is why ``_RESOLUTION_INCOMPLETE_STATUSES`` already lets workflow
        # selection proceed on it (#955) — but the final answer stayed blocked,
        # so a screening run loaded its skill and was then refused a conclusion.
        # Consumers remain blocked on ambiguous in ``authorize_tool_call``, so
        # such a run still cannot fetch a quote to misattribute.
        if (
            self._identity_required
            and self._identities
            and status in {"unresolved", "conflicting", "invalidated"}
        ):
            issues.append(
                {
                    "code": "identity_not_locked",
                    "status": status,
                    "message": f"Instrument identity is {status}; a final market conclusion requires locked identity.",
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
                    "message": (
                        f"Locked listed identity {', '.join(symbols)} was relabelled as private/unlisted "
                        "without a conflicting resolver result."
                    ),
                }
            )
        return issues

    def _validate_unsourced_symbols(self, content: str) -> list[dict[str, Any]]:
        """Reject figures attached to an instrument no tool in this run handled.

        This is the mechanically decidable half of "what the tools did not
        return, you do not supply" (#886/#887). Naming a symbol is left alone —
        prose may legitimately mention an index or a peer — but the moment a
        clause pairs an unhandled canonical symbol with a figure, the figure has
        no possible origin other than model memory.

        Args:
            content: Candidate assistant answer.

        Returns:
            One issue per distinct unsourced symbol carrying figures.
        """
        issues: list[dict[str, Any]] = []
        reported: set[str] = set()
        for line, line_offset in _lines_with_offsets(content):
            for segment, span_start, span_end in _clause_spans(line, line_offset):
                unknown = sorted(
                    symbol
                    for symbol in _scan_symbols(segment) - self._session_symbols - reported
                    if symbol.rsplit(".", 1)[0] not in self._session_symbol_roots
                )
                if not unknown or not self._numbers_without_dates_or_percent(segment):
                    continue
                # Accept figures that are attributed to an external source
                # (e.g., "The paper reports a Sharpe ratio of 1.8.") rather
                # than model memory. Scoped to the clause: a line-level check
                # would let a citation in one clause launder an invented
                # sibling metric in the next.
                if _ATTRIBUTION_RE.search(segment):
                    continue
                for symbol in unknown:
                    reported.add(symbol)
                    issues.append(
                        {
                            "code": "unsourced_symbol_figures",
                            "symbol": symbol,
                            "claim": segment.strip()[:200],
                            "span": [span_start, span_end],
                            "message": (
                                f"No tool call in this session passed in or returned {symbol}, "
                                "yet the answer attaches figures to it. Retrieve it, or report "
                                "it as not retrieved."
                            ),
                        }
                    )
        return issues

    def _validate_analysis_claims(self, content: str) -> list[dict[str, Any]]:
        """Reject backtest/analysis metrics with no kind-scoped evidence.

        #1336: after failed or deduplicated market-data calls the model may
        still present return-volatility / drawdown / probability figures as
        measured facts. A metric figure is legitimate only when the run's
        evidence contains the same *kind* of figure — recorded from a
        completed analysis result (backtest artifacts, factor/shadow/quantlib
        output) or from any successful tool's numeric output (e.g.
        ``portfolio_risk_xray``, flattened into ``_evidence``). Kind scoping
        is what keeps an observed price from standing in for an invented
        volatility figure. Definitional prose ("夏普比率大于 1.0 通常被认为
        较好"), explicitly forward-looking forecasts, and categorical
        historical-window facts without any window-producing analysis are
        handled separately here.

        Args:
            content: Candidate assistant answer.

        Returns:
            One issue per metric-bearing clause or table cell.
        """
        issues: list[dict[str, Any]] = []
        positions = _lines_with_offsets(content)
        lines = [line for line, _ in positions]
        consumed: set[int] = set()
        price_records = self._comparable_price_records()
        for header, rows, row_indices in self._pipe_tables(lines):
            consumed.update(row_indices)
            columns = [
                (cell, _metric_kind_for_text(cell), bool(_FORECAST_FRAME_RE.search(cell)))
                for cell in header
            ]
            has_kind_header = any(kind for _, kind, _ in columns)
            for row, row_line_index in zip(rows, row_indices):
                row_start = positions[row_line_index][1]
                row_span = (row_start, row_start + len(lines[row_line_index]))
                cells = row + [""] * (len(columns) - len(row))
                if not has_kind_header:
                    # Generic header ("指标 | 数值", "Metric | Value"): a cell
                    # naming a metric kind is a row LABEL and claims exactly
                    # one adjacent value cell (right first — "label, value"
                    # order — then left, for value-first layouts). Validating
                    # every cell after the label would grab annotation columns
                    # ("备注 | 较去年提升 2%") that prose never attributes to
                    # the label; parity with the prose verdict is the bar.
                    row_kinds = [
                        (index, _metric_kind_for_text(cell))
                        for index, cell in enumerate(cells)
                        if _metric_kind_for_text(cell) is not None
                    ]
                    claimed: set[int] = set()
                    for label_index, row_kind in row_kinds:
                        # A label cell that smuggles its own measurement
                        # ("| 年化收益率 18.2% | - |") is prose-identical to
                        # "年化收益率为 18.2%" — validate the label's own
                        # numbers against its kind too.
                        self._check_table_cell(
                            cells[label_index], row_kind, cells[label_index], issues,
                            row_span, price_records,
                        )
                        # A forecast frame in the LABEL ("| 预计夏普比率 | 1.2 |")
                        # frames the claimed value clause-wide, exactly as prose
                        # exempts the whole clause and the metric-header path
                        # skips a forecast-framed column. Without this the generic
                        # path is stricter than both of its siblings.
                        label_is_forecast = bool(
                            _FORECAST_FRAME_RE.search(cells[label_index])
                        )
                        for value_index in (label_index + 1, label_index - 1):
                            if (
                                0 <= value_index < len(cells)
                                and value_index not in claimed
                                and _metric_kind_for_text(cells[value_index]) is None
                            ):
                                claimed.add(value_index)
                                if label_is_forecast:
                                    continue
                                self._check_table_cell(
                                    cells[label_index], row_kind, cells[value_index], issues,
                                    row_span, price_records,
                                )
                    continue
                for (cell_text, kind, header_forecast), cell in zip(columns, cells):
                    if kind is None or header_forecast:
                        continue
                    self._check_table_cell(
                        cell_text, kind, cell, issues, row_span, price_records
                    )
        for index, (line, line_offset) in enumerate(positions):
            if index in consumed:
                continue
            line_symbol = self._symbol_for_claim(line, price_records)
            for segment, span_start, span_end in _clause_spans(line, line_offset):
                if _CATEGORICAL_WINDOW_RE.search(segment) and _NUMBER_RE.search(segment):
                    if not any(
                        entry.get("tool") in _ANALYSIS_WINDOW_TOOLS
                        for entry in self._analysis_completed
                    ):
                        match = _NUMBER_RE.search(segment)
                        issues.append(
                            {
                                "code": "analysis_claim_unavailable",
                                "claim": segment.strip()[:200],
                                "span": [span_start, span_end],
                                "value": match.group(0) if match else None,
                                "message": (
                                    "No backtest completed in this session, yet the "
                                    "answer states a categorical historical-window "
                                    "fact. Mark the analysis as incomplete and omit "
                                    "this claim."
                                ),
                            }
                        )
                    continue
                if _DEFINITION_FRAME_RE.search(segment):
                    continue
                # Attributed figures ("The paper reports a Sharpe of 1.8",
                # or the marker in a neighbouring clause: "据研究显示，策略
                # 年化收益 18.2%") are citations, not invented measurements
                # — skip the gate.
                if _ATTRIBUTION_RE.search(segment):
                    continue
                values = self._measure_numbers(segment)
                if not values:
                    continue
                if not _ANALYSIS_METRIC_RE.search(segment):
                    continue
                if _FORECAST_FRAME_RE.search(segment):
                    continue
                kind = _metric_kind_for_text(segment)
                unsupported = [
                    value
                    for value in values
                    if not self._analysis_value_observed(value, kind)
                    and not self._is_observed_price_figure(
                        value, price_records, line_symbol, segment
                    )
                ]
                if not unsupported:
                    continue
                # A return figure may be arithmetic on sourced inputs rather
                # than an invented backtest metric (#1338 review): an explicit
                # formula anchored to observed values, or growth between two
                # observed endpoints stated in the same line.
                # A drawdown stated against an observed high ("较 5 月高点
                # 1.053 元已回撤约 37%") is the same arithmetic on the same
                # sourced endpoints, and was rejected as an invented backtest
                # metric because only the return branch carried the exemption
                # (owner-forwarded 159516.SZ run, 2026-09-09). Drawdown is
                # compared by magnitude, as ``_analysis_value_observed`` does.
                # Neither exemption is available to a STRATEGY-subject
                # metric. A strategy's max drawdown is a property of an
                # equity curve, and no arithmetic on two observed prints of
                # the instrument is that curve: "回测显示该策略最大回撤 5.9%
                # （从 1.180 跌至 1.110）" was released as grounded with no
                # backtest in the session at all. Such a figure keeps the
                # unchanged backtest-evidence requirement above.
                if kind in {"return", "drawdown"} and not _STRATEGY_SUBJECT_RE.search(
                    segment
                ):
                    if _DERIVATION_RE.search(segment):
                        # As in the price path: only the values the formula
                        # justifies are exempt, not every figure beside them.
                        # Scoped to the CLAUSE, which is where the claim
                        # itself was measured — a line-scoped exemption let
                        # "基于 40 × 1.171 = 46.84 计算，策略最大回撤 40%"
                        # ground the metric in the next clause.
                        justified = self._derivation_justified_values(
                            segment, price_records, line_symbol
                        )
                        unsupported = [
                            value
                            for value in unsupported
                            if not _matches_any(value, justified)
                        ]
                        if not unsupported:
                            continue
                    # The operand scan stays LINE-scoped: the endpoints of a
                    # move are routinely written in two clauses ("当前 0.666
                    # 元，较 5 月高点 1.053 元已回撤约 37%").
                    operands = self._observed_operands_in_line(
                        line, price_records, line_symbol
                    )
                    if len(operands) >= 2:
                        derived = self._returns_derived_from_observed(
                            unsupported,
                            price_records,
                            line_symbol,
                            operands=operands,
                            magnitude=kind == "drawdown",
                        )
                        # The endpoints themselves are grounded by the same
                        # exemption: "较高点 1.180 元已回撤约 6%" states the
                        # observed high beside the ratio it derives.
                        unsupported = [
                            value
                            for value in unsupported
                            if value not in derived
                            and not (
                                not str(value).endswith(("%", "％"))
                                and _matches_any(value, operands, rel=1e-9)
                            )
                        ]
                        if not unsupported:
                            continue
                issues.append(
                    {
                        "code": "analysis_claim_unavailable",
                        "claim": segment.strip()[:200],
                        "span": [span_start, span_end],
                        "value": unsupported[0],
                        "kind": kind,
                        "message": (
                            "No supporting analysis evidence (a completed "
                            "backtest result or observed risk metric) exists for "
                            "this figure. Mark the analysis as incomplete and omit "
                            "these figures."
                        ),
                    }
                )
        return issues

    def _check_table_cell(
        self,
        label: str,
        kind: str | None,
        cell: str,
        issues: list[dict[str, Any]],
        span: tuple[int, int] | None = None,
        price_records: Sequence[EvidenceRecord] = (),
    ) -> None:
        """Reject one table cell whose numeric value is an unsupported metric.

        Shared by the metric-headed and generic-header (label, value) table
        paths. A forecast or definitional annotation exempts only that cell.

        Args:
            label: The metric label the value is attached to (header cell or
                row's first cell), for the issue claim text.
            kind: The resolved metric kind, already derived from ``label``.
            cell: One value cell to validate.
            issues: Accumulator for ``analysis_claim_unavailable`` issues.
            span: ``(start, end)`` of the table ROW inside the validated
                answer. The claim text is the human-readable ``label: cell``,
                which is not a substring of the draft ("| 最大回撤 | 12% |"),
                so without the span the release path could never locate a
                metric table and every metrics report fell back to the canned
                refusal.
            price_records: Comparable observed price evidence, so a cell that
                merely restates observed closes ("| 峰值→当前 | 0.997 → 0.605 |")
                is not reported as an invented drawdown figure and then cut
                out of the released answer. Same rule as the prose path.
        """
        if kind is None:
            return
        values = _PolicyMixin._measure_numbers(cell)
        if not values:
            return
        # A forecast annotation inside the cell ("预计 12.4%") exempts only
        # that cell, never its neighbours.
        if _FORECAST_FRAME_RE.search(cell) or _DEFINITION_FRAME_RE.search(cell):
            return
        # A cell RESTATING endpoints ("| 峰值→当前 | 1.180 → 1.110 |") is not
        # a metric figure, and cutting it out of the released answer removed
        # two correctly quoted closes. The restatement shape is what the
        # exemption is for, so it needs the cell to hold a PAIR: a single
        # number in a metric cell IS that metric's value, and exempting it on
        # numeric equality alone released "| 夏普比率 | 1.171 |" as grounded
        # because 1.171 happened to be an observed close. On a $12.50 stock
        # the collision is "| 最大回撤 | 12.5 |".
        unsupported = [
            value
            for value in values
            if not self._analysis_value_observed(value, kind)
            and not (
                len(values) >= 2
                and self._matches_observed_price(value, price_records, None)
            )
        ]
        if not unsupported:
            return
        issues.append(
            {
                "code": "analysis_claim_unavailable",
                "claim": f"{label}: {cell}"[:200],
                "span": list(span) if span else None,
                "value": unsupported[0],
                "kind": kind,
                "message": (
                    "No supporting analysis evidence (a completed "
                    "backtest result or observed risk metric) exists "
                    "for this figure. Mark the analysis as incomplete "
                    "and omit these figures."
                ),
            }
        )

    @staticmethod
    def _measure_numbers(text: str) -> list[str]:
        """Extract measurement-shaped numbers (decimal or percent) from a claim."""
        masked = _LOCALIZED_DATE_RE.sub(" ", text)
        masked = _DATE_RE.sub(" ", masked)
        masked = _SHORT_DATE_RE.sub(" ", masked)
        masked = _DASH_DATE_RE.sub(" ", masked)
        return [
            match.group(0).replace(" ", "").replace(",", "")
            for match in _MEASURE_NUMBER_RE.finditer(masked)
        ]

    @staticmethod
    def _pipe_tables(
        lines: Sequence[str],
    ) -> list[tuple[list[str], list[list[str]], list[int]]]:
        """Yield (header cells, row cells, row line indices) per pipe table."""
        tables: list[tuple[list[str], list[list[str]], list[int]]] = []
        index = 0
        while index < len(lines):
            if lines[index].count("|") < 2:
                index += 1
                continue
            block_start = index
            block: list[str] = []
            while index < len(lines) and lines[index].count("|") >= 2:
                block.append(lines[index])
                index += 1
            rows: list[list[str]] = []
            row_indices: list[int] = []
            for offset, block_line in enumerate(block[1:], start=1):
                cells = _PolicyMixin._table_cells(block_line)
                if not cells or all(
                    _TABLE_SEPARATOR_RE.fullmatch(cell.strip()) for cell in cells
                ):
                    continue
                rows.append(cells)
                row_indices.append(block_start + offset)
            tables.append((_PolicyMixin._table_cells(block[0]), rows, row_indices))
        return tables

    def _analysis_value_observed(self, raw: str, kind: str | None) -> bool:
        """Return True when a claim measurement matches kind-scoped evidence.

        Tools disagree on scale: ``compute_risk_xray`` returns fractions
        (annualized_vol 0.182, max_drawdown -0.094) while answers quote
        percents (18.2%, -9.4%). Try both the value and its percent scaling
        so an observed 0.182 grounds an 18.2% claim and vice versa. Drawdown
        sign conventions disagree too (positive vs negative fraction), so
        magnitude is compared for that kind. Only evidence of the claim's own
        kind counts: an observed price never grounds a volatility figure.
        """
        try:
            value = float(raw.replace("%", "").replace("％", "").replace(",", ""))
        except ValueError:
            return True
        observed: list[float] = []
        for record in self._analysis_metrics:
            if record.get("metric") == kind and record.get("value") is not None:
                observed.append(float(record["value"]))
        for record in self._evidence:
            if record.status != "observed" or record.value is None:
                continue
            if _metric_kind_for_path(record.field) != kind:
                continue
            observed.append(float(record.value))
        candidates = {value, value / 100.0}
        if kind == "drawdown":
            candidates |= {abs(value), abs(value) / 100.0}

        def close(candidate: float, item: float) -> bool:
            return abs(candidate - item) <= max(abs(item) * 0.005, 1e-9)

        return any(close(candidate, item) for candidate in candidates for item in observed)

    def _is_observed_price_figure(
        self,
        raw: str,
        records: Sequence[EvidenceRecord],
        symbol: str | None,
        text: str,
    ) -> bool:
        """Whether a figure in an analysis clause is literally an observed price.

        ``_validate_analysis_claims`` reports the FIRST unmatched number in a
        clause that mentions a metric, and an observed close sharing a clause
        with an ungrounded metric was that number: "最新收盘 1.171 元且策略最大
        回撤 12%" reported 1.171, so the release path cut the correctly quoted
        close and footnoted it as unverifiable. A value the ledger observed is
        not an invented backtest metric — its own check is
        ``_compare_price_claim``, which compares any price-word figure against
        observed OHLC at the 0.5% band. A percent-written figure is never
        exempted here, so every drawdown / return / volatility / win-rate
        percentage keeps the full analysis check.

        Numeric equality is not enough on its own. A run holds hundreds of
        observed OHLC values spanning the instrument's price range, so ANY
        percent-free metric written in that range collides with one:
        "策略夏普比率 1.171" and "策略最大回撤 1.171" were released as grounded
        against an observed close, with no backtest in the session. The figure
        must also be WRITTEN as a price — a price word ahead of it or a
        currency unit glued to it — which is exactly the shape that has its
        own check in ``_compare_price_claim``.

        Args:
            raw: A figure as ``_measure_numbers`` returns it.
            records: Comparable observed price evidence.
            symbol: Symbol resolved for the line, if any.
            text: The clause the figure was written in.

        Returns:
            True when the figure is written without a percent sign, reads as a
            price in its clause, and equals an observed price value for this
            symbol.
        """
        figure = str(raw).strip()
        if not _figure_reads_as_a_price(text, figure):
            return False
        return self._matches_observed_price(figure, records, symbol)

    def _matches_observed_price(
        self,
        raw: str,
        records: Sequence[EvidenceRecord],
        symbol: str | None,
    ) -> bool:
        """Whether a percent-free figure equals an observed price value.

        Args:
            raw: A figure as ``_measure_numbers`` returns it.
            records: Comparable observed price evidence.
            symbol: Symbol resolved for the claim, if any.

        Returns:
            True when the figure carries no percent sign and matches evidence.
        """
        figure = str(raw).strip()
        if figure.endswith(("%", "％")):
            return False
        candidates = list(records)
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        observed = [
            float(record.value) for record in candidates if record.value is not None
        ]
        return _matches_any(figure, observed, rel=1e-9)

    def _observed_operands_in_line(
        self,
        line: str,
        records: Sequence[EvidenceRecord],
        symbol: str | None,
    ) -> list[float]:
        """Observed values that literally appear as numbers in this clause.

        This is the structural half of the derivation exemption. Keying it on
        a growth PHRASE ("从…到" / "from…to") made the gate stricter for every
        wording the list happened to miss, which is the same per-language
        drift that ``test_grounding_language_parity`` exists to stop: the
        Chinese "第一日收盘 100.0 美元，第二日收盘 112.4 美元，收益率 12.4%"
        states the identical derivation and was rejected. Requiring the
        operands themselves to be present and sourced is language-independent
        and strictly narrower than a phrase list, because a bare
        "cumulative return of 12.4%" carries no operands at all.
        """
        candidates = [record for record in records if record.value is not None]
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        observed = {float(record.value) for record in candidates}
        if not observed:
            return []
        present: set[float] = set()
        for raw in self._numbers_without_dates_or_percent(line):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            for candidate in observed:
                if abs(value - candidate) <= max(abs(candidate) * 1e-9, 1e-9):
                    present.add(candidate)
        return sorted(present)

    def _returns_derived_from_observed(
        self,
        claimed: Sequence[str],
        records: Sequence[EvidenceRecord],
        symbol: str | None,
        *,
        operands: Sequence[float] | None = None,
        magnitude: bool = False,
    ) -> set[str]:
        """Return the figures that equal growth between observed endpoints.

        #1338 review: "AAPL.US 从 2026-08-03 的 100.0 涨到 2026-09-02 的
        112.4，区间收益率为 12.4%" states arithmetic on sourced inputs. Only
        a match against a pair of observed values grounds the figure; the
        caller must already have verified the from/to frame, so a bare
        unanchored return claim never reaches here.

        The match tolerates the precision the figure was WRITTEN at: "约 37%"
        for a derived 36.75% asserts that the value rounds to 37, and the flat
        ±0.5% relative band (±0.18 points there) rejected every integer-percent
        rounding a model makes. Half a unit of the last written digit is the
        bound, so "37%" accepts 36.5–37.5 and "36.8%" accepts 36.75–36.85; a
        figure that is simply wrong ("40%") is as far away as before.

        Args:
            claimed: Raw figures from the clause, as ``_measure_numbers`` returns them.
            records: Observed evidence the operands may come from.
            symbol: Symbol the clause names, if any.
            operands: Observed values literally present in the clause.
            magnitude: Compare absolute values (drawdown sign conventions differ).

        Returns:
            The subset of ``claimed`` the endpoint arithmetic justifies. It
            used to be one bool for the whole clause, and the caller skipped
            every figure beside the justified one: "at 0.666 down about 37%
            from the 1.053 May high with a max drawdown of 60%" released the
            60% because the 37% checked out.
        """
        if operands is not None:
            observed = sorted(set(operands))
        else:
            candidates = [record for record in records if record.value is not None]
            if symbol:
                candidates = [
                    record for record in candidates if record.symbol == symbol
                ]
            observed = sorted({float(record.value) for record in candidates})
        if len(observed) < 2:
            return set()
        values: list[tuple[str, float, float, bool]] = []
        for raw in claimed:
            raw_text = str(raw).strip()
            text = raw_text.rstrip("%％")
            try:
                value = float(text)
            except ValueError:
                continue
            decimals = len(text.split(".", 1)[1]) if "." in text else 0
            values.append((str(raw), value, 0.5 * 10.0 ** (-decimals), raw_text != text))

        def close(value: float, derived: float, half_unit: float) -> bool:
            if magnitude:
                # A drawdown is a fall. Taking absolute values so "回撤 37%"
                # can match a derived -36.75% also made the INVERSE
                # derivation match: the same two endpoints read low→high give
                # +58.11%, and "最大回撤约 58%" was released as grounded
                # (attack2 probe, 2026-09-09). Only the falling direction may
                # ground a drawdown.
                if derived > 0:
                    return False
                value, derived = abs(value), abs(derived)
            return abs(value - derived) <= max(abs(derived) * 0.005, half_unit, 1e-9)

        justified: set[str] = set()
        for base in observed:
            for target in observed:
                if target == base:
                    continue
                derived = (target - base) / base
                for raw, value, half_unit, is_percent in values:
                    # ``half_unit`` is half a unit of the last digit the figure
                    # was WRITTEN with, so it is only meaningful against the
                    # derivation expressed in those same units. A figure
                    # carrying "%" is percentage points and is compared only
                    # against ``derived * 100``; running it against the
                    # fraction too gave an integer percent a 0.5 band in
                    # FRACTION units — 50 percentage points — and "区间收益率
                    # 约 0%" validated against a real +58% move (attack4
                    # probe, 2026-09-09). A bare figure stays ambiguous and is
                    # tried both ways, each against its own consistent band.
                    # A bare figure always carries a decimal point —
                    # ``_MEASURE_NUMBER_RE`` extracts an integer only with a
                    # percent sign on it — so the fraction reading is never
                    # taken with a 0.5 half unit (which would be 50
                    # percentage points wide).
                    if not is_percent and close(value, derived, half_unit):
                        justified.add(raw)
                    if close(value, derived * 100.0, half_unit):
                        justified.add(raw)
        return justified

    @staticmethod
    def _occurrence_is_derived(
        segment: str,
        formulas: Sequence[tuple[int, int, list[float], float]],
        value: float,
        start: int,
        end: int,
    ) -> bool:
        """Whether this figure, HERE, is part of a formula that justifies it.

        Two things are decided per occurrence rather than per value.

        The position: a number is exempt only where it sits inside the
        formula. "建议买入价 0.95 元（0.95 × 1.171 = 1.11245 参考）" is arithmetic
        that derives nothing, and under a value-scoped exemption the entry
        price outside the bracket inherited the exemption its own copy inside
        the bracket earned.

        The role: a formula's RESULT is a number the answer proposes, which is
        legitimate for an entry price and never legitimate for a market print.
        "2026-06-23 收盘价 = 1.171 × 0.80 = 0.937" claims a close the ledger
        holds as 1.137, and the exemption released it. Under an observation
        label the operands stay exempt and the result is compared.

        Args:
            segment: The clause the figure was found in.
            formulas: ``_derivation_formulas`` output for that clause.
            value: The figure.
            start: Where the figure starts in ``segment``.
            end: Where it ends.

        Returns:
            True when the occurrence may skip the price comparison.
        """
        for formula_start, formula_end, inputs, result in formulas:
            if not (formula_start <= start and end <= formula_end):
                continue
            if _matches_any(value, inputs):
                return True
            if _matches_any(value, [result]) and not _labels_its_result_observed(
                segment, formula_start
            ):
                return True
        return False

    def _validate_price_claims(self, content: str) -> list[dict[str, Any]]:
        """Check Markdown OHLC tables and price prose against observed records.

        Comparison runs against every observed quote in the run, whichever tool
        produced it. The provenance demands below stay keyed on ``get_market_data``
        evidence, whose ``source``/``currency``/venue fields are authoritative;
        a generic tool's fallback source is its own name, and requiring the
        answer to spell that out would reject correct prose.
        """
        issues, table_lines = self._validate_price_tables(content)
        records = self._comparable_price_records()
        # A report names its subject once and then writes prose about it. Both
        # narrower scopes are tried first; this is the last resort, and it only
        # resolves when the whole answer names exactly one evidence symbol.
        document_symbol = self._symbol_for_claim(content, records)
        has_price_claim = any(
            self._numbers_without_dates_or_percent(line)
            for index, line in enumerate(content.splitlines())
            if index in table_lines
        )
        for index, (line, line_offset) in enumerate(_lines_with_offsets(content)):
            if index in table_lines or "|" in line:
                continue
            line_symbol = self._symbol_for_claim(line, records)
            for segment, span_start, span_end in _clause_spans(line, line_offset):
                if not _PRICE_CONTEXT_RE.search(segment):
                    continue
                values = self._direct_price_values(segment)
                if not values:
                    continue
                has_price_claim = True
                symbol = (
                    self._symbol_for_claim(segment, records)
                    or line_symbol
                    or document_symbol
                )
                # Only the formula's own operands and result are exempt, at
                # the position where they sit; a figure riding along in the
                # same clause is still compared.
                formulas = self._derivation_formulas(segment, records, symbol)
                # NO attribution exemption here, deliberately. A paper's
                # Sharpe is a figure this run could never have observed, so
                # citing it is legitimate; a PRICE is exactly what this run
                # does observe, so "analysts say TSLA.US last traded at
                # 412.35" is the laundering shape this gate exists to catch —
                # adding a citation subject must not buy a fabricated quote a
                # way through. The exemption stays in the analysis gate only.
                #
                # An indicator reading is admissible evidence only for a
                # LEVEL claim; a clause claiming a market print is answered by
                # OHLC evidence alone.
                indicator_admissible = bool(
                    _PRICE_LEVEL_WORD_RE.search(segment)
                ) and not _OBSERVED_PRICE_WORD_RE.search(segment)
                for value, value_start, value_end in values:
                    if self._occurrence_is_derived(
                        segment, formulas, value, value_start, value_end
                    ):
                        continue
                    issue = self._compare_price_claim(
                        value=value,
                        records=records,
                        field_name=None,
                        date_value=None,
                        symbol=symbol,
                        claim=segment.strip(),
                        span=(span_start, span_end),
                        indicator_admissible=indicator_admissible,
                    )
                    if issue:
                        issues.append(issue)
        market_records = self._price_records()
        if has_price_claim and market_records:
            issues.extend(self._validate_price_provenance(content, market_records))
        return self._dedupe_issues(issues)

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
                    "message": (
                        "A price claim must surface its locked canonical symbol and venue suffix."
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
                for alias in _SOURCE_ALIASES.get(
                    source.casefold(), (source.casefold(),)
                )
            )
        ]
        if missing_sources:
            issues.append(
                {
                    "code": "data_source_not_surfaced",
                    "sources": missing_sources,
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
        return code == "CNY" and bool(_BARE_YUAN_RE.search(content))

    def _validate_price_tables(
        self,
        content: str,
    ) -> tuple[list[dict[str, Any]], set[int]]:
        """Validate field/date-specific claims in Markdown OHLC tables."""
        positions = _lines_with_offsets(content)
        lines = [line for line, _ in positions]
        issues: list[dict[str, Any]] = []
        consumed: set[int] = set()
        index = 0
        records = self._comparable_price_records()
        while index + 1 < len(lines):
            header = self._table_cells(lines[index])
            separator = self._table_cells(lines[index + 1])
            if not header or not separator or len(header) != len(separator):
                index += 1
                continue
            if not all(_TABLE_SEPARATOR_RE.match(cell.replace(" ", "")) for cell in separator):
                index += 1
                continue
            field_columns = {
                position: _TABLE_FIELD_ALIASES[cell.strip().casefold()]
                for position, cell in enumerate(header)
                if cell.strip().casefold() in _TABLE_FIELD_ALIASES
            }
            if not field_columns:
                index += 1
                continue
            date_column = next(
                (position for position, cell in enumerate(header) if cell.strip().casefold() in _DATE_HEADERS),
                None,
            )
            symbol_column = next(
                (position for position, cell in enumerate(header) if cell.strip().casefold() in _SYMBOL_HEADERS),
                None,
            )
            consumed.update({index, index + 1})
            row_index = index + 2
            while row_index < len(lines):
                row = self._table_cells(lines[row_index])
                if not row or len(row) != len(header):
                    break
                consumed.add(row_index)
                date_value = row[date_column].strip() if date_column is not None else None
                symbol = _normalize_symbol(row[symbol_column]) if symbol_column is not None else None
                for position, field_name in field_columns.items():
                    values = self._numbers_without_dates_or_percent(row[position])
                    if len(values) != 1:
                        continue
                    row_start = positions[row_index][1]
                    issue = self._compare_price_claim(
                        value=values[0],
                        records=records,
                        field_name=field_name,
                        date_value=date_value,
                        symbol=symbol,
                        claim=row[position].strip(),
                        # A table cell's claim text is a bare number, so it
                        # can never be located by substring search — "1.10"
                        # matched inside "21.10 亿元" and rewrote an untouched
                        # turnover figure into "2（略※）". The row's own span
                        # is the anchor; the value inside it is what is cut.
                        span=(row_start, row_start + len(lines[row_index])),
                    )
                    if issue:
                        issues.append(issue)
                row_index += 1
            index = max(row_index, index + 1)
        return issues, consumed

    @staticmethod
    def _table_cells(line: str) -> list[str]:
        """Split one Markdown table row, or return an empty list."""
        if "|" not in line:
            return []
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def _compare_price_claim(
        self,
        *,
        value: float,
        records: list[EvidenceRecord],
        field_name: str | None,
        date_value: str | None,
        symbol: str | None,
        claim: str,
        span: tuple[int, int] | None = None,
        indicator_admissible: bool = False,
    ) -> dict[str, Any] | None:
        """Compare one unlabelled observed claim to the closest evidence value.

        Args:
            value: The figure the answer states.
            records: Comparable observed price evidence.
            field_name: OHLC field the claim is keyed to (table column), if any.
            date_value: Trade date the claim is keyed to, if any.
            symbol: Symbol explicitly resolved for the claim, if any.
            claim: Clause or cell text, for the issue message.
            span: ``(start, end)`` of the claim inside the validated answer.
            indicator_admissible: The prose clause states a price LEVEL
                (支撑位 / 均线 / bollinger band), the one claim an indicator
                reading may answer. Every other price claim — a named OHLC
                field, a spot quote, a proposed entry — is compared against
                OHLC evidence only, which is the rule the table path already
                applies by filtering on ``field_name``. The table path passes
                a ``field_name`` for every cell it validates and so leaves
                this at its default.
        """
        resolved_symbol = symbol
        candidates = records
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        symbols = sorted({record.symbol for record in candidates if record.symbol})
        # An indicator reading is symbol-bound in a way an OHLC bar's union is
        # not. With evidence for two instruments and a clause naming neither,
        # 562500's sma_20 grounded "600519.SH … 其均线在 1.150 元"; the union
        # below was argued for OHLC quotes, not for a derived level attached
        # to one symbol. Drop indicator records whenever the claim's symbol
        # was not explicitly resolved and the run holds more than one.
        if resolved_symbol is None and len({r.symbol for r in records if r.symbol}) > 1:
            candidates = [record for record in candidates if record.field != "indicator"]
        if not indicator_admissible:
            candidates = [record for record in candidates if record.field != "indicator"]
        if not symbol and len(symbols) == 1:
            symbol = symbols[0]
        # An unattributed claim used to be rejected outright once the run held
        # evidence for more than one symbol. That is every comparison report:
        # "Apple's closing price was 313.33 USD. Microsoft closed higher." names
        # its subject by company name, and the clause was refused although the
        # value was exactly the observed close sitting in evidence. Such a claim
        # is now checked against the union of the observed quotes instead, so a
        # number the run never observed is still caught below — it simply has to
        # match nothing at all rather than nothing under one chosen symbol.
        if field_name:
            candidates = [record for record in candidates if record.field == field_name]
        if date_value:
            candidates = [
                record
                for record in candidates
                if record.timestamp
                and _timestamp_matches_claim_date(record.timestamp, date_value)
            ]
        if not candidates:
            return {
                "code": "numeric_claim_unavailable",
                "claim": claim,
                "span": list(span) if span else None,
                "value": value,
                "symbol": symbol,
                "field": field_name,
                "date": date_value,
                "message": f"Price claim {value:g} has no matching observed tool evidence.",
            }
        observed = [float(record.value) for record in candidates if record.value is not None]
        if any(abs(value - item) <= max(abs(item) * 0.005, 1e-9) for item in observed):
            return None
        return {
            "code": "numeric_claim_conflict",
            "claim": claim,
            "span": list(span) if span else None,
            "value": value,
            "symbol": symbol,
            "field": field_name,
            "date": date_value,
            "observed_min": min(observed),
            "observed_max": max(observed),
            "source_tool_call_ids": sorted({record.call_id for record in candidates}),
            "message": (
                f"Price claim {value:g} conflicts with observed {field_name or 'OHLC'} "
                f"evidence {min(observed):g}–{max(observed):g}."
            ),
        }

    @staticmethod
    def _masked_candidate_text(text: str) -> str:
        """Mask every non-price digit run, preserving string length and offset.

        Each mask match is replaced by an equal-length run of spaces, so a
        number's offset in the returned string is its offset in ``text`` — the
        structural price-claim scan needs that alignment.
        """
        masked = text
        for pattern in (
            _MD_LIST_ITEM_RE,
            _RATE_FORMULA_IDENTITY_RE,
            _CANONICAL_SYMBOL_RE,
            _LOCALIZED_DATE_RE,
            _DATE_RE,
            _SHORT_DATE_RE,
            _DASH_DATE_RE,
            _PERCENT_RANGE_RE,
            _PERCENTAGE_POINT_RE,
            _ORDER_LEVEL_RE,
            _AGGREGATE_AMOUNT_RE,
            _LABELLED_SCORE_RE,
            _INDICATOR_VALUE_RE,
            _SIGNAL_VALUE_RE,
            _PROSPECTIVE_LEVEL_RE,
            _REFERENCE_LEVEL_RE,
            _SINCE_REFERENCE_RE,
            _LINE_REFERENCE_RE,
            _NUMBERED_HEADING_RE,
            _RATIO_RE,
            _FX_RATE_RE,
            _QUANTITY_WITH_UNIT_RE,
        ):
            masked = pattern.sub(lambda m: " " * (m.end() - m.start()), masked)
        return masked

    @staticmethod
    def _numbers_without_dates_or_percent(text: str) -> list[float]:
        """Extract the numbers in a claim that could plausibly be prices.

        Digits that belong to a canonical symbol, a calendar date, an aggregate
        amount, a labelled score, a named indicator reading, a unit-bearing
        quantity, or a percentage are masked first. Left unmasked they are
        compared against observed OHLC ranges and reject a correct draft:
        ``000543.SZ`` alone contributes 543, and a well-formed verdict line
        contributes its confidence score and every moving-average window it
        names (#1001).

        Args:
            text: One claim segment or table cell.

        Returns:
            Candidate price values, in order of appearance.
        """
        masked = _PolicyMixin._masked_candidate_text(text)
        values: list[float] = []
        for match in _NUMBER_RE.finditer(masked):
            tail = masked[match.end() :].lstrip()
            if tail.startswith(("%", "％")):
                continue
            try:
                values.append(float(match.group(0).replace(",", "")))
            except ValueError:
                continue
        return values

    @staticmethod
    def _direct_price_values(text: str) -> list[tuple[float, int, int]]:
        """Numbers in a price segment that read as asserted observed values.

        Returned as ``(value, start, end)``: the masking that decides which
        digits are candidates preserves length, so each offset is the number's
        offset in ``text``. The caller needs the position, not only the value —
        a derivation exempts the occurrence sitting inside it, not every copy
        of the same number in the clause.

        ``_numbers_without_dates_or_percent`` returns every non-masked number;
        this further drops numbers that are formula operands rather than claims
        (#1354). For each surviving number, the span between the nearest
        preceding price-context word and the number decides:

        * a sentence boundary (``. ``/``! ``/``? ``) in the span — the number
          belongs to a later sentence the price word cannot reach ("close was
          210. In 2024 …" must not claim 2024);
        * otherwise, a closed formula marker in the span turns the number into
          an operand, unless an observation binder ("was", "at", 报收/收于/…) after
          the last marker re-attaches it to the price word.

        "close/SMA50 > 1" and "close above SMA50 and was 2500" are decided in
        opposite directions by the binder; "close was 2500" (no marker) stays
        a claim either way.
        """
        price_words = list(_PRICE_CONTEXT_RE.finditer(text))
        masked = _PolicyMixin._masked_candidate_text(text)
        values: list[tuple[float, int, int]] = []
        for match in _NUMBER_RE.finditer(masked):
            tail = masked[match.end() :].lstrip()
            if tail.startswith(("%", "％")):
                continue
            try:
                value = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            preceding = [w for w in price_words if w.end() <= match.start()]
            if preceding:
                span = text[preceding[-1].end() : match.start()]
                if _SENTENCE_BOUNDARY_RE.search(span):
                    continue
                markers = list(_FORMULA_MARKER_RE.finditer(span))
                if markers and not _OBSERVATION_BINDER_RE.search(
                    span[markers[-1].end() :]
                ):
                    continue
            values.append((value, match.start(), match.end()))
        return values

    def _derivation_formulas(
        self,
        text: str,
        records: Sequence[EvidenceRecord],
        symbol: str | None,
    ) -> list[tuple[int, int, list[float], float]]:
        """Return every arithmetically valid, observation-anchored formula.

        Each entry is ``(start, end, operands, result)`` — the formula's own
        character span inside ``text``, the operands it consumes and the result
        it produces. The span is what makes the exemption safe. A value-scoped
        exemption ("this number equals something a formula justifies") let a
        figure be exempt EVERYWHERE in the clause once it appeared anywhere in
        an equation, and a model told by the correction prompt to show its
        arithmetic can satisfy that with arithmetic that derives nothing:
        "建议买入价 0.95 元（0.95 × 1.171 = 1.11245 参考）" is true and released
        an entry price 19% below every observed bar. The occurrence inside the
        formula is exempt; the one beside it is compared like any other claim.

        There is no keyword or shape precondition. A `基于` / `based on` was
        once required and an added `_ARITHMETIC_EQUATION_RE` relaxed it to
        "any two-operand equation"; both are subsumed by what decides below —
        a result separator, a parseable arithmetic run of at least two
        operands (``_evaluate_formula``), an input matching observed evidence,
        and a result that is correct. Deleting the regex changed no verdict in
        the suite and removes one entry from the regex catalogue.

        Args:
            text: The clause (or line) to inspect.
            records: Comparable observed price evidence.
            symbol: Symbol resolved for the claim, if any.

        Returns:
            ``(start, end, operands, result)`` per valid formula, in order.
        """
        candidates = list(records)
        if symbol:
            candidates = [record for record in candidates if record.symbol == symbol]
        candidate_symbols = {record.symbol for record in candidates if record.symbol}
        if not symbol and len(candidate_symbols) > 1:
            return []
        observed = [
            float(record.value) for record in candidates if record.value is not None
        ]
        if not observed:
            return []

        formulas: list[tuple[int, int, list[float], float]] = []
        for equals in _RESULT_SEPARATOR_RE.finditer(text):
            # An indicator identifier carries digits ("SMA20", "MA5"); left in
            # place they leak into the arithmetic run ("20 1.150 × 0.95") and
            # the formula fails to parse, so a derivation from an observed
            # moving average was never accepted. The blanking is
            # LENGTH-PRESERVING because the offsets below are the exemption's
            # anchor: collapsing "SMA20" to one space shifted every span after
            # it and the formula was excluded from its own protection.
            prefix = _INDICATOR_IDENTIFIER_RE.sub(
                lambda match: " " * (match.end() - match.start()),
                text[: equals.start()],
            )
            left = re.search(r"([0-9.,+\-*/×÷()\s]+)$", prefix)
            right = re.match(
                r"\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
                text[equals.end() :],
            )
            if not left or not right:
                continue
            evaluated = self._evaluate_formula(left.group(1))
            if evaluated is None:
                continue
            computed, inputs = evaluated
            try:
                claimed = float(right.group(1).replace(",", ""))
            except ValueError:
                continue
            if not any(
                abs(item - value) <= max(abs(value) * 0.005, 1e-9)
                for item in inputs
                for value in observed
            ):
                continue
            if abs(computed - claimed) <= max(abs(computed) * 0.005, 1e-9):
                formulas.append(
                    (left.start(1), equals.end() + right.end(), inputs, claimed)
                )
        return formulas

    def _derivation_justified_values(
        self,
        text: str,
        records: Sequence[EvidenceRecord],
        symbol: str | None,
    ) -> list[float]:
        """Every operand and result of every valid formula in ``text``.

        The flat form, for the callers that only ask "is this figure one a
        formula here justifies" without a position to check it at.
        """
        justified: list[float] = []
        for _, _, inputs, result in self._derivation_formulas(text, records, symbol):
            justified.extend(inputs)
            justified.append(result)
        return justified

    @staticmethod
    def _evaluate_formula(expression: str) -> tuple[float, list[float]] | None:
        """Evaluate a numeric ``+ - * /`` expression without executing code."""
        normalized = expression.replace("×", "*").replace("÷", "/").replace(",", "").strip()
        try:
            tree = ast.parse(normalized, mode="eval")
        except (SyntaxError, ValueError):
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
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div),
            ):
                left_value = visit(node.left)
                right_value = visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left_value + right_value
                if isinstance(node.op, ast.Sub):
                    return left_value - right_value
                if isinstance(node.op, ast.Mult):
                    return left_value * right_value
                if right_value == 0:
                    raise ValueError("division by zero")
                return left_value / right_value
            raise ValueError("unsupported formula")

        try:
            value = visit(tree)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        if len(inputs) < 2 or not math.isfinite(value):
            return None
        return value, inputs

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
