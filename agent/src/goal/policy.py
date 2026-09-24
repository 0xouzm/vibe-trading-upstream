"""Policy helpers for finance research goals."""

from __future__ import annotations

import re

_EXECUTION_PATTERNS = (
    re.compile(r"\b(place|submit|execute|send)\b.{0,40}\b(order|trade)\b", re.I),
    # No bare asset noun (shares/contracts/btc/...) here: this is a research
    # product whose whole purpose is evaluating whether to buy/sell/short a
    # security, so "buy NVDA shares" alone in a research objective is the
    # normal case, not an execution request. Only fire on an actual
    # execution-imminent cue alongside the buy/sell verb.
    re.compile(
        r"\b(buy|sell|short|long)\b.{0,40}\b(now|immediately|right away|market order|limit order)\b",
        re.I,
    ),
    re.compile(r"(下单|市价单|限价单|马上买|立即买|现在买|马上卖|立即卖|现在卖)"),
)


def normalize_required_text(value: str, field_name: str) -> str:
    """Strip and validate a required text field.

    Args:
        value: User supplied text.
        field_name: Field name for the error message.

    Returns:
        The stripped value.

    Raises:
        ValueError: If the stripped value is empty.
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def reject_live_execution_objective(objective: str) -> None:
    """Reject direct live-trading or order-execution goal text.

    Args:
        objective: Research goal objective.

    Raises:
        ValueError: If the objective looks like an execution request.
    """
    text = objective.strip()
    for pattern in _EXECUTION_PATTERNS:
        if pattern.search(text):
            raise ValueError("live trading or execution goals are not supported")
