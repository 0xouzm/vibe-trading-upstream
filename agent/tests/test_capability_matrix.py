"""The generated broker capability matrix must match the profile registry (#1626)."""

from __future__ import annotations

from src.trading.capability_matrix import DOC_PATH, render
from src.trading.profiles import BUILTIN_PROFILES


def test_capability_matrix_is_current() -> None:
    """CI drift guard: the committed doc equals a fresh render.

    Fails red when a connector profile changed without regenerating:
    python -m src.trading.capability_matrix
    """
    assert DOC_PATH.is_file(), f"missing generated doc {DOC_PATH}"
    assert DOC_PATH.read_text(encoding="utf-8") == render(), (
        "docs/broker-capabilities.md is stale; regenerate with "
        "python -m src.trading.capability_matrix"
    )


def test_capability_matrix_covers_every_profile() -> None:
    rendered = render()
    rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("| ") and "---" not in line and "Connector" not in line
    ]
    assert len(rows) == len(BUILTIN_PROFILES)
    for profile in BUILTIN_PROFILES:
        assert f"| {profile.id} |" in rendered
        assert f"| {profile.connector} |" in rendered
