"""The generated broker capability matrix must match the profile registry (#1626)."""

from __future__ import annotations

from src.trading.capability_matrix import README_PATH, render, render_block, sync_readme
from src.trading.profiles import BUILTIN_PROFILES


def test_capability_matrix_block_in_readme_is_current() -> None:
    """CI drift guard: the README's marked block equals a fresh render.

    Fails red when a connector profile changed without regenerating:
    python -m src.trading.capability_matrix
    """
    text = README_PATH.read_text(encoding="utf-8")
    assert render_block() in text, (
        "README's broker-capability-matrix block is stale; regenerate with python -m src.trading.capability_matrix"
    )


def test_capability_matrix_covers_every_connector() -> None:
    rendered = render()
    rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("| ") and "---" not in line and "Connector" not in line
    ]
    connectors = {p.connector for p in BUILTIN_PROFILES}
    assert len(rows) == len(connectors)
    for connector in connectors:
        assert f"| {connector} |" in rendered


def test_sync_readme_replaces_only_the_marked_block() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    synced = sync_readme(text)
    assert synced == text  # already current
    # drift the block's content while keeping the markers, as a profile edit
    # without regeneration would
    without = text.replace(render(), "stale\n")
    assert sync_readme(without) == text  # regenerates back to current
