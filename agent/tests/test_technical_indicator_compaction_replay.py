from __future__ import annotations

from src.agent.loop import AgentLoop
from src.tools.technical_indicator_tool import TechnicalIndicatorTool


def test_technical_indicators_opt_into_readonly_compaction_replay() -> None:
    """The real technical-indicator tool can restore an identical lost result.

    Technical indicators are read-only but repeatable, so the generic replay
    gate requires the explicit replay_after_compaction opt-in added for this
    tool. Keep this test on the concrete class so that removing or renaming the
    contract cannot silently return identical post-compaction calls to a fresh
    market-data fetch.
    """
    tool = TechnicalIndicatorTool()
    loop = object.__new__(AgentLoop)

    assert tool.is_readonly is True
    assert tool.repeatable is True
    assert tool.replay_after_compaction is True
    assert (
        loop._readonly_replay_allowed(
            tool,
            {"symbol": "AAPL", "interval": "1d", "lookback": 200},
        )
        is True
    )
