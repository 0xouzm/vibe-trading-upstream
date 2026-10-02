import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.loop import AgentLoop
from src.agent.memory import WorkspaceMemory


def test_successful_strategy_file_write_records_the_active_model(
    tmp_path: Path,
) -> None:
    loop = object.__new__(AgentLoop)
    loop.memory = WorkspaceMemory(run_dir=str(tmp_path))
    loop._written_files = set()
    loop._active_model_id = "gpt-5.6-luna"
    loop._active_model_source = "configured"
    loop._llm_runtime = SimpleNamespace(
        provider="openai",
        configured_model="gpt-5.6-luna",
    )

    loop._record_written_target({"path": "config.json"})
    loop._llm_runtime = SimpleNamespace(
        provider="openrouter",
        configured_model="anthropic/claude-sonnet",
    )
    loop._active_model_id = "claude-sonnet-4-5"
    loop._active_model_source = "provider_response"
    loop._record_written_target({"path": "code/signal_engine.py"})

    provenance = json.loads(
        (tmp_path / "strategy_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance == {
        "files": {
            "config.json": {
                "provider": "openai",
                "model_id": "gpt-5.6-luna",
                "model_source": "configured",
            },
            "code/signal_engine.py": {
                "provider": "openrouter",
                "model_id": "claude-sonnet-4-5",
                "model_source": "provider_response",
            },
        }
    }


def test_strategy_file_write_on_a_constructed_loop_records_the_configured_model(
    tmp_path: Path,
) -> None:
    """A real AgentLoop must carry the provenance fields from construction on.

    The test above sets them by hand; this one goes through ``__init__`` so a
    loop that never assigns them fails here instead of in a user's backtest.
    """
    from src.agent.tools import ToolRegistry

    loop = AgentLoop(
        registry=ToolRegistry(),
        llm=None,
        memory=WorkspaceMemory(run_dir=str(tmp_path)),
    )

    loop._record_written_target({"path": "config.json"})

    provenance = json.loads(
        (tmp_path / "strategy_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["files"]["config.json"] == {
        "provider": loop._llm_runtime.provider or None,
        "model_id": loop._llm_runtime.configured_model or None,
        "model_source": "configured",
    }
