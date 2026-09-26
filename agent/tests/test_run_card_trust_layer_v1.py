import hashlib
import json

from backtest.run_card import write_run_card


def test_run_card_v1_hashes_tool_inputs_and_cites_metric_artifact(tmp_path) -> None:
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "artifacts"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text("sharpe\n1.23\n", encoding="utf-8")
    trace = {
        "tool": "backtest",
        "args": {"run_dir": "/private/local/run", "api_key": "trace-secret"},
        "started_at": "2026-09-27T00:00:00Z",
        "ended_at": "2026-09-27T00:00:01Z",
        "status": "ok",
        "result": {"sharpe": 1.23, "payload": "raw-result"},
    }

    card = write_run_card(
        run_dir,
        {"codes": ["BIL"]},
        {"sharpe": 1.23},
        tool_traces=[trace],
    )

    trace_record = card["tool_traces"][0]
    assert card["schema_version"] == "1.0"
    assert set(trace_record) == {
        "tool",
        "args_hash",
        "started_at",
        "ended_at",
        "status",
        "result_hash",
    }
    assert trace_record["tool"] == "backtest"
    assert trace_record["started_at"] == trace["started_at"]
    assert trace_record["ended_at"] == trace["ended_at"]
    assert trace_record["status"] == "ok"
    assert len(trace_record["args_hash"]) == 64
    assert len(trace_record["result_hash"]) == 64
    for key in ("args", "result"):
        canonical = json.dumps(
            trace[key],
            sort_keys=True,
            default=str,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert trace_record[f"{key}_hash"] == expected_hash
    assert card["citations"] == [
        {"metric": "sharpe", "artifact_id": "artifacts/metrics.csv"}
    ]

    serialized = json.dumps(card)
    on_disk = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    assert on_disk == card
    assert "/private/local/run" not in serialized
    assert "trace-secret" not in serialized
    assert "raw-result" not in serialized

    same_payload_different_key_order = {
        **trace,
        "args": {"api_key": "trace-secret", "run_dir": "/private/local/run"},
        "result": {"payload": "raw-result", "sharpe": 1.23},
    }
    reordered_card = write_run_card(
        tmp_path / "reordered",
        {"codes": ["BIL"]},
        {"sharpe": 1.23},
        tool_traces=[same_payload_different_key_order],
    )
    assert reordered_card["tool_traces"][0]["args_hash"] == trace_record["args_hash"]
    assert (
        reordered_card["tool_traces"][0]["result_hash"] == trace_record["result_hash"]
    )

    same_shape_trace = {
        **trace,
        "args": {**trace["args"], "_private_state": "first"},
        "result": {**trace["result"], "_private_state": "first"},
    }
    same_shape_card = write_run_card(
        tmp_path / "same-shape",
        {"codes": ["BIL"]},
        {"sharpe": 1.23},
        tool_traces=[same_shape_trace],
    )
    changed_private_trace = {
        **same_shape_trace,
        "args": {**same_shape_trace["args"], "_private_state": "different"},
        "result": {**same_shape_trace["result"], "_private_state": "different"},
    }
    changed_card = write_run_card(
        tmp_path / "other",
        {"codes": ["BIL"]},
        {"sharpe": 1.23},
        tool_traces=[changed_private_trace],
    )
    assert "citations" not in changed_card
    assert (
        changed_card["tool_traces"][0]["args_hash"]
        != same_shape_card["tool_traces"][0]["args_hash"]
    )
    assert (
        changed_card["tool_traces"][0]["result_hash"]
        != same_shape_card["tool_traces"][0]["result_hash"]
    )
