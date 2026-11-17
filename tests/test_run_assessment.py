"""Test that the fake run script works end to end. No model, no network."""

import json

from scripts.run_assessment import main


def test_fake_run_writes_a_valid_audit_record(tmp_path):
    path = main(tmp_path / "audit")

    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["assessment_id"] == "demo-run"
    assert record["policy_documents"][0]["name"] == "demo_policy.txt"
    assert record["alignment_result"] == {"findings": [], "llm_failures": 0}

    steps = {entry["step"]: entry["status"] for entry in record["step_log"]}
    assert steps["agent3_alignment"] == "completed"
    assert "We keep your data for two years." not in json.dumps(
        record["policy_documents"]
    )