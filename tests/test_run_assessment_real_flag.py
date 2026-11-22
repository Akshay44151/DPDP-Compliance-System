
"""Tests for real-mode CLI and full-real pipeline wiring.

All model and external dependencies are mocked. No API calls are made.
"""

import scripts.run_assessment as runner
from types import SimpleNamespace


def test_cli_dispatches_to_full_real(monkeypatch):
    calls = []

    def fake_full_real(project, policy, output_dir):
        calls.append((project, policy, output_dir))
        return 0

    monkeypatch.setattr(runner, "run_full_real", fake_full_real)

    result = runner.cli([
        "--full-real",
        "--project", "sample_project",
        "--policy", "sample_policy.txt",
        "sample_output",
    ])

    assert result == 0
    assert calls == [
        ("sample_project", "sample_policy.txt", "sample_output")
    ]


def test_cli_dispatches_to_real(monkeypatch):
    calls = []

    def fake_real(project, policy, output_dir):
        calls.append((project, policy, output_dir))
        return 0

    monkeypatch.setattr(runner, "run_real", fake_real)

    result = runner.cli([
        "--real",
        "--project", "sample_project",
        "--policy", "sample_policy.txt",
    ])

    assert result == 0
    assert calls == [
        ("sample_project", "sample_policy.txt", None)
    ]


def test_cli_rejects_both_real_modes():
    try:
        runner.cli(["--real", "--full-real"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError(
            "--real and --full-real should be mutually exclusive"
        )


def test_full_real_pipeline_wires_agents_without_external_calls(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "sample_project"
    project.mkdir()

    policy = tmp_path / "sample_policy.txt"
    policy.write_text(
        "We keep personal data for two years.",
        encoding="utf-8",
    )

    model = object()
    retriever = object()
    calls = {}

    monkeypatch.setattr(
        runner,
        "_load_model",
        lambda: model,
    )

    # Replace all real agent constructors with harmless test doubles.
    monkeypatch.setattr(
        "agents.policy_agent.agent.PolicyExtractionAgent",
        lambda supplied_model: ("policy_agent", supplied_model),
    )
    monkeypatch.setattr(
        "agents.implementation_agent.agent.ImplementationExtractionAgent",
        lambda supplied_model: ("implementation_agent", supplied_model),
    )
    monkeypatch.setattr(
        "agents.alignment_agent.agent.AlignmentAgent",
        lambda supplied_model: ("alignment_agent", supplied_model),
    )
    monkeypatch.setattr(
        "agents.legal_rag.retriever.LegalRetriever",
        lambda: retriever,
    )
    monkeypatch.setattr(
        "agents.legal_rag.agent.LegalAnalysisAgent",
        lambda retriever=None: ("legal_agent", retriever),
    )

    def fake_analysis_stages(
        state,
        policy_agent,
        implementation_factory,
        alignment_factory,
    ):
        calls["analysis"] = {
            "state": state,
            "policy_agent": policy_agent,
            "implementation_agent": implementation_factory(),
            "alignment_agent": alignment_factory(),
        }

    def fake_final_stage(
        *,
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    ):
        calls["final"] = {
            "state": state,
            "legal_agent": legal_agent,
            "scoring_engine": scoring_engine,
            "report_agent": report_agent,
        }

    def fake_finish(state, output_dir, heading, default_folder):
        calls["finish"] = {
            "state": state,
            "output_dir": output_dir,
            "heading": heading,
            "default_folder": default_folder,
        }

    monkeypatch.setattr(
        runner,
        "run_analysis_stages",
        fake_analysis_stages,
    )
    monkeypatch.setattr(
        runner,
        "run_final_assessment_stage",
        fake_final_stage,
    )
    monkeypatch.setattr(runner, "_finish", fake_finish)

    monkeypatch.setattr(
        runner,
        "summarize_assessment",
        lambda state: SimpleNamespace(is_complete=True),
    )

    result = runner.run_full_real(
        project,
        policy,
        tmp_path / "output",
    )

    assert result == 0

    analysis = calls["analysis"]
    assert analysis["policy_agent"] == ("policy_agent", model)
    assert analysis["implementation_agent"] == (
        "implementation_agent",
        model,
    )
    assert analysis["alignment_agent"] == ("alignment_agent", model)

    final = calls["final"]
    assert final["state"] is analysis["state"]
    assert final["legal_agent"] == ("legal_agent", retriever)

    assert isinstance(final["scoring_engine"], runner.ComplianceScoringEngine)
    assert isinstance(final["report_agent"], runner.ReportGenerationAgent)

    finish = calls["finish"]
    assert finish["state"] is analysis["state"]
    assert finish["output_dir"] == tmp_path / "output"
    assert finish["heading"] == "FULL REAL pipeline run finished."
    assert finish["default_folder"] == "dpdp_audit_full_real"
    monkeypatch.setattr(
        runner,
        "summarize_assessment",
        lambda state: SimpleNamespace(is_complete=False),
    )

    result = runner.run_full_real(
        project,
        policy,
        tmp_path / "output",
    )

    assert result == 1



"""Test that the fake run script works end to end. No model, no network."""

import json

from scripts.run_assessment import main


def test_fake_run_writes_a_valid_audit_record(tmp_path):
    path = main(tmp_path / "audit")

    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["assessment_id"] == "demo-run"
    assert record["policy_documents"][0]["name"] == "demo_policy.txt"

    # Check the important Agent 3 fields without rejecting valid
    # additional fields from the expanded AlignmentResult schema.
    alignment = record["alignment_result"]
    assert alignment["findings"] == []
    assert alignment["llm_failures"] == 0
    assert alignment["failed_calls"] == []
    assert alignment["policy_text_truncated"] is False

    # The final stages should also be represented in the audit.
    assert record["legal_analysis_original"] is not None
    assert record["legal_analysis_gated"] is not None
    assert record["compliance_score"] is not None
    assert record["compliance_report"] is not None

    steps = {
        entry["step"]: entry["status"]
        for entry in record["step_log"]
    }
    assert steps["agent3_alignment"] == "completed"
    assert steps["legal_analysis"] == "completed"
    assert steps["compliance_scoring"] == "completed"
    assert steps["report_generation"] == "completed"

    assert "We keep your data for two years." not in json.dumps(
        record["policy_documents"]
    )