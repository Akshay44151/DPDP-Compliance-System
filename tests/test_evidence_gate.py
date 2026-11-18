"""Tests for orchestrator/evidence_gate.py. No model, key or network needed."""

from orchestrator.evidence_gate import (
    BLOCKED,
    NOT_GATED,
    PASS,
    REVIEW_ONLY,
    evaluate_findings,
    worst_decision,
)

CATEGORY_KINDS = {
    "consent": ["consent_logic", "consent_ui", "data_write", "outbound_email"],
    "data_minimization": ["model_field", "form_field", "form_input"],
}


def fact(fid, fact_type="present", category="consent"):
    return {"id": fid, "category": category, "fact_type": fact_type}


def report(**overrides):
    base = {
        "files_scanned": 12,
        "coverage": {"consent_logic": 2, "data_write": 3},
        "facts": [fact("F-CONSENT-01", "present")],
        "llm_failures": 0,
        "failed_categories": [],
    }
    base.update(overrides)
    return base


def finding(fid="A-1", **overrides):
    base = {
        "id": fid,
        "kind": "claim_check",
        "category": "consent",
        "status": "violated",
        "fact_ids": ["F-CONSENT-01"],
    }
    base.update(overrides)
    return base


def result(*findings, **overrides):
    base = {
        "findings": list(findings),
        "failed_calls": [],
        "policy_text_truncated": False,
    }
    base.update(overrides)
    return base


def gate(impl, align):
    return evaluate_findings(impl, align, CATEGORY_KINDS)


def test_present_fact_on_clean_scan_passes():
    out = gate(report(), result(finding()))
    assert out.decision_for("A-1").decision == PASS
    assert out.warnings == ()


def test_partial_status_is_gated_like_violated():
    impl = report(facts=[fact("F-CONSENT-01", "absent")])
    out = gate(impl, result(finding(status="partial")))
    assert out.decision_for("A-1").decision == REVIEW_ONLY


def test_absent_only_facts_are_review_only():
    impl = report(facts=[fact("F-CONSENT-01", "absent")])
    out = gate(impl, result(finding()))
    decision = out.decision_for("A-1")
    assert decision.decision == REVIEW_ONLY
    assert "absence" in decision.reasons[0]


def test_mixed_absent_and_present_facts_pass():
    impl = report(
        facts=[fact("F-CONSENT-01", "absent"), fact("F-CONSENT-02", "present")]
    )
    out = gate(impl, result(finding(fact_ids=["F-CONSENT-01", "F-CONSENT-02"])))
    assert out.decision_for("A-1").decision == PASS


def test_finding_without_facts_is_review_only():
    out = gate(report(), result(finding(fact_ids=[])))
    assert out.decision_for("A-1").decision == REVIEW_ONLY


def test_failed_category_is_blocked():
    impl = report(failed_categories=["consent"], llm_failures=1)
    out = gate(impl, result(finding()))
    decision = out.decision_for("A-1")
    assert decision.decision == BLOCKED
    assert any("consent" in r for r in decision.reasons)
    assert any("failed model calls" in w for w in out.warnings)


def test_failed_category_does_not_block_other_categories():
    impl = report(
        failed_categories=["data_minimization"],
        coverage={"consent_logic": 1},
    )
    out = gate(impl, result(finding()))
    assert out.decision_for("A-1").decision == PASS


def test_no_files_scanned_is_blocked():
    out = gate(report(files_scanned=0), result(finding()))
    assert out.decision_for("A-1").decision == BLOCKED


def test_truncation_boundary():
    at_cap = report(coverage={"consent_logic": 30, "data_write": 30})
    over_cap = report(coverage={"consent_logic": 30, "data_write": 31})
    assert gate(at_cap, result(finding())).decision_for("A-1").decision == PASS
    assert gate(over_cap, result(finding())).decision_for("A-1").decision == BLOCKED


def test_truncation_counts_only_the_categorys_own_kinds():
    impl = report(coverage={"consent_logic": 2, "model_field": 500})
    out = gate(impl, result(finding()))
    assert out.decision_for("A-1").decision == PASS


def test_unknown_fact_id_is_blocked():
    out = gate(report(), result(finding(fact_ids=["F-CONSENT-99"])))
    decision = out.decision_for("A-1")
    assert decision.decision == BLOCKED
    assert "F-CONSENT-99" in decision.reasons[0]


def test_failed_verdict_call_is_blocked():
    align = result(finding(), failed_calls=["verdict:consent"])
    assert gate(report(), align).decision_for("A-1").decision == BLOCKED


def test_unmapped_category_is_review_only():
    impl = report(facts=[fact("F-X-01", "present", "grievance_redressal")])
    f = finding(category="grievance_redressal", fact_ids=["F-X-01"])
    out = gate(impl, result(f))
    assert out.decision_for("A-1").decision == REVIEW_ONLY


def test_undeclared_and_other_statuses_are_not_gated():
    findings = [
        finding("U-1", kind="undeclared_behavior", status="undeclared", fact_ids=[]),
        finding("A-2", status="aligned"),
        finding("A-3", status="unverifiable", fact_ids=[]),
    ]
    out = gate(report(), result(*findings))
    assert [d.decision for d in out.decisions] == [NOT_GATED] * 3


def test_missing_agent2_report_blocks_gated_findings():
    out = gate(None, result(finding()))
    assert out.decision_for("A-1").decision == BLOCKED
    assert any("Agent 2 report is missing" in w for w in out.warnings)


def test_missing_agent3_result_gives_no_decisions():
    out = gate(report(), None)
    assert out.decisions == ()
    assert out.warnings


def test_run_level_warnings():
    align = result(
        finding(), policy_text_truncated=True, failed_calls=["undeclared"]
    )
    out = gate(report(), align)
    assert len(out.warnings) == 2
    assert any("policy text" in w for w in out.warnings)
    assert any("undeclared" in w for w in out.warnings)


def test_gate_does_not_change_its_inputs():
    impl, align = report(), result(finding())
    before = (repr(impl), repr(align))
    gate(impl, align)
    assert (repr(impl), repr(align)) == before


def test_worst_decision_and_to_dict():
    out = gate(
        report(),
        result(finding("A-1"), finding("A-2", fact_ids=[])),
    )
    assert worst_decision(out.decisions) == REVIEW_ONLY
    assert out.to_dict()["decisions"][1]["decision"] == REVIEW_ONLY