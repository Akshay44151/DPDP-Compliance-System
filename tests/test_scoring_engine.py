from agents.legal_rag.schemas import (
    LegalAnalysisResult,
    LegalAssessmentStatus as S,
    LegalFinding,
)
from agents.scoring.engine import ComplianceScoringEngine


def finding(status, fid):
    return LegalFinding(
        alignment_finding_id=fid,
        status=status,
        legal_reasoning="Test finding",
    )


def test_calculates_score_and_coverage_correctly():
    result = LegalAnalysisResult(
        findings=[
            finding(S.POTENTIAL_COMPLIANCE, "A1"),
            finding(S.POTENTIAL_NON_COMPLIANCE, "A2"),
            finding(S.INSUFFICIENT_EVIDENCE, "A3"),
            finding(S.REQUIRES_REVIEW, "A4"),
            finding(S.NOT_APPLICABLE, "A5"),
        ]
    )

    score = ComplianceScoringEngine().calculate(result)

    assert score.score == 50.0
    assert score.assessed_findings == 2
    assert score.assessment_coverage_percent == 50.0
    assert score.insufficient_evidence_count == 1
    assert score.requires_review_count == 1
    assert score.not_applicable_count == 1


def test_no_score_when_all_findings_are_insufficient():
    result = LegalAnalysisResult(
        findings=[
            finding(S.INSUFFICIENT_EVIDENCE, "A1"),
            finding(S.INSUFFICIENT_EVIDENCE, "A2"),
        ]
    )

    score = ComplianceScoringEngine().calculate(result)

    assert score.score is None
    assert score.assessed_findings == 0
    assert score.assessment_coverage_percent == 0.0


def test_empty_assessment_returns_no_score():
    result = LegalAnalysisResult(findings=[])

    score = ComplianceScoringEngine().calculate(result)

    assert score.score is None
    assert score.total_findings == 0
    assert score.assessment_coverage_percent is None
