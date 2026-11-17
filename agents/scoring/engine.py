from agents.legal_rag.schemas import LegalAnalysisResult, LegalAssessmentStatus
from agents.scoring.schemas import ComplianceScore


class ComplianceScoringEngine:
    """Calculates a transparent, provisional score from Agent 4 findings."""

    def calculate(self, result: LegalAnalysisResult) -> ComplianceScore:
        findings = result.findings

        counts = {
            status: sum(1 for finding in findings if finding.status == status)
            for status in LegalAssessmentStatus
        }

        compliant = counts[LegalAssessmentStatus.POTENTIAL_COMPLIANCE]
        non_compliant = counts[LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE]
        insufficient = counts[LegalAssessmentStatus.INSUFFICIENT_EVIDENCE]
        review = counts[LegalAssessmentStatus.REQUIRES_REVIEW]
        not_applicable = counts[LegalAssessmentStatus.NOT_APPLICABLE]

        assessed = compliant + non_compliant
        applicable = len(findings) - not_applicable

        score = round((compliant / assessed) * 100, 2) if assessed else None
        coverage = (
            round((assessed / applicable) * 100, 2)
            if applicable else None
        )

        return ComplianceScore(
            score=score,
            total_findings=len(findings),
            assessed_findings=assessed,
            potential_compliance_count=compliant,
            potential_non_compliance_count=non_compliant,
            insufficient_evidence_count=insufficient,
            requires_review_count=review,
            not_applicable_count=not_applicable,
            assessment_coverage_percent=coverage,
            score_description=(
                "Provisional indicator based only on findings with a potential "
                "compliance or potential non-compliance assessment. It is not "
                "a legal certification. Insufficient evidence and review "
                "findings are excluded from the score."
            ),
        )
