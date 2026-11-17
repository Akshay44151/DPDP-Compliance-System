from agents.legal_rag.schemas import LegalAnalysisResult, LegalAssessmentStatus as S
from agents.report_agent.schemas import (
    ComplianceReport,
    ReportGenerationRequest,
    ReportRecommendation,
)


class ReportGenerationAgent:
    """Agent 5: builds a structured report from validated assessment results."""

    DISCLAIMER = (
        "This report is an automated informational assessment, not legal advice "
        "or a legal certification. Findings should be reviewed by qualified personnel."
    )

    def generate(self, request: ReportGenerationRequest) -> ComplianceReport:
        if not isinstance(request.legal_analysis, LegalAnalysisResult):
            raise TypeError("legal_analysis must be a LegalAnalysisResult.")

        analysis = request.legal_analysis
        recommendations = []

        for finding in analysis.findings:
            if finding.status == S.POTENTIAL_NON_COMPLIANCE:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="high",
                    action="Review the identified implementation gap and plan corrective measures.",
                    reason=finding.legal_reasoning,
                ))
            elif finding.status == S.INSUFFICIENT_EVIDENCE:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="medium",
                    action="Collect additional implementation evidence and repeat the assessment.",
                    reason=finding.legal_reasoning,
                ))
            elif finding.status == S.REQUIRES_REVIEW:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="high",
                    action="Refer this finding for qualified legal or compliance review.",
                    reason=finding.legal_reasoning,
                ))
            elif finding.status == S.POTENTIAL_COMPLIANCE:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="low",
                    action="Retain supporting evidence and periodically revalidate the implementation.",
                    reason=finding.legal_reasoning,
                ))

        coverage = request.score.assessment_coverage_percent
        coverage_text = (
            f"{coverage:.2f}%"
            if coverage is not None
            else "unavailable"
        )

        review_count = request.score.requires_review_count

        if request.score.score is None:
            summary = (
                "No provisional compliance score could be calculated because "
                "there were no findings with an assessable outcome. "
                f"Assessment coverage: {coverage_text}. "
                f"{review_count} findings require review."
            )
        else:
            summary = (
                f"The provisional assessment score is {request.score.score:.2f}% "
                f"with assessment coverage of {coverage_text}, based on "
                f"{request.score.assessed_findings} assessable findings. "
                f"{review_count} findings require review. "
                f"{request.score.insufficient_evidence_count} findings have "
                "insufficient evidence and are excluded from the score."
            )

        return ComplianceReport(
            assessment_id=request.assessment_id,
            score=request.score,
            findings=analysis.findings,
            recommendations=recommendations,
            sources_consulted=analysis.sources_consulted,
            executive_summary=summary,
            disclaimer=self.DISCLAIMER,
            failed_calls=analysis.failed_calls,
            retrieval_failures=analysis.retrieval_failures,
        )
