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

        if request.score.score is None:
            summary = (
                "No provisional compliance score could be calculated because "
                "there were no findings with an assessable outcome."
            )
        else:
            summary = (
                f"The provisional assessment score is {request.score.score:.2f}% "
                f"based on {request.score.assessed_findings} assessable findings. "
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
        )
