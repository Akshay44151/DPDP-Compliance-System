"""Generate a structured, readable PDF from a DPDP audit record."""

from io import BytesIO
from xml.sax.saxutils import escape


def make_pdf(data):
    """Create a concise report using final evidence-gated legal findings."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("reportlab"):
            raise RuntimeError(
                "PDF export requires ReportLab. "
                "Install dependencies with pip install -r requirements.txt."
            ) from error
        raise

    if not isinstance(data, dict):
        raise ValueError("The audit record must be a JSON object.")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="DPDP Compliance Assessment Report",
        author="DPDP Compliance Assessment System",
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "DPDPTitle", parent=base["Title"], fontSize=19,
            leading=24, alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"), spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "DPDPSection", parent=base["Heading1"], fontSize=13,
            leading=17, textColor=colors.HexColor("#17365D"),
            spaceBefore=13, spaceAfter=7, keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "DPDPSubsection", parent=base["Heading2"], fontSize=10.5,
            leading=14, textColor=colors.HexColor("#24527A"),
            spaceBefore=8, spaceAfter=4, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "DPDPBody", parent=base["BodyText"], fontSize=8.5,
            leading=12, spaceAfter=5, splitLongWords=True,
        ),
        "small": ParagraphStyle(
            "DPDPSmall", parent=base["BodyText"], fontSize=7.5,
            leading=10, textColor=colors.HexColor("#555555"),
            spaceAfter=3,
        ),
    }

    story = [
        Paragraph("DPDP Compliance Assessment Report", styles["title"]),
        Paragraph(
            "Automated assessment results and supporting evidence",
            styles["small"],
        ),
        Spacer(1, 8),
    ]

    def text(value, key=None):
        if value is None:
            value = "Not available"
        elif isinstance(value, bool):
            value = "Yes" if value else "No"
        elif isinstance(value, dict):
            parts = []
            for item_key, item_value in value.items():
                formatted = text(item_value, key=item_key)
                parts.append(
                    f"{item_key.replace('_', ' ').title()}: {formatted}"
                )
            value = "; ".join(parts) if parts else "None"
        elif isinstance(value, list):
            value = "; ".join(text(item) for item in value) if value else "None"
        else:
            value = str(value)

        if key in ("target", "project_path", "path"):
            lowered = str(value).lower()
            if (
                "/tmp/" in lowered
                or "\\appdata\\local\\temp\\" in lowered
                or lowered.startswith("/tmp")
            ):
                value = "Temporary workspace (path hidden)"

        return escape(str(value)).replace("\n", "<br/>")

    def section(title):
        story.append(Paragraph(text(title), styles["section"]))

    def subsection(title):
        story.append(Paragraph(text(title), styles["subsection"]))

    def field(label, value):
        story.append(Paragraph(
            f"<b>{text(label)}:</b> {text(value)}",
            styles["body"],
        ))

    def bullet_items(items, empty_message="No entries recorded."):
        if not isinstance(items, list) or not items:
            story.append(Paragraph(empty_message, styles["small"]))
            return
        for item in items:
            if isinstance(item, dict):
                parts = []
                for key, value in item.items():
                    if key in ("project_path", "path"):
                        continue
                    parts.append(
                        f"<b>{text(key.replace('_', ' ').title())}:</b> "
                        f"{text(value, key=key)}"
                    )
                if parts:
                    story.append(Paragraph(" • ".join(parts), styles["body"]))
            else:
                story.append(Paragraph("• " + text(item), styles["body"]))

    # 1. Assessment overview
    section("1. Assessment Overview")
    field("Assessment ID", data.get("assessment_id"))
    field("Created at", data.get("created_at"))

    score = data.get("compliance_score")
    score = score if isinstance(score, dict) else {}

    metrics = [
        ["Metric", "Result"],
        ["Total findings", score.get("total_findings", "Not available")],
        ["Assessed findings", score.get("assessed_findings", "Not available")],
        ["Assessment coverage", str(score.get("assessment_coverage_percent", "Not available")) + "%"],
        ["Minimum coverage required", str(score.get("minimum_coverage_percent", "Not available")) + "%"],
        ["Potential compliance", score.get("potential_compliance_count", 0)],
        ["Potential non-compliance", score.get("potential_non_compliance_count", 0)],
        ["Insufficient evidence", score.get("insufficient_evidence_count", 0)],
        ["Requires review", score.get("requires_review_count", 0)],
        ["Not applicable", score.get("not_applicable_count", 0)],
    ]

    overview_table = Table(metrics, colWidths=[75 * mm, 90 * mm], repeatRows=1)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 7))

    if score.get("score") is None:
        reason = score.get(
            "score_withheld_reason",
            "No findings have a legally assessable outcome.",
        )
        story.append(Paragraph(
            "Score withheld " + chr(0x2014) + " " + text(reason),
            styles["body"],
        ))
    else:
        field("Compliance score", score.get("score"))

    # 2. Final legal findings - gated analysis is the sole source.
    gated = data.get("legal_analysis_gated")
    gated = gated if isinstance(gated, dict) else {}
    findings = gated.get("findings")
    findings = findings if isinstance(findings, list) else []

    section("2. Final Legal Findings")
    story.append(Paragraph(
        "The findings below reflect the final evidence-gated legal analysis. "
        "Preliminary findings are not presented as final outcomes.",
        styles["small"],
    ))

    if findings:
        rows = [["Finding ID", "Final status"]]
        for index, finding in enumerate(findings, start=1):
            finding = finding if isinstance(finding, dict) else {}
            rows.append([
                str(finding.get("alignment_finding_id", f"Finding {index}")),
                str(finding.get("status", "Not available")),
            ])

        findings_table = Table(
            rows, repeatRows=1, colWidths=[65 * mm, 100 * mm]
        )
        findings_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C9D6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(findings_table)

        for index, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                continue

            finding_id = finding.get(
                "alignment_finding_id", f"Finding {index}"
            )
            subsection(
                f"{finding_id} - {finding.get('status', 'Not available')}"
            )

            for key, label in [
                ("legal_reasoning", "Legal reasoning"),
                ("applicable_provisions", "Applicable provisions"),
                ("evidence_fact_ids", "Evidence references"),
                ("citations", "Legal citations"),
                ("confidence", "Confidence"),
                ("limitations", "Limitations"),
            ]:
                if key in finding:
                    field(label, finding[key])
    else:
        story.append(Paragraph(
            "No final legal findings are available.",
            styles["body"],
        ))

    # 3. Policy extraction
    section("3. Privacy Policy Analysis")
    claims = data.get("policy_claims", [])
    rejected = data.get("rejected_claims", [])
    documents = data.get("policy_documents", [])

    field("Verified policy claims", len(claims) if isinstance(claims, list) else 0)
    field("Rejected claims", len(rejected) if isinstance(rejected, list) else 0)

    if documents:
        subsection("Policy documents")
        bullet_items(documents)

    if claims:
        subsection("Extracted policy claims")
        bullet_items(claims)

    if rejected:
        subsection("Rejected claims")
        bullet_items(rejected)

    # 4. Implementation evidence
    section("4. Implementation Evidence")
    implementation = data.get("implementation_report")
    implementation = implementation if isinstance(implementation, dict) else {}

    field("Files scanned", implementation.get("files_scanned"))
    field("Coverage", implementation.get("coverage"))
    field("LLM retries", implementation.get("llm_retries", 0))
    field("LLM failures", implementation.get("llm_failures", 0))

    evidence = implementation.get("evidence", [])
    if evidence:
        subsection("Evidence items")
        bullet_items(evidence)

    facts = implementation.get("facts", [])
    if facts:
        subsection("Implementation facts")
        bullet_items(facts)

    failed_categories = implementation.get("failed_categories", [])
    if failed_categories:
        subsection("Categories not successfully analysed")
        bullet_items(failed_categories)

    # 5. Evidence gate
    section("5. Evidence Gate Results")
    evidence_gate = data.get("evidence_gate")
    evidence_gate = evidence_gate if isinstance(evidence_gate, dict) else {}

    counts = evidence_gate.get("counts")
    if counts:
        field("Gate counts", counts)

    decisions = evidence_gate.get("decisions", [])
    if decisions:
        subsection("Gate decisions")
        bullet_items(decisions)

    gate_warnings = evidence_gate.get("warnings", [])
    if gate_warnings:
        subsection("Evidence gate warnings")
        bullet_items(gate_warnings)

    gate_application = data.get("legal_gate_application")
    gate_application = gate_application if isinstance(gate_application, dict) else {}
    changes = gate_application.get("changes", [])
    if changes:
        subsection("Applied legal gate changes")
        bullet_items(changes)

    # 6. Recommendations - prefer compliance report recommendations.
    section("6. Recommendations")
    compliance = data.get("compliance_report")
    compliance = compliance if isinstance(compliance, dict) else {}
    recommendations = compliance.get("recommendations", [])

    if recommendations:
        bullet_items(recommendations)
    else:
        story.append(Paragraph(
            "No recommendations were recorded.",
            styles["body"],
        ))

    # 7. Sources and execution
    section("7. Sources and Execution Summary")

    sources = gated.get("sources_consulted", [])
    if sources:
        subsection("Sources consulted")
        bullet_items(sources)
    else:
        story.append(Paragraph(
            "No external legal sources were recorded as consulted.",
            styles["small"],
        ))

    step_log = data.get("step_log", [])
    if step_log:
        subsection("Assessment execution steps")
        bullet_items(step_log)

    summary = data.get("summary")
    summary = summary if isinstance(summary, dict) else {}

    for key, label in [
        ("is_complete", "Assessment complete"),
        ("report_generated", "Report generated"),
        ("failed_steps", "Failed steps"),
        ("warnings", "Warnings"),
        ("incomplete_reasons", "Incomplete reasons"),
    ]:
        if key in summary:
            field(label, summary[key])

    failed_calls = gated.get("failed_calls", [])
    retrieval_failures = gated.get("retrieval_failures", [])

    if failed_calls:
        subsection("Failed legal analysis calls")
        bullet_items(failed_calls)

    if retrieval_failures:
        subsection("Legal retrieval failures")
        bullet_items(retrieval_failures)

    # Disclaimer
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This is an automated informational assessment, "
        "not legal advice or a legal certification. Findings, evidence and "
        "source citations should be reviewed by qualified personnel.",
        styles["small"],
    ))

    doc.build(story)
    return buffer.getvalue()
