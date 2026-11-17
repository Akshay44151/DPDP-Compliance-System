"""Run the front half of the pipeline with FAKE agents (Phase A).

No model, no key, no network. It proves that the orchestrator pieces fit
together: Agents 1 and 2 in parallel, then Agent 3, then the summary and
the audit record.

Run from the project root:
    python -m scripts.run_assessment
    python -m scripts.run_assessment some\\folder     (where to write the audit file)

By default the audit file goes to a folder inside your system temp folder,
so nothing is added to the repository.
"""

import sys
import tempfile
from pathlib import Path

from agents.policy_agent.schemas import (
    PolicyCategory,
    PolicyClaim,
    PolicyExtractionResult,
)
from agents.policy_agent.verification import ExtractionReport
from orchestrator.analysis_stage import run_analysis_stages
from orchestrator.audit import build_audit_record, write_audit_record
from orchestrator.models import AssessmentRequest, AssessmentState, PolicyDocument
from orchestrator.summary import summarize_assessment

POLICY_NAME = "demo_policy.txt"
POLICY_TEXT = "We keep your data for two years."


class FakeExtractor:
    """Stands in for Agent 1. Returns one fixed claim."""

    def extract_with_report(self, *, document_text: str, source: str):
        claim = PolicyClaim(
            claim="Data is kept for two years.",
            category=PolicyCategory.OTHER,
            evidence="We keep your data for two years.",
            source=source,
        )
        return ExtractionReport(result=PolicyExtractionResult(claims=[claim]))


class FakeImplementationReport:
    """Stands in for Agent 2's report."""

    llm_failures = 0
    llm_retries = 0
    facts: list = []

    def model_dump(self, *, mode: str = "python"):
        return {"llm_failures": 0, "facts": []}


class FakeImplementationAgent:
    def run(self, project_path):
        return FakeImplementationReport()


class FakeAlignmentResult:
    """Stands in for Agent 3's result. No findings."""

    llm_failures = 0
    findings: list = []

    def model_dump(self, *, mode: str = "python"):
        return {"findings": [], "llm_failures": 0}


class FakeAlignmentAgent:
    def run(self, policy, implementation, *, policy_text=None):
        return FakeAlignmentResult()


def main(output_dir: str | Path | None = None) -> Path:
    """Run the fake pipeline, print the summary, write the audit file."""
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="demo-run",
            policy_documents=[PolicyDocument(name=POLICY_NAME, text=POLICY_TEXT)],
            project_path=".",
        )
    )

    run_analysis_stages(
        state,
        FakeExtractor(),
        FakeImplementationAgent,
        FakeAlignmentAgent,
    )

    summary = summarize_assessment(state)
    record = build_audit_record(state, summary)

    folder = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "dpdp_audit_demo"
    path = write_audit_record(record, folder)

    print("FAKE agents only. No model was called.")
    print()
    for entry in state.step_log:
        print(f"{entry.step} [{entry.target}] {entry.status.value} (attempt {entry.attempt})")
    print()
    print(summary.model_dump_json(indent=2))
    print()
    print(f"Audit record written to: {path}")
    return path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)