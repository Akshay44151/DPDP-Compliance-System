"""Run the front half of the pipeline: fake agents by default, real with --real.

Default (no model, no key, no network): fake agents prove that the pieces
fit together: Agents 1 and 2 in parallel, then Agent 3, then the summary
and the audit record.

With --real, Agents 1, 2 and 3 use the real model from your .env. This
makes API calls. It says nothing about Agents 4 and 5 (not included).

Run from the project root:
    python -m scripts.run_assessment
    python -m scripts.run_assessment some\\folder     (where to write the audit file)
    python -m scripts.run_assessment --real
    python -m scripts.run_assessment --real --project APP_FOLDER --policy POLICY.txt

By default the audit file goes to a folder inside your system temp folder,
so nothing is added to the repository.
"""

import argparse
import sys
import tempfile
from datetime import datetime
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT / "tests" / "fixtures" / "apps" / "acmenotes_violating"
DEFAULT_POLICY = ROOT / "tests" / "fixtures" / "synthetic_privacy_policy.txt"

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


def _finish(state: AssessmentState, output_dir, heading: str, default_folder: str) -> Path:
    """Print the step log and summary, write the audit record."""
    summary = summarize_assessment(state)
    record = build_audit_record(state, summary)

    folder = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / default_folder
    path = write_audit_record(record, folder)

    print(heading)
    print()
    for entry in state.step_log:
        print(f"{entry.step} [{entry.target}] {entry.status.value} (attempt {entry.attempt})")
    print()
    print(summary.model_dump_json(indent=2))
    print()
    print(f"Audit record written to: {path}")
    return path


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

    return _finish(state, output_dir, "FAKE agents only. No model was called.", "dpdp_audit_demo")


def _load_model():
    """Load the real chat model from .env. Imported here so fake mode never needs it."""
    from llm.langchain_model import get_langchain_model

    return get_langchain_model()


def run_real(project: str | Path, policy: str | Path, output_dir=None) -> int:
    """Run Agents 1, 2 and 3 with the real model. Returns 0 on success, 1 on a setup error."""
    project_path = Path(project)
    policy_path = Path(policy)

    # Cheap checks first, before anything that could call the network.
    if not project_path.is_dir():
        print(f"Project folder not found: {project_path}")
        return 1
    if not policy_path.is_file():
        print(f"Policy file not found: {policy_path}")
        return 1
    policy_text = policy_path.read_text(encoding="utf-8")
    if not policy_text.strip():
        print(f"Policy file is empty: {policy_path}")
        return 1

    print("REAL model. This makes API calls (Agents 1, 2 and 3 only).")
    print(f"Project: {project_path}")
    print(f"Policy:  {policy_path} ({len(policy_text)} characters)")
    print()

    try:
        model = _load_model()
    except ValueError as error:
        print(f"Cannot start the real run: {error}")
        return 1

    from agents.alignment_agent.agent import AlignmentAgent
    from agents.implementation_agent.agent import ImplementationExtractionAgent
    from agents.policy_agent.agent import PolicyExtractionAgent

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id=f"real-{stamp}",
            policy_documents=[PolicyDocument(name=policy_path.name, text=policy_text)],
            project_path=str(project_path),
        )
    )

    run_analysis_stages(
        state,
        PolicyExtractionAgent(model),
        lambda: ImplementationExtractionAgent(model),
        lambda: AlignmentAgent(model),
    )

    _finish(state, output_dir, "REAL model run finished.", "dpdp_audit_real")
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the front half of the pipeline.")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="folder for the audit file (default: system temp folder)")
    parser.add_argument("--real", action="store_true",
                        help="use the real model (makes API calls)")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT),
                        help="Django project folder to scan (only with --real)")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY),
                        help="policy text file (only with --real)")
    args = parser.parse_args(argv)

    if not args.real:
        main(args.output_dir)
        return 0
    return run_real(args.project, args.policy, args.output_dir)


if __name__ == "__main__":
    sys.exit(cli())