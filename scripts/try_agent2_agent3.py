"""Run Agent 2 then Agent 3 against a Django project using the real Groq client.

Usage (from repo root, venv active):
    python scripts/try_agent2_agent3.py
    python scripts/try_agent2_agent3.py tests/fixtures/apps/acmenotes_baseline
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.alignment_agent.agent import AlignmentAgent  # noqa: E402
from agents.implementation_agent.agent import ImplementationExtractionAgent  # noqa: E402
from agents.policy_agent.schemas import PolicyExtractionResult  # noqa: E402
from llm.groq_client import GroqLLMClient  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("app", nargs="?", default="tests/fixtures/apps/acmenotes_violating")
    p.add_argument("--policy", default="tests/fixtures/agent1_sample_output.json",
                   help="Agent 1 output JSON")
    p.add_argument("--save", help="write full results to this JSON file")
    args = p.parse_args()

    llm = GroqLLMClient()
    policy = PolicyExtractionResult(claims=json.loads(Path(args.policy).read_text("utf-8"))["claims"])

    impl = ImplementationExtractionAgent(llm).run(args.app)
    print(f"\n== Agent 2: {impl.files_scanned} files, evidence coverage {impl.coverage}")
    for f in impl.facts:
        print(f"  [{f.category.value}] ({f.fact_type}) {f.statement}")

    result = AlignmentAgent(llm).run(policy, impl)
    print(f"\n== Agent 3: {result.summary}")
    for a in result.findings:
        if a.status.value == "out_of_scope":
            continue
        subject = a.claim.claim if a.claim else "(code behaviour not covered by the policy)"
        print(f"  {a.status.value.upper():12} [{a.category.value}] {subject}\n"
              f"               -> {a.reasoning}")

    if args.save:
        Path(args.save).write_text(json.dumps(
            {"implementation": impl.model_dump(mode="json"),
             "alignment": result.model_dump(mode="json")}, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
