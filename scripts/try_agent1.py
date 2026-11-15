"""Manual experiment: run Agent 1 on a real policy using the real Groq API.

Not a unit test. It makes one real API call.
Run from the project root:  python -m scripts.try_agent1
"""

from pathlib import Path

from agents.policy_agent.agent import PolicyExtractionAgent

POLICY_PATH = Path("tests/fixtures/synthetic_privacy_policy.txt")


def main() -> None:
    document_text = POLICY_PATH.read_text(encoding="utf-8")

    agent = PolicyExtractionAgent()
    report = agent.extract_with_report(
        document_text=document_text,
        source=POLICY_PATH.name,
    )

    print(f"Verified claims : {len(report.result.claims)}")
    print(f"Rejected claims : {len(report.rejected_claims)}\n")

    for i, claim in enumerate(report.result.claims, start=1):
        print(f"--- Verified claim {i} ---")
        print(f"category : {claim.category.value}")
        print(f"claim    : {claim.claim}")
        print(f"evidence : {claim.evidence}\n")

    for i, claim in enumerate(report.rejected_claims, start=1):
        print(f"--- REJECTED claim {i} (evidence not found in document) ---")
        print(f"claim    : {claim.claim}")
        print(f"evidence : {claim.evidence}\n")


if __name__ == "__main__":
    main()