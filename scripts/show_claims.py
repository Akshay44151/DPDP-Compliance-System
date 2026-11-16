"""Print every verified claim from one run on the hard policy (1 Groq call).

Run from the project root:  python -m scripts.show_claims
"""

from pathlib import Path

from agents.policy_agent.agent import PolicyExtractionAgent

POLICY = Path("tests/fixtures/hard_privacy_policy.txt")


def main() -> None:
    agent = PolicyExtractionAgent()
    report = agent.extract_with_report(
        document_text=POLICY.read_text(encoding="utf-8"),
        source=POLICY.name,
    )
    for i, c in enumerate(report.result.claims, start=1):
        print(f"{i}. [{c.category.value}] {c.evidence}")


if __name__ == "__main__":
    main()