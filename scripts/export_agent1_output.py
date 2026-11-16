"""Run Agent 1 once and save its output as JSON for other team members.

Makes one real Groq call. Run from the project root:
    python -m scripts.export_agent1_output
"""

import json
from pathlib import Path

from agents.policy_agent.agent import PolicyExtractionAgent

POLICY = Path("tests/fixtures/hard_privacy_policy.txt")
OUTPUT = Path("tests/fixtures/agent1_sample_output.json")


def main() -> None:
    agent = PolicyExtractionAgent()
    report = agent.extract_with_report(
        document_text=POLICY.read_text(encoding="utf-8"),
        source=POLICY.name,
    )
    payload = {
        "source": POLICY.name,
        "claims": [c.model_dump(mode="json") for c in report.result.claims],
        "rejected_claims": [c.model_dump(mode="json") for c in report.rejected_claims],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(payload['claims'])} claims to {OUTPUT}")


if __name__ == "__main__":
    main()