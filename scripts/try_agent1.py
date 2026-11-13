"""Manual experiment: run Agent 1 on a real policy using the real Groq API.

Not a unit test. It makes one real API call.
Run from the project root:  python -m scripts.try_agent1
"""

import re
from pathlib import Path

from agents.policy_agent.agent import PolicyExtractionAgent

POLICY_PATH = Path("tests/fixtures/synthetic_privacy_policy.txt")


def normalise(text: str) -> str:
    """Collapse all whitespace so line breaks don't cause false mismatches."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    document_text = POLICY_PATH.read_text(encoding="utf-8")

    agent = PolicyExtractionAgent()  # builds the real Groq model from .env
    result = agent.extract(
        document_text=document_text,
        source=POLICY_PATH.name,
    )

    haystack = normalise(document_text)
    print(f"Claims returned: {len(result.claims)}\n")

    for i, claim in enumerate(result.claims, start=1):
        found = normalise(claim.evidence) in haystack
        print(f"--- Claim {i} ---")
        print(f"category : {claim.category.value}")
        print(f"claim    : {claim.claim}")
        print(f"evidence : {claim.evidence}")
        print(f"source   : {claim.source}")
        print(f"evidence is exact text from policy? {'YES' if found else 'NO  <-- check this'}")
        print()


if __name__ == "__main__":
    main()
    