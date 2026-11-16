"""Show what the model actually returned when a structured call fails.

Makes up to 3 real Groq calls on the hard policy.
Run from the project root:  python -m scripts.diagnose_failure
"""

from pathlib import Path

from agents.policy_agent.agent import PolicyExtractionAgent

POLICY = Path("tests/fixtures/hard_privacy_policy.txt")


def failed_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return str(error.get("failed_generation", ""))
    return str(exc)


def main() -> None:
    agent = PolicyExtractionAgent()
    text = POLICY.read_text(encoding="utf-8")

    for attempt in range(1, 4):
        try:
            report = agent.extract_with_report(
                document_text=text, source=POLICY.name
            )
            print(f"attempt {attempt}: OK, {len(report.result.claims)} claims")
        except Exception as exc:
            raw = failed_text(exc)
            print(f"attempt {attempt}: FAILED ({type(exc).__name__})")
            print(f"  failed output length: {len(raw)} characters")
            print(f"  first 200: {raw[:200]!r}")
            print(f"  last 300 : {raw[-300:]!r}\n")


if __name__ == "__main__":
    main()