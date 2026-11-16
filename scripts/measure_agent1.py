"""Measure Agent 1's recall and consistency on a policy with planted claims.

Makes several real Groq calls. Run from the project root:
    python -m scripts.measure_agent1
    python -m scripts.measure_agent1 5     (number of runs, default 5)
"""

import json
import sys
from pathlib import Path

from agents.policy_agent.agent import PolicyExtractionAgent
from agents.policy_agent.verification import normalise_for_matching

KEY_PATH = Path("tests/fixtures/hard_policy_key.json")


def contains(evidence: str, snippet: str) -> bool:
    return normalise_for_matching(snippet) in normalise_for_matching(evidence)


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    document_path = Path(key["document"])
    document_text = document_path.read_text(encoding="utf-8")
    planted = key["planted_claims"]
    decoys = key["decoys"]

    agent = PolicyExtractionAgent()

    found_counts = {p["id"]: 0 for p in planted}
    decoy_counts = {d["id"]: 0 for d in decoys}
    total_claims = 0
    total_rejected = 0
    ok_runs = 0
    failed_runs = 0

    for run in range(1, runs + 1):
        try:
            report = agent.extract_with_report(
                document_text=document_text,
                source=document_path.name,
            )
        except Exception as exc:
            failed_runs += 1
            print(f"run {run}: FAILED ({type(exc).__name__}) {str(exc)[:150]}")
            continue

        ok_runs += 1
        evidences = [c.evidence for c in report.result.claims]
        total_claims += len(evidences)
        total_rejected += len(report.rejected_claims)

        found_now = 0
        for p in planted:
            if any(contains(e, p["evidence"]) for e in evidences):
                found_counts[p["id"]] += 1
                found_now += 1
        for d in decoys:
            if any(contains(e, d["evidence"]) for e in evidences):
                decoy_counts[d["id"]] += 1

        print(
            f"run {run}: {len(evidences)} verified, "
            f"{len(report.rejected_claims)} rejected, "
            f"{found_now}/{len(planted)} planted claims found"
        )

    print(f"\nSuccessful runs: {ok_runs}/{runs}   Failed runs: {failed_runs}")
    if ok_runs == 0:
        print("No successful runs, so there is nothing to score.")
        return

    print(f"\nPer planted claim (found in how many of {ok_runs} good runs):")
    for p in planted:
        n = found_counts[p["id"]]
        flag = "" if n == ok_runs else "   <-- not always found"
        print(f"  {p['id']} {n}/{ok_runs}  {p['topic']}{flag}")

    print("\nDecoys (should be 0 everywhere):")
    for d in decoys:
        n = decoy_counts[d["id"]]
        flag = "   <-- PROBLEM" if n else ""
        print(f"  {d['id']} {n}/{ok_runs}  {d['topic']}{flag}")

    recall = sum(found_counts.values()) / (len(planted) * ok_runs)
    print(f"\nAverage recall over {ok_runs} good runs: {recall:.0%}")
    print(f"Total verified claims: {total_claims}, rejected: {total_rejected}")


if __name__ == "__main__":
    main()