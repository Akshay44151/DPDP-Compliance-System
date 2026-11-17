"""Run Agent 2 then Agent 3 against a Django project using the shared LangChain model.

Usage (from repo root, venv active):
    python scripts/try_agent2_agent3.py
    python scripts/try_agent2_agent3.py tests/fixtures/apps/acmenotes_baseline -q
    python scripts/try_agent2_agent3.py APP --policy-text long_policy_tmp.txt --save after_x_1.json
    python scripts/try_agent2_agent3.py APP --drop-category transparency_notice

Works on older commits too: attributes that do not exist there are reported as n/a.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.alignment_agent.agent import AlignmentAgent  # noqa: E402
from agents.implementation_agent.agent import ImplementationExtractionAgent  # noqa: E402
from agents.policy_agent.schemas import PolicyExtractionResult  # noqa: E402
from llm.langchain_model import get_langchain_model  # noqa: E402


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return head + (" (+uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("app", nargs="?", default="tests/fixtures/apps/acmenotes_violating")
    p.add_argument("--policy", default="tests/fixtures/agent1_sample_output.json",
                   help="Agent 1 output JSON")
    p.add_argument("--policy-text", help="plain-text policy file passed to Agent 3 as policy_text")
    p.add_argument("--drop-category", action="append", default=[],
                   help="remove claims of this category before Agent 3 (repeatable)")
    p.add_argument("-q", "--quiet", action="store_true", help="skip per-fact and per-finding lines")
    p.add_argument("--save", help="write full results to this JSON file")
    args = p.parse_args()

    commit = git_commit()
    model = get_langchain_model()
    raw_claims = json.loads(Path(args.policy).read_text("utf-8"))["claims"]
    if args.drop_category:
        raw_claims = [c for c in raw_claims if c.get("category") not in args.drop_category]
    policy = PolicyExtractionResult(claims=raw_claims)
    policy_text = Path(args.policy_text).read_text("utf-8") if args.policy_text else None

    agent2, agent3 = ImplementationExtractionAgent(model), AlignmentAgent(model)
    print(f"commit: {commit} | app: {args.app} | claims: {len(raw_claims)}"
          f"{' | dropped: ' + ','.join(args.drop_category) if args.drop_category else ''}")

    impl = agent2.run(args.app)
    failed_cats = getattr(impl, "failed_categories", None)
    print(f"\n== Agent 2: {impl.files_scanned} files, evidence coverage {impl.coverage}")
    print(f"   LLM retries: {impl.llm_retries}, failures: {impl.llm_failures}, "
          f"failed categories: {failed_cats if failed_cats is not None else 'n/a'}")
    if not args.quiet:
        for f in impl.facts:
            print(f"  [{f.category.value}] ({f.fact_type}) {f.statement}")

    result = agent3.run(policy, impl, policy_text=policy_text) if policy_text \
        else agent3.run(policy, impl)
    unknown_ids = getattr(agent3, "unknown_verdict_ids", None)
    failed_calls = getattr(agent3, "failed_calls", None)
    invalid = sum(1 for a in result.findings if getattr(a, "reason_code", None) == "invalid_model_output")
    print(f"\n== Agent 3: {result.summary}")
    print(f"   LLM retries: {result.llm_retries}, failures: {result.llm_failures}, "
          f"failed calls: {failed_calls if failed_calls is not None else 'n/a'}")
    print(f"   unmatched verdict ids: "
          f"{len(unknown_ids) if unknown_ids is not None else 'n/a'} {unknown_ids or ''}")
    print(f"   findings with reason_code invalid_model_output: {invalid}")
    if policy_text:
        print(f"   policy_text_chars: {getattr(result, 'policy_text_chars', 'n/a')}, "
              f"truncated: {getattr(result, 'policy_text_truncated', 'n/a')}")
    if impl.llm_failures or result.llm_failures:
        print("   WARNING: some LLM calls failed; results are incomplete. Last errors:")
        print(f"     Agent 2: {agent2.last_error}\n     Agent 3: {agent3.last_error}")
    if not args.quiet:
        for a in result.findings:
            if a.status.value == "out_of_scope":
                continue
            subject = a.claim.claim if a.claim else "(code behaviour not covered by the policy)"
            print(f"  {a.status.value.upper():12} [{a.category.value}] {subject}\n"
                  f"               -> {a.reasoning}")

    if args.save:
        Path(args.save).write_text(json.dumps(
            {"run_info": {"commit": commit, "app": args.app, "argv": sys.argv[1:],
                          "unknown_verdict_ids": unknown_ids,
                          "invalid_model_output_findings": invalid,
                          "failed_categories": failed_cats},
             "implementation": impl.model_dump(mode="json"),
             "alignment": result.model_dump(mode="json")}, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()