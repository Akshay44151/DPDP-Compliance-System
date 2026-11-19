"""Evidence gate: decide which Agent 3 findings may go on to legal reasoning.

Deterministic Python. Reads two results that were already turned into dicts
(Agent 2's report and Agent 3's result) and never changes them. For each
claim_check finding that says "violated" or "partial", it returns one
decision:

- blocked      The evidence cannot support a gap at all (a failed or cut-off
               scan, or a fact id that does not exist). Treat as
               insufficient evidence.
- review_only  Nothing wrong with the run, but the finding rests only on
               "not found by the scan" (no facts, or only absent facts). The
               scanner reads .py and .html files with fixed detectors, so
               absence is never proof. Needs human review, never a potential
               non-compliance on its own.
- pass         Backed by at least one present or partial fact on a clean,
               complete scan of that category.

Other findings (aligned, unverifiable, undeclared_behavior, and so on) are
not gated: undeclared findings rest on raw evidence of presence.

A passing decision is not a legal conclusion and not a compliance result.
It only means the evidence is not ruled out by the checks above.

Nothing here imports an agent, so fake mode and tests need no model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PASS = "pass"
REVIEW_ONLY = "review_only"
BLOCKED = "blocked"
NOT_GATED = "not_gated"

# Agent 3 statuses (lower case, as in model_dump(mode="json")) that could
# lead to a potential gap and therefore need the evidence checks.
GATED_STATUSES = ("violated", "partial")

DEFAULT_MAX_EVIDENCE_PER_CATEGORY = 60

_SEVERITY = {NOT_GATED: 0, PASS: 0, REVIEW_ONLY: 1, BLOCKED: 2}


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict on one Agent 3 finding."""

    finding_id: str
    decision: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class GateResult:
    """All decisions plus run-level warnings."""

    decisions: tuple[GateDecision, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def decision_for(self, finding_id: str) -> GateDecision | None:
        for decision in self.decisions:
            if decision.finding_id == finding_id:
                return decision
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "warnings": list(self.warnings),
        }


def category_was_truncated(
    category: str,
    coverage: Mapping[str, int],
    category_kinds: Mapping[str, Sequence[str]],
    max_evidence: int = DEFAULT_MAX_EVIDENCE_PER_CATEGORY,
) -> bool | None:
    """True if Agent 2 had to cut this category's evidence.

    Agent 2 counts evidence for the whole project, then keeps only the first
    max_evidence items of the category's kinds. So the category was cut when
    the counts add up to more than that. Returns None if the category has no
    known kinds, because then it cannot be checked.
    """
    kinds = category_kinds.get(category)
    if kinds is None:
        return None
    total = sum(_as_int(coverage.get(kind, 0)) for kind in kinds)
    return total > max_evidence


def evaluate_findings(
    implementation_report: Mapping[str, Any] | None,
    alignment_result: Mapping[str, Any] | None,
    category_kinds: Mapping[str, Sequence[str]],
    *,
    max_evidence_per_category: int = DEFAULT_MAX_EVIDENCE_PER_CATEGORY,
) -> GateResult:
    """Gate every claim_check finding that says violated or partial.

    category_kinds maps a PolicyCategory value (a string) to the evidence
    kinds Agent 2 uses for it. Pass the mapping in, so this module never
    imports Agent 2.
    """
    if alignment_result is None:
        return GateResult(warnings=("Agent 3 result is missing; nothing to gate.",))

    warnings = _run_warnings(implementation_report, alignment_result)

    report = implementation_report or {}
    coverage = report.get("coverage") or {}
    failed_categories = {str(c) for c in (report.get("failed_categories") or [])}
    files_scanned = _as_int(report.get("files_scanned", 0))
    facts_by_id = {
        fact["id"]: fact
        for fact in (report.get("facts") or [])
        if isinstance(fact, dict) and "id" in fact
    }
    failed_calls = {str(c) for c in (alignment_result.get("failed_calls") or [])}

    decisions: list[GateDecision] = []
    for finding in alignment_result.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        finding_id = str(finding.get("id", ""))

        if (
            finding.get("kind") != "claim_check"
            or finding.get("status") not in GATED_STATUSES
        ):
            decisions.append(GateDecision(finding_id, NOT_GATED))
            continue

        blocked: list[str] = []
        review: list[str] = []
        category = str(finding.get("category", ""))

        if implementation_report is None:
            blocked.append("Agent 2 report is missing.")
        else:
            if files_scanned <= 0:
                blocked.append("No files were scanned.")
            if category in failed_categories:
                blocked.append(
                    f"Agent 2 could not complete the {category} category."
                )
            truncated = category_was_truncated(
                category, coverage, category_kinds, max_evidence_per_category
            )
            if truncated:
                blocked.append(
                    f"Evidence for {category} was cut at "
                    f"{max_evidence_per_category} items; later files were not seen."
                )
            elif truncated is None:
                review.append(
                    f"No evidence kinds are known for {category}, so scan "
                    "coverage could not be checked."
                )

        if f"verdict:{category}" in failed_calls:
            blocked.append(f"Agent 3's {category} verdict call failed.")

        fact_ids = [str(f) for f in (finding.get("fact_ids") or [])]
        missing = [f for f in fact_ids if f not in facts_by_id]
        if missing:
            blocked.append(
                "Fact id(s) not found in the Agent 2 report: "
                + ", ".join(sorted(missing))
                + "."
            )

        if not fact_ids:
            review.append("The finding cites no implementation facts.")
        elif not missing and all(
            facts_by_id[f].get("fact_type") == "absent" for f in fact_ids
        ):
            review.append(
                "Every cited fact is an absence: the scan did not find "
                "something, which is not proof it does not exist."
            )

        if blocked:
            decisions.append(GateDecision(finding_id, BLOCKED, tuple(blocked + review)))
        elif review:
            decisions.append(GateDecision(finding_id, REVIEW_ONLY, tuple(review)))
        else:
            decisions.append(GateDecision(finding_id, PASS))

    return GateResult(decisions=tuple(decisions), warnings=tuple(warnings))


def worst_decision(decisions: Sequence[GateDecision]) -> str:
    """The most severe decision in a list (blocked > review_only > pass)."""
    worst = NOT_GATED
    for decision in decisions:
        if _SEVERITY[decision.decision] > _SEVERITY[worst]:
            worst = decision.decision
    return worst


def _run_warnings(
    implementation_report: Mapping[str, Any] | None,
    alignment_result: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if implementation_report is None:
        warnings.append("Agent 2 report is missing; every gated finding is blocked.")
    elif _as_int(implementation_report.get("llm_failures", 0)) > 0:
        warnings.append(
            "Agent 2 had failed model calls; some categories may have no facts."
        )
    if alignment_result.get("policy_text_truncated"):
        warnings.append(
            "The policy text was cut before Agent 3 read it; claim checks may "
            "miss commitments made in the cut part."
        )
    if "undeclared" in (alignment_result.get("failed_calls") or []):
        warnings.append("The undeclared-behaviour check failed and may be incomplete.")
    return warnings


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0