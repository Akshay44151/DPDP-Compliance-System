"""Drop exact duplicate policy claims before Agent 3 (Phase A).

Agent 3 gives one verdict per claim id, so claims with the same source,
category and evidence would share one verdict anyway. The orchestrator
drops the extra copies first, so each claim is counted once, and logs
every drop. Deterministic Python. Nothing inside Agent 1 or Agent 3
changes.

The rule is make_claim_id itself, so it matches Agent 3 exactly. The
wording of the "claim" field is not part of the id.
"""

from agents.policy_agent.claim_id import make_claim_id
from agents.policy_agent.schemas import PolicyCategory
from orchestrator.models import AssessmentState, StepStatus

STEP_NAME = "dedup_claims"


def drop_duplicate_claims(state: AssessmentState) -> int:
    """Keep the first copy of each claim, drop later copies, log each drop.

    Returns how many claims were dropped. Order is preserved. Rejected
    claims are not touched.
    """
    seen: set[str] = set()
    kept = []
    dropped = 0

    for claim in state.policy_claims:
        claim_id = make_claim_id(claim.source, claim.category, claim.evidence)
        if claim_id in seen:
            dropped += 1
            state.log(
                STEP_NAME,
                StepStatus.COMPLETED,
                target=claim.source,
                message=(
                    f"Dropped duplicate claim {claim_id} "
                    f"(category {PolicyCategory(claim.category).value}): same "
                    "source, category and evidence as an earlier claim."
                ),
            )
            continue
        seen.add(claim_id)
        kept.append(claim)

    state.policy_claims[:] = kept
    return dropped