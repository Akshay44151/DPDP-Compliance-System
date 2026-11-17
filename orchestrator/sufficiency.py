"""Evidence sufficiency: map Agent 3's alignment statuses to outcomes.

This module is deterministic Python. No LLM is involved.

Principle: evidence not found does not mean non-compliant. Only a status that
shows the code supports the claim becomes COMPLIANT. Anything unclear becomes
INSUFFICIENT_EVIDENCE, never COMPLIANT and never GAP.

"COMPLIANT" here means the code matches what the policy claims. It is not a
legal judgment; that belongs to Agent 4.

The status names below are OUR copy of Agent 3's AlignmentStatus values. We do
not import Person 2's code, so nothing breaks while their branch is unmerged.
If Person 2 renames or adds a status, this file must be updated.
"""

from enum import Enum


class Agent3Status(str, Enum):
    """Copy of the status strings Agent 3 (alignment) can return."""

    ALIGNED = "aligned"
    PARTIAL = "partial"
    VIOLATED = "violated"
    UNVERIFIABLE = "unverifiable"
    OUT_OF_SCOPE = "out_of_scope"
    UNDECLARED = "undeclared"


class Outcome(str, Enum):
    """The result for one finding, after the sufficiency rules."""

    COMPLIANT = "compliant"
    GAP = "gap"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_ASSESSED = "not_assessed"  # outside the evaluated principles; never retried


STATUS_TO_OUTCOME: dict[Agent3Status, Outcome] = {
    # Code supports the claim.
    Agent3Status.ALIGNED: Outcome.COMPLIANT,
    # Code supports only part of the claim: a real mismatch, not missing evidence.
    Agent3Status.PARTIAL: Outcome.GAP,
    # Code contradicts the claim.
    Agent3Status.VIOLATED: Outcome.GAP,
    # The extracted facts do not cover the claim (or the model call failed).
    Agent3Status.UNVERIFIABLE: Outcome.INSUFFICIENT_EVIDENCE,
    # Agent 2 does not evaluate this category, so it is not scored or retried.
    Agent3Status.OUT_OF_SCOPE: Outcome.NOT_ASSESSED,
    # Depends on Agent 1 not having missed a claim, which we have seen it do.
    # Revisit once Person 2 adds the check against the full policy text.
    Agent3Status.UNDECLARED: Outcome.INSUFFICIENT_EVIDENCE,
}


def outcome_for_status(status: str | Agent3Status) -> Outcome:
    """Return the outcome for an Agent 3 status.

    An unknown status string is treated as INSUFFICIENT_EVIDENCE. It must
    never become COMPLIANT or GAP by accident.
    """
    try:
        return STATUS_TO_OUTCOME[Agent3Status(status)]
    except ValueError:
        return Outcome.INSUFFICIENT_EVIDENCE