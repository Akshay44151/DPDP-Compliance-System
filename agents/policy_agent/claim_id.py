"""Deterministic claim IDs, assigned by code and never by the model.

The ID is a short hash of the document name, the category and the
normalised evidence text. It does not depend on the wording of the
`claim` field, so it survives the model rewording a claim. It does NOT
survive the model merging or splitting claims, or filing the same
evidence under a different category.
"""

import hashlib

from agents.policy_agent.schemas import PolicyCategory
from agents.policy_agent.verification import normalise_for_matching


def make_claim_id(source: str, category: PolicyCategory, evidence: str) -> str:
    key = (
        f"{source}|{PolicyCategory(category).value}|"
        f"{normalise_for_matching(evidence)}"
    )
    return "pc_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]