"""Deterministic checks on Agent 1's output.

The LLM is asked to quote the policy exactly, but nothing forces it to.
These functions verify the quotes with plain code, so "evidence" in the
rest of the system means "text that really appears in the document".
"""

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from agents.policy_agent.schemas import PolicyClaim, PolicyExtractionResult

# Different dash characters are mapped to a normal "-".
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
# Curly quotes are mapped to straight ones.
_QUOTES = {ord(c): "'" for c in "\u2018\u2019"} | {
    ord(c): '"' for c in "\u201c\u201d"
}


def normalise_for_matching(text: str) -> str:
    """Remove formatting differences without changing the words."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASHES).translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


def evidence_in_document(evidence: str, document_text: str) -> bool:
    """True only if the evidence appears in the document as real text."""
    needle = normalise_for_matching(evidence)
    if not needle:
        return False
    return needle in normalise_for_matching(document_text)


def split_verified(
    claims: list[PolicyClaim],
    document_text: str,
) -> tuple[list[PolicyClaim], list[PolicyClaim]]:
    """Split claims into (verified, rejected) by checking their evidence."""
    verified: list[PolicyClaim] = []
    rejected: list[PolicyClaim] = []
    for claim in claims:
        if evidence_in_document(claim.evidence, document_text):
            verified.append(claim)
        else:
            rejected.append(claim)
    return verified, rejected


class ExtractionReport(BaseModel):
    """Agent 1's verified output plus any claims it rejected."""

    model_config = ConfigDict(extra="forbid")

    result: PolicyExtractionResult
    rejected_claims: list[PolicyClaim] = Field(default_factory=list)