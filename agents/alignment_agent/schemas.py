from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.implementation_agent.schemas import CodeLocation
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"            # code supports the claim
    PARTIAL = "partial"            # code supports part of the claim
    VIOLATED = "violated"          # code contradicts the claim
    UNVERIFIABLE = "unverifiable"  # facts don't cover the claim
    OUT_OF_SCOPE = "out_of_scope"  # claim category is outside the five evaluated principles
    UNDECLARED = "undeclared"      # code does something no policy claim covers
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # undisclosed-looking practice, but the policy was not fully available


class AlignmentFinding(BaseModel):
    """One policy-vs-code finding. Input to Agent 4 (legal reasoning)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["claim_check", "undeclared_behavior"]
    category: PolicyCategory
    status: AlignmentStatus
    reason_code: Literal[
        "model_assessed", "evidence_gap", "model_failure", "invalid_model_output",
        "unsupported_evidence", "out_of_scope", "undeclared_practice", "coverage_gap",
    ] = Field(default="model_assessed", description="Why the finding was emitted or downgraded.")
    claim: PolicyClaim | None = None
    claim_id: str | None = Field(
        default=None,
        description="make_claim_id(source, category, evidence) of the claim; None for undeclared behaviour.",
    )
    fact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)  # raw scanner evidence behind the finding
    code_locations: list[CodeLocation] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class AlignmentResult(BaseModel):
    """Structured output returned by Agent 3."""

    findings: list[AlignmentFinding]
    summary: dict[str, int]  # status -> count
    llm_retries: int = 0   # calls that needed a second attempt (invalid JSON)
    llm_failures: int = 0  # calls that gave up; inspect failed_calls for their stage/category
    failed_calls: list[str] = Field(
        default_factory=list,
        description="Exhausted call IDs: verdict:<category-value> or undeclared.",
    )
    policy_text_chars: int = 0  # length of the policy text passed in (0 if none)
    policy_text_truncated: bool = False  # True if it was cut to max_policy_chars; a cut text is not coverage