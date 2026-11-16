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


class AlignmentFinding(BaseModel):
    """One policy-vs-code finding. Input to Agent 4 (legal reasoning)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["claim_check", "undeclared_behavior"]
    category: PolicyCategory
    status: AlignmentStatus
    claim: PolicyClaim | None = None
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
    llm_failures: int = 0  # calls that gave up; their claims became "unverifiable"