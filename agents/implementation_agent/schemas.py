from typing import Literal
 
from pydantic import BaseModel, ConfigDict, Field
 
from agents.policy_agent.schemas import PolicyCategory
 
# The five DPDP principles the project evaluates, expressed in Member 1's categories
# so Agent 3 can match claims to facts with a plain equality check.
IN_SCOPE_CATEGORIES: list[PolicyCategory] = [
    PolicyCategory.TRANSPARENCY_NOTICE,
    PolicyCategory.CONSENT,
    PolicyCategory.PURPOSE_LIMITATION,
    PolicyCategory.DATA_MINIMIZATION,
    PolicyCategory.STORAGE_RETENTION,
]
 
 
class CodeLocation(BaseModel):
    file: str
    line: int
 
 
class EvidenceItem(BaseModel):
    """A raw, deterministic observation from the codebase (no LLM involved)."""
 
    id: str
    kind: str
    location: CodeLocation
    snippet: str
    detail: dict = Field(default_factory=dict)
 
 
class ImplementationFact(BaseModel):
    """What the code actually does, tagged by policy category. Input to Agent 3."""
 
    model_config = ConfigDict(extra="forbid")
 
    id: str
    category: PolicyCategory
    statement: str
    fact_type: Literal["present", "absent", "partial"]
    evidence_ids: list[str] = Field(default_factory=list)
    locations: list[CodeLocation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
 
 
class ImplementationReport(BaseModel):
    """Structured output returned by Agent 2."""
 
    project_path: str
    files_scanned: int
    coverage: dict[str, int]  # evidence kind -> count (justifies "absent" facts)
    evidence: list[EvidenceItem]
    facts: list[ImplementationFact]
 
