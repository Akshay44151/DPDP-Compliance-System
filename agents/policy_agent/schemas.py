
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PolicyCategory(str, Enum):
    """Categories for privacy and data-protection claims."""

    TRANSPARENCY_NOTICE = "transparency_notice"
    CONSENT = "consent"
    PURPOSE_LIMITATION = "purpose_limitation"
    DATA_MINIMIZATION = "data_minimization"
    DATA_ACCURACY = "data_accuracy"
    STORAGE_RETENTION = "storage_retention"
    SECURITY_SAFEGUARDS = "security_safeguards"
    DATA_PRINCIPAL_RIGHTS = "data_principal_rights"
    GRIEVANCE_REDRESSAL = "grievance_redressal"
    CHILDREN_DATA = "children_data"
    DATA_BREACH = "data_breach"
    DATA_SHARING = "data_sharing"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"
    OTHER = "other"


class PolicyClaim(BaseModel):
    """A privacy-related claim extracted from a policy document."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(
        min_length=1,
        description="A concise statement of what the policy document claims."
    )

    category: PolicyCategory = Field(
        description="The privacy or data-protection category of the claim."
    )

    evidence: str = Field(
        min_length=1,
        description="The exact supporting excerpt from the source document."
    )

    source: str = Field(
        min_length=1,
        description="The document name or source identifier."
    )


class PolicyExtractionResult(BaseModel):
    """Structured output returned by Agent 1."""

    model_config = ConfigDict(extra="forbid")

    claims: list[PolicyClaim] = Field(
        description="Privacy-related claims extracted from the document."
    )
