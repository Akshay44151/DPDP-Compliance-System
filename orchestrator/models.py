"""Data models for the central orchestrator (Phase A, step 1).

These are validated containers, not agents. They describe:
  1. what the user submits (AssessmentRequest),
  2. everything known about one assessment run (AssessmentState),
  3. an audit log of what the orchestrator did (StepRecord).

Agent 1's real output (PolicyClaim) is included. The results of Agents 2
and 3 are kept as plain dictionaries for now, so we do not import code
that is not in main yet. Fields for the other agents are added later,
once their output formats are agreed with their owners. Nothing is
guessed here.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.policy_agent.schemas import PolicyClaim


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PolicyDocument(BaseModel):
    """One policy or governance document supplied by the user."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)  # e.g. "privacy_policy.txt"
    text: str = Field(min_length=1)  # the full document text


class AssessmentRequest(BaseModel):
    """Everything the user submits to start an assessment."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(min_length=1)
    policy_documents: list[PolicyDocument] = Field(min_length=1)
    # Folder of the code project for Agent 2. None means no code analysis.
    project_path: str | None = None

    @field_validator("policy_documents")
    @classmethod
    def _names_must_be_unique(
        cls, documents: list[PolicyDocument]
    ) -> list[PolicyDocument]:
        # A document's name becomes the "source" of its claims, so two
        # documents with the same name would make evidence ambiguous.
        names = [doc.name for doc in documents]
        if len(names) != len(set(names)):
            raise ValueError("Policy document names must be unique.")
        return documents


class StepStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class StepRecord(BaseModel):
    """One entry in the orchestrator's audit log."""

    model_config = ConfigDict(extra="forbid")

    step: str                       # e.g. "agent1_policy_extraction"
    target: str = ""                # e.g. the document name being processed
    status: StepStatus
    attempt: int = Field(default=1, ge=1)
    message: str = ""               # e.g. the error text for a failure
    timestamp: datetime = Field(default_factory=_now)


class AssessmentState(BaseModel):
    """The orchestrator's single record of an assessment in progress."""

    model_config = ConfigDict(extra="forbid")

    request: AssessmentRequest
    policy_claims: list[PolicyClaim] = Field(default_factory=list)
    rejected_claims: list[PolicyClaim] = Field(default_factory=list)
    # Agent 2's report as a plain dict (report.model_dump(mode="json")).
    # None means the implementation step has not produced a report.
    implementation_report: dict[str, Any] | None = None
    # Agent 3's result as a plain dict (result.model_dump(mode="json")).
    # None means the alignment step has not produced a result.
    alignment_result: dict[str, Any] | None = None
        # The evidence gate's decisions (GateResult.to_dict()).
    # None means the gate did not run or crashed.
    evidence_gate: dict[str, Any] | None = None
    step_log: list[StepRecord] = Field(default_factory=list)

    def log(
        self,
        step: str,
        status: StepStatus,
        *,
        target: str = "",
        attempt: int = 1,
        message: str = "",
    ) -> StepRecord:
        """Add an audit entry and return it."""
        record = StepRecord(
            step=step,
            target=target,
            status=status,
            attempt=attempt,
            message=message,
        )
        self.step_log.append(record)
        return record