"""Contracts that the orchestrator depends on (Phase A, step 2).

The orchestrator calls agents through these small interfaces instead of
importing concrete agent classes. Tests can then pass in fakes, and a
teammate's agent can be plugged in later without changing the
orchestrator.

Agent 1 has a real, agreed contract. Agent 3 has a contract that is OUR
copy of what Person 2 described. We do not import their code, so nothing
here depends on their branch. Contracts for Agents 2, 4 and 5 are added
once their code is visible. Nothing is guessed.
"""

from typing import Any, Protocol, runtime_checkable

from agents.policy_agent.schemas import PolicyExtractionResult
from agents.policy_agent.verification import ExtractionReport


@runtime_checkable
class PolicyExtractor(Protocol):
    """Anything that can turn one policy document into verified claims.

    PolicyExtractionAgent (Agent 1) satisfies this contract as it is.
    """

    def extract_with_report(
        self,
        *,
        document_text: str,
        source: str,
    ) -> ExtractionReport:
        ...


class AlignmentOutput(Protocol):
    """The parts of Agent 3's result that the orchestrator reads."""

    llm_failures: int  # model calls that gave up; their claims are unverifiable
    findings: list[Any]

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        ...


@runtime_checkable
class AlignmentChecker(Protocol):
    """Anything that can compare policy claims with implementation facts.

    AlignmentAgent (Agent 3) is expected to satisfy this. The agent keeps
    its failure counters on the object, so use one instance per run and
    never share it between threads.
    """

    def run(
        self,
        policy: PolicyExtractionResult,
        implementation: Any,
        *,
        policy_text: str | None = None,
    ) -> AlignmentOutput:
        ...