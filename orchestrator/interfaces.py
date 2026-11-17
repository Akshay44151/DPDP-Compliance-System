"""Contracts that the orchestrator depends on (Phase A, step 2).

The orchestrator calls agents through these small interfaces instead of
importing concrete agent classes. Tests can then pass in fakes, and a
teammate's agent can be plugged in later without changing the
orchestrator.

Only Agent 1 has a contract here, because Agent 1's output format is
real and agreed. Contracts for Agents 2 to 5 are added once their owners
share their schemas. Nothing is guessed.
"""

from typing import Protocol, runtime_checkable

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