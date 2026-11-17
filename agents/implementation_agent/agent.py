"""Agent 2: Implementation Extraction Agent (LangChain).

Stage 1 (scanner.py): deterministic evidence from Django code.
Stage 2 (this file): a LangChain chain (prompt | model.with_structured_output) turns evidence
into per-category ImplementationFacts, validated with Pydantic, with evidence ids checked
against the real scan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from groq import BadRequestError
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, ValidationError

from agents.implementation_agent.scanner import scan_project
from agents.implementation_agent.schemas import (EvidenceItem, ImplementationFact,
                                                 ImplementationReport)
from agents.policy_agent.schemas import PolicyCategory as C
from llm.langchain_model import get_langchain_model

# Which evidence kinds are relevant to which policy category.
CATEGORY_KINDS: dict[C, list[str]] = {
    C.TRANSPARENCY_NOTICE: ["privacy_notice_link", "purpose_text", "consent_ui",
                            "data_sharing", "form_input", "model_field"],
    C.CONSENT: ["consent_logic", "consent_ui", "data_write", "outbound_email"],
    C.PURPOSE_LIMITATION: ["purpose_marker", "purpose_text", "data_write", "data_sharing",
                           "outbound_email", "personal_data_logged"],
    C.DATA_MINIMIZATION: ["model_field", "form_field", "form_input"],
    C.STORAGE_RETENTION: ["retention_logic", "retention_field", "setting",
                          "personal_data_logged"],
}
MAX_EVIDENCE_PER_CATEGORY = 60

SYSTEM_PROMPT = """You are the Implementation Extraction Agent in a DPDP Act 2023 compliance \
pipeline. You are given static-analysis evidence from a Django application for ONE privacy \
category. Describe what the code actually does. Rules:
- State only what the evidence supports. Do NOT judge compliance and do NOT read any privacy policy.
- Every fact must cite evidence ids from the list. Never invent ids.
- If the evidence shows something is missing (e.g. coverage count is 0 for a relevant kind, \
or data is written or emailed with no consent reference), emit a fact with fact_type "absent" or \
"partial" and cite the nearest evidence (e.g. the data_write items), or none if there is none.
- Keep each statement to one concrete sentence.
- Be concise: group related items into ONE fact (for example list all personal-data model fields \
in a single statement, and all form inputs in another). One fact per distinct behaviour.
- Never emit two facts that say the same thing, and emit at most 8 facts.
- Emit an "absent" fact only for a control this category would normally need (one fact per missing \
control), not one per evidence kind with a zero count.
- Describe only observed behaviour. Do not speculate about intent or possible problems \
(no "suggests", "likely", "potential").
- An evidence item with an empty code snippet only shows that a file or name exists. Never infer \
what a file contains, or that it is empty, from a missing snippet.
- Be specific: in a fact about data sent to an external service, name the target URL or host from \
the evidence; in a fact about a database write, list every field in its "fields" detail; in a fact \
about an outbound email, give its subject; in a fact about a withdrawal or unsubscribe control, \
name it.
- If COVERAGE lists data_sharing with a count of 0, emit one "absent" fact stating that no code \
sends personal data to external services."""


HUMAN_PROMPT = "{task_input}"  # rendered evidence goes in as a variable, so code braces are safe

# Failures that mean "the model's structured output was unusable"; anything else (auth, network,
# rate limit) is a real error and is allowed to propagate.
RETRYABLE = (OutputParserException, ValidationError, BadRequestError)


class _RawFact(BaseModel):
    model_config = ConfigDict(extra="forbid")  # all fields required: strict-schema friendly

    statement: str
    fact_type: Literal["present", "absent", "partial"]
    evidence_ids: list[str]
    confidence: float


class _RawResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[_RawFact]


class ImplementationExtractionAgent:
    def __init__(self, model=None) -> None:
        self.model = model or get_langchain_model()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
        self.chain = self.prompt | self.model.with_structured_output(_RawResponse)
        self.retries = 0
        self.failures = 0
        self.last_error: str | None = None

    def run(self, project_path: str | Path) -> ImplementationReport:
        self.retries = self.failures = 0
        evidence, files, coverage = scan_project(project_path)
        by_id = {e.id: e for e in evidence}
        facts: list[ImplementationFact] = []
        for category, kinds in CATEGORY_KINDS.items():
            relevant = [e for e in evidence if e.kind in kinds][:MAX_EVIDENCE_PER_CATEGORY]
            raw = self._ask(category, relevant, coverage, kinds)
            for i, rf in enumerate(raw.facts, 1):
                ids = [x for x in rf.evidence_ids if x in by_id]  # drop hallucinated ids
                facts.append(ImplementationFact(
                    id=f"F-{category.name}-{i:02d}",
                    category=category,
                    statement=rf.statement,
                    fact_type=rf.fact_type if rf.fact_type in ("present", "absent", "partial") else "partial",
                    evidence_ids=ids,
                    locations=[by_id[x].location for x in ids],
                    confidence=min(max(rf.confidence, 0.0), 1.0),
                ))
        return ImplementationReport(project_path=str(project_path), files_scanned=files,
                                    coverage=coverage, evidence=evidence, facts=facts,
                                    llm_retries=self.retries, llm_failures=self.failures)

    def _ask(self, category: C, evidence: list[EvidenceItem],
             coverage: dict[str, int], kinds: list[str]) -> _RawResponse:
        task_input = self._render(category, evidence, coverage, kinds)
        last_err = ""
        for attempt in range(2):  # one retry when the structured output is unusable
            text = task_input if not last_err else (
                task_input + f"\n\nYour previous reply was invalid: {last_err}. "
                "Return the required structured output only.")
            try:
                return _RawResponse.model_validate(self.chain.invoke({"task_input": text}))
            except RETRYABLE as exc:
                last_err = self.last_error = str(exc)[:200]
                if attempt == 0:
                    self.retries += 1
        self.failures += 1
        return _RawResponse(facts=[])  # fail soft; Agent 3 treats missing facts as unverifiable

    @staticmethod
    def _render(category: C, evidence: list[EvidenceItem],
                coverage: dict[str, int], kinds: list[str]) -> str:
        cov = {k: coverage.get(k, 0) for k in kinds}
        lines = [f"CATEGORY: {category.value}",
                 f"COVERAGE (evidence count per kind across the whole project): {json.dumps(cov)}",
                 "EVIDENCE:"]
        if not evidence:
            lines.append("(none found)")
        for e in evidence:
            lines.append(f"{e.id} [{e.kind}] {e.location.file}:{e.location.line} "
                         f"detail={json.dumps(e.detail)} code={e.snippet!r}")
        return "\n".join(lines)