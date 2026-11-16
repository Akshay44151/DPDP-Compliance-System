"""Agent 2: Implementation Extraction Agent.

Stage 1 (scanner.py): deterministic evidence from Django code.
Stage 2 (this file): LLM turns evidence into per-category ImplementationFacts,
validated with Pydantic, with evidence ids checked against the real scan.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agents.implementation_agent.scanner import scan_project
from agents.implementation_agent.schemas import (EvidenceItem, ImplementationFact,
                                                 ImplementationReport)
from agents.policy_agent.schemas import PolicyCategory as C

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
- Keep each statement to one concrete sentence."""


class _RawFact(BaseModel):
    statement: str
    fact_type: str
    evidence_ids: list[str] = []
    confidence: float = 0.5


class _RawResponse(BaseModel):
    facts: list[_RawFact]


class ImplementationExtractionAgent:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client  # anything with generate(*, system_prompt, user_prompt, response_schema=None)

    def run(self, project_path: str | Path) -> ImplementationReport:
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
                                    coverage=coverage, evidence=evidence, facts=facts)

    def _ask(self, category: C, evidence: list[EvidenceItem],
             coverage: dict[str, int], kinds: list[str]) -> _RawResponse:
        user_prompt = self._render(category, evidence, coverage, kinds)
        schema = _RawResponse.model_json_schema()
        last_err = ""
        for _ in range(2):  # one retry on invalid JSON
            prompt = user_prompt if not last_err else (
                user_prompt + f"\n\nYour previous reply was invalid: {last_err}. Return valid JSON only.")
            text = self.llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt,
                                     response_schema=schema)
            try:
                return _RawResponse.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = str(exc)[:200]
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
