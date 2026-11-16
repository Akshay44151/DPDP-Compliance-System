"""Agent 3: Alignment Agent.

Compares policy claims (Agent 1) with implementation facts (Agent 2), one category at a time.
Deterministic guards keep the LLM honest: every verdict must cite real fact ids,
every claim gets exactly one finding, and out-of-scope claims never reach the LLM.
"""
from __future__ import annotations

import json
from collections import Counter

from pydantic import BaseModel, ValidationError

from agents.alignment_agent.schemas import AlignmentFinding, AlignmentResult, AlignmentStatus
from agents.implementation_agent.schemas import (IN_SCOPE_CATEGORIES, ImplementationFact,
                                                 ImplementationReport)
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim, PolicyExtractionResult

SYSTEM_PROMPT = """You are the Alignment Agent in a DPDP Act 2023 compliance pipeline. You compare \
what a privacy policy PROMISES (claims) with what the application code ACTUALLY DOES (facts) for \
ONE privacy category.

For every claim give one verdict:
- "aligned": the facts show the code does what the claim says.
- "partial": the facts support only part of the claim.
- "violated": the facts show the code contradicts the claim (including facts of type "absent" \
for something the claim promises, e.g. claim promises consent but data is used with no consent check).
- "unverifiable": the facts say nothing relevant to this claim.
Rules:
- aligned, partial and violated MUST cite fact ids from the list. Never invent ids.
- Judge only from the facts given. Do not assume code behaviour that is not listed.
- Separately, list "undeclared" items: facts of type "present" or "partial" showing the code \
collects, uses or shares personal data in a way that NO claim in this category mentions \
(e.g. a sensitive field the policy never discloses). Do not repeat facts already used in a verdict.
- Keep reasoning to one or two sentences. confidence is between 0 and 1."""


class _Verdict(BaseModel):
    claim_id: str
    status: str
    fact_ids: list[str] = []
    reasoning: str = ""
    confidence: float = 0.5


class _Undeclared(BaseModel):
    fact_ids: list[str]
    reasoning: str = ""
    confidence: float = 0.5


class _Raw(BaseModel):
    verdicts: list[_Verdict]
    undeclared: list[_Undeclared] = []


EVIDENCE_BASED = {AlignmentStatus.ALIGNED, AlignmentStatus.PARTIAL, AlignmentStatus.VIOLATED}


class AlignmentAgent:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    # ---------- public ----------
    def run(self, policy: PolicyExtractionResult, implementation: ImplementationReport) -> AlignmentResult:
        findings: list[AlignmentFinding] = []

        def new_id() -> str:
            return f"A-{len(findings) + 1:03d}"

        for claim in policy.claims:
            if claim.category not in IN_SCOPE_CATEGORIES:
                findings.append(AlignmentFinding(
                    id=new_id(), kind="claim_check", category=claim.category,
                    status=AlignmentStatus.OUT_OF_SCOPE, claim=claim,
                    reasoning="Category is outside the five principles evaluated by this framework."))

        for category in IN_SCOPE_CATEGORIES:
            claims = [c for c in policy.claims if c.category == category]
            facts = [f for f in implementation.facts if f.category == category]
            if not claims and not facts:
                continue
            if not facts:  # nothing to compare against
                for c in claims:
                    findings.append(AlignmentFinding(
                        id=new_id(), kind="claim_check", category=category,
                        status=AlignmentStatus.UNVERIFIABLE, claim=c,
                        reasoning="No implementation facts were extracted for this category."))
                continue
            raw = self._ask(category, claims, facts)
            findings.extend(self._to_findings(category, claims, facts, raw, new_id))

        summary = dict(Counter(f.status.value for f in findings))
        return AlignmentResult(findings=findings, summary=summary)

    # ---------- internals ----------
    def _to_findings(self, category, claims, facts, raw: _Raw, new_id) -> list[AlignmentFinding]:
        facts_by_id = {f.id: f for f in facts}
        verdicts = {}
        for v in raw.verdicts:
            verdicts.setdefault(v.claim_id, v)  # first verdict per claim wins
        out: list[AlignmentFinding] = []

        for i, claim in enumerate(claims, 1):
            v = verdicts.get(f"C{i}")
            if v is None:
                out.append(self._finding(new_id(), category, AlignmentStatus.UNVERIFIABLE, claim, [],
                                         "The model returned no verdict for this claim.", 0.0, facts_by_id))
                continue
            try:
                status = AlignmentStatus(v.status)
            except ValueError:
                status = AlignmentStatus.UNVERIFIABLE
            if status in (AlignmentStatus.OUT_OF_SCOPE, AlignmentStatus.UNDECLARED):
                status = AlignmentStatus.UNVERIFIABLE
            ids = [x for x in v.fact_ids if x in facts_by_id]
            reasoning = v.reasoning
            if status in EVIDENCE_BASED and not ids:  # unsupported verdict -> downgrade
                status = AlignmentStatus.UNVERIFIABLE
                reasoning = f"Downgraded: no valid supporting fact cited. {reasoning}".strip()
            out.append(self._finding(new_id(), category, status, claim, ids, reasoning,
                                     v.confidence, facts_by_id))

        for u in raw.undeclared:
            ids = [x for x in u.fact_ids if x in facts_by_id]
            if not ids:
                continue
            out.append(self._finding(new_id(), category, AlignmentStatus.UNDECLARED, None, ids,
                                     u.reasoning, u.confidence, facts_by_id))
        return out

    @staticmethod
    def _finding(fid, category, status, claim, ids, reasoning, confidence, facts_by_id) -> AlignmentFinding:
        return AlignmentFinding(
            id=fid,
            kind="undeclared_behavior" if status == AlignmentStatus.UNDECLARED else "claim_check",
            category=category, status=status, claim=claim, fact_ids=ids,
            code_locations=[loc for x in ids for loc in facts_by_id[x].locations],
            reasoning=reasoning, confidence=min(max(confidence, 0.0), 1.0))

    def _ask(self, category: PolicyCategory, claims: list[PolicyClaim],
             facts: list[ImplementationFact]) -> _Raw:
        user_prompt = self._render(category, claims, facts)
        schema = _Raw.model_json_schema()
        last_err = ""
        for _ in range(2):  # one retry on invalid JSON
            prompt = user_prompt if not last_err else (
                user_prompt + f"\n\nYour previous reply was invalid: {last_err}. Return valid JSON only.")
            text = self.llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt,
                                     response_schema=schema)
            try:
                return _Raw.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = str(exc)[:200]
        return _Raw(verdicts=[])  # fail soft: every claim becomes "unverifiable"

    @staticmethod
    def _render(category, claims, facts) -> str:
        lines = [f"CATEGORY: {category.value}", "POLICY CLAIMS:"]
        if not claims:
            lines.append("(the policy makes no claim in this category)")
        for i, c in enumerate(claims, 1):
            lines.append(f"C{i}: {c.claim}  | policy excerpt: {c.evidence[:200]!r}")
        lines.append("IMPLEMENTATION FACTS:")
        for f in facts:
            where = ", ".join(f"{l.file}:{l.line}" for l in f.locations[:3]) or "no location"
            lines.append(f"{f.id} [{f.fact_type}] {f.statement} ({where})")
        return "\n".join(lines)
