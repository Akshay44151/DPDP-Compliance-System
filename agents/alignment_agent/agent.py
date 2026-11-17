"""Agent 3: Alignment Agent (LangChain).

Pass 1 (per category): compare each policy claim with implementation facts -> a verdict per claim.
Pass 2 (once, whole project): find code behaviour that NO claim in the whole policy discloses.
Deterministic guards keep the LLM honest: every verdict must cite real fact ids, every claim gets
exactly one finding, out-of-scope claims never reach the LLM, and "undeclared" items that merely
repeat facts already explained by a verdict are dropped.
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Literal

from groq import BadRequestError
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, ValidationError

from agents.alignment_agent.schemas import AlignmentFinding, AlignmentResult, AlignmentStatus
from agents.implementation_agent.schemas import (IN_SCOPE_CATEGORIES, EvidenceItem,
                                                 ImplementationFact, ImplementationReport)
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim, PolicyExtractionResult
from llm.langchain_model import get_langchain_model

VERDICT_PROMPT = """You are the Alignment Agent in a DPDP Act 2023 compliance pipeline. You compare \
what a privacy policy PROMISES (claims) with what the application code ACTUALLY DOES (facts) for \
ONE privacy category.

For every claim give one verdict:
- "aligned": the facts show the code does what the claim says.
- "partial": the facts support only part of the claim.
- "violated": the facts show the code contradicts the claim (including facts of type "absent" \
for something the claim promises, e.g. claim promises consent but data is used with no consent check).
- "unverifiable": the facts say nothing relevant to this claim.
Rules:
- A fact of type "absent" can also SUPPORT a claim that promises something does not happen (for \
example the claim says personal data is not shared with advertisers and a fact says no code sends \
data to external services: that is "aligned").
- Do not hedge. If the facts show the code doing what the claim promises it does NOT do (for \
example personal data going to an external advertising or tracking URL when the claim says data is \
not used for advertising), the verdict is "violated". Use "partial" only when part of the claim is \
clearly supported and another part is not.
- aligned, partial and violated MUST cite fact ids from the list. Never invent ids.
- Judge only from the facts given. Do not assume code behaviour that is not listed.
- The claim wording is only a label; judge by the policy excerpt (the verbatim quote).
- Keep reasoning to one or two sentences. confidence is between 0 and 1."""

UNDECLARED_PROMPT = """You are the Alignment Agent in a DPDP Act 2023 compliance pipeline. You are \
given ALL claims of a privacy policy and the personal-data practices found in the application code \
(each collected field, data use or data transfer is one line with an id).

List "undeclared" practices: code that collects, logs, shares or sends personal data in a way that NO \
claim anywhere in the policy discloses (for example a sensitive identifier the policy never mentions, \
or personal data sent to a third party the policy does not name).
Rules:
- Check every claim before listing something. A practice that any claim already mentions is NOT undeclared.
- Group related lines into ONE item per distinct practice (for example every undisclosed identifier \
field together) and cite every id in the group.
- Do not list lines under ALREADY REPORTED.
- Cite ids from the list only. Keep reasoning to one or two sentences. confidence is between 0 and 1."""

HUMAN_PROMPT = "{task_input}"  # rendered claims/facts go in as a variable, so braces are safe

# Failures that mean "the model's structured output was unusable"; anything else (auth, network,
# rate limit) is a real error and is allowed to propagate.
RETRYABLE = (OutputParserException, ValidationError, BadRequestError)


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # all fields required: strict-schema friendly

    claim_id: str
    status: Literal["aligned", "partial", "violated", "unverifiable"]
    fact_ids: list[str]
    reasoning: str
    confidence: float


class _VerdictList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[_Verdict]


class _Undeclared(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[str]
    reasoning: str
    confidence: float


class _UndeclaredList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    undeclared: list[_Undeclared]


# Evidence kinds that are personal-data practices, and the category an undisclosed one falls under.
PRACTICE_CATEGORY = {
    "model_field": PolicyCategory.TRANSPARENCY_NOTICE,         # collected but not disclosed
    "data_sharing": PolicyCategory.PURPOSE_LIMITATION,
    "outbound_email": PolicyCategory.PURPOSE_LIMITATION,
    "personal_data_logged": PolicyCategory.PURPOSE_LIMITATION,
}

EVIDENCE_BASED = {AlignmentStatus.ALIGNED, AlignmentStatus.PARTIAL, AlignmentStatus.VIOLATED}


class AlignmentAgent:
    def __init__(self, model=None) -> None:
        self.model = model or get_langchain_model()
        self.verdict_chain = ChatPromptTemplate.from_messages(
            [("system", VERDICT_PROMPT), ("human", HUMAN_PROMPT)]
        ) | self.model.with_structured_output(_VerdictList)
        self.undeclared_chain = ChatPromptTemplate.from_messages(
            [("system", UNDECLARED_PROMPT), ("human", HUMAN_PROMPT)]
        ) | self.model.with_structured_output(_UndeclaredList)
        self.retries = 0
        self.failures = 0
        self.last_error: str | None = None
        self.failed_calls: list[str] = []

    # ---------- public ----------
    def run(self, policy: PolicyExtractionResult, implementation: ImplementationReport) -> AlignmentResult:
        self.retries = self.failures = 0
        self.last_error = None
        self.failed_calls = []
        findings: list[AlignmentFinding] = []
        counter = itertools.count(1)

        def new_id() -> str:
            return f"A-{next(counter):03d}"

        for claim in policy.claims:
            if claim.category not in IN_SCOPE_CATEGORIES:
                findings.append(AlignmentFinding(
                    id=new_id(), kind="claim_check", category=claim.category,
                    status=AlignmentStatus.OUT_OF_SCOPE, claim=claim,
                    reason_code="out_of_scope",
                    reasoning="Category is outside the five principles evaluated by this framework."))

        # Pass 1: verdicts, one category at a time
        for category in IN_SCOPE_CATEGORIES:
            claims = [c for c in policy.claims if c.category == category]
            facts = [f for f in implementation.facts if f.category == category]
            if not claims:
                continue
            if not facts:  # nothing to compare against
                for c in claims:
                    findings.append(AlignmentFinding(
                        id=new_id(), kind="claim_check", category=category,
                        status=AlignmentStatus.UNVERIFIABLE, claim=c,
                        reason_code="evidence_gap",
                        reasoning="No implementation facts were extracted for this category."))
                continue
            call_id = f"verdict:{category.value}"
            raw = self._ask(self.verdict_chain, self._render_verdict(category, claims, facts),
                            _VerdictList, call_id)
            new = self._verdict_findings(
                category, claims, facts, raw, new_id, model_failed=call_id in self.failed_calls)
            findings.extend(new)

        # Pass 2: undeclared behaviour, with the whole policy in view
        reported = {e for f in findings if f.status == AlignmentStatus.VIOLATED for e in f.evidence_ids}
        findings.extend(self._undeclared_findings(policy, implementation, reported, new_id))

        summary = dict(Counter(f.status.value for f in findings))
        return AlignmentResult(findings=findings, summary=summary,
                               llm_retries=self.retries, llm_failures=self.failures,
                               failed_calls=self.failed_calls)

    # ---------- pass 1 ----------
    def _verdict_findings(self, category, claims, facts, raw: _VerdictList, new_id,
                          model_failed: bool = False) -> list[AlignmentFinding]:
        facts_by_id = {f.id: f for f in facts}
        verdicts = {}
        for v in raw.verdicts:
            verdicts.setdefault(v.claim_id, v)  # first verdict per claim wins
        out: list[AlignmentFinding] = []
        for i, claim in enumerate(claims, 1):
            v = verdicts.get(f"C{i}")
            if v is None:
                reason_code = "model_failure" if model_failed else "invalid_model_output"
                reasoning = ("The model call failed after retries; this claim could not be assessed."
                             if model_failed else "The model returned no verdict for this claim.")
                out.append(self._finding(new_id(), category, AlignmentStatus.UNVERIFIABLE, claim, [],
                                         reasoning, 0.0, facts_by_id, reason_code))
                continue
            try:
                status = AlignmentStatus(v.status)
            except ValueError:
                status = AlignmentStatus.UNVERIFIABLE
            reason_code = "model_assessed"
            if status in (AlignmentStatus.OUT_OF_SCOPE, AlignmentStatus.UNDECLARED):
                status = AlignmentStatus.UNVERIFIABLE
                reason_code = "invalid_model_output"
            elif status == AlignmentStatus.UNVERIFIABLE:
                reason_code = "evidence_gap"
            ids = [x for x in v.fact_ids if x in facts_by_id]
            reasoning = v.reasoning
            if status in EVIDENCE_BASED and not ids:  # unsupported verdict -> downgrade
                status = AlignmentStatus.UNVERIFIABLE
                reasoning = f"Downgraded: no valid supporting fact cited. {reasoning}".strip()
                reason_code = "unsupported_evidence"
            out.append(self._finding(new_id(), category, status, claim, ids, reasoning,
                                     v.confidence, facts_by_id, reason_code))
        return out

    # ---------- pass 2 ----------
    def _undeclared_findings(self, policy, implementation, reported: set[str], new_id):
        practices = [e for e in implementation.evidence if e.kind in PRACTICE_CATEGORY]
        if not practices:
            return []
        raw = self._ask(self.undeclared_chain,
                        self._render_undeclared(policy.claims, practices, reported),
                        _UndeclaredList, "undeclared")
        by_id = {e.id: e for e in practices}
        seen = set(reported)  # violations are already reported; later items may not repeat earlier ones
        out: list[AlignmentFinding] = []
        for u in raw.undeclared:
            ids = [x for x in dict.fromkeys(u.evidence_ids) if x in by_id and x not in seen]
            if not ids:
                continue
            seen.update(ids)
            out.append(AlignmentFinding(
                id=new_id(), kind="undeclared_behavior", category=PRACTICE_CATEGORY[by_id[ids[0]].kind],
                status=AlignmentStatus.UNDECLARED, reason_code="undeclared_practice",
                claim=None, evidence_ids=ids,
                fact_ids=[f.id for f in implementation.facts if set(f.evidence_ids) & set(ids)],
                code_locations=[by_id[x].location for x in ids],
                reasoning=u.reasoning, confidence=min(max(u.confidence, 0.0), 1.0)))
        return out

    # ---------- shared ----------
    @staticmethod
    def _finding(fid, category, status, claim, ids, reasoning, confidence, facts_by_id,
                 reason_code: str = "model_assessed") -> AlignmentFinding:
        return AlignmentFinding(
            id=fid,
            kind="undeclared_behavior" if status == AlignmentStatus.UNDECLARED else "claim_check",
            category=category, status=status, reason_code=reason_code, claim=claim, fact_ids=ids,
            evidence_ids=list(dict.fromkeys(e for x in ids for e in facts_by_id[x].evidence_ids)),
            code_locations=[loc for x in ids for loc in facts_by_id[x].locations],
            reasoning=reasoning, confidence=min(max(confidence, 0.0), 1.0))

    def _ask(self, chain, task_input: str, model_cls, call_id: str):
        last_err = ""
        for attempt in range(2):  # one retry when the structured output is unusable
            text = task_input if not last_err else (
                task_input + f"\n\nYour previous reply was invalid: {last_err}. "
                "Return the required structured output only.")
            try:
                return model_cls.model_validate(chain.invoke({"task_input": text}))
            except RETRYABLE as exc:
                last_err = self.last_error = str(exc)[:200]
                if attempt == 0:
                    self.retries += 1
        self.failures += 1
        self.failed_calls.append(call_id)
        return model_cls.model_validate({k: [] for k in model_cls.model_fields})  # fail soft

    @staticmethod
    def _fact_line(f: ImplementationFact) -> str:
        where = ", ".join(f"{l.file}:{l.line}" for l in f.locations[:3]) or "no location"
        return f"{f.id} [{f.fact_type}] {f.statement} ({where})"

    def _render_verdict(self, category, claims, facts) -> str:
        lines = [f"CATEGORY: {category.value}", "POLICY CLAIMS:"]
        for i, c in enumerate(claims, 1):
            lines.append(f"C{i}: policy excerpt: {c.evidence[:400]!r}  | label: {c.claim}")
        lines.append("IMPLEMENTATION FACTS:")
        lines.extend(self._fact_line(f) for f in facts)
        return "\n".join(lines)

    @staticmethod
    def _practice(e: EvidenceItem) -> str:
        d = e.detail
        if e.kind == "model_field":
            return f"stores field {d.get('model')}.{d.get('field')} ({d.get('field_type')})"
        if e.kind == "data_sharing":
            return f"sends data to an external service via {d.get('call')}: {e.snippet[:150]!r}"
        if e.kind == "outbound_email":
            return (f"function {d.get('function')} sends email "
                    f"(references consent: {d.get('references_consent')})")
        if e.kind == "personal_data_logged":
            return f"writes personal data to logs ({', '.join(d.get('personal_tokens', []))}): {e.snippet[:100]!r}"
        return e.snippet[:150]

    def _render_undeclared(self, claims: list[PolicyClaim], practices: list[EvidenceItem],
                           reported: set[str]) -> str:
        lines = ["TASK: undeclared", "ALL POLICY CLAIMS (every category):"]
        for c in claims:
            lines.append(f"- [{c.category.value}] {c.evidence[:300]!r}")
        lines.append("CODE PRACTICES:")
        for e in practices:
            lines.append(f"{e.id} [{e.kind}] {e.location.file}:{e.location.line} {self._practice(e)}")
        lines.append(f"ALREADY REPORTED: {', '.join(sorted(reported)) or '(none)'}")
        return "\n".join(lines)