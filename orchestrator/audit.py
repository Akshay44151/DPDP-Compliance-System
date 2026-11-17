"""Audit record (Phase A).

Builds one JSON-friendly record of a finished assessment and writes it to
disk, so a run can be checked later. Deterministic Python, no LLM.

Rules:
- The record stores policy document NAMES and CHARACTER COUNTS only. The
  policy text itself is never written.
- It stores only what is in the AssessmentState plus the summary. Nothing
  from .env, no keys.
- A missing result stays None. It is never turned into an empty result.
- A file is never overwritten. Writing the same record twice fails loudly.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.models import AssessmentState
from orchestrator.summary import AssessmentSummary

AUDIT_VERSION = 1


def _safe_name(text: str) -> str:
    """Make text safe to use inside a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return cleaned or "assessment"


def build_audit_record(
    state: AssessmentState,
    summary: AssessmentSummary,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Return a plain dict that json.dumps can write."""
    created = created_at or datetime.now(timezone.utc)
    request = state.request

    documents = [
        {"name": document.name, "chars": len(document.text)}
        for document in request.policy_documents
    ]

    return {
        "audit_version": AUDIT_VERSION,
        "created_at": created.isoformat(),
        "assessment_id": request.assessment_id,
        "project_path": request.project_path,
        "policy_documents": documents,
        "policy_claims": [
            claim.model_dump(mode="json") for claim in state.policy_claims
        ],
        "rejected_claims": [
            claim.model_dump(mode="json") for claim in state.rejected_claims
        ],
        "implementation_report": state.implementation_report,
        "alignment_result": state.alignment_result,
        "step_log": [
            record.model_dump(mode="json") for record in state.step_log
        ],
        "summary": summary.model_dump(mode="json"),
    }


def write_audit_record(record: dict[str, Any], directory: str | Path) -> Path:
    """Write the record as a JSON file and return its path.

    The folder is created if needed. If the file already exists the write
    fails with FileExistsError instead of replacing it.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    name = (
        f"{_safe_name(str(record['assessment_id']))}"
        f"__{_safe_name(str(record['created_at']))}.json"
    )
    path = folder / name

    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path