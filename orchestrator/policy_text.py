"""Join the policy documents into one string for Agent 3 (Phase A).

Agent 3 will accept an optional `policy_text` so that its undeclared
behaviour check can read the whole policy as well as the extracted
claims. A dropped claim then cannot cause a false alarm.

Each document gets a header line naming its file, so Agent 3 can tell
which document said what. The document text itself is passed through
unchanged. It is untrusted data: this function does not clean it, and
the agent that reads it must treat it as text to analyse, never as
instructions.
"""

from orchestrator.models import AssessmentState

HEADER_TEMPLATE = "=== Document: {name} ==="


def combined_policy_text(state: AssessmentState) -> str:
    """Return every policy document as one string, with a header per file."""
    parts = [
        f"{HEADER_TEMPLATE.format(name=document.name)}\n{document.text}"
        for document in state.request.policy_documents
    ]
    return "\n\n".join(parts)