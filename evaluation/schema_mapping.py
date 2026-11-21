
"""Mappings between evaluation answer keys and project schemas."""

from agents.alignment_agent.schemas import AlignmentStatus
from agents.policy_agent.schemas import PolicyCategory


def map_expected_status(value: str) -> AlignmentStatus:
    """Convert an answer-key status into the project's AlignmentStatus."""
    try:
        return AlignmentStatus(value)
    except ValueError as exc:
        allowed = [status.value for status in AlignmentStatus]
        raise ValueError(
            f"Unknown expected status {value!r}. Allowed values: {allowed}"
        ) from exc


def map_expected_category(value: str) -> PolicyCategory:
    """Convert an answer-key category into the project's PolicyCategory."""
    try:
        return PolicyCategory(value)
    except ValueError as exc:
        allowed = [category.value for category in PolicyCategory]
        raise ValueError(
            f"Unknown policy category {value!r}. Allowed values: {allowed}"
        ) from exc
