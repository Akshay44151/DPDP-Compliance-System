"""Deterministic retention parser (no LLM).

Reads the already-verified evidence text of a storage_retention claim and
returns each retention commitment as a period in days plus a trigger.
Because it never calls a model, it cannot affect Agent 1's recall, and a
period or trigger is only reported when the text clearly states it.

Conversion rules (to be agreed with Agent 2's owner):
    1 week = 7 days, 1 month = 30 days, 1 year = 365 days.
Note that "24 months" gives 720 days but "2 years" gives 730.
"""

import re
from dataclasses import dataclass

from agents.policy_agent.schemas import PolicyCategory, PolicyClaim

DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "sixty": 60, "ninety": 90,
}

_NUMBER = r"\d+|" + "|".join(_NUMBER_WORDS)
_DURATION = re.compile(
    rf"\b({_NUMBER})[\s-]+(day|week|month|year)s?\b", re.IGNORECASE
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")

_TRIGGERS = [
    (
        "last_login",
        re.compile(r"last\s+(log[\s-]?in|sign[\s-]?in)", re.IGNORECASE),
    ),
    (
        "account_deletion",
        re.compile(
            r"account\s+(deletion|closure|is\s+deleted|is\s+closed)"
            r"|delet\w+\s+(of\s+)?(your|the)\s+account"
            r"|clos\w+\s+(of\s+)?(your|the)\s+account",
            re.IGNORECASE,
        ),
    ),
    (
        "collection",
        re.compile(r"\bcollection\b|\bcollected\b|we\s+collect", re.IGNORECASE),
    ),
    (
        "purpose_fulfilled",
        re.compile(
            r"purpose\s+(is\s+|has\s+been\s+)?"
            r"(fulfilled|achieved|served|completed)"
            r"|no\s+longer\s+(needed|necessary|required)",
            re.IGNORECASE,
        ),
    ),
]


@dataclass(frozen=True)
class RetentionCommitment:
    period_days: int
    trigger: str  # last_login / account_deletion / collection / purpose_fulfilled / other
    text: str     # the sentence the commitment was read from


def _to_number(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBER_WORDS[word.lower()]


def _find_trigger(window: str) -> str:
    """Pick the trigger phrase that appears first in the text after the period."""
    best_name = "other"
    best_start = len(window) + 1
    for name, pattern in _TRIGGERS:
        match = pattern.search(window)
        if match and match.start() < best_start:
            best_name, best_start = name, match.start()
    return best_name


def parse_retention(evidence: str) -> list[RetentionCommitment]:
    """Return every retention commitment found in the evidence text."""
    if not evidence or not evidence.strip():
        return []

    commitments: list[RetentionCommitment] = []
    seen: set[tuple[int, str, str]] = set()

    for sentence in _SENTENCES.split(evidence.strip()):
        matches = list(_DURATION.finditer(sentence))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(sentence)
            window = sentence[match.end():end]
            days = _to_number(match.group(1)) * DAYS_PER_UNIT[match.group(2).lower()]
            if days <= 0:
                continue

            commitment = RetentionCommitment(
                period_days=days,
                trigger=_find_trigger(window),
                text=sentence.strip(),
            )
            key = (commitment.period_days, commitment.trigger, commitment.text)
            if key not in seen:
                seen.add(key)
                commitments.append(commitment)

    return commitments


def parse_claim_retention(claim: PolicyClaim) -> list[RetentionCommitment]:
    """Parse a claim, but only if it is a storage_retention claim."""
    if claim.category != PolicyCategory.STORAGE_RETENTION:
        return []
    return parse_retention(claim.evidence)