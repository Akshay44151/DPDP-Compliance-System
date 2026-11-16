"""Split a policy document into sections at numbered headings.

Pure code, no LLM. A heading is a short line like "4. Retention":
a number, a dot or bracket, then a capitalised phrase with no
sentence punctuation. Text before the first heading (the title) is
kept with the first section, and text after the last heading
(including any trailing notice) stays in the last section.
"""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^\s*\d+[.)]\s+[A-Z][^.!?\n]{0,80}$")


@dataclass(frozen=True)
class Section:
    heading: str
    text: str
    start_line: int  # 1-based line in the original document


def split_into_sections(document_text: str) -> list[Section]:
    """Split at numbered headings; fall back to one section if none."""
    if not document_text or not document_text.strip():
        return []

    lines = document_text.splitlines()
    heading_rows = [i for i, line in enumerate(lines) if _HEADING.match(line)]

    if len(heading_rows) < 2:
        return [
            Section(
                heading="(whole document)",
                text=document_text.strip(),
                start_line=1,
            )
        ]

    sections: list[Section] = []
    for n, row in enumerate(heading_rows):
        end = heading_rows[n + 1] if n + 1 < len(heading_rows) else len(lines)
        begin = 0 if n == 0 else row
        sections.append(
            Section(
                heading=lines[row].strip(),
                text="\n".join(lines[begin:end]).strip(),
                start_line=begin + 1,
            )
        )
    return sections