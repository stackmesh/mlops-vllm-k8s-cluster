from __future__ import annotations

import re
from dataclasses import dataclass

CITATION = re.compile(r"\[([A-Za-z0-9_-]+)\]")


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str
    citations: tuple[str, ...]
    unknown: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "citations": list(self.citations),
            "unknown": list(self.unknown),
        }


def enforce_citations(answer: str, allowed_ids: set[str]) -> GuardrailResult:
    """Fail closed unless every bracketed id was actually retrieved.

    This is a post-generation check, not a logit constraint. It drops answers
    that cite documents outside the retrieved set or that cite nothing.
    """
    citations = tuple(dict.fromkeys(CITATION.findall(answer)))
    if not allowed_ids:
        return GuardrailResult(False, "no supporting context was retrieved", citations, citations)
    unknown = tuple(citation for citation in citations if citation not in allowed_ids)
    if unknown:
        listed = ", ".join(unknown)
        return GuardrailResult(
            False,
            f"answer cited documents outside the retrieved set: {listed}",
            citations,
            unknown,
        )
    if not citations:
        return GuardrailResult(False, "answer omitted citations", citations, ())
    return GuardrailResult(True, "citations match the retrieved set", citations, ())
