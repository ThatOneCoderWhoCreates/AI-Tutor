"""Step 7 (built last): check every sentence of an answer against the lecture text.

Each sentence (claim) gets supported=True/False with a short reason.
"""
from __future__ import annotations

import re

from .llm import ask_llm_json
from .schemas import Chunk, Claim

PROMPT = """TASK: verify
Decide whether the CLAIM is supported by the EVIDENCE from a lecture.
"Supported" means the evidence states it or clearly implies it. If the evidence does not say it, it is NOT supported,
even if the claim is true in general.
Reply with JSON only: {{"supported": true or false, "reason": "one short sentence"}}

CLAIM: {claim}

EVIDENCE:
{evidence}
"""


def split_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|(?<=\])\s+(?=[A-Z])", answer.strip())
    claims = []
    for s in sentences:
        clean = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", s).strip()
        if len(clean.split()) >= 3:
            claims.append(clean)
    return claims


def _validate(data):
    if not isinstance(data, dict) or "supported" not in data:
        raise ValueError('Expected {"supported": true/false, "reason": "..."}')
    sup = data["supported"]
    if isinstance(sup, str):
        sup = sup.strip().lower() in ("true", "yes", "supported")
    return bool(sup), str(data.get("reason", ""))


def verify_answer(answer: str, evidence: list[Chunk]) -> list[Claim]:
    if not evidence:
        return []
    ev_text = "\n".join(f"- {c.text}" for c in evidence)
    out = []
    for claim in split_claims(answer):
        try:
            supported, reason = ask_llm_json(PROMPT.format(claim=claim, evidence=ev_text), validate=_validate)
        except RuntimeError as exc:
            supported, reason = False, f"could not check: {exc}"
        out.append(Claim(text=claim, supported=supported, reason=reason))
    return out
