"""Step 5: answer a question using only the retrieved lecture chunks."""
from __future__ import annotations

import re

from .. import fallback
from ..llm import ask_llm
from ..schemas import AskResponse, Chunk, Citation
from .index import Retriever, tokenize

NOT_COVERED = "The lecture does not cover this."

SYSTEM = (
    "You are a friendly teaching assistant. You only know what is in the lecture sources you are given. "
    "Never use outside knowledge."
)

PROMPT = """TASK: answer
Answer the student's question using ONLY the lecture sources below.
Rules:
- Use only facts stated in SOURCES. Do not add anything from your own knowledge.
- End every sentence with the number of the source it comes from, in square brackets, like [2].
- The question may use different wording from the lecture: singular or plural ("linked list" and "linked lists"),
  a synonym, an abbreviation or a spelling mistake. If SOURCES discuss the same idea, answer from them.
- Only if SOURCES do not discuss the idea at all, reply with exactly this sentence: {not_covered}
- Use simple English, 2 to 5 sentences.

QUESTION: {question}

SOURCES:
{sources}
"""


def fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def format_sources(chunks: list[Chunk]) -> str:
    blocks = []
    for n, c in enumerate(chunks, start=1):
        slide = c.slide_text.splitlines()[0] if c.slide_text else ""
        header = f"[{n}] (lecture {c.lecture_id}, {fmt_time(c.start)}-{fmt_time(c.end)}"
        header += f", slide: {slide})" if slide else ")"
        blocks.append(f"{header}\n{c.text}")
    return "\n".join(blocks)


def normalize_citations(answer: str) -> str:
    """Small local models write citations in many styles. Turn them all into [n].

    "[Source 2]", "[source 2, 3]", "(Source 2)", "[2][3]" -> "[2]", "[2, 3]", "[2]", "[2] [3]"
    """
    answer = re.sub(r"[\[(]\s*sources?\s*(\d+(?:\s*(?:,|and)\s*(?:sources?\s*)?\d+)*)\s*[\])]",
                    lambda m: "[" + ", ".join(re.findall(r"\d+", m.group(1))) + "]", answer, flags=re.I)
    return re.sub(r"\]\[", "] [", answer)


def parse_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    nums: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for part in group.split(","):
            n = int(part)
            if 1 <= n <= len(chunks) and n not in nums:
                nums.append(n)
    return [
        Citation(n=n, chunk_id=chunks[n - 1].id, lecture_id=chunks[n - 1].lecture_id,
                 start=chunks[n - 1].start, end=chunks[n - 1].end, concept=chunks[n - 1].concept)
        for n in nums
    ]


REFUSAL = re.compile(
    r"\b(lectures?|sources?|excerpts?|materials?|context|provided text|given text)\b[^.]{0,40}?"
    r"\b(does not|doesn't|do not|don't|did not|didn't)\b"
    r"|\bnot (covered|mentioned|discussed|addressed|included) in\b"
    r"|\bno information\b|\b(cannot|can't|unable to) answer\b"
    r"|\bwithout (mentioning|discussing|covering)\b|\bonly (covers|discusses|mentions)\b",
    re.I,
)


def is_refusal(reply: str, citations: list[Citation]) -> bool:
    """True when the model is saying the lecture does not have the answer.

    Small models often add a citation even to "The lecture does not cover this [1]",
    so a short reply that is mostly a refusal counts as one even with a citation.
    """
    text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", reply).strip()
    if not text:
        return True
    if not REFUSAL.search(text):
        return False
    # A refusal opens with one ("The sources do not discuss X. They only cover Y. I cannot answer.").
    # A real answer that also mentions a gap ("A heap keeps ... The lecture does not cover heap sort.")
    # opens with the answer, so it is kept.
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    flagged = [bool(REFUSAL.search(x)) for x in sentences]
    if not flagged[0]:
        return False
    return not citations or len(text) <= 160 or sum(flagged) * 2 >= len(flagged)


def _grounded(question: str, chunks: list[Chunk], hint: str = "") -> AskResponse | None:
    prompt = PROMPT.format(question=question, sources=format_sources(chunks), not_covered=NOT_COVERED)
    if hint:
        prompt = prompt.replace("\nQUESTION:", f"\nNOTE: {hint}\n\nQUESTION:", 1)
    reply = normalize_citations(ask_llm(prompt, system=SYSTEM))
    citations = parse_citations(reply, chunks)
    if is_refusal(reply, citations):
        return None
    return AskResponse(answer=reply, citations=citations, source="lecture")


def chunks_for_named_topic(question: str, retriever: Retriever, lecture_id: str | None = None,
                           limit: int = 4) -> list[Chunk]:
    """Chunks of the course topic whose name appears in the question, ignoring singular/plural.
    The longest matching topic name wins ("linked lists" over "lists")."""
    q = " " + " ".join(tokenize(question)) + " "
    best, best_len = None, 0
    for c in retriever.chunks:
        if not c.concept or (lecture_id and c.lecture_id != lecture_id):
            continue
        name = " ".join(tokenize(c.concept))
        if name and f" {name} " in q and len(name) > best_len:
            best, best_len = c.concept, len(name)
    if not best:
        return []
    found = [c for c in retriever.chunks if c.concept == best and (not lecture_id or c.lecture_id == lecture_id)]
    return sorted(found, key=lambda c: (c.lecture_id, c.start))[:limit]


def answer_question(question: str, retriever: Retriever, lecture_id: str | None = None,
                    k: int | None = None, mode: str = "hybrid", history: list[dict] | None = None,
                    use_fallback: bool = True) -> tuple[AskResponse, list[Chunk]]:
    chunks = retriever.search(question, k=k, mode=mode, lecture_id=lecture_id)
    resp = None
    if chunks:
        resp = _grounded(question, chunks)
    if resp is None:
        # The model said "not covered", but the question names a topic of the course
        # ("what is linked list" vs the topic "linked lists"). Try once more with that topic's chunks.
        topic_chunks = chunks_for_named_topic(question, retriever, lecture_id)
        if topic_chunks:
            topic = topic_chunks[0].concept
            resp = _grounded(question, topic_chunks,
                             hint=f"The question asks about \"{topic}\", and these sources are the lecture's "
                                  f"explanation of {topic}. Answer from them.")
            chunks = topic_chunks if resp else chunks
    if resp is None:
        resp = AskResponse(answer=NOT_COVERED, citations=[], source="none")
        if use_fallback and fallback.config().provider != "none":
            topics = list(dict.fromkeys(c.concept for c in retriever.chunks if c.concept))
            try:
                ga = fallback.general_answer(question, history=history, topics=topics)
                resp = AskResponse(answer=f"{fallback.PREFIX}\n\n{ga.text}", citations=[], source="general",
                                   answered_by=ga.answered_by,
                                   note=("Skipped: " + " ".join(ga.skipped)) if ga.skipped else None)
            except fallback.FallbackError as exc:
                resp.note = f"General answer unavailable: {exc}"
    return resp, chunks
