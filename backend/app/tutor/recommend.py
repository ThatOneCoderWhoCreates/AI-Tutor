"""'What should I study next?': pick the weakest topic and the video segment that teaches it."""
from __future__ import annotations

from ..schemas import Chunk, NextResponse
from ..store import merge_segments
from .bkt import MASTERED, load_student, mastery_of


def next_topic(student_id: str, cmap: dict[str, list[Chunk]]) -> NextResponse:
    state = load_student(student_id)
    candidates = []
    for order, (concept, chunks) in enumerate(cmap.items()):
        p = mastery_of(state, concept)
        if p < MASTERED:
            candidates.append((p, order, concept, chunks))

    if not candidates:
        return NextResponse(concept=None, mastery=1.0, reason="You have mastered every topic. Nice work.", segments=[])

    p, _order, concept, chunks = min(candidates, key=lambda x: (x[0], x[1]))
    tries, right = state["attempts"].get(concept, [0, 0])
    if tries == 0:
        reason = f"You have not practiced \"{concept}\" yet, and it is the next topic in the course."
    else:
        reason = (f"Your mastery of \"{concept}\" is {round(p * 100)}%, the lowest in the course "
                  f"({right} of {tries} practice answers correct).")
    return NextResponse(concept=concept, mastery=round(p, 4), reason=reason, segments=merge_segments(chunks))
