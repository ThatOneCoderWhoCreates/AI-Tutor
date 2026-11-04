from app.llm import extract_json
from app.rag.chunker import build_chunks
from app.trust import split_claims
from app.tutor.bkt import PARAMS, bkt_update


def test_bkt_correct_answers_raise_mastery():
    p = PARAMS.p_init
    for _ in range(5):
        new_p = bkt_update(p, True)
        assert new_p > p
        p = new_p
    assert p > 0.9


def test_bkt_wrong_answer_lowers_mastery():
    assert bkt_update(0.6, False) < 0.6


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_chunker_splits_on_slide_change_and_max_length():
    segments = [{"start": i * 10.0, "end": i * 10.0 + 10, "text": f"sentence {i}."} for i in range(20)]
    slides = [{"start": 0.0, "text": "Intro"}, {"start": 50.0, "text": "Gradient Descent"}]
    chunks = build_chunks("t", segments, slides, max_seconds=60)
    assert chunks[0].start == 0 and chunks[0].end == 50
    assert chunks[0].slide_text == "Intro"
    assert chunks[1].start == 50 and chunks[1].slide_text == "Gradient Descent"
    assert all(c.end - c.start <= 60 for c in chunks)
    assert [c.id for c in chunks][:2] == ["t_001", "t_002"]


def test_quiz_validator_accepts_common_small_model_mistakes():
    from app.tutor.quiz import _validator

    validate = _validator({"c1"}, "c1")
    out = validate({"questions": [
        {"question": "Q1?", "options": ["A) one", "B) two", "C) three", "D) four"], "answer": "B"},
        {"question": "Q2?", "options": ["red", "green", "blue", "pink"], "correct_answer": "C) blue",
         "source_chunk_id": "made-up"},
    ]})
    assert out[0]["options"] == ["one", "two", "three", "four"] and out[0]["answer_index"] == 1
    assert out[1]["answer_index"] == 2
    assert out[1]["source_chunk_id"] == "c1"


def test_split_claims_removes_citation_marks():
    claims = split_claims("Gradient descent walks downhill on the loss [1]. The learning rate sets the step size [2].")
    assert claims == ["Gradient descent walks downhill on the loss.", "The learning rate sets the step size."]


def test_stemming_matches_singular_and_plural():
    from app.rag.index import stem, tokenize

    assert tokenize("linked list") == tokenize("linked lists")
    assert stem("queues") == "queue" and stem("queries") == "query" and stem("hashes") == "hash"
    assert stem("class") == "class" and stem("analysis") == "analysis" and stem("status") == "status"
