"""FastAPI backend. Run from the backend folder:
    python -m uvicorn app.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import settings
from .rag.answer import answer_question
from .rag.index import Retriever
from .schemas import (
    AnswerRequest, AnswerResponse, AskRequest, AskResponse, ConceptInfo, NextRequest, NextResponse,
    QuizQuestionPublic, QuizRequest, QuizResponse, RenameRequest, Segment, YouTubeRequest,
    BKTFitRequest, FinetuneRequest, UseModelRequest,
)
from .store import (
    concept_map, lecture_dir, list_lectures, load_chunks, merge_segments, read_json, rename_concept, slugify,
    write_json,
)
from . import jobs
from .rag.embed import active_embedding_model, set_active_embedding_model
from .trust import verify_answer
from .tutor import bkt_fit
from .tutor.bkt import DEFAULT, active_params, load_student, mastery_of, record_answer, replay_all_students, reset_params
from .tutor.quiz import find_question, get_quiz
from .tutor.recommend import next_topic

_index_lock = threading.Lock()


def rebuild_index(app: FastAPI, fresh: bool = False) -> int:
    """fresh=True makes a new Retriever, needed after switching the search model."""
    with _index_lock:
        retriever = None if fresh else getattr(app.state, "retriever", None)
        retriever = retriever or Retriever()
        retriever.build(load_chunks())
        app.state.retriever = retriever
        return len(retriever.chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (settings.lectures_dir, settings.students_dir, settings.quizzes_dir):
        d.mkdir(parents=True, exist_ok=True)
    n = rebuild_index(app)
    print(f"[tutor] LLM={settings.llm_provider}:{settings.llm_model} "
          f"embeddings={settings.embedding_backend} store={settings.vector_store} chunks={n}")
    yield


app = FastAPI(title="Lecture-Grounded Adaptive Tutor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    """Model problems (Ollama off, model not pulled, bad JSON) reach the UI as a readable message."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _retriever() -> Retriever:
    return app.state.retriever


# ---------------- general ----------------
def _ollama_status() -> str:
    """'ready', 'no-model' (Ollama runs but the model is not pulled) or 'offline'."""
    import httpx

    try:
        tags = httpx.get(f"{settings.ollama_host}/api/tags", timeout=2).json()
    except Exception:
        return "offline"
    names = {m.get("name", "") for m in tags.get("models", [])}
    wanted = settings.llm_model
    if wanted in names or f"{wanted}:latest" in names:
        return "ready"
    return "no-model"


@app.get("/health")
def health():
    return {
        "ok": True,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model if settings.llm_provider != "mock" else "mock",
        "llm_status": _ollama_status() if settings.llm_provider == "ollama" else "ready",
        "chunks": len(_retriever().chunks),
        "fallback": _fallback_status(),
    }


def _fallback_status() -> dict:
    from . import fallback

    cfg = fallback.config()
    return {"provider": cfg.provider, "model": cfg.model, "label": cfg.label, "ready": cfg.ready, "problem": cfg.problem,
            "chain": fallback.status()}


@app.post("/reindex")
def reindex():
    return {"chunks": rebuild_index(app)}


# ---------------- lectures ----------------
@app.get("/lectures")
def lectures():
    by_lecture: dict[str, list] = {}
    for c in _retriever().chunks:
        by_lecture.setdefault(c.lecture_id, []).append(c)
    out = []
    for lec in list_lectures():
        cs = by_lecture.get(lec["id"], [])
        meta = read_json(lecture_dir(lec["id"]) / "meta.json", {}) or {}
        lec.update(chunks=len(cs), topics=len({c.concept for c in cs if c.concept}),
                   source_url=meta.get("source_url"), error=meta.get("error"))
        out.append(lec)
    return out


def _run_relabel(lecture_id: str):
    from .tutor.concepts import relabel_lecture

    try:
        relabel_lecture(lecture_id)
    except Exception as exc:
        print(f"[tutor] relabel failed for {lecture_id}: {exc}")
    finally:
        rebuild_index(app)


@app.post("/lectures/{lecture_id}/relabel")
def relabel(lecture_id: str, background: BackgroundTasks):
    meta = read_json(lecture_dir(lecture_id) / "meta.json", None)
    if meta is None or meta.get("status") != "ready":
        raise HTTPException(400, "Only a lecture that is ready can be relabeled.")
    meta.update(status="labeling", error=None)
    write_json(lecture_dir(lecture_id) / "meta.json", meta)
    background.add_task(_run_relabel, lecture_id)
    return {"id": lecture_id, "status": "labeling"}


@app.delete("/lectures/{lecture_id}")
def delete_lecture(lecture_id: str):
    d = lecture_dir(slugify(lecture_id))
    if not d.exists() or d.resolve().parent != settings.lectures_dir.resolve():
        raise HTTPException(404, "Unknown lecture")
    shutil.rmtree(d)
    return {"deleted": lecture_id, "chunks": rebuild_index(app)}


@app.get("/lectures/{lecture_id}/status")
def lecture_status(lecture_id: str):
    meta = read_json(lecture_dir(lecture_id) / "meta.json", None)
    if meta is None:
        raise HTTPException(404, "Unknown lecture")
    return {"id": lecture_id, "status": meta.get("status"), "error": meta.get("error")}


@app.get("/media/{lecture_id}")
def media(lecture_id: str):
    meta = read_json(lecture_dir(lecture_id) / "meta.json", {}) or {}
    video = meta.get("video")
    path = lecture_dir(lecture_id) / video if video else None
    if not path or not path.exists():
        raise HTTPException(404, "This lecture has no video file")
    return FileResponse(path)


def _run_ingest(path: Path, lecture_id: str, title: str):
    from .rag.ingest import ingest_video

    try:
        ingest_video(path, lecture_id, title)
        rebuild_index(app)
    except Exception as exc:
        print(f"[tutor] ingest failed for {lecture_id}: {exc}")
    finally:
        path.unlink(missing_ok=True)


@app.post("/lectures/upload")
async def upload_lecture(background: BackgroundTasks, file: UploadFile = File(...),
                         lecture_id: str = Form(...), title: str = Form("")):
    lecture_id = slugify(lecture_id)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = settings.data_dir / f"upload_{lecture_id}{suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    lecture_dir(lecture_id).mkdir(parents=True, exist_ok=True)
    background.add_task(_run_ingest, tmp, lecture_id, title or lecture_id)
    return {"id": lecture_id, "status": "queued"}


def _run_youtube(url: str, lecture_id: str, title: str | None, use_captions: bool):
    from .rag.youtube import ingest_youtube

    try:
        ingest_youtube(url, lecture_id, title, use_captions=use_captions)
        rebuild_index(app)
    except Exception as exc:
        print(f"[tutor] YouTube ingest failed for {lecture_id}: {exc}")


@app.post("/lectures/youtube")
def youtube_lecture(req: YouTubeRequest, background: BackgroundTasks):
    from .rag.youtube import video_id

    vid = video_id(req.url)
    if not vid:
        raise HTTPException(400, "That does not look like a YouTube video link")
    lecture_id = slugify(req.lecture_id) if req.lecture_id else f"yt-{vid.lower()}"
    lecture_dir(lecture_id).mkdir(parents=True, exist_ok=True)
    background.add_task(_run_youtube, req.url, lecture_id, req.title or None, req.use_captions)
    return {"id": lecture_id, "status": "queued"}


# ---------------- topics ----------------
@app.get("/concepts", response_model=list[ConceptInfo])
def concepts(student_id: str | None = None, lecture_id: str | None = None):
    chunks = [c for c in _retriever().chunks if not lecture_id or c.lecture_id == lecture_id]
    state = load_student(student_id) if student_id else None
    out = []
    for order, (name, cs) in enumerate(concept_map(chunks).items()):
        info = ConceptInfo(name=name, order=order, segments=merge_segments(cs))
        if state:
            info.mastery = round(mastery_of(state, name), 4)
            info.attempts, info.correct = state["attempts"].get(name, [0, 0])
        out.append(info)
    return out


@app.post("/concepts/rename")
def concepts_rename(req: RenameRequest):
    if not req.new.strip():
        raise HTTPException(400, "New name is empty")
    changed = rename_concept(req.old, req.new)
    rebuild_index(app)
    return {"changed_chunks": changed}


# ---------------- the three agreed endpoints ----------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    history = [t.model_dump() for t in req.history]
    resp, retrieved = answer_question(req.question, _retriever(), lecture_id=req.lecture_id, history=history)
    if req.verify and resp.source == "lecture" and resp.citations:
        cited = {c.chunk_id for c in resp.citations}
        evidence = [c for c in retrieved if c.id in cited] or retrieved
        resp.claims = verify_answer(resp.answer, evidence)
    return resp


@app.post("/quiz", response_model=QuizResponse)
def quiz(req: QuizRequest):
    cmap = concept_map(_retriever().chunks)
    if req.concept not in cmap:
        raise HTTPException(404, f"Unknown topic: {req.concept}")
    seen = set()
    if req.student_id:
        seen = {h["question_id"] for h in load_student(req.student_id)["history"]}
    questions = get_quiz(req.concept, cmap[req.concept], req.n, seen, fresh=req.fresh)
    return QuizResponse(
        concept=req.concept,
        questions=[QuizQuestionPublic(id=q["id"], question=q["question"], options=q["options"]) for q in questions],
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    q = find_question(req.question_id)
    if not q:
        raise HTTPException(404, "Unknown question id")
    correct = req.choice == q["answer_index"]
    mastery = record_answer(req.student_id, q["concept"], q["id"], correct)
    chunk = _retriever().by_id.get(q.get("source_chunk_id", ""))
    source = Segment(lecture_id=chunk.lecture_id, start=chunk.start, end=chunk.end) if chunk else None
    return AnswerResponse(correct=correct, correct_index=q["answer_index"], explanation=q.get("explanation", ""),
                          concept=q["concept"], mastery=round(mastery, 4), source=source)


@app.post("/next", response_model=NextResponse)
def next_(req: NextRequest):
    return next_topic(req.student_id, concept_map(_retriever().chunks))


@app.get("/progress/{student_id}")
def progress(student_id: str):
    state = load_student(student_id)
    return {"student_id": student_id, "mastery": state["mastery"], "attempts": state["attempts"],
            "history": state["history"][-50:]}


# ---------------- Lab: model training ----------------
@app.get("/lab/bkt")
def lab_bkt():
    seqs = bkt_fit.sequences_from_app()
    return {
        "active": active_params(),
        "default": DEFAULT.to_dict(),
        "last_fit": bkt_fit.last_fit(),
        "app_data": {"students": len({s.student for s in seqs}), "sequences": len(seqs),
                     "answers": sum(len(s.answers) for s in seqs)},
    }


@app.post("/lab/bkt/fit")
def lab_bkt_fit(req: BKTFitRequest):
    concepts = list(concept_map(_retriever().chunks)) or None
    return bkt_fit.run_fit(req.source, n_students=req.students, seed=req.seed, concepts=concepts)


@app.post("/lab/bkt/apply")
def lab_bkt_apply():
    return bkt_fit.apply_last_fit()


@app.post("/lab/bkt/reset")
def lab_bkt_reset():
    reset_params()
    return {"reset": True, "students_updated": replay_all_students()}


def _can_finetune() -> tuple[bool, str]:
    if settings.embedding_backend != "sentence-transformers":
        return False, "Fine-tuning needs EMBEDDING_BACKEND=sentence-transformers in backend/.env."
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"PyTorch or sentence-transformers is not working ({str(exc)[:120]}). Run: pip install -r requirements.txt"
    return True, ""


@app.get("/lab/embeddings")
def lab_embeddings():
    from .rag import finetune

    ok, why = _can_finetune()
    tuned = finetune.output_dir(finetune.base_model_name())
    active = active_embedding_model()
    return {
        "base_model": settings.embedding_model,
        "active_model": active,
        "using_finetuned": Path(active).resolve() == tuned.resolve() if tuned.exists() else False,
        "finetuned_available": tuned.exists() and any(tuned.iterdir()),
        "finetuned_path": str(tuned),
        "last_run": finetune.last_run(),
        "job": jobs.status("finetune"),
        "can_train": ok,
        "reason": why,
        "chunks": len(_retriever().chunks),
    }


@app.post("/lab/embeddings/train")
def lab_embeddings_train(req: FinetuneRequest):
    from .rag import finetune

    ok, why = _can_finetune()
    if not ok:
        raise HTTPException(400, why)
    return jobs.start("finetune", lambda progress: finetune.run(epochs=req.epochs, progress=progress))


@app.post("/lab/embeddings/use")
def lab_embeddings_use(req: UseModelRequest):
    from .rag import finetune

    if req.model == "finetuned":
        tuned = finetune.output_dir(finetune.base_model_name())
        if not tuned.exists():
            raise HTTPException(400, "Train the fine-tuned model first.")
        set_active_embedding_model(str(tuned))
    else:
        set_active_embedding_model(None)
    n = rebuild_index(app, fresh=True)
    return {"active_model": active_embedding_model(), "chunks": n}
