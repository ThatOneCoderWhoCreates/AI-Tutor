# Week 1 Contract: How the Three Parts Plug Together

These three agreements let Role A (Lecture RAG), Role B (Tutor engine) and Role C (App and quality) work in parallel. Change them only after telling the whole team.

The code version of this contract is `backend/app/schemas.py`.

## 1. The chunk (A produces it, B reads it, C displays it)

```json
{
  "id": "lec1_007",
  "lecture_id": "lec1",
  "text": "Gradient descent works by...",
  "slide_text": "Gradient Descent",
  "start": 760,
  "end": 845,
  "concept": "gradient descent"
}
```

| Field | Meaning |
|---|---|
| `id` | `<lecture_id>_<3-digit number>`, unique across the course |
| `lecture_id` | Folder name under `backend/data/lectures/` (added to the PDF's version so chunks from many lectures can live in one index) |
| `text` | What the lecturer said in this time range (from Whisper) |
| `slide_text` | OCR (Optical Character Recognition) text of the slide(s) on screen; first line is treated as the slide title |
| `start`, `end` | Seconds from the start of the video |
| `concept` | Topic label, lowercase, 1 to 4 words. Chunks with the same label belong to the same topic |

Stored at `backend/data/lectures/<lecture_id>/chunks.json`. The fake file `backend/data/lectures/demo/chunks.json` follows this format and is what B and C build against until real lectures are ingested.

## 2. Endpoints

Full interactive documentation runs at `http://localhost:8000/docs` when the backend is up.

The three agreed endpoints:

| Endpoint | In | Out |
|---|---|---|
| `POST /ask` | `{"question", "lecture_id"?, "student_id"?, "verify"?, "history"?: [{"role": "user" or "assistant", "content"}]}` | `{"answer", "citations": [{"n", "chunk_id", "lecture_id", "start", "end", "concept"}], "claims"?, "source": "lecture" or "general" or "none", "note"?}` |
| `POST /quiz` | `{"concept", "n", "student_id"?}` | `{"concept", "questions": [{"id", "question", "options"}]}` (the correct answer is never sent to the browser) |
| `POST /next` | `{"student_id"}` | `{"concept", "mastery", "reason", "segments": [{"lecture_id", "start", "end"}]}` |

Supporting endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /answer` | `{"student_id", "question_id", "choice"}` → correct or not, explanation, new mastery, video segment to rewatch |
| `GET /concepts?student_id=` | All topics in course order, with segments and the student's mastery |
| `POST /concepts/rename` | `{"old", "new"}` rename a topic, or merge it into an existing one |
| `GET /lectures`, `GET /lectures/{id}/status` | Lecture list and processing status |
| `POST /lectures/upload` | Multipart upload (`file`, `lecture_id`, `title`); runs the pipeline in the background |
| `POST /lectures/youtube` | `{"url", "lecture_id"?, "title"?, "use_captions"?}` downloads a YouTube lecture and runs the pipeline in the background |
| `GET /media/{lecture_id}` | Streams the lecture video (supports seeking) |
| `GET /progress/{student_id}` | Mastery, attempts and recent answer history |
| `GET /health`, `POST /reindex` | Status; rebuild the search index |
| `DELETE /lectures/{id}` | Remove a lecture and rebuild the search index |
| `GET /lab/bkt`, `POST /lab/bkt/fit`, `POST /lab/bkt/apply`, `POST /lab/bkt/reset` | Learn BKT numbers with EM (`{"source": "app" or "simulated", "students"}`), use them, or go back to defaults |
| `GET /lab/embeddings`, `POST /lab/embeddings/train`, `POST /lab/embeddings/use` | Fine-tune the search model in the background (`{"epochs"}`), check progress, switch between `"base"` and `"finetuned"` |

## 3. One place for the LLM call

Every module calls `ask_llm(prompt)` or `ask_llm_json(prompt, validate=...)` from `backend/app/llm.py`. Nothing else talks to a model directly.

The provider (`ollama` or `mock`) and the model name (for example `qwen3.5:9b`) live only in `backend/.env`. That file is listed in `.gitignore`, so each teammate can use a different model without changing code.

Every prompt starts with a line `TASK: <name>` (answer, quiz, concepts, verify). The offline mock model uses this to reply in the right format, so keep it when you edit prompts.
