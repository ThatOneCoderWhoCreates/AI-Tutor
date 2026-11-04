# Lecture-Grounded Adaptive Tutor

Add a lecture from YouTube or a video file and get a tutor that:

- answers questions using only the lecture, with a clickable timestamp for every answer
- gives optional practice questions per topic
- tracks how well you know each topic with Bayesian Knowledge Tracing (BKT)
- tells you what to study next and plays the exact part of the video that teaches it

Everything runs on your own computer. The language model is Qwen running locally through Ollama, so there is no API (Application Programming Interface) key and no cost.

Two parts of the system are trained on your own data: the search model is fine-tuned on your lectures, and the knowledge-tracing numbers are learned from student answers. Both run from the **Lab** page.

For a role-by-role explanation of what was built (Role A, B and C), see `docs/IMPLEMENTATION.md`.

## Setup (run once)

The app works on macOS and Windows. Two scripts do the same steps on both: `install.py` sets everything up, and `start.py` runs the app.

### macOS

In Terminal:

```
brew install python@3.12 ffmpeg tesseract node ollama git
brew services start ollama
cd ~/Downloads/lecture-tutor
/opt/homebrew/bin/python3.12 install.py
```

### Windows

In PowerShell. After the `winget` lines, close PowerShell and open a new one so the new programs are found:

```
winget install Python.Python.3.12 Gyan.FFmpeg UB-Mannheim.TesseractOCR OpenJS.NodeJS.LTS Ollama.Ollama Git.Git
```

Then, in the new PowerShell window (change the path to where the project is):

```
cd $HOME\Downloads\lecture-tutor
py -3.12 install.py
```

### What install.py does

1. Creates the Python environment in `backend/.venv` and installs the Python packages (several minutes the first time).
2. Creates `backend/.env` from `.env.example` (an existing `.env` is never overwritten).
3. Installs the frontend packages.
4. Checks for ffmpeg, Tesseract and Ollama and prints the install command for anything missing. On Windows it also looks in the usual install folders, so it works before PATH is updated.
5. Downloads the local model named in `backend/.env` (`qwen3.5:9b`, several GB).

Running it again is safe.

## Run the app

macOS:

```
cd ~/Downloads/lecture-tutor
python3 start.py
```

Windows (PowerShell):

```
cd $HOME\Downloads\lecture-tutor
py start.py
```

`start.py` starts Ollama if it is not running, then the backend (port 8000) and the frontend (port 5173), and opens http://localhost:5173. Press Ctrl+C once to stop everything. The first start downloads the search model BAAI/bge-small-en-v1.5 (about 130 MB).

<details>
<summary>Running the two parts by hand instead</summary>

Backend, macOS:

```
cd ~/Downloads/lecture-tutor/backend
source .venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

Backend, Windows (PowerShell):

```
cd $HOME\Downloads\lecture-tutor\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Frontend, both systems (second terminal):

```
cd lecture-tutor/frontend
npm run dev
```

</details>

The sidebar has five pages: **Study** (video, chapters, chat), **Practice** (quizzes), **Progress**, **Lectures** (add and manage lectures) and **Lab** (training). The badge at the bottom of the sidebar shows the model. If it says "Ollama offline", open the Ollama app (or run `ollama serve`).

The demo lecture uses fake chunks and has no video, so the player shows a simulated timeline. API documentation is at http://localhost:8000/docs.

**Model size.** `qwen3.5:9b` needs about 16 GB of RAM to run comfortably. On a computer with 8 GB, set `LLM_MODEL=qwen2.5:3b` in `backend/.env` and run `ollama pull qwen2.5:3b`.

**Speech to text.** `WHISPER_BACKEND=auto` uses mlx-whisper on Apple-silicon Macs (uses the Mac's GPU) and faster-whisper on Windows, Intel Macs and Linux. faster-whisper runs on the CPU, so transcribing a long lecture takes longer on Windows laptops; `WHISPER_MODEL=base` is faster if needed.

**Tests** (optional): macOS `backend/.venv/bin/python -m pytest -q backend/tests`, Windows `backend\.venv\Scripts\python.exe -m pytest -q backend\tests`. They use a built-in fake model, so they do not need Ollama.

## Commands on Windows

The commands in the rest of this file are written for macOS. On Windows (PowerShell), make two changes:

- Instead of `source .venv/bin/activate`, run `.venv\Scripts\Activate.ps1`. If PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.
- Use backslashes in paths (`backend\eval\results`) and `$HOME\Downloads` instead of `~/Downloads`.

Everything else (`python -m ...`, `ollama ...`, `npm ...`, `curl ...`) is the same.

## Add a lecture from YouTube

In the app: open the **Lectures** page, paste the link, press **Download and process**.

Or from the terminal, which shows progress for each step:

```
cd lecture-tutor/backend
source .venv/bin/activate
python -m app.rag.ingest --url "https://www.youtube.com/watch?v=VIDEO_ID" --id lec1
curl -X POST http://localhost:8000/reindex
```

Keep the link in double quotes. Without them, zsh misreads the `?` and `&` characters in it.

Add `--captions` to use YouTube's own captions instead of Whisper. This is much faster but usually less accurate.

What happens:

1. yt-dlp downloads the video at 720p.
2. Whisper turns speech into text with timestamps. This is the slowest step: roughly 3 to 6 minutes for a one-hour lecture on an M-series Mac with the `small` model.
3. Slide changes are detected and each slide is read with Tesseract OCR (Optical Character Recognition).
4. Transcript and slide text are merged into chunks of up to 90 seconds.
5. Qwen gives each chunk a topic name.

Results are cached in `backend/data/lectures/lec1/`, so a second run is fast. Add `--force` to redo everything.

Topic names will not be perfect. On the **Lectures** page, press **Edit topics** to rename a topic, or type another topic's exact name to merge the two.

A video file works the same way:

```
python -m app.rag.ingest ~/Downloads/lecture1.mp4 --id lec2 --title "Lecture 2"
```

## Answers for questions outside the lectures

By default the tutor only answers from your lectures. To also answer other questions, connect a second model. The reply then starts with "[It was not covered in this lecture, but this is the answer to your question]" so it is clear the answer did not come from the lecture. Only the question and the last few chat messages are sent to that service, never the lecture text.

You can list several models. They are tried in order: if one reaches its free limit or fails, the next one answers, and the one that failed is skipped for about a minute. The chat shows which model answered.

Recommended setup: Gemini first, Groq second, and the local model last (always available, no key).

1. Get a free Gemini key at https://aistudio.google.com/apikey (sign in, press **Create API key**).
2. Optional: get a free Groq key at https://console.groq.com/keys.
3. Open `backend/.env` with `open -e ~/Downloads/lecture-tutor/backend/.env` and set these lines (leave `GROQ_API_KEY` empty if you skipped step 2; it is then skipped):

```
FALLBACK_PROVIDER=gemini,groq,local
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

4. Restart the backend. The bottom of the sidebar shows the chain, for example "Outside-lecture answers: Gemini → Groq → Local qwen3.5:9b".

To pick a specific model, write `provider:model`, for example `FALLBACK_PROVIDER=gemini:gemini-3.8-flash,gemini,groq,local`. Two Gemini models have separate free limits, so this also adds capacity. Other options: `openrouter:<free model name>` with `OPENROUTER_API_KEY`. Keep keys only in `.env`. It is listed in `.gitignore`, so it is not uploaded to GitHub.

## Change the model

The default is `qwen3.5:9b`, chosen after comparing it with `qwen2.5:7b` (see `docs/IMPLEMENTATION.md`). To use another model, edit `backend/.env` and restart the backend, for example:

```
LLM_MODEL=qwen2.5:7b
```

Any Ollama chat model works. Run `ollama pull <name>` first. To work on the interface without any model, set `LLM_PROVIDER=mock`.

## Train the models

Both jobs can be run from the **Lab** page with one button each, or from the terminal.

**Learn the BKT (Bayesian Knowledge Tracing) numbers with EM (Expectation-Maximization).** First test the method on simulated students with known numbers, then learn from real answers once at least 5 people have practiced:

```
cd lecture-tutor/backend
source .venv/bin/activate
python -m app.tutor.bkt_fit --source simulated --students 400
python -m app.tutor.bkt_fit --source app --apply
```

**Fine-tune the search model on your lectures.** Add one or two real lectures first. Qwen writes practice search questions for every chunk (a few minutes), then bge-small is trained on them (usually a few minutes on an M-series Mac):

```
python -m app.rag.finetune --epochs 3 --use
curl -X POST http://localhost:8000/reindex
```

`--use` switches the app to the fine-tuned model. You can switch back on the Lab page.

## Evaluate (for the report)

```
cd lecture-tutor/backend
source .venv/bin/activate
python -m eval.run_eval eval/questions_lecture1.json --answers --quiz --lecture lecture1 --models qwen2.5:7b qwen3.5:9b
python -m eval.mock_student --rounds 8
```

This writes to `backend/eval/results/`:

- `report.md`: tables ready to paste into the report
- `answers_<model>.csv`: every question with the model's answer, time taken and whether the timestamp was right
- `mock_student.png` and `mock_student.csv`: chart of a simulated student's mastery rising over study rounds

What is measured:

| Part | Metric | Meaning |
|---|---|---|
| Search | Hit@1, Hit@k, MRR (Mean Reciprocal Rank) | Does search find the right part of the lecture? Compares vector-only (baseline), keyword-only (BM25) and hybrid |
| Answers | timestamp accuracy | The answer cites a chunk inside the correct time range |
| Answers | false refusal, correct refusal | Says "not covered" when it should not, and when it should |
| Answers | claim support | Share of answer sentences the trust layer confirms against the lecture |
| Quiz | first try valid, quiz success | How often the model returns usable questions |
| Speed | median seconds | Time per answer or per quiz on your machine |

For your real lectures, copy `eval/questions_demo.json` to `eval/questions_lectures.json` and write 30 to 50 questions, each with the time range where the answer is taught. Add a few questions the lecture does not cover, with `"expect": "not_covered"`.

To compare the base and fine-tuned search models on those questions:

```
python -m eval.run_eval eval/questions_lectures.json --embedding-models BAAI/bge-small-en-v1.5 finetuned
```

## Who works where

| Role | Files |
|---|---|
| A: Lecture RAG (Retrieval-Augmented Generation) | `backend/app/rag/` |
| B: Tutor engine | `backend/app/tutor/`, `backend/app/trust.py` |
| C: App and quality | `backend/app/main.py`, `frontend/src/`, `backend/eval/` |
| Shared (change together) | `backend/app/schemas.py`, `backend/app/llm.py`, `docs/CONTRACT.md` |

## Known limitations

- Whisper, Qwen and the YouTube download could not be run in the environment where this code was written (no access to those servers). They are covered by tests that use fake replacements, but the first real run on your Mac is the true test.
- Prompts were written for small local models, but they have not yet been tuned on real Qwen output. Check `answers_<model>.csv` after the first evaluation run and adjust the prompts in `app/rag/answer.py` and `app/tutor/quiz.py` if needed.
- Slide detection compares frames once per second. A webcam overlay can create extra slide boundaries. If a lecture gets too many short chunks, raise `SCENE_THRESHOLD` in `.env` (for example to 0.03).
- Student data is stored in JSON files with no login. This is fine for a demo, not for real users.
