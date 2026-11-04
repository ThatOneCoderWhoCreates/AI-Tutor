export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

const post = (path, body = {}) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const q = encodeURIComponent;

export const api = {
  health: () => request("/health"),
  lectures: () => request("/lectures"),
  lectureStatus: (id) => request(`/lectures/${id}/status`),
  deleteLecture: (id) => request(`/lectures/${id}`, { method: "DELETE" }),
  relabel: (id) => post(`/lectures/${id}/relabel`),
  concepts: (studentId, lectureId) =>
    request(`/concepts?student_id=${q(studentId)}${lectureId ? `&lecture_id=${q(lectureId)}` : ""}`),
  rename: (oldName, newName) => post("/concepts/rename", { old: oldName, new: newName }),
  ask: (question, studentId, verify, history = []) => post("/ask", { question, student_id: studentId, verify, history }),
  quiz: (concept, n, studentId) => post("/quiz", { concept, n, student_id: studentId }),
  answer: (studentId, questionId, choice) => post("/answer", { student_id: studentId, question_id: questionId, choice }),
  next: (studentId) => post("/next", { student_id: studentId }),
  upload: (file, lectureId, title) => {
    const form = new FormData();
    form.append("file", file);
    form.append("lecture_id", lectureId);
    form.append("title", title);
    return request("/lectures/upload", { method: "POST", body: form });
  },
  youtube: (url, lectureId, title, useCaptions) =>
    post("/lectures/youtube", { url, lecture_id: lectureId || null, title: title || null, use_captions: useCaptions }),
  progress: (studentId) => request(`/progress/${q(studentId)}`),
  mediaUrl: (lectureId) => `${API}/media/${lectureId}`,

  labBkt: () => request("/lab/bkt"),
  labBktFit: (source, students) => post("/lab/bkt/fit", { source, students }),
  labBktApply: () => post("/lab/bkt/apply"),
  labBktReset: () => post("/lab/bkt/reset"),
  labEmbeddings: () => request("/lab/embeddings"),
  labEmbeddingsTrain: (epochs) => post("/lab/embeddings/train", { epochs }),
  labEmbeddingsUse: (model) => post("/lab/embeddings/use", { model }),
};

export function fmtTime(seconds) {
  const s = Math.round(seconds || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

export const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

export function masteryLevel(c) {
  if (!c.attempts) return { key: "new", label: "Not started" };
  if ((c.mastery ?? 0) >= 0.95) return { key: "high", label: "Mastered" };
  if ((c.mastery ?? 0) >= 0.6) return { key: "mid", label: "Almost there" };
  return { key: "low", label: "Needs practice" };
}

// Topics with only the segments that belong to one lecture, in lecture order.
export function conceptsForLecture(concepts, lectureId) {
  return concepts
    .map((c) => ({ ...c, segments: c.segments.filter((s) => s.lecture_id === lectureId) }))
    .filter((c) => c.segments.length)
    .sort((a, b) => a.segments[0].start - b.segments[0].start);
}
