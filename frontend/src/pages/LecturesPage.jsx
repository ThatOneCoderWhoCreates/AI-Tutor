import React, { useEffect, useState } from "react";
import { api, fmtTime } from "../api.js";
import UploadPanel from "../components/UploadPanel.jsx";

const STATUS = {
  queued: "Queued",
  downloading: "Downloading",
  transcribing: "Transcribing",
  reading_slides: "Reading slides",
  chunking: "Chunking",
  labeling: "Finding topics",
  ready: "Ready",
  error: "Failed",
  missing: "No data",
};

// Rename a topic, or type another topic's exact name to merge the two.
function TopicEditor({ lectureId, studentId, onChanged }) {
  const [topics, setTopics] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [error, setError] = useState("");

  const load = () => api.concepts(studentId, lectureId).then(setTopics).catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, [lectureId]);

  const save = async (oldName) => {
    const newName = (drafts[oldName] || "").trim();
    if (!newName || newName === oldName) return;
    try {
      await api.rename(oldName, newName);
      setDrafts((d) => ({ ...d, [oldName]: undefined }));
      await load();
      onChanged();
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="topic-editor">
      <p className="muted small">
        Topic names were chosen by the local model. Fix any that are wrong. To merge two topics, type the other
        topic's exact name. Changes apply to every lecture that uses the topic.
      </p>
      {error && <p className="error">{error}</p>}
      <ul>
        {topics.map((t) => (
          <li key={t.name}>
            <span className="muted small">{fmtTime(t.segments[0]?.start)}</span>
            <input value={drafts[t.name] ?? t.name} onChange={(e) => setDrafts({ ...drafts, [t.name]: e.target.value })}
              onKeyDown={(e) => e.key === "Enter" && save(t.name)} />
            <button className="secondary" disabled={!drafts[t.name] || drafts[t.name] === t.name} onClick={() => save(t.name)}>
              Save
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function LectureCard({ lec, studentId, onOpen, onChanged }) {
  const [editing, setEditing] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const busy = !["ready", "error", "missing"].includes(lec.status);

  const remove = async () => {
    await api.deleteLecture(lec.id);
    onChanged();
  };

  return (
    <div className="card lecture">
      <div className="lecture-top">
        <div>
          <h3>{lec.title}</h3>
          <p className="muted small">
            {lec.id}
            {lec.duration ? ` · ${fmtTime(lec.duration)}` : ""}
            {lec.status === "ready" ? ` · ${lec.topics} topics · ${lec.chunks} chunks` : ""}
            {lec.source_url && <> · <a href={lec.source_url} target="_blank" rel="noreferrer">source</a></>}
          </p>
        </div>
        <span className={`pill ${lec.status === "ready" ? "high" : lec.status === "error" ? "low" : "mid"}`}>
          {busy && <span className="spinner" />}
          {STATUS[lec.status] || lec.status}
        </span>
      </div>
      {lec.error && <p className="error small">{lec.error}</p>}
      <div className="lecture-actions">
        {lec.status === "ready" && <button className="primary" onClick={() => onOpen(lec.id)}>Open</button>}
        {lec.status === "ready" && (
          <button className="secondary" onClick={() => setEditing(!editing)}>{editing ? "Done editing" : "Edit topics"}</button>
        )}
        {lec.status === "ready" && (
          <button className="secondary" title="Ask the local model to choose the topics again. Practice questions for these topics are written again."
            onClick={async () => { await api.relabel(lec.id); setEditing(false); onChanged(); }}>
            Find topics again
          </button>
        )}
        <span className="spacer" />
        {!busy && (confirm ? (
          <>
            <span className="small muted">Delete this lecture?</span>
            <button className="danger" onClick={remove}>Delete</button>
            <button className="link" onClick={() => setConfirm(false)}>Cancel</button>
          </>
        ) : (
          <button className="link danger-link" onClick={() => setConfirm(true)}>Delete</button>
        ))}
      </div>
      {editing && <TopicEditor lectureId={lec.id} studentId={studentId} onChanged={onChanged} />}
    </div>
  );
}

export default function LecturesPage({ lectures, refresh, openLecture, studentId }) {
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Lectures</h1>
          <p className="muted">Add lectures from YouTube or a video file, check their topics, and remove old ones.</p>
        </div>
      </header>
      <div className="lectures-grid">
        <div className="lecture-list">
          {lectures.length === 0 && <p className="muted">No lectures yet.</p>}
          {lectures.map((l) => (
            <LectureCard key={l.id} lec={l} studentId={studentId} onOpen={openLecture} onChanged={refresh} />
          ))}
        </div>
        <UploadPanel onReady={() => refresh()} />
      </div>
    </div>
  );
}
