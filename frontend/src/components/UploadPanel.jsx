import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const LABELS = {
  queued: "Queued",
  downloading: "Downloading from YouTube",
  transcribing: "Transcribing speech (slowest step)",
  reading_slides: "Reading slides",
  chunking: "Splitting into chunks",
  labeling: "Finding topics with the local model",
  ready: "Ready",
  error: "Failed",
};

// Add a lecture from a YouTube link or a video file. The backend runs the pipeline in the background.
export default function UploadPanel({ onReady }) {
  const [mode, setMode] = useState("youtube");
  const [url, setUrl] = useState("");
  const [useCaptions, setUseCaptions] = useState(false);
  const [file, setFile] = useState(null);
  const [lectureId, setLectureId] = useState("");
  const [title, setTitle] = useState("");
  const [job, setJob] = useState(null);
  const reset = () => {
    setUrl("");
    setFile(null);
    setLectureId("");
    setTitle("");
  };
  const [error, setError] = useState("");

  useEffect(() => {
    if (!job || job.status === "ready" || job.status === "error") return;
    const t = setInterval(async () => {
      try {
        const s = await api.lectureStatus(job.id);
        setJob(s);
        if (s.status === "ready") {
          reset();
          onReady(s.id);
        }
        if (s.status === "error") onReady(null);
      } catch {}
    }, 3000);
    return () => clearInterval(t);
  }, [job]);

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      if (mode === "youtube") {
        if (!url.trim()) return setError("Paste a YouTube link.");
        setJob(await api.youtube(url.trim(), lectureId.trim(), title.trim(), useCaptions));
        onReady(null);
      } else {
        if (!file || !lectureId.trim()) return setError("Choose a video and give it a short id like lec1.");
        setJob(await api.upload(file, lectureId.trim(), title.trim()));
        onReady(null);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const busy = job && !["ready", "error"].includes(job.status);

  return (
    <form className="card upload" onSubmit={submit}>
      <div className="card-head"><h3>Add a lecture</h3></div>
      <div className="segmented">
        <button type="button" className={mode === "youtube" ? "active" : ""} onClick={() => setMode("youtube")}>
          YouTube link
        </button>
        <button type="button" className={mode === "file" ? "active" : ""} onClick={() => setMode("file")}>
          Video file
        </button>
      </div>

      {mode === "youtube" ? (
        <>
          <label>
            YouTube link
            <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://www.youtube.com/watch?v=..." />
          </label>
          <label className="check">
            <input type="checkbox" checked={useCaptions} onChange={(e) => setUseCaptions(e.target.checked)} />
            Use YouTube captions instead of Whisper (much faster, usually less accurate)
          </label>
        </>
      ) : (
        <label>
          Video file
          <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        </label>
      )}

      <label>
        <span>Lecture id {mode === "youtube" && <span className="muted">(optional)</span>}</span>
        <input value={lectureId} onChange={(e) => setLectureId(e.target.value)} placeholder="lec1" />
      </label>
      <label>
        <span>Title {mode === "youtube" && <span className="muted">(optional, uses the video title)</span>}</span>
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Lecture 1: Gradient Descent" />
      </label>
      <button className="primary" type="submit" disabled={busy}>
        {mode === "youtube" ? "Download and process" : "Upload and process"}
      </button>
      <p className="muted small">
        Everything runs on this computer. A one-hour lecture takes several minutes with Whisper.
      </p>
      {error && <p className="error">{error}</p>}
      {job && (
        <p className={job.status === "error" ? "error" : "status"}>
          {busy && <span className="spinner" />}
          {LABELS[job.status] || job.status}
          {job.error ? `: ${job.error}` : ""}
        </p>
      )}
    </form>
  );
}
