import React, { useState } from "react";
import { conceptsForLecture, fmtTime, masteryLevel, pct } from "../api.js";
import AskPanel from "../components/AskPanel.jsx";
import NextBanner from "../components/NextBanner.jsx";
import VideoPlayer from "../components/VideoPlayer.jsx";

function Welcome({ go, onClose }) {
  return (
    <div className="card welcome">
      <div className="card-head">
        <h3>How to use Lecture Tutor</h3>
        <button className="link" onClick={onClose}>Hide</button>
      </div>
      <ol className="steps">
        <li><strong>Add a lecture.</strong> Paste a YouTube link on the <a href="#/lectures" onClick={() => go("lectures")}>Lectures</a> page. It is split into topics automatically.</li>
        <li><strong>Watch and ask.</strong> Ask questions here. Every answer has a timestamp you can click.</li>
        <li><strong>Practice.</strong> Take short quizzes. The tutor tracks what you know and suggests what to study next.</li>
      </ol>
      <p className="muted small">Until then, try the demo lecture below.</p>
    </div>
  );
}

export default function StudyPage({ lectures, lectureId, setLectureId, concepts, next, seek, seekTo, practice,
                                    studentId, messages, setMessages, go, showWelcome, hideWelcome }) {
  const [time, setTime] = useState(0);
  const lecture = lectures.find((l) => l.id === lectureId);
  const chapters = conceptsForLecture(concepts, lectureId);
  const ready = lectures.filter((l) => l.status === "ready");

  const currentIdx = chapters.findIndex((c) => c.segments.some((s) => time >= s.start && time < s.end));

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Study</h1>
          <p className="muted">Watch the lecture, jump to any topic, and ask questions.</p>
        </div>
        <label className="select">
          <span>Lecture</span>
          <select value={lectureId || ""} onChange={(e) => setLectureId(e.target.value)}>
            {ready.length === 0 && <option value="">No lectures yet</option>}
            {ready.map((l) => <option key={l.id} value={l.id}>{l.title}</option>)}
          </select>
        </label>
      </header>

      {showWelcome && <Welcome go={go} onClose={hideWelcome} />}
      <NextBanner next={next} seekTo={seekTo} practice={practice} />

      <div className="study-grid">
        <div className="study-main">
          <VideoPlayer lecture={lecture} seek={seek} chapters={chapters} onTime={setTime} />
          <div className="card">
            <div className="card-head">
              <h3>Chapters</h3>
              <span className="muted small">{chapters.length} topics</span>
            </div>
            {chapters.length === 0 && <p className="muted">This lecture has no topics yet.</p>}
            <ul className="chapters">
              {chapters.map((c, i) => {
                const level = masteryLevel(c);
                const s = c.segments[0];
                return (
                  <li key={c.name} className={i === currentIdx ? "current" : ""}>
                    <button className="chapter-time" onClick={() => seekTo(s.lecture_id, s.start, s.end)}>
                      {fmtTime(s.start)}
                    </button>
                    <button className="chapter-name" onClick={() => seekTo(s.lecture_id, s.start, s.end)}>
                      {c.name}
                      {c.segments.length > 1 && <span className="muted small"> · {c.segments.length} parts</span>}
                    </button>
                    <span className={`pill ${level.key}`} title={level.label}>{c.attempts ? pct(c.mastery) : "new"}</span>
                    <button className="chip ghost" onClick={() => practice(c.name)}>Practice</button>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
        <AskPanel studentId={studentId} onSeek={seekTo} messages={messages} setMessages={setMessages}
          suggestions={chapters.map((c) => c.name)} />
      </div>
    </div>
  );
}
