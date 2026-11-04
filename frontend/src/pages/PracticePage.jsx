import React, { useMemo } from "react";
import { masteryLevel, pct } from "../api.js";
import { IconBack } from "../components/Icons.jsx";
import NextBanner from "../components/NextBanner.jsx";
import PracticePanel from "../components/PracticePanel.jsx";

function TopicCard({ c, recommended, onClick }) {
  const level = masteryLevel(c);
  const accuracy = c.attempts ? Math.round((c.correct / c.attempts) * 100) : null;
  return (
    <button className={`topic-card ${recommended ? "recommended" : ""}`} onClick={onClick}>
      <div className="topic-card-top">
        <span className={`pill ${level.key}`}>{level.label}</span>
        {recommended && <span className="pill accent">Recommended</span>}
      </div>
      <h3>{c.name}</h3>
      <div className="mastery"><div className={`mastery-fill ${level.key}`} style={{ width: pct(c.mastery) }} /></div>
      <div className="topic-card-meta">
        <span>{pct(c.mastery)} mastery</span>
        <span className="muted">{c.attempts ? `${c.attempts} answers · ${accuracy}% right` : "not practiced yet"}</span>
      </div>
    </button>
  );
}

export default function PracticePage({ concepts, next, topic, setTopic, count, studentId, refresh, seekTo, practice }) {
  const startMastery = useMemo(
    () => concepts.find((c) => c.name === topic)?.mastery ?? null,
    // keep the value from when the session started
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [topic]
  );

  if (topic) {
    return (
      <div className="page narrow">
        <button className="back" onClick={() => setTopic(null)}><IconBack /> All topics</button>
        <PracticePanel concept={topic} count={count} studentId={studentId} onAnswered={refresh} onSeek={seekTo}
          startMastery={startMastery} />
      </div>
    );
  }

  const sorted = [...concepts].sort((a, b) => a.order - b.order);
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Practice</h1>
          <p className="muted">Pick a topic for a short quiz. Questions are written from the lecture itself.</p>
        </div>
      </header>
      <NextBanner next={next} seekTo={seekTo} practice={practice} compact />
      {sorted.length === 0 && <p className="muted">No topics yet. Add a lecture first.</p>}
      <div className="topic-grid">
        {sorted.map((c) => (
          <TopicCard key={c.name} c={c} recommended={next?.concept === c.name} onClick={() => setTopic(c.name)} />
        ))}
      </div>
    </div>
  );
}
