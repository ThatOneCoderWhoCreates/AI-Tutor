import React, { useEffect, useState } from "react";
import { api, masteryLevel, pct } from "../api.js";
import MasteryChart from "../components/MasteryChart.jsx";

export default function ProgressPage({ studentId, concepts, refreshKey, seekTo, practice, go }) {
  const [data, setData] = useState(null);
  const [bkt, setBkt] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.progress(studentId).then(setData).catch((e) => setError(e.message));
    api.labBkt().then(setBkt).catch(() => {});
  }, [studentId, refreshKey]);

  const history = data?.history || [];
  const total = history.length;
  const right = history.filter((h) => h.correct).length;
  const mastered = concepts.filter((c) => (c.mastery ?? 0) >= 0.95).length;
  const practicedNames = [...new Set(history.map((h) => h.concept))];

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Progress</h1>
          <p className="muted">How well you know each topic, estimated from your practice answers.</p>
        </div>
      </header>
      {error && <p className="error">{error}</p>}

      <div className="stats">
        <div className="stat"><strong>{total}</strong><span>answers</span></div>
        <div className="stat"><strong>{total ? Math.round((right / total) * 100) : 0}%</strong><span>correct</span></div>
        <div className="stat"><strong>{mastered}/{concepts.length}</strong><span>topics mastered</span></div>
        <div className="stat"><strong>{practicedNames.length}</strong><span>topics practiced</span></div>
      </div>

      <div className="progress-grid">
        <div className="card">
          <div className="card-head"><h3>Mastery after each answer</h3></div>
          {total === 0 ? (
            <p className="muted">No practice yet. Answer a few questions and this chart fills in.</p>
          ) : (
            <MasteryChart history={history.slice(-60)} topics={practicedNames} />
          )}
          <p className="muted small">
            Estimated with Bayesian Knowledge Tracing (BKT) using{" "}
            {bkt?.active?.source === "fitted" ? "parameters learned from student answers" : "default parameters"}.
            A topic counts as mastered at 95%. <a href="#/lab" onClick={() => go("lab")}>Details in Lab</a>
          </p>
        </div>

        <div className="card">
          <div className="card-head"><h3>Recent answers</h3></div>
          {total === 0 && <p className="muted">Nothing yet.</p>}
          <ul className="recent">
            {history.slice(-10).reverse().map((h, i) => (
              <li key={i}>
                <span className={h.correct ? "ok" : "bad"}>{h.correct ? "✓" : "✗"}</span>
                <span className="recent-topic">{h.concept}</span>
                <span className="muted">{pct(h.mastery)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><h3>All topics</h3></div>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Topic</th><th>Mastery</th><th>Answers</th><th>Right</th><th>Status</th><th /></tr>
            </thead>
            <tbody>
              {[...concepts].sort((a, b) => a.order - b.order).map((c) => {
                const level = masteryLevel(c);
                const s = c.segments[0];
                return (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td>
                      <div className="bar-cell">
                        <div className="mastery"><div className={`mastery-fill ${level.key}`} style={{ width: pct(c.mastery) }} /></div>
                        <span className="small">{pct(c.mastery)}</span>
                      </div>
                    </td>
                    <td>{c.attempts}</td>
                    <td>{c.attempts ? `${Math.round((c.correct / c.attempts) * 100)}%` : "–"}</td>
                    <td><span className={`pill ${level.key}`}>{level.label}</span></td>
                    <td className="row-actions">
                      {s && <button className="chip ghost" onClick={() => seekTo(s.lecture_id, s.start, s.end)}>Watch</button>}
                      <button className="chip ghost" onClick={() => practice(c.name)}>Practice</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
