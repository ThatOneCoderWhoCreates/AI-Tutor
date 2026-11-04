import React from "react";

const PALETTE = ["#1f6f66", "#c48a1c", "#b3412f", "#5b5fc7", "#7a8f2e", "#a14f8b", "#2c7fb8", "#8c6d31"];

// Mastery after each answer, one line per topic. Pure SVG, no chart library.
export default function MasteryChart({ history, topics, height = 160 }) {
  if (!history.length) return null;
  const W = 600;
  const H = height;
  const PAD = { l: 32, r: 10, t: 10, b: 22 };
  const x = (i) => PAD.l + (i / Math.max(1, history.length - 1)) * (W - PAD.l - PAD.r);
  const y = (p) => PAD.t + (1 - p) * (H - PAD.t - PAD.b);

  const lines = topics.map((topic, ti) => {
    let last = null;
    const pts = [];
    history.forEach((h, i) => {
      if (h.concept === topic) last = h.mastery;
      if (last != null) pts.push([x(i), y(last)]);
    });
    return { topic, color: PALETTE[ti % PALETTE.length], pts };
  });

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mastery after each answer">
        {[0, 0.5, 0.95].map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className={v === 0.95 ? "grid goal" : "grid"} />
            <text x={PAD.l - 6} y={y(v) + 3} className="axis" textAnchor="end">{Math.round(v * 100)}%</text>
          </g>
        ))}
        <text x={W - PAD.r} y={H - 4} className="axis" textAnchor="end">answers →</text>
        <text x={W - PAD.r} y={y(0.95) - 4} className="axis" textAnchor="end">mastered</text>
        {lines.map((l) =>
          l.pts.length ? (
            <polyline key={l.topic} points={l.pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={l.color}
              strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
          ) : null
        )}
      </svg>
      <div className="legend">
        {lines.filter((l) => l.pts.length).map((l) => (
          <span key={l.topic}><i style={{ background: l.color }} />{l.topic}</span>
        ))}
      </div>
    </div>
  );
}
