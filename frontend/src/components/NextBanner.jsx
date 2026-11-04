import React from "react";
import { fmtTime, pct } from "../api.js";
import { IconPlay } from "./Icons.jsx";

// "What should I study next?" shown automatically. Updates after every answer.
export default function NextBanner({ next, seekTo, practice, compact = false }) {
  if (!next) return null;
  if (!next.concept) {
    return (
      <div className="next-banner done">
        <span className="eyebrow">Up next</span>
        <p>{next.reason}</p>
      </div>
    );
  }
  const seg = next.segments[0];
  return (
    <div className="next-banner">
      <div className="next-text">
        <span className="eyebrow">Recommended next</span>
        <p>
          <strong>{next.concept}</strong> <span className="muted">· mastery {pct(next.mastery)}</span>
        </p>
        {!compact && <p className="muted small">{next.reason}</p>}
      </div>
      <div className="next-actions">
        {seg && (
          <button className="primary" onClick={() => seekTo(seg.lecture_id, seg.start, seg.end)}>
            <IconPlay /> Watch {fmtTime(seg.start)} to {fmtTime(seg.end)}
          </button>
        )}
        <button className="secondary" onClick={() => practice(next.concept, 2)}>Quick check</button>
      </div>
    </div>
  );
}
