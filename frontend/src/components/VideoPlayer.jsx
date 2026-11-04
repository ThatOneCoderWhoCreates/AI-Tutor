import React, { useEffect, useRef, useState } from "react";
import { api, fmtTime } from "../api.js";

// Plays a lecture and jumps to timestamps. If a segment has an end time,
// playback pauses there automatically ("watch 12:40 to 14:05").
// Lectures without a video file (the fake demo) show a simulated timeline instead.
export default function VideoPlayer({ lecture, seek, chapters, onTime }) {
  const videoRef = useRef(null);
  const stopAt = useRef(null);
  const [time, setTime] = useState(0);
  const duration = lecture?.duration || Math.max(1, ...chapters.flatMap((c) => c.segments.map((s) => s.end)));

  const report = (t) => {
    setTime(t);
    onTime?.(t);
  };

  useEffect(() => {
    if (!seek || !lecture || seek.lectureId !== lecture.id) return;
    stopAt.current = seek.end ?? null;
    report(seek.start);
    const v = videoRef.current;
    if (!v) return;
    const go = () => {
      v.currentTime = seek.start;
      v.play().catch(() => {});
    };
    if (v.readyState >= 1) go();
    else v.addEventListener("loadedmetadata", go, { once: true });
  }, [seek, lecture]);

  const onTimeUpdate = () => {
    const v = videoRef.current;
    report(v.currentTime);
    if (stopAt.current != null && v.currentTime >= stopAt.current) {
      v.pause();
      stopAt.current = null;
    }
  };

  const strip = chapters
    .flatMap((c) => c.segments.map((s) => ({ ...s, concept: c })))
    .sort((a, b) => a.start - b.start);
  const current = strip.find((s) => time >= s.start && time < s.end);

  const jump = (s) => {
    stopAt.current = null;
    if (videoRef.current) {
      videoRef.current.currentTime = s.start;
      videoRef.current.play().catch(() => {});
    }
    report(s.start);
  };

  if (!lecture) {
    return (
      <div className="player">
        <div className="fake-video"><div className="fake-video-inner"><span className="muted">No lecture selected</span></div></div>
      </div>
    );
  }

  return (
    <div className="player">
      {lecture.has_video ? (
        <video ref={videoRef} key={lecture.id} src={api.mediaUrl(lecture.id)} controls onTimeUpdate={onTimeUpdate} />
      ) : (
        <div className="fake-video">
          <div className="fake-video-inner">
            <span className="eyebrow">Demo lecture (no video file)</span>
            <span className="fake-time">{fmtTime(time)}</span>
            <span className="muted">{current ? `Now: ${current.concept.name}` : "Click a chapter to jump"}</span>
          </div>
        </div>
      )}
      <div className="chapter-strip" title="Topics along the lecture timeline">
        {strip.map((s, i) => (
          <button key={i} className={`chapter ${current === s ? "active" : ""}`}
            style={{ flexGrow: Math.max(1, s.end - s.start) }} onClick={() => jump(s)}
            title={`${s.concept.name} (${fmtTime(s.start)} to ${fmtTime(s.end)})`}>
            <span>{s.concept.name}</span>
          </button>
        ))}
      </div>
      <div className="playhead-row">
        <div className="playhead" style={{ left: `${Math.min(100, (time / duration) * 100)}%` }} />
      </div>
    </div>
  );
}
