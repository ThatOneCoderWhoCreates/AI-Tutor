import React, { useEffect, useRef, useState } from "react";
import { api, fmtTime } from "../api.js";
import { IconPlay } from "./Icons.jsx";

const GENERAL_PREFIX = "[It was not covered in this lecture, but this is the answer to your question]";

// Chat with the lecture. Every answer links to the exact timestamps it came from.
// "Check claims" runs the trust layer and marks each sentence supported or not.
// Messages live in App so the chat survives switching pages.
export default function AskPanel({ studentId, onSeek, messages, setMessages, suggestions = [] }) {
  const [question, setQuestion] = useState("");
  const [verify, setVerify] = useState(false);
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages, busy]);

  const send = async (text) => {
    const q = (text ?? question).trim();
    if (!q || busy) return;
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const history = messages.slice(-6).map((m) =>
        m.role === "user" ? { role: "user", content: m.text } : { role: "assistant", content: m.answer || "" }
      );
      const res = await api.ask(q, studentId, verify, history);
      setMessages((m) => [...m, { role: "tutor", ...res }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "tutor", answer: `Something went wrong: ${err.message}`, citations: [], error: true }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card ask">
      <div className="card-head">
        <h3>Ask the lecture</h3>
        {messages.length > 0 && <button className="link" onClick={() => setMessages([])}>Clear</button>}
      </div>
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p>Answers come only from this course's lectures, with a timestamp you can click.</p>
            {suggestions.length > 0 && (
              <div className="suggestions">
                {suggestions.slice(0, 4).map((s) => (
                  <button key={s} className="chip ghost" onClick={() => send(`Explain ${s} in simple words`)}>
                    Explain {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg user">{m.text}</div>
          ) : (
            <div key={i} className={`msg tutor ${m.error ? "error" : ""}`}>
              {m.claims ? (
                <ul className="claims">
                  {m.claims.map((c, j) => (
                    <li key={j} className={c.supported ? "ok" : "bad"} title={c.reason}>
                      <span className="mark">{c.supported ? "✓" : "?"}</span> {c.text}
                    </li>
                  ))}
                </ul>
              ) : m.source === "general" ? (
                <>
                  <p className="general-note">{GENERAL_PREFIX}</p>
                  <p className="general-body">{m.answer.replace(GENERAL_PREFIX, "").trim()}</p>
                  {m.answered_by && <p className="muted small">Answered by {m.answered_by}</p>}
                </>
              ) : (
                <p>{m.answer.replace(/\s*\[\d+(?:\s*,\s*\d+)*\]/g, "")}</p>
              )}
              {m.note && <p className="muted small note">{m.note}</p>}
              {m.citations?.length > 0 && (
                <div className="citations">
                  {m.citations.map((c) => (
                    <button key={c.chunk_id} className="chip" onClick={() => onSeek(c.lecture_id, c.start, c.end)}>
                      <IconPlay /> {fmtTime(c.start)} to {fmtTime(c.end)} · {c.concept}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        )}
        {busy && <div className="msg tutor muted"><span className="spinner" />Thinking…</div>}
        <div ref={endRef} />
      </div>
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about the lecture…" />
        <button type="submit" className="primary" disabled={busy}>Ask</button>
      </form>
      <label className="check small">
        <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} />
        Check each sentence against the lecture (slower)
      </label>
    </div>
  );
}
