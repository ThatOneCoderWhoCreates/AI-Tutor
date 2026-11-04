import React, { useEffect, useState } from "react";
import { api, fmtTime, pct } from "../api.js";
import { IconPlay } from "./Icons.jsx";

// One topic's practice session. One question at a time; mastery updates after each answer.
export default function PracticePanel({ concept, studentId, count, onAnswered, onSeek, startMastery }) {
  const [questions, setQuestions] = useState([]);
  const [index, setIndex] = useState(0);
  const [result, setResult] = useState(null);
  const [chosen, setChosen] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [answers, setAnswers] = useState([]);

  const load = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    setChosen(null);
    setIndex(0);
    setAnswers([]);
    try {
      const res = await api.quiz(concept, count, studentId);
      setQuestions(res.questions);
    } catch (e) {
      setError(e.message);
      setQuestions([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [concept, count, studentId]);

  const q = questions[index];

  const choose = async (i) => {
    if (result) return;
    setChosen(i);
    try {
      const res = await api.answer(studentId, q.id, i);
      setResult(res);
      setAnswers((a) => [...a, res.correct]);
      onAnswered();
    } catch (e) {
      setError(e.message);
    }
  };

  const next = () => {
    setResult(null);
    setChosen(null);
    setIndex((x) => x + 1);
  };

  const finished = result && index + 1 >= questions.length;
  const score = answers.filter(Boolean).length;

  return (
    <div className="card practice">
      <div className="practice-head">
        <div>
          <span className="eyebrow">Practice</span>
          <h2>{concept}</h2>
        </div>
        {questions.length > 0 && (
          <div className="dots" aria-label={`Question ${index + 1} of ${questions.length}`}>
            {questions.map((_, i) => (
              <span key={i} className={i < answers.length ? (answers[i] ? "ok" : "bad") : i === index ? "now" : ""} />
            ))}
          </div>
        )}
      </div>

      {loading && <p className="muted"><span className="spinner" />Writing questions from the lecture. The first time can take a little while with a local model.</p>}
      {error && <p className="error">{error}</p>}

      {!loading && q && (
        <div className="question">
          <p className="q-text">{q.question}</p>
          <div className="options">
            {q.options.map((opt, i) => {
              let cls = "option";
              if (result) {
                if (i === result.correct_index) cls += " correct";
                else if (i === chosen) cls += " wrong";
              }
              return (
                <button key={i} className={cls} onClick={() => choose(i)} disabled={!!result}>
                  <span className="opt-letter">{"ABCD"[i]}</span> <span>{opt}</span>
                </button>
              );
            })}
          </div>

          {result && (
            <div className={`feedback ${result.correct ? "good" : "bad"}`}>
              <p><strong>{result.correct ? "Correct." : "Not quite."}</strong> {result.explanation}</p>
              <div className="feedback-row">
                <span>Mastery of this topic: <strong>{pct(result.mastery)}</strong></span>
                {result.source && (
                  <button className="chip" onClick={() => onSeek(result.source.lecture_id, result.source.start, result.source.end)}>
                    <IconPlay /> Rewatch {fmtTime(result.source.start)}
                  </button>
                )}
              </div>
              {!finished && <button className="primary" onClick={next}>Next question</button>}
            </div>
          )}

          {finished && (
            <div className="summary">
              <p>
                You got <strong>{score} of {questions.length}</strong> right.
                {startMastery != null && <> Mastery went from {pct(startMastery)} to <strong>{pct(result.mastery)}</strong>.</>}
              </p>
              <button className="primary" onClick={load}>More questions</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
