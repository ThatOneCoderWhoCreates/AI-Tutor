import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const PARAMS = [
  ["p_init", "Already knows", "Chance the student knows the topic before practicing"],
  ["p_learn", "Learns per question", "Chance of learning the topic from one practice question"],
  ["p_slip", "Slip", "Chance of a wrong answer even when the topic is known"],
  ["p_guess", "Guess", "Chance of a right answer without knowing the topic"],
];

const n3 = (v) => (v == null ? "n/a" : Number(v).toFixed(3));
const n2 = (v) => (v == null ? "n/a" : Number(v).toFixed(2));

function BestCell({ value, best, fmt = n3 }) {
  return <td className={best ? "best" : ""}>{fmt(value)}</td>;
}

function BktCard({ onChanged }) {
  const [info, setInfo] = useState(null);
  const [students, setStudents] = useState(300);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const load = () => api.labBkt().then(setInfo).catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  const act = async (label, fn) => {
    setBusy(label);
    setError("");
    try {
      await fn();
      await load();
      onChanged();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy("");
    }
  };

  if (!info) return <div className="card"><p className="muted">Loading…</p></div>;
  const fit = info.last_fit;
  const active = info.active;
  const metrics = fit?.test_metrics;
  const rows = metrics ? [["Default numbers", metrics.default], ["Learned, one set for all topics", metrics.fitted_global],
    ["Learned, one set per topic", metrics.fitted_per_topic]] : [];
  const best = (key) => Math.max(...rows.map(([, m]) => m[key] ?? -Infinity));

  return (
    <div className="card lab-card">
      <div className="card-head">
        <h3>Knowledge tracing: learn the BKT numbers</h3>
        <span className={`pill ${active.source === "fitted" ? "high" : "new"}`}>
          {active.source === "fitted" ? "Using learned numbers" : "Using default numbers"}
        </span>
      </div>
      <p className="muted">
        Mastery is estimated with Bayesian Knowledge Tracing (BKT), which uses four numbers per topic. Instead of
        guessing them, Expectation-Maximization (EM) learns them from real answers: it finds the numbers that best
        explain the right and wrong answers students actually gave.
      </p>

      <div className="table-wrap">
        <table className="table compact">
          <thead><tr><th>Number</th><th>Default</th><th>In use (all topics)</th></tr></thead>
          <tbody>
            {PARAMS.map(([k, label, hint]) => (
              <tr key={k}>
                <td title={hint}>{label} <span className="muted small">({k})</span></td>
                <td>{n2(info.default[k])}</td>
                <td>{n3(active.global[k])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="lab-actions">
        <button className="primary" disabled={!!busy || info.app_data.answers === 0}
          onClick={() => act("app", () => api.labBktFit("app", 300))}>
          {busy === "app" ? "Learning…" : "Learn from app answers"}
        </button>
        <span className="muted small">
          {info.app_data.answers} answer{info.app_data.answers === 1 ? "" : "s"} from {info.app_data.students} student
          {info.app_data.students === 1 ? "" : "s"} so far
          {info.app_data.students < 5 && ". Answers from at least 5 students are needed to measure improvement"}
        </span>
      </div>
      <div className="lab-actions">
        <button className="secondary" disabled={!!busy} onClick={() => act("sim", () => api.labBktFit("simulated", students))}>
          {busy === "sim" ? "Running…" : "Test the method on simulated students"}
        </button>
        <label className="inline">
          <input type="number" min="20" max="3000" step="10" value={students}
            onChange={(e) => setStudents(Number(e.target.value) || 300)} /> students
        </label>
      </div>
      <p className="muted small">
        The simulated test makes fake students with known, hidden numbers, then checks whether EM recovers them.
        This shows the method works before there is enough real data.
      </p>
      {error && <p className="error">{error}</p>}

      {fit && (
        <div className="fit-result">
          <h4>
            Last result: {fit.source === "simulated" ? "simulated students" : "app answers"}
            <span className="muted small"> · {fit.students} students, {fit.answers} answers, {fit.seconds}s</span>
          </h4>

          <div className="table-wrap">
            <table className="table compact">
              <thead>
                <tr><th>Topic</th><th>Sequences</th>{PARAMS.map(([k, label]) => <th key={k} title={k}>{label}</th>)}</tr>
              </thead>
              <tbody>
                <tr className="strong">
                  <td>All topics together</td><td>{fit.sequences}</td>
                  {PARAMS.map(([k]) => <td key={k}>{n3(fit.global[k])}</td>)}
                </tr>
                {Object.entries(fit.topic_info).map(([topic, ti]) => (
                  <React.Fragment key={topic}>
                    <tr>
                      <td>{topic}</td><td>{ti.sequences}</td>
                      {PARAMS.map(([k]) => <td key={k}>{ti.fitted ? n3(fit.per_topic[topic][k]) : "uses all-topic"}</td>)}
                    </tr>
                    {fit.true_per_topic?.[topic] && (
                      <tr className="truth">
                        <td>true hidden values</td><td />
                        {PARAMS.map(([k]) => <td key={k}>{n3(fit.true_per_topic[topic][k])}</td>)}
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>

          {metrics && (
            <>
              <h4>Prediction on {fit.test_students} held-out students</h4>
              <div className="table-wrap">
                <table className="table compact">
                  <thead><tr><th>Numbers used</th><th>Log-likelihood per answer</th><th>Accuracy</th><th>AUC</th></tr></thead>
                  <tbody>
                    {rows.map(([name, m]) => (
                      <tr key={name}>
                        <td>{name}</td>
                        <BestCell value={m.log_likelihood_per_answer} best={m.log_likelihood_per_answer === best("log_likelihood_per_answer")} />
                        <BestCell value={m.accuracy} best={m.accuracy === best("accuracy")} />
                        <BestCell value={m.auc} best={m.auc === best("auc")} />
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted small">
                These students were not used for learning. Each answer is predicted from the answers before it.
                Higher is better in every column. AUC (Area Under the ROC Curve) is the chance that a right answer
                got a higher predicted score than a wrong one.
              </p>
            </>
          )}

          <div className="lab-actions">
            {fit.source === "app" && (
              <button className="primary" disabled={!!busy} onClick={() => act("apply", api.labBktApply)}>
                Use these numbers
              </button>
            )}
            {active.source === "fitted" && (
              <button className="secondary" disabled={!!busy} onClick={() => act("reset", api.labBktReset)}>
                Go back to default numbers
              </button>
            )}
            {fit.source === "app" && <span className="muted small">Mastery for all students is recalculated.</span>}
          </div>
        </div>
      )}
    </div>
  );
}

function EmbeddingCard({ onChanged }) {
  const [info, setInfo] = useState(null);
  const [epochs, setEpochs] = useState(3);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = () => api.labEmbeddings().then(setInfo).catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  const running = info?.job?.state === "running";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [running]);

  const act = async (fn) => {
    setBusy(true);
    setError("");
    try {
      await fn();
      await load();
      onChanged();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  if (!info) return <div className="card"><p className="muted">Loading…</p></div>;
  const run = info.last_run;
  const job = info.job;
  const metricRows = [["hit@1", "Hit@1"], ["hit@3", "Hit@3"], ["mrr", "MRR"]];

  return (
    <div className="card lab-card">
      <div className="card-head">
        <h3>Search model: fine-tune on your lectures</h3>
        <span className={`pill ${info.using_finetuned ? "high" : "new"}`}>
          {info.using_finetuned ? "Using fine-tuned model" : "Using base model"}
        </span>
      </div>
      <p className="muted">
        The search model turns text into numbers so questions can be matched with the right part of a lecture. The
        base model learned from general web text. Fine-tuning teaches it your lectures: the local model writes a few
        practice questions for every chunk, and the search model is trained to match each question with its chunk.
      </p>
      <p className="small">Base model: <code>{info.base_model}</code></p>

      {!info.can_train && <p className="error">{info.reason}</p>}
      {info.chunks < 30 && (
        <p className="muted small">
          You have {info.chunks} chunks. Add one or two real lectures first; with the demo alone the result is not meaningful.
        </p>
      )}

      <div className="lab-actions">
        <button className="primary" disabled={!info.can_train || running || busy}
          onClick={() => act(() => api.labEmbeddingsTrain(epochs))}>
          {running ? "Training…" : "Fine-tune on my lectures"}
        </button>
        <label className="inline">
          <select value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} disabled={running}>
            {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n} epoch{n > 1 ? "s" : ""}</option>)}
          </select>
        </label>
      </div>

      {(running || job.state === "error" || job.state === "interrupted") && (
        <div className={`job ${job.state}`}>
          <p className="small">{running && <span className="spinner" />}{job.message}</p>
          {running && job.progress != null && (
            <div className="progress-bar"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
          )}
        </div>
      )}
      {error && <p className="error">{error}</p>}

      {run && (
        <div className="fit-result">
          <h4>
            Last training run
            <span className="muted small"> · {run.train_pairs} question-chunk pairs, {run.epochs} epochs on {run.device}, {run.seconds}s</span>
          </h4>
          <div className="table-wrap">
            <table className="table compact">
              <thead><tr><th>Measured on {run.val_questions} questions from {run.val_chunks} held-out chunks</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
              <tbody>
                {metricRows.map(([k, label]) => {
                  const b = run.before[k];
                  const a = run.after[k];
                  const d = a != null && b != null ? a - b : null;
                  return (
                    <tr key={k}>
                      <td>{label}</td><td>{n3(b)}</td><td>{n3(a)}</td>
                      <td className={d > 0 ? "up" : d < 0 ? "down" : ""}>{d == null ? "n/a" : `${d >= 0 ? "+" : ""}${d.toFixed(3)}`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            Training loss by epoch: {run.history.filter((h) => h.loss != null).map((h) => h.loss.toFixed(3)).join(", ")}.
            The best epoch ({run.best_epoch}) was kept. Held-out chunks were never used for training. For the report,
            also run the evaluation script on your own hand-written questions.
          </p>
          <div className="lab-actions">
            {info.finetuned_available && !info.using_finetuned && (
              <button className="primary" disabled={busy} onClick={() => act(() => api.labEmbeddingsUse("finetuned"))}>
                Use fine-tuned model
              </button>
            )}
            {info.using_finetuned && (
              <button className="secondary" disabled={busy} onClick={() => act(() => api.labEmbeddingsUse("base"))}>
                Switch back to base model
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function LabPage({ refresh }) {
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Lab</h1>
          <p className="muted">Train the two parts of the system that learn from data, and see whether they improve.</p>
        </div>
      </header>
      <div className="lab-grid">
        <BktCard onChanged={refresh} />
        <EmbeddingCard onChanged={refresh} />
      </div>
    </div>
  );
}
