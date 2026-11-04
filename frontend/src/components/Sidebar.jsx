import React from "react";
import { IconLab, IconLectures, IconPractice, IconProgress, IconStudy } from "./Icons.jsx";

export const NAV = [
  { id: "study", label: "Study", hint: "Watch and ask", Icon: IconStudy },
  { id: "practice", label: "Practice", hint: "Quiz yourself", Icon: IconPractice },
  { id: "progress", label: "Progress", hint: "What you know", Icon: IconProgress },
  { id: "lectures", label: "Lectures", hint: "Add and manage", Icon: IconLectures },
  { id: "lab", label: "Lab", hint: "Train the models", Icon: IconLab },
];

function ModelBadge({ health }) {
  if (!health) return <span className="badge">connecting…</span>;
  if (health.llm_provider === "mock") return <span className="badge">mock model</span>;
  const map = {
    ready: [health.llm_model, "ok", "Model is ready"],
    offline: ["Ollama offline", "bad", "Start it with: brew services start ollama"],
    "no-model": [`${health.llm_model} not pulled`, "bad", `Run: ollama pull ${health.llm_model}`],
  };
  const [text, cls, title] = map[health.llm_status] || [health.llm_model, "", ""];
  return <span className={`badge ${cls}`} title={title}>{text}</span>;
}

export default function Sidebar({ page, go, studentId, setStudentId, health, badges }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="logo">◐</span>
        <span className="brand-name">Lecture Tutor</span>
      </div>
      <nav className="nav">
        {NAV.map(({ id, label, hint, Icon }) => (
          <a key={id} href={`#/${id}`} title={label} className={page === id ? "active" : ""} onClick={() => go(id)}>
            <Icon />
            <span className="nav-text">
              <span className="nav-label">{label}</span>
              <span className="nav-hint">{hint}</span>
            </span>
            {badges[id] ? <span className="nav-badge">{badges[id]}</span> : null}
          </a>
        ))}
      </nav>
      <div className="sidebar-foot">
        <label className="student">
          <span>Student</span>
          <input value={studentId} onChange={(e) => setStudentId(e.target.value || "student-1")} />
        </label>
        <ModelBadge health={health} />
        {health?.fallback && (
          <span className={`fallback-line ${health.fallback.ready ? "" : "off"}`} title={health.fallback.problem || "Answers questions the lectures do not cover"}>
            Outside-lecture answers: {health.fallback.ready ? `${health.fallback.label}${health.fallback.model && health.fallback.provider !== "local" ? ` (${health.fallback.model})` : ""}` : "off"}
          </span>
        )}
      </div>
    </aside>
  );
}
