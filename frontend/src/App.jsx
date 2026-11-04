import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Sidebar, { NAV } from "./components/Sidebar.jsx";
import LabPage from "./pages/LabPage.jsx";
import LecturesPage from "./pages/LecturesPage.jsx";
import PracticePage from "./pages/PracticePage.jsx";
import ProgressPage from "./pages/ProgressPage.jsx";
import StudyPage from "./pages/StudyPage.jsx";

const PAGES = NAV.map((n) => n.id);
const pageFromHash = () => {
  const h = window.location.hash.replace(/^#\/?/, "");
  return PAGES.includes(h) ? h : "study";
};

function loadPref(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function savePref(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

const isBusy = (l) => !["ready", "error", "missing"].includes(l.status);

export default function App() {
  const [page, setPage] = useState(pageFromHash);
  const [studentId, setStudentId] = useState(() => loadPref("tutor.student", "student-1"));
  const [health, setHealth] = useState(null);
  const [lectures, setLectures] = useState([]);
  const [lectureId, setLectureId] = useState(() => loadPref("tutor.lecture", ""));
  const [concepts, setConcepts] = useState([]);
  const [next, setNext] = useState(null);
  const [seek, setSeek] = useState(null);
  const [practiceTopic, setPracticeTopic] = useState(null);
  const [practiceCount, setPracticeCount] = useState(3);
  const [messages, setMessages] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState("");
  const [showWelcome, setShowWelcome] = useState(() => loadPref("tutor.welcome", "show") === "show");

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (p) => {
    if (window.location.hash !== `#/${p}`) window.location.hash = `#/${p}`;
    setPage(p);
    window.scrollTo(0, 0);
  };

  const refresh = useCallback(async () => {
    try {
      const [h, lecs, cons, nx] = await Promise.all([
        api.health(), api.lectures(), api.concepts(studentId), api.next(studentId),
      ]);
      setHealth(h);
      setLectures(lecs);
      setConcepts(cons);
      setNext(nx);
      setRefreshKey((k) => k + 1);
      setError("");
      setLectureId((cur) => {
        const ready = lecs.filter((l) => l.status === "ready");
        if (ready.some((l) => l.id === cur)) return cur;
        return (ready.find((l) => l.id !== "demo") || ready[0])?.id || "";
      });
    } catch (e) {
      setError(`Cannot reach the backend (${e.message}). Start it with: python -m uvicorn app.main:app --reload --port 8000`);
    }
  }, [studentId]);

  useEffect(() => {
    refresh();
    savePref("tutor.student", studentId);
  }, [refresh]);

  useEffect(() => {
    if (lectureId) savePref("tutor.lecture", lectureId);
  }, [lectureId]);

  // keep lecture status fresh on every page while something is processing
  const processing = lectures.some(isBusy);
  useEffect(() => {
    if (!processing) return;
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [processing, refresh]);

  const seekTo = (lecId, start, end) => {
    setLectureId(lecId);
    setSeek({ lectureId: lecId, start, end, nonce: Math.random() });
    go("study");
  };

  const practice = (concept, count = 3) => {
    setPracticeTopic(concept);
    setPracticeCount(count);
    go("practice");
  };

  const openLecture = (id) => {
    setLectureId(id);
    go("study");
  };

  const hideWelcome = () => {
    setShowWelcome(false);
    savePref("tutor.welcome", "hide");
  };

  const realLectures = lectures.filter((l) => l.id !== "demo" && l.status === "ready").length;
  const badges = { lectures: processing ? "…" : null };

  return (
    <div className="app">
      <Sidebar page={page} go={go} studentId={studentId} setStudentId={setStudentId} health={health} badges={badges} />
      <main className="main">
        {error && <div className="banner error">{error}</div>}
        {page === "study" && (
          <StudyPage lectures={lectures} lectureId={lectureId} setLectureId={setLectureId} concepts={concepts} next={next}
            seek={seek} seekTo={seekTo} practice={practice} studentId={studentId} messages={messages}
            setMessages={setMessages} go={go} showWelcome={showWelcome && realLectures === 0} hideWelcome={hideWelcome} />
        )}
        {page === "practice" && (
          <PracticePage concepts={concepts} next={next} topic={practiceTopic} setTopic={setPracticeTopic}
            count={practiceCount} studentId={studentId} refresh={refresh} seekTo={seekTo} practice={practice} />
        )}
        {page === "progress" && (
          <ProgressPage studentId={studentId} concepts={concepts} refreshKey={refreshKey} seekTo={seekTo}
            practice={practice} go={go} />
        )}
        {page === "lectures" && (
          <LecturesPage lectures={lectures} refresh={refresh} openLecture={openLecture} studentId={studentId} />
        )}
        {page === "lab" && <LabPage refresh={refresh} />}
      </main>
    </div>
  );
}
