import React from "react";

const base = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8,
  strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };

export const IconStudy = () => (
  <svg {...base}><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M10 8.5v5l4-2.5z" /><path d="M8 21h8" /></svg>
);
export const IconPractice = () => (
  <svg {...base}><rect x="4" y="3" width="16" height="18" rx="2" /><path d="m8 12 2.5 2.5L16 9" /></svg>
);
export const IconProgress = () => (
  <svg {...base}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
);
export const IconLectures = () => (
  <svg {...base}><path d="M4 5h16M4 12h16M4 19h10" /><path d="m17 16 4 3-4 3z" /></svg>
);
export const IconLab = () => (
  <svg {...base}><path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3" /><path d="M7 15h10" /></svg>
);
export const IconPlay = () => (
  <svg {...base} width={14} height={14}><path d="M7 5v14l11-7z" fill="currentColor" stroke="none" /></svg>
);
export const IconBack = () => (
  <svg {...base} width={16} height={16}><path d="M15 18 9 12l6-6" /></svg>
);
