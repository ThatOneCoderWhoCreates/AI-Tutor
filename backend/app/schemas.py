"""Shared data shapes. This file IS the week-1 contract between the three roles.

Role A produces Chunk objects, Role B reads them, Role C displays them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str                       # "lec1_007"
    lecture_id: str               # "lec1"
    text: str                     # what the lecturer said in this time range
    slide_text: str = ""          # OCR text of the slide on screen
    start: float                  # seconds from the start of the video
    end: float
    concept: str = ""             # topic label, lowercase, e.g. "gradient descent"


class Segment(BaseModel):
    lecture_id: str
    start: float
    end: float


class Citation(BaseModel):
    n: int
    chunk_id: str
    lecture_id: str
    start: float
    end: float
    concept: str


class Claim(BaseModel):
    text: str
    supported: bool
    reason: str = ""


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    lecture_id: str | None = None
    student_id: str | None = None
    verify: bool = False
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    claims: list[Claim] | None = None
    # lecture = answered from the lecture, general = answered by the fallback model,
    # none = not covered and no fallback answer
    source: Literal["lecture", "general", "none"] = "lecture"
    note: str | None = None
    answered_by: str | None = None


class QuizRequest(BaseModel):
    concept: str
    n: int = Field(default=3, ge=1, le=10)
    student_id: str | None = None
    fresh: bool = False


class QuizQuestionPublic(BaseModel):
    id: str
    question: str
    options: list[str]


class QuizResponse(BaseModel):
    concept: str
    questions: list[QuizQuestionPublic]


class AnswerRequest(BaseModel):
    student_id: str
    question_id: str
    choice: int


class AnswerResponse(BaseModel):
    correct: bool
    correct_index: int
    explanation: str
    concept: str
    mastery: float
    source: Segment | None = None


class NextRequest(BaseModel):
    student_id: str


class NextResponse(BaseModel):
    concept: str | None
    mastery: float
    reason: str
    segments: list[Segment]


class ConceptInfo(BaseModel):
    name: str
    order: int
    segments: list[Segment]
    mastery: float | None = None
    attempts: int = 0
    correct: int = 0


class RenameRequest(BaseModel):
    old: str
    new: str


class YouTubeRequest(BaseModel):
    url: str
    lecture_id: str | None = None
    title: str | None = None
    use_captions: bool = False


class BKTFitRequest(BaseModel):
    source: Literal["app", "simulated"] = "app"
    students: int = Field(default=300, ge=10, le=5000)
    seed: int = 0


class FinetuneRequest(BaseModel):
    epochs: int = Field(default=3, ge=1, le=10)


class UseModelRequest(BaseModel):
    model: Literal["base", "finetuned"]
