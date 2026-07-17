import datetime
import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    resume_profile_json = Column(Text, nullable=False)
    starting_difficulty_tier = Column(String, nullable=False)
    current_difficulty_tier = Column(String, nullable=False)
    total_planned_questions = Column(Integer, nullable=False)
    current_phase = Column(String, default="skills")  # "skills", "project_transition", "projects"
    questions_asked_count = Column(Integer, default=0)
    planned_topics_json = Column(Text, nullable=False)  # List of topics and project titles to ask
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    questions = relationship("Question", back_populates="session", cascade="all, delete-orphan")

class Question(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    question_text = Column(Text, nullable=False)
    concept_being_tested = Column(String, nullable=True)
    expected_answer_points_json = Column(Text, nullable=True)  # List of strings
    topic = Column(String, nullable=True)
    difficulty_tier = Column(String, nullable=False)
    source_chunk_id = Column(String, nullable=True)
    source_book = Column(String, nullable=True)
    source_section_title = Column(String, nullable=True)
    source_chunk_text = Column(Text, nullable=True)
    question_type = Column(String, nullable=False)  # "skill", "project", "transition"
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("Session", back_populates="questions")
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")

class Answer(Base):
    __tablename__ = "answers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question_id = Column(String, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)
    answer_text = Column(Text, nullable=False)
    quality_signal = Column(String, nullable=False)  # "strong", "developing", "needs_review"
    covered_points_json = Column(Text, nullable=False)
    missed_points_json = Column(Text, nullable=False)
    brief_feedback = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    question = relationship("Question", back_populates="answers")
