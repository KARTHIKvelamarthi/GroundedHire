import os
import sys
import unittest
import tempfile
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure parent directory is in path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.database import run_migrations, Base
from backend.models import Session as DBSession, Question as DBQuestion, Answer as DBAnswer


class TestDatabaseMigration(unittest.TestCase):
    def setUp(self):
        # Create a temporary file-based SQLite database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        
        # Connect with raw sqlite3 to create the OLD schema (before Step 1)
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Old sessions table definition (without company, jd_text, jd_profile, topic_plan)
        cursor.execute("""
            CREATE TABLE sessions (
                id VARCHAR PRIMARY KEY,
                candidate_name VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                resume_profile_json TEXT NOT NULL,
                starting_difficulty_tier VARCHAR NOT NULL,
                current_difficulty_tier VARCHAR NOT NULL,
                total_planned_questions INTEGER NOT NULL,
                current_phase VARCHAR,
                questions_asked_count INTEGER,
                planned_topics_json TEXT NOT NULL,
                created_at DATETIME
            )
        """)
        
        # Old questions table definition (without source, resume_ref)
        cursor.execute("""
            CREATE TABLE questions (
                id VARCHAR PRIMARY KEY,
                session_id VARCHAR NOT NULL,
                question_text TEXT NOT NULL,
                concept_being_tested VARCHAR,
                expected_answer_points_json TEXT,
                topic VARCHAR,
                difficulty_tier VARCHAR NOT NULL,
                source_chunk_id VARCHAR,
                source_book VARCHAR,
                source_section_title VARCHAR,
                source_chunk_text TEXT,
                question_type VARCHAR NOT NULL,
                sequence_number INTEGER NOT NULL,
                created_at DATETIME,
                FOREIGN KEY(session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        """)
        
        # Old answers table definition
        cursor.execute("""
            CREATE TABLE answers (
                id VARCHAR PRIMARY KEY,
                question_id VARCHAR NOT NULL,
                answer_text TEXT NOT NULL,
                quality_signal VARCHAR NOT NULL,
                covered_points_json TEXT NOT NULL,
                missed_points_json TEXT NOT NULL,
                brief_feedback TEXT NOT NULL,
                created_at DATETIME,
                FOREIGN KEY(question_id) REFERENCES questions (id) ON DELETE CASCADE
            )
        """)
        
        # Insert a pre-existing session and question
        cursor.execute("""
            INSERT INTO sessions (
                id, candidate_name, role, resume_profile_json,
                starting_difficulty_tier, current_difficulty_tier,
                total_planned_questions, current_phase, questions_asked_count,
                planned_topics_json
            ) VALUES (
                'test-session-1', 'Alice Candidate', 'ai_ml_engineer', '{}',
                'intermediate', 'intermediate', 6, 'skills', 1, '[]'
            )
        """)
        
        cursor.execute("""
            INSERT INTO questions (
                id, session_id, question_text, difficulty_tier,
                question_type, sequence_number
            ) VALUES (
                'test-question-1', 'test-session-1', 'Explain overfitting.',
                'intermediate', 'skill', 1
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Setup SQLAlchemy engine targeting this temporary database
        self.engine = create_engine(
            f"sqlite:///{self.temp_db_path}",
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    def tearDown(self):
        self.engine.dispose()
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except OSError:
                pass

    def test_migration_adds_missing_columns_to_old_schema(self):
        # 1. Verify missing columns before migration
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        session_cols_before = [row[1] for row in cursor.execute("PRAGMA table_info(sessions)").fetchall()]
        question_cols_before = [row[1] for row in cursor.execute("PRAGMA table_info(questions)").fetchall()]
        conn.close()
        
        self.assertNotIn("company", session_cols_before)
        self.assertNotIn("jd_text", session_cols_before)
        self.assertNotIn("jd_profile", session_cols_before)
        self.assertNotIn("topic_plan", session_cols_before)
        self.assertNotIn("source", question_cols_before)
        self.assertNotIn("resume_ref", question_cols_before)
        
        # 2. Run migration
        run_migrations(self.engine)
        
        # 3. Verify columns exist after migration
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        session_cols_after = [row[1] for row in cursor.execute("PRAGMA table_info(sessions)").fetchall()]
        question_cols_after = [row[1] for row in cursor.execute("PRAGMA table_info(questions)").fetchall()]
        conn.close()
        
        for expected in ["company", "jd_text", "jd_profile", "topic_plan"]:
            self.assertIn(expected, session_cols_after, f"Missing '{expected}' in sessions")
            
        for expected in ["source", "resume_ref"]:
            self.assertIn(expected, question_cols_after, f"Missing '{expected}' in questions")

        # 4. Verify pre-existing data was preserved
        db = self.SessionLocal()
        existing_session = db.query(DBSession).filter_by(id="test-session-1").first()
        self.assertIsNotNone(existing_session)
        self.assertEqual(existing_session.candidate_name, "Alice Candidate")
        self.assertIsNone(existing_session.company)
        self.assertIsNone(existing_session.jd_text)
        self.assertIsNone(existing_session.jd_profile)
        self.assertIsNone(existing_session.topic_plan)
        
        existing_q = db.query(DBQuestion).filter_by(id="test-question-1").first()
        self.assertIsNotNone(existing_q)
        self.assertEqual(existing_q.question_text, "Explain overfitting.")
        self.assertIsNone(existing_q.source)
        self.assertIsNone(existing_q.resume_ref)
        db.close()

    def test_migration_is_idempotent(self):
        # Run migration once
        run_migrations(self.engine)
        
        # Run migration a second and third time
        run_migrations(self.engine)
        run_migrations(self.engine)
        
        # Verify columns still exist and are not duplicated or corrupted
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        session_cols = [row[1] for row in cursor.execute("PRAGMA table_info(sessions)").fetchall()]
        question_cols = [row[1] for row in cursor.execute("PRAGMA table_info(questions)").fetchall()]
        conn.close()
        
        # Check column counts
        self.assertEqual(session_cols.count("company"), 1)
        self.assertEqual(session_cols.count("jd_text"), 1)
        self.assertEqual(session_cols.count("jd_profile"), 1)
        self.assertEqual(session_cols.count("topic_plan"), 1)
        self.assertEqual(question_cols.count("source"), 1)
        self.assertEqual(question_cols.count("resume_ref"), 1)

    def test_insert_and_query_with_new_fields(self):
        # Run migration
        run_migrations(self.engine)
        
        db = self.SessionLocal()
        new_session = DBSession(
            id="test-session-2",
            candidate_name="Bob Engineer",
            role="ai_ml_engineer",
            resume_profile_json='{"skills": ["python"]}',
            starting_difficulty_tier="intermediate",
            current_difficulty_tier="intermediate",
            total_planned_questions=8,
            planned_topics_json='["classification"]',
            company="DeepMind",
            jd_text="Looking for an ML engineer with strong PyTorch experience.",
            jd_profile={"required_skills": ["PyTorch"], "seniority_hint": "intermediate"},
            topic_plan=[{"topic": "classification", "bucket": "verify", "source": "skills", "resume_ref": "PyTorch"}]
        )
        db.add(new_session)
        
        new_question = DBQuestion(
            id="test-question-2",
            session_id="test-session-2",
            question_text="How does backpropagation compute gradients in deep networks?",
            difficulty_tier="intermediate",
            question_type="skill",
            sequence_number=1,
            source="jd_gap",
            resume_ref="PyTorch"
        )
        db.add(new_question)
        db.commit()
        
        # Query back and verify
        fetched_session = db.query(DBSession).filter_by(id="test-session-2").first()
        self.assertEqual(fetched_session.company, "DeepMind")
        self.assertEqual(fetched_session.jd_text, "Looking for an ML engineer with strong PyTorch experience.")
        self.assertEqual(fetched_session.jd_profile.get("seniority_hint"), "intermediate")
        self.assertEqual(fetched_session.topic_plan[0]["bucket"], "verify")
        
        fetched_q = db.query(DBQuestion).filter_by(id="test-question-2").first()
        self.assertEqual(fetched_q.source, "jd_gap")
        self.assertEqual(fetched_q.resume_ref, "PyTorch")
        db.close()


if __name__ == "__main__":
    unittest.main()

