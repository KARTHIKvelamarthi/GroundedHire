import os
import sys
import io
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure root and core are in path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
for d in [_current_dir, _parent_dir, os.path.join(_parent_dir, "core")]:
    if d not in sys.path:
        sys.path.insert(0, d)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import (
    start_session,
    get_next_question,
    get_session_plan,
    get_session_summary,
)
from backend.database import Base, run_migrations
from backend.models import Session as DBSession, Question as DBQuestion


class TestBackendJDIntegration(unittest.TestCase):
    def setUp(self):
        # Create an isolated in-memory SQLite database per test
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        run_migrations(self.engine)
        self.db = self.SessionLocal()

        self.mock_resume_profile = {
            "candidate_name": "Jane Doe",
            "starting_difficulty_tier": "intermediate",
            "raw_skills": ["PyTorch", "Python", "scikit-learn"],
            "mapped_topics": ["deep_learning", "classification", "model_selection"],
            "projects": [
                {
                    "title": "Audio Classifier",
                    "description": "Audio classifier built with PyTorch.",
                    "technologies_used": ["PyTorch", "Python"],
                    "related_topics": ["deep_learning", "classification"]
                }
            ]
        }

        self.mock_jd_profile = {
            "usable": True,
            "role_title": "Staff ML Engineer",
            "company": "DeepTech",
            "seniority_hint": "advanced",
            "skills_by_priority": {
                "high": ["PyTorch", "Clustering"],
                "moderate": ["Evaluation Metrics"],
                "low_inferred": ["Optimization"]
            },
            "topic_weights": {
                "deep_learning": 1.0,
                "clustering": 1.0,
                "evaluation_metrics": 0.6,
                "optimization": 0.3
            },
            "inferred_only_topics": ["optimization"],
            "source": "llm"
        }

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _create_mock_upload_file(self, filename="resume.pdf"):
        mock_file = MagicMock()
        mock_file.filename = filename
        mock_file.file = io.BytesIO(b"%PDF-1.4 dummy content")
        return mock_file

    @patch("backend.main.resume_parser.parse_resume")
    def test_session_start_without_jd(self, mock_parse_resume):
        mock_parse_resume.return_value = self.mock_resume_profile
        mock_file = self._create_mock_upload_file()

        data = start_session(
            role="ai_ml_engineer",
            file=mock_file,
            company=None,
            jd_text=None,
            db=self.db
        )

        self.assertIn("session_id", data)
        self.assertIsNone(data["company"])
        self.assertFalse(data["jd_usable"])
        self.assertEqual(data["starting_difficulty_tier"], "intermediate")
        self.assertIsInstance(data["topic_plan"], list)
        self.assertTrue(len(data["topic_plan"]) >= 4)
        for item in data["topic_plan"]:
            self.assertEqual(item["bucket"], "spot_check")
            self.assertEqual(item["source"], "skills")

        # Verify DB session record
        db_sess = self.db.query(DBSession).filter(DBSession.id == data["session_id"]).first()
        self.assertIsNotNone(db_sess)
        self.assertIsNone(db_sess.company)
        self.assertIsNone(db_sess.jd_text)
        self.assertIsNone(db_sess.jd_profile)
        self.assertIsNotNone(db_sess.topic_plan)

    @patch("backend.main.jd_parser.parse_jd")
    @patch("backend.main.resume_parser.parse_resume")
    def test_session_start_with_jd_and_company(self, mock_parse_resume, mock_parse_jd):
        mock_parse_resume.return_value = self.mock_resume_profile
        mock_parse_jd.return_value = self.mock_jd_profile
        mock_file = self._create_mock_upload_file()

        jd_text = "Requirements: PyTorch, Clustering algorithms. Preferred: Evaluation metrics."
        data = start_session(
            role="ai_ml_engineer",
            file=mock_file,
            company="DeepTech",
            jd_text=jd_text,
            db=self.db
        )

        self.assertEqual(data["company"], "DeepTech")
        self.assertTrue(data["jd_usable"])
        # Seniority shifted from intermediate to intermediate-dense due to advanced JD hint
        self.assertEqual(data["starting_difficulty_tier"], "intermediate-dense")
        self.assertIn("at DeepTech", data["greeting_message"])

        # Check buckets in topic_plan
        plan = data["topic_plan"]
        buckets = {item["bucket"] for item in plan}
        self.assertIn("verify", buckets)
        self.assertIn("jd_gap", buckets)

        # Verify DB session record
        db_sess = self.db.query(DBSession).filter(DBSession.id == data["session_id"]).first()
        self.assertEqual(db_sess.company, "DeepTech")
        self.assertEqual(db_sess.jd_text, jd_text)
        self.assertEqual(db_sess.jd_profile["role_title"], "Staff ML Engineer")

    @patch("backend.main.jd_parser.parse_jd")
    @patch("backend.main.resume_parser.parse_resume")
    def test_get_session_plan_endpoint(self, mock_parse_resume, mock_parse_jd):
        mock_parse_resume.return_value = self.mock_resume_profile
        mock_parse_jd.return_value = self.mock_jd_profile
        mock_file = self._create_mock_upload_file()

        start_res = start_session(
            role="ai_ml_engineer",
            file=mock_file,
            company="DeepTech",
            jd_text="Requirements: PyTorch, Clustering.",
            db=self.db
        )
        sess_id = start_res["session_id"]

        plan_data = get_session_plan(sess_id, db=self.db)

        self.assertEqual(plan_data["session_id"], sess_id)
        self.assertEqual(plan_data["company"], "DeepTech")
        self.assertTrue(plan_data["jd_used"])
        self.assertEqual(plan_data["starting_difficulty_tier"], "intermediate-dense")
        self.assertIsInstance(plan_data["plan"], list)
        self.assertTrue(len(plan_data["plan"]) >= 4)

    @patch("backend.main.question_generator.generate_question")
    @patch("backend.main.jd_parser.parse_jd")
    @patch("backend.main.resume_parser.parse_resume")
    def test_next_question_tags_source_and_resume_ref(self, mock_parse_resume, mock_parse_jd, mock_gen_q):
        mock_parse_resume.return_value = self.mock_resume_profile
        mock_parse_jd.return_value = self.mock_jd_profile
        mock_gen_q.return_value = {
            "question": "Can you explain how deep neural networks are trained with backpropagation?",
            "concept_being_tested": "backpropagation",
            "expected_answer_points": ["gradient descent", "chain rule"],
            "topic": "deep_learning",
            "difficulty_tier": "intermediate-dense",
            "role": "ai_ml_engineer",
            "source_chunk_id": "chunk_123",
            "source_book": "Deep Learning Book",
            "source_page_start": 10,
            "source_page_end": 12,
            "source_section_title": "Backprop",
            "source_chunk_text": "Backpropagation computes gradients...",
            "copy_overlap_ratio": 0.05,
            "copy_overlap_warning_triggered": False,
            "generated_at": "2026-11-01T20:00:00Z"
        }
        mock_file = self._create_mock_upload_file()

        start_res = start_session(
            role="ai_ml_engineer",
            file=mock_file,
            company="DeepTech",
            jd_text="Requirements: PyTorch, Clustering.",
            db=self.db
        )
        sess_id = start_res["session_id"]

        q_data = get_next_question(sess_id, db=self.db)

        self.assertIn("question_id", q_data)
        self.assertEqual(q_data["topic"], "deep_learning")
        self.assertEqual(q_data["source"], "skills")
        self.assertEqual(q_data["resume_ref"], "PyTorch")

        # Verify DB question record has source and resume_ref
        db_q = self.db.query(DBQuestion).filter(DBQuestion.id == q_data["question_id"]).first()
        self.assertIsNotNone(db_q)
        self.assertEqual(db_q.source, "skills")
        self.assertEqual(db_q.resume_ref, "PyTorch")

    @patch("backend.main.question_generator.generate_question")
    @patch("backend.main.jd_parser.parse_jd")
    @patch("backend.main.resume_parser.parse_resume")
    def test_next_question_jd_gap_skips_project_context(self, mock_parse_resume, mock_parse_jd, mock_gen_q):
        mock_parse_resume.return_value = self.mock_resume_profile
        # Make clustering the only required skill so it lands in JD gap
        jd_gap_profile = {
            "usable": True,
            "role_title": "Clustering Specialist",
            "company": "DeepTech",
            "seniority_hint": "intermediate",
            "skills_by_priority": {"high": ["Clustering"], "moderate": [], "low_inferred": []},
            "topic_weights": {"clustering": 1.0},
            "inferred_only_topics": [],
            "source": "llm"
        }
        mock_parse_jd.return_value = jd_gap_profile
        mock_gen_q.return_value = {
            "question": "What is the k-means clustering algorithm?",
            "concept_being_tested": "k-means",
            "expected_answer_points": ["centroids", "distance"],
            "topic": "clustering",
            "difficulty_tier": "beginner",
            "role": "ai_ml_engineer",
            "source_chunk_id": "chunk_456",
            "source_book": "Pattern Recognition",
            "source_page_start": 20,
            "source_page_end": 22,
            "source_section_title": "Clustering",
            "source_chunk_text": "K-means partitions observations...",
            "copy_overlap_ratio": 0.05,
            "copy_overlap_warning_triggered": False,
            "generated_at": "2026-11-01T20:00:00Z"
        }
        mock_file = self._create_mock_upload_file()

        start_res = start_session(
            role="ai_ml_engineer",
            file=mock_file,
            company="DeepTech",
            jd_text="Requirements: Clustering.",
            db=self.db
        )
        sess_id = start_res["session_id"]

        q_data = get_next_question(sess_id, db=self.db)

        self.assertEqual(q_data["topic"], "clustering")
        self.assertEqual(q_data["source"], "jd_gap")
        self.assertIsNone(q_data["resume_ref"])

        # Check call kwargs to generate_question: bucket must be "jd_gap" and candidate_context must be None
        call_kwargs = mock_gen_q.call_args.kwargs
        self.assertEqual(call_kwargs.get("bucket"), "jd_gap")
        self.assertIsNone(call_kwargs.get("candidate_context"))
        self.assertEqual(call_kwargs.get("company"), "DeepTech")

    @patch("backend.main.answer_validator.summarize_session")
    @patch("backend.main.question_generator.generate_question")
    @patch("backend.main.jd_parser.parse_jd")
    @patch("backend.main.resume_parser.parse_resume")
    def test_session_summary_includes_jd_and_plan(
        self, mock_parse_resume, mock_parse_jd, mock_gen_q, mock_summarize
    ):
        mock_parse_resume.return_value = self.mock_resume_profile
        mock_parse_jd.return_value = self.mock_jd_profile
        mock_gen_q.return_value = {
            "question": "Can you explain backpropagation?",
            "concept_being_tested": "backpropagation",
            "expected_answer_points": ["gradient"],
            "topic": "deep_learning",
            "difficulty_tier": "intermediate-dense",
            "role": "ai_ml_engineer",
            "source_chunk_id": "chunk_123",
            "source_book": "Deep Learning Book",
            "source_page_start": 10,
            "source_page_end": 12,
            "source_section_title": "Backprop",
            "source_chunk_text": "Backpropagation computes gradients...",
            "copy_overlap_ratio": 0.05,
            "copy_overlap_warning_triggered": False,
            "generated_at": "2026-11-01T20:00:00Z"
        }
        mock_summarize.return_value = {
            "total_questions": 1,
            "topics_covered": ["deep_learning"],
            "quality_distribution": {"strong": 1, "developing": 0, "needs_review": 0},
            "per_topic_summary": {},
            "overall_impression": "Strong candidate."
        }
        mock_file = self._create_mock_upload_file()

        start_res = start_session(
            role="ai_ml_engineer",
            file=mock_file,
            company="DeepTech",
            jd_text="Requirements: PyTorch, Clustering.",
            db=self.db
        )
        sess_id = start_res["session_id"]

        # Generate 1 question
        get_next_question(sess_id, db=self.db)

        s_data = get_session_summary(sess_id, db=self.db)

        self.assertEqual(s_data["company"], "DeepTech")
        self.assertTrue(s_data["jd_used"])
        self.assertIsInstance(s_data["topic_plan"], list)
        self.assertTrue(len(s_data["qa_pairs"]) > 0)
        self.assertEqual(s_data["qa_pairs"][0]["source"], "skills")
        self.assertEqual(s_data["qa_pairs"][0]["resume_ref"], "PyTorch")


if __name__ == "__main__":
    unittest.main()
