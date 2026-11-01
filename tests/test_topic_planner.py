import os
import sys
import unittest

# Ensure tests and core directories are on path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_core_dir = os.path.join(_parent_dir, "core")
for d in [_current_dir, _parent_dir, _core_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from core.topic_planner import (
    PlanItem,
    plan_topics,
    calculate_starting_difficulty,
    normalize_tier_name,
    find_matching_resume_skill,
)
from core.resume_parser import FIXED_TAXONOMY


class TestTopicPlanner(unittest.TestCase):
    def setUp(self):
        self.sample_resume_profile = {
            "candidate_name": "Alice ML",
            "starting_difficulty_tier": "intermediate",
            "raw_skills": ["PyTorch", "scikit-learn", "Python", "Docker"],
            "mapped_topics": ["deep_learning", "neural_networks", "classification", "model_selection"],
            "projects": [
                {
                    "title": "Vision Classifier",
                    "description": "Image classification using PyTorch ResNet.",
                    "technologies_used": ["PyTorch", "Python"],
                    "related_topics": ["deep_learning", "classification"]
                }
            ]
        }

        self.sample_jd_profile = {
            "usable": True,
            "role_title": "Senior AI/ML Engineer",
            "company": "DeepTech AI",
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

    def test_plan_item_properties(self):
        item = PlanItem(
            topic="deep_learning",
            bucket="verify",
            priority_score=3.0,
            reason="Required by JD and verified on resume",
            source="skills",
            resume_ref="PyTorch"
        )
        self.assertEqual(item.topic, "deep_learning")
        self.assertEqual(item.bucket, "verify")
        self.assertEqual(item.priority_score, 3.0)
        self.assertEqual(item["topic"], "deep_learning")
        d = item.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["source"], "skills")
        self.assertEqual(d["resume_ref"], "PyTorch")

    def test_no_jd_parity(self):
        # 1. With jd_profile=None
        plan_none = plan_topics(self.sample_resume_profile, jd_profile=None)
        self.assertTrue(len(plan_none) >= 4)
        for item in plan_none:
            self.assertEqual(item.bucket, "spot_check")
            self.assertEqual(item.source, "skills")
            self.assertIn(item.topic, FIXED_TAXONOMY)

        # 2. With jd_profile having usable=False
        unusable_jd = {"usable": False, "role_title": "N/A", "topic_weights": {}}
        plan_unusable = plan_topics(self.sample_resume_profile, jd_profile=unusable_jd)
        self.assertEqual([p.topic for p in plan_none], [p.topic for p in plan_unusable])

    def test_all_three_buckets(self):
        plan = plan_topics(self.sample_resume_profile, self.sample_jd_profile, n_questions=8)
        self.assertEqual(len(plan), 8)

        buckets = {item.bucket for item in plan}
        self.assertIn("verify", buckets)
        self.assertIn("jd_gap", buckets)
        self.assertIn("spot_check", buckets)

        # Verify: deep_learning is on resume AND required by JD (weight 1.0)
        verify_items = [item for item in plan if item.bucket == "verify"]
        verify_topics = [item.topic for item in verify_items]
        self.assertIn("deep_learning", verify_topics)
        for v in verify_items:
            self.assertEqual(v.source, "skills")
            self.assertIn("Required by JD and verified", v.reason)

        # JD Gap: clustering is required by JD (weight 1.0) but absent from resume
        gap_items = [item for item in plan if item.bucket == "jd_gap"]
        gap_topics = [item.topic for item in gap_items]
        self.assertIn("clustering", gap_topics)
        for g in gap_items:
            self.assertEqual(g.source, "jd_gap")
            self.assertIsNone(g.resume_ref)
            self.assertIn("Required by JD but absent from resume", g.reason)

        # Spot check: classification or model_selection is on resume but not required by JD
        spot_items = [item for item in plan if item.bucket == "spot_check"]
        self.assertTrue(len(spot_items) > 0)
        for s in spot_items:
            self.assertEqual(s.source, "skills")

    def test_jd_gap_cap_at_two(self):
        # JD requiring 4 topics not on the candidate's resume
        jd_with_many_gaps = {
            "usable": True,
            "role_title": "ML Engineer",
            "topic_weights": {
                "clustering": 1.0,
                "dimensionality_reduction": 1.0,
                "ensemble_methods": 1.0,
                "probability_statistics": 1.0,
            },
            "inferred_only_topics": [],
            "source": "llm"
        }
        plan = plan_topics(self.sample_resume_profile, jd_with_many_gaps, n_questions=8)
        gap_items = [item for item in plan if item.bucket == "jd_gap"]
        self.assertLessEqual(len(gap_items), 2)

    def test_inferred_only_topics_never_trigger_jd_gap(self):
        # JD with a low_inferred topic (weight 0.3) absent from resume
        jd_inferred_gap = {
            "usable": True,
            "role_title": "ML Engineer",
            "topic_weights": {
                "dimensionality_reduction": 0.3,
                "clustering": 0.3
            },
            "inferred_only_topics": ["dimensionality_reduction", "clustering"],
            "source": "llm"
        }
        plan = plan_topics(self.sample_resume_profile, jd_inferred_gap, n_questions=6)
        gap_items = [item for item in plan if item.bucket == "jd_gap"]
        # Inferred-only topics must NOT trigger jd_gap
        self.assertEqual(len(gap_items), 0)

    def test_clamp_behavior(self):
        # Clamp up to 6 when n_questions=4
        plan_low = plan_topics(self.sample_resume_profile, self.sample_jd_profile, n_questions=4)
        self.assertEqual(len(plan_low), 6)

        # Clamp down to 10 when n_questions=15
        plan_high = plan_topics(self.sample_resume_profile, self.sample_jd_profile, n_questions=15)
        self.assertEqual(len(plan_high), 10)

        # Exact match when within [6, 10]
        plan_exact = plan_topics(self.sample_resume_profile, self.sample_jd_profile, n_questions=7)
        self.assertEqual(len(plan_exact), 7)

    def test_deduplication(self):
        plan = plan_topics(self.sample_resume_profile, self.sample_jd_profile, n_questions=10)
        topics = [item.topic for item in plan]
        self.assertEqual(len(topics), len(set(topics)), "Plan items must contain no duplicate topics")

    def test_calculate_starting_difficulty(self):
        # 1. Base beginner, JD beginner -> beginner
        self.assertEqual(calculate_starting_difficulty("beginner", "beginner"), "beginner")

        # 2. Base beginner, JD advanced -> intermediate (max 1-tier shift, not straight to intermediate-dense)
        self.assertEqual(calculate_starting_difficulty("beginner", "advanced"), "intermediate")

        # 3. Base beginner, JD None/unknown -> beginner
        self.assertEqual(calculate_starting_difficulty("beginner", None), "beginner")
        self.assertEqual(calculate_starting_difficulty("beginner", "unknown"), "beginner")

        # 4. Base intermediate-dense, JD beginner -> intermediate (max 1-tier downward shift)
        self.assertEqual(calculate_starting_difficulty("intermediate-dense", "beginner"), "intermediate")

        # 5. Base intermediate, JD advanced -> intermediate-dense
        self.assertEqual(calculate_starting_difficulty("intermediate", "advanced"), "intermediate-dense")

        # 6. Base intermediate, JD beginner -> beginner
        self.assertEqual(calculate_starting_difficulty("intermediate", "beginner"), "beginner")

        # 7. Base intermediate, JD intermediate -> intermediate
        self.assertEqual(calculate_starting_difficulty("intermediate", "intermediate"), "intermediate")

    def test_empty_resume_profile(self):
        empty_resume = {
            "candidate_name": "Unknown",
            "starting_difficulty_tier": "intermediate",
            "raw_skills": [],
            "mapped_topics": [],
            "projects": []
        }
        plan = plan_topics(empty_resume, self.sample_jd_profile, n_questions=6)
        self.assertEqual(len(plan), 6)
        for item in plan:
            self.assertIn(item.topic, FIXED_TAXONOMY)


if __name__ == "__main__":
    unittest.main()

