import os
import sys
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Set, Tuple

logger = logging.getLogger("core.topic_planner")

# Ensure core and parent directory are on search path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from resume_parser import FIXED_TAXONOMY, SKILL_TO_TOPICS_MAP
from ingest import assign_topics

# Standard difficulty tiers
DIFFICULTY_ORDER = ["beginner", "intermediate", "intermediate-dense"]

DEFAULT_PADDING_TOPICS = [
    "classification",
    "overfitting_regularization",
    "deep_learning",
    "neural_networks",
    "optimization",
    "model_selection"
]


@dataclass
class PlanItem:
    """
    Represents a planned interview question topic within the interview session plan.
    """
    topic: str
    bucket: str  # "verify" | "jd_gap" | "spot_check"
    priority_score: float
    reason: str
    source: str  # "skills" | "jd_gap" | "project"
    resume_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def normalize_tier_name(tier_or_seniority: str) -> str:
    """
    Normalizes seniority/tier names to one of:
    'beginner', 'intermediate', 'intermediate-dense'
    """
    t = str(tier_or_seniority or "").strip().lower()
    if t in ["beginner", "entry", "junior"]:
        return "beginner"
    elif t in ["advanced", "intermediate-dense", "senior", "lead", "staff"]:
        return "intermediate-dense"
    return "intermediate"


def calculate_starting_difficulty(
    resume_seniority: str,
    jd_seniority_hint: Optional[str] = None
) -> str:
    """
    Calculates the starting difficulty tier for the interview session:
    - Base is inferred from the candidate's resume ('beginner', 'intermediate', 'intermediate-dense').
    - If a usable JD provides a valid seniority_hint ('beginner', 'intermediate', 'advanced'),
      it may shift the starting tier by AT MOST one tier.
    - If jd_seniority_hint is None or 'unknown', resume-inferred tier is kept unchanged.
    """
    base_tier = normalize_tier_name(resume_seniority)
    try:
        base_idx = DIFFICULTY_ORDER.index(base_tier)
    except ValueError:
        base_idx = 1

    if not jd_seniority_hint or jd_seniority_hint.lower() == "unknown":
        return DIFFICULTY_ORDER[base_idx]

    target_tier = normalize_tier_name(jd_seniority_hint)
    try:
        target_idx = DIFFICULTY_ORDER.index(target_tier)
    except ValueError:
        return DIFFICULTY_ORDER[base_idx]

    # Shift by at most 1 tier in the direction of the JD target
    if target_idx > base_idx:
        new_idx = min(base_idx + 1, len(DIFFICULTY_ORDER) - 1)
    elif target_idx < base_idx:
        new_idx = max(base_idx - 1, 0)
    else:
        new_idx = base_idx

    return DIFFICULTY_ORDER[new_idx]


def find_matching_resume_skill(topic: str, raw_skills: List[str]) -> Optional[str]:
    """
    Finds the first skill in raw_skills that maps to the given topic.
    """
    for skill in raw_skills:
        s_lower = skill.strip().lower()
        # Direct assign_topics check
        if topic in assign_topics(skill):
            return skill
        # Mapping check
        if topic in SKILL_TO_TOPICS_MAP.get(s_lower, []):
            return skill
    return None


def plan_topics(
    resume_profile: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]] = None,
    n_questions: Optional[int] = None
) -> List[PlanItem]:
    """
    Constructs an optimized, prioritized question plan by comparing the candidate's
    resume against the target Job Description (if available).
    
    Buckets:
    - 'verify': Present on resume AND required by JD (high priority).
    - 'jd_gap': Required by JD but ABSENT from resume (max 2 per session, lower tier probing).
    - 'spot_check': Present on resume but not in JD (lower priority).
    
    Constraints:
    - Clamped to 6-10 total planned questions when n_questions is specified.
    - Topics are deduplicated.
    - Inferred-only topics from JD (weight <= 0.3) MUST NOT trigger jd_gap.
    - If jd_profile is None or not usable, reproduces old resume-only planning exactly.
    """
    raw_skills = resume_profile.get("raw_skills", [])
    resume_topics = [t for t in resume_profile.get("mapped_topics", []) if t in FIXED_TAXONOMY]
    project_targets = [p for p in resume_profile.get("projects", []) if p.get("related_topics")]
    
    # Check if JD is usable
    is_jd_usable = bool(jd_profile and jd_profile.get("usable"))

    # Determine question target count
    if n_questions is not None:
        target_count = max(6, min(10, n_questions))
    else:
        if not is_jd_usable:
            # Baseline resume-only planning:
            # skill_topics takes min(6, len(resume_topics))
            # dynamic padding until len(skill_topics) + len(project_targets) >= 6
            candidate_topics = list(resume_topics[:6])
            for d_topic in DEFAULT_PADDING_TOPICS:
                if len(candidate_topics) + min(3, len(project_targets)) >= 6:
                    break
                if d_topic not in candidate_topics:
                    candidate_topics.append(d_topic)
            target_count = len(candidate_topics)
        else:
            # With JD, target 6 skill topics so total questions (skills + projects) is 6-10
            target_count = max(6, min(10, len(resume_topics) if len(resume_topics) >= 6 else 6))

    plan_items: List[PlanItem] = []
    seen_topics: Set[str] = set()

    # --- MODE A: NO USABLE JD (Reproduce current resume-only behavior exactly) ---
    if not is_jd_usable:
        candidate_topics = list(resume_topics[:6])
        for d_topic in DEFAULT_PADDING_TOPICS:
            if len(candidate_topics) >= target_count:
                break
            if d_topic not in candidate_topics:
                candidate_topics.append(d_topic)

        # In case target_count > len(candidate_topics), fill with FIXED_TAXONOMY
        for t in FIXED_TAXONOMY:
            if len(candidate_topics) >= target_count:
                break
            if t not in candidate_topics:
                candidate_topics.append(t)

        for idx, topic in enumerate(candidate_topics[:target_count]):
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            ref_skill = find_matching_resume_skill(topic, raw_skills)
            reason = f"Listed on your resume" + (f" ({ref_skill})" if ref_skill else "")
            plan_items.append(
                PlanItem(
                    topic=topic,
                    bucket="spot_check",
                    priority_score=round(1.0 - (idx * 0.1), 2),
                    reason=reason,
                    source="skills",
                    resume_ref=ref_skill
                )
            )
        return plan_items

    # --- MODE B: JD-ALIGNED PLANNING ---
    jd_topic_weights: Dict[str, float] = jd_profile.get("topic_weights", {})
    inferred_only_topics: Set[str] = set(jd_profile.get("inferred_only_topics", []))
    
    # 1. Bucket: VERIFY (On resume AND required by JD with weight >= 0.6)
    verify_candidates: List[Tuple[str, float, Optional[str]]] = []
    for topic in resume_topics:
        if topic in jd_topic_weights and jd_topic_weights[topic] >= 0.6:
            ref_skill = find_matching_resume_skill(topic, raw_skills)
            score = 2.0 + jd_topic_weights[topic]
            verify_candidates.append((topic, score, ref_skill))
            
    # Sort verify candidates by weight descending
    verify_candidates.sort(key=lambda x: x[1], reverse=True)

    # 2. Bucket: JD_GAP (Required by JD with weight >= 0.6, but ABSENT from resume; CAPPED AT 2)
    # Note: Inferred-only topics (weight <= 0.3) MUST NOT trigger jd_gap.
    gap_candidates: List[Tuple[str, float]] = []
    for topic, weight in jd_topic_weights.items():
        if topic not in resume_topics and topic in FIXED_TAXONOMY:
            if weight >= 0.6 and topic not in inferred_only_topics:
                gap_candidates.append((topic, weight))

    # Sort gaps by weight descending
    gap_candidates.sort(key=lambda x: x[1], reverse=True)

    # Reserve slots for up to 2 gap questions so they are not crowded out by verify items
    max_gaps = min(2, len(gap_candidates))
    verify_slot_limit = max(0, target_count - max_gaps)

    for topic, score, ref_skill in verify_candidates:
        if len(plan_items) >= verify_slot_limit:
            break
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        ref_text = f" ({ref_skill})" if ref_skill else ""
        plan_items.append(
            PlanItem(
                topic=topic,
                bucket="verify",
                priority_score=score,
                reason=f"Required by JD and verified on your resume{ref_text}",
                source="skills",
                resume_ref=ref_skill
            )
        )

    # Add JD gaps (capped at 2)
    jd_gap_count = 0
    for topic, weight in gap_candidates:
        if jd_gap_count >= 2 or len(plan_items) >= target_count:
            break
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        jd_gap_count += 1
        plan_items.append(
            PlanItem(
                topic=topic,
                bucket="jd_gap",
                priority_score=round(1.5 + weight, 2),
                reason="Required by JD but absent from resume; probing technical foundation",
                source="jd_gap",
                resume_ref=None
            )
        )

    # If space remains, add remaining verify candidates
    for topic, score, ref_skill in verify_candidates:
        if len(plan_items) >= target_count:
            break
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        ref_text = f" ({ref_skill})" if ref_skill else ""
        plan_items.append(
            PlanItem(
                topic=topic,
                bucket="verify",
                priority_score=score,
                reason=f"Required by JD and verified on your resume{ref_text}",
                source="skills",
                resume_ref=ref_skill
            )
        )

    # 3. Bucket: SPOT_CHECK (On resume but NOT in JD requirements)
    spot_check_candidates: List[Tuple[str, float, Optional[str]]] = []
    for idx, topic in enumerate(resume_topics):
        if topic not in seen_topics:
            ref_skill = find_matching_resume_skill(topic, raw_skills)
            score = round(1.0 - (idx * 0.05), 2)
            spot_check_candidates.append((topic, score, ref_skill))

    for topic, score, ref_skill in spot_check_candidates:
        if len(plan_items) >= target_count:
            break
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        ref_text = f" ({ref_skill})" if ref_skill else ""
        plan_items.append(
            PlanItem(
                topic=topic,
                bucket="spot_check",
                priority_score=score,
                reason=f"Listed on your resume{ref_text}; spot-checking technical depth",
                source="skills",
                resume_ref=ref_skill
            )
        )

    # 4. Dynamic Padding: If still below target_count, pad with general ML topics
    for d_topic in DEFAULT_PADDING_TOPICS:
        if len(plan_items) >= target_count:
            break
        if d_topic not in seen_topics:
            seen_topics.add(d_topic)
            plan_items.append(
                PlanItem(
                    topic=d_topic,
                    bucket="spot_check",
                    priority_score=0.5,
                    reason="Fundamental machine learning concept",
                    source="skills",
                    resume_ref=None
                )
            )

    # 5. Final fallback padding with FIXED_TAXONOMY if needed
    for f_topic in FIXED_TAXONOMY:
        if len(plan_items) >= target_count:
            break
        if f_topic not in seen_topics:
            seen_topics.add(f_topic)
            plan_items.append(
                PlanItem(
                    topic=f_topic,
                    bucket="spot_check",
                    priority_score=0.4,
                    reason="Machine learning taxonomy foundation",
                    source="skills",
                    resume_ref=None
                )
            )

    return plan_items
