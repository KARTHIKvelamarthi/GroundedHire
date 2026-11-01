import os
import sys
import json
import shutil
import uuid
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Configure clean, minimal terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("backend.main")

# Suppress noisy third-party library logging
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Ensure parent directory and core directory are in python search path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
core_dir = os.path.join(parent_dir, "core")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# Import workspace modules
import resume_parser
import question_generator
import answer_validator
import llm_client
import jd_parser
import topic_planner

# Import local database and models
from .database import engine, Base, get_db, run_migrations
from .models import Session as DBSession, Question as DBQuestion, Answer as DBAnswer

# Create FastAPI instance
app = FastAPI(title="GroundedHire Backend API")

@app.on_event("startup")
async def startup_event():
    run_migrations(engine)
    # Re-apply logger levels at startup to override any uvicorn defaults
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow React dev server or any origin locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables and execute safe startup migration
Base.metadata.create_all(bind=engine)
run_migrations(engine)

# Difficulty levels ordering
DIFFICULTY_ORDER = ["beginner", "intermediate", "intermediate-dense"]

def adjust_difficulty(current_tier: str, quality_signal: str) -> str:
    """
    Shifts the difficulty level up or down based on candidate performance.
    """
    try:
        current_idx = DIFFICULTY_ORDER.index(current_tier.lower())
    except ValueError:
        current_idx = 1  # Default to intermediate
        
    if quality_signal == "strong":
        new_idx = min(current_idx + 1, len(DIFFICULTY_ORDER) - 1)
        return DIFFICULTY_ORDER[new_idx]
    elif quality_signal == "needs_review":
        new_idx = max(current_idx - 1, 0)
        return DIFFICULTY_ORDER[new_idx]
        
    # 'developing' keeps current difficulty
    return current_tier

# Request Model for Answer Submission
class AnswerSubmit(BaseModel):
    question_id: str
    answer_text: str

@app.post("/session/start")
def start_session(
    role: str = Form(...),
    file: UploadFile = File(...),
    company: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Starts a new interview session:
    1. Saves and parses the candidate's resume
    2. Parses optional Job Description (JD) and company name
    3. Constructs an aligned questions plan (skills/verify/jd_gap/spot_check + transitions + projects)
    4. Creates a session entry in the database
    5. Returns initialization parameters and a greeting message
    """
    if role not in ["ai_ml_engineer", "data_science_applied_ml"]:
        raise HTTPException(status_code=400, detail="Invalid role specified.")

    logger.info(f"Parsing resume: {file.filename}")

    # Save uploaded file temporarily
    temp_dir = os.path.join(parent_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Parse resume using the existing parser module
        resume_profile = resume_parser.parse_resume(temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # 1. Parse optional Job Description (JD)
    sanitized_company = company.strip() if (company and company.strip()) else None
    jd_profile = None
    if jd_text and jd_text.strip():
        try:
            jd_profile = jd_parser.parse_jd(jd_text, sanitized_company)
        except Exception as e:
            logger.warning(f"Failed to parse JD, proceeding without JD: {e}")
            jd_profile = {"usable": False, "error": str(e)}

    # 2. Extract candidate and difficulty data
    candidate_name = resume_profile.get("candidate_name") or "Candidate"
    resume_seniority = resume_profile.get("starting_difficulty_tier") or "intermediate"
    jd_seniority_hint = jd_profile.get("seniority_hint") if (jd_profile and jd_profile.get("usable")) else None
    starting_difficulty = topic_planner.calculate_starting_difficulty(resume_seniority, jd_seniority_hint)

    # 3. Plan topics using topic_planner (verify, jd_gap, spot_check)
    plan_items = topic_planner.plan_topics(resume_profile, jd_profile)
    topic_plan = [item.to_dict() if hasattr(item, "to_dict") else item for item in plan_items]
    skill_topics = [item["topic"] for item in topic_plan]

    # Plan projects phase
    project_targets = [p for p in resume_profile.get("projects", []) if p.get("related_topics")]
    if len(project_targets) > 3:
        project_targets = project_targets[:3]

    # Compute total planned questions count (clamp between 6 and 10)
    total_planned = len(skill_topics) + (1 if project_targets else 0) + len(project_targets)
    total_planned_questions = max(6, min(10, total_planned))

    # Build plan object for backward compatibility
    plan = {
        "skill_topics": skill_topics,
        "project_targets": project_targets
    }

    # 4. Create Session in Database
    new_session = DBSession(
        candidate_name=candidate_name,
        role=role,
        company=sanitized_company,
        jd_text=jd_text,
        jd_profile=jd_profile,
        topic_plan=topic_plan,
        resume_profile_json=json.dumps(resume_profile),
        starting_difficulty_tier=starting_difficulty,
        current_difficulty_tier=starting_difficulty,
        total_planned_questions=total_planned_questions,
        current_phase="skills",
        questions_asked_count=0,
        planned_topics_json=json.dumps(plan)
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    # Generate Greeting Message
    company_phrase = f" at {sanitized_company}" if sanitized_company else ""
    greeting_message = (
        f"Hi {candidate_name}, you've applied for the {role} role{company_phrase}. "
        f"Based on your profile, I'd estimate you're currently at an {starting_difficulty} level in this field. "
        f"Let's get started — I'll ask a few questions about your core skills first, then we'll dig into some of your projects."
    )

    return {
        "session_id": new_session.id,
        "candidate_name": candidate_name,
        "role": role,
        "company": sanitized_company,
        "starting_difficulty_tier": starting_difficulty,
        "total_planned_questions": total_planned_questions,
        "greeting_message": greeting_message,
        "jd_usable": bool(jd_profile and jd_profile.get("usable")),
        "topic_plan": topic_plan
    }

@app.get("/session/{session_id}/next-question")
def get_next_question(session_id: str, db: Session = Depends(get_db)):
    """
    Returns the next question based on current phase and session progress.
    Increments questions_asked_count and saves the generated question.
    """
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    logger.info(f"Retrieving next question for session_id={session_id}")

    if session.questions_asked_count >= session.total_planned_questions:
        return {
            "question_id": None,
            "question_text": None,
            "topic": None,
            "difficulty_tier": None,
            "source_book": None,
            "source_section_title": None,
            "question_number": session.questions_asked_count,
            "total_planned_questions": session.total_planned_questions,
            "interview_complete": True
        }
        
    # Extract plan and profile
    plan = json.loads(session.planned_topics_json)
    skill_topics = plan.get("skill_topics", [])
    project_targets = plan.get("project_targets", [])
    resume_profile = json.loads(session.resume_profile_json)
    
    # Determine phase
    phase = "skills"
    topic = ""
    concept_being_tested = ""
    question_text = ""
    expected_answer_points = []
    source_chunk_id = None
    source_book = None
    source_section_title = None
    source_chunk_text = None
    question_type = "skill"
    source = "skills"
    resume_ref = None
    
    # Collect used chunks
    used_chunk_ids = {q.source_chunk_id for q in session.questions if q.source_chunk_id}
    
    # 1. Skills Phase
    if session.questions_asked_count < len(skill_topics):
        phase = "skills"
        question_type = "skill"

        # Check topic_plan for rich plan item (bucket, source, resume_ref)
        plan_item = None
        if session.topic_plan and session.questions_asked_count < len(session.topic_plan):
            plan_item = session.topic_plan[session.questions_asked_count]

        if plan_item:
            topic = plan_item.get("topic")
            bucket = plan_item.get("bucket", "spot_check")
            source = plan_item.get("source", "skills")
            resume_ref = plan_item.get("resume_ref")
        else:
            topic = skill_topics[session.questions_asked_count]
            bucket = "spot_check"
            source = "skills"
            resume_ref = None

        candidate_context = {
            "candidate_name": session.candidate_name,
            "raw_skills": resume_profile.get("raw_skills", []),
            "projects": resume_profile.get("projects", []),
            "mapped_topics": resume_profile.get("mapped_topics", [])
        }
        
        try:
            q_res = question_generator.generate_question(
                role=session.role,
                topic=topic,
                difficulty_tier=session.current_difficulty_tier,
                candidate_context=candidate_context if bucket != "jd_gap" else None,
                used_chunk_ids=used_chunk_ids,
                bucket=bucket,
                company=session.company
            )
            question_text = q_res["question"]
            concept_being_tested = q_res["concept_being_tested"]
            expected_answer_points = q_res["expected_answer_points"]
            source_chunk_id = q_res["source_chunk_id"]
            source_book = q_res["source_book"]
            source_section_title = q_res["source_section_title"]
            source_chunk_text = q_res["source_chunk_text"]
        except Exception as e:
            # Fallback question if parser or LLM fails
            question_text = f"Could you explain the core concepts, common algorithms, and trade-offs related to {topic}?"
            concept_being_tested = topic
            expected_answer_points = [f"Explain {topic}", f"Discuss key algorithms", "Detail trade-offs"]
            source_chunk_text = f"Fallback content for topic: {topic}"
            
    # 2. Transition Phase
    elif session.questions_asked_count == len(skill_topics) and project_targets:
        phase = "project_transition"
        question_type = "transition"
        first_proj = project_targets[0]
        proj_title = first_proj.get("title") or "your project"
        proj_desc = first_proj.get("description") or ""
        source = "project"
        resume_ref = proj_title
        
        prompt = f"""You are an advanced technical interviewer conducting a candidate interview.
We are transitioning from testing core technical skills to discussing the candidate's project: '{proj_title}'.
Write a short (1 sentence) natural transition line introducing that we will now discuss the project '{proj_title}', then ask an open-ended question about their role, architecture, or approach in it.
Also generate 3 expected key points that the candidate should cover in their answer.

Return ONLY a JSON object with keys "question" and "expected_points":
{{
  "question": "A 1-sentence transition + open-ended question",
  "expected_points": ["point 1", "point 2", "point 3"]
}}

Candidate project description:
{proj_desc}
"""
        try:
            raw_res = llm_client.call_llm(prompt, temperature=0.7, json_mode=True)
            data = json.loads(raw_res)
            question_text = data["question"]
            expected_answer_points = data.get("expected_points", ["Describe project architecture", "Explain candidate's individual contribution", "Detail technologies used"])
        except Exception:
            question_text = f"Now that we've covered your technical skills, I'd like to transition to your project '{proj_title}'. Could you describe your role, the main architecture, and the biggest technical challenge you faced?"
            expected_answer_points = ["Describe project architecture", "Explain candidate's individual contribution", "Detail technologies used"]
            
        concept_being_tested = f"Project Transition: {proj_title}"
        topic = proj_title
        
    # 3. Projects Phase
    else:
        phase = "projects"
        question_type = "project"
        
        # Calculate index relative to projects targets list
        project_idx = session.questions_asked_count - len(skill_topics) - (1 if project_targets else 0)
        
        if 0 <= project_idx < len(project_targets):
            target_project = project_targets[project_idx]
            proj_title = target_project.get("title") or "your project"
            topic = target_project.get("related_topics", ["supervised_learning"])[0]
            source = "project"
            resume_ref = proj_title
            
            project_candidate_context = {
                "candidate_name": session.candidate_name,
                "raw_skills": resume_profile.get("raw_skills", []),
                "projects": [target_project],  # Restrict to this project only to force pairing
                "mapped_topics": resume_profile.get("mapped_topics", [])
            }
            
            try:
                q_res = question_generator.generate_question(
                    role=session.role,
                    topic=topic,
                    difficulty_tier=session.current_difficulty_tier,
                    candidate_context=project_candidate_context,
                    used_chunk_ids=used_chunk_ids,
                    company=session.company
                )
                question_text = q_res["question"]
                concept_being_tested = q_res["concept_being_tested"]
                expected_answer_points = q_res["expected_answer_points"]
                source_chunk_id = q_res["source_chunk_id"]
                source_book = q_res["source_book"]
                source_section_title = q_res["source_section_title"]
                source_chunk_text = q_res["source_chunk_text"]
            except Exception as e:
                question_text = f"Regarding your project '{proj_title}', could you elaborate on how you handled key model selection or optimization decisions?"
                concept_being_tested = f"Project Deep-dive: {proj_title}"
                expected_answer_points = ["Explain decisions", "Identify trade-offs", "Detail outcomes"]
                source_chunk_text = f"Fallback content for project: {proj_title}"
        else:
            # Fallback if questions count exceeds the expected plan length
            topic = "model_selection"
            source = "skills"
            resume_ref = None
            try:
                q_res = question_generator.generate_question(
                    role=session.role,
                    topic=topic,
                    difficulty_tier=session.current_difficulty_tier,
                    used_chunk_ids=used_chunk_ids,
                    company=session.company
                )
                question_text = q_res["question"]
                concept_being_tested = q_res["concept_being_tested"]
                expected_answer_points = q_res["expected_answer_points"]
                source_chunk_id = q_res["source_chunk_id"]
                source_book = q_res["source_book"]
                source_section_title = q_res["source_section_title"]
                source_chunk_text = q_res["source_chunk_text"]
            except Exception:
                question_text = "Could you explain how cross-validation helps prevent model overfitting?"
                concept_being_tested = "cross-validation"
                expected_answer_points = ["Explain validation", "Define overfitting"]
                source_chunk_text = "Fallback text for cross-validation"

    # Update session status
    session.current_phase = phase
    session.questions_asked_count += 1
    
    # Save question to DB
    new_question = DBQuestion(
        session_id=session.id,
        question_text=question_text,
        concept_being_tested=concept_being_tested,
        expected_answer_points_json=json.dumps(expected_answer_points),
        topic=topic,
        difficulty_tier=session.current_difficulty_tier,
        source_chunk_id=source_chunk_id,
        source_book=source_book,
        source_section_title=source_section_title,
        source_chunk_text=source_chunk_text,
        question_type=question_type,
        sequence_number=session.questions_asked_count,
        source=source,
        resume_ref=resume_ref
    )
    db.add(new_question)
    db.commit()
    db.refresh(new_question)
    
    return {
        "question_id": new_question.id,
        "question_text": question_text,
        "topic": topic,
        "difficulty_tier": new_question.difficulty_tier,
        "source_book": source_book,
        "source_section_title": source_section_title,
        "question_number": session.questions_asked_count,
        "total_planned_questions": session.total_planned_questions,
        "source": new_question.source,
        "resume_ref": new_question.resume_ref,
        "interview_complete": False
    }

@app.post("/session/{session_id}/answer")
def submit_answer(
    session_id: str,
    payload: AnswerSubmit,
    db: Session = Depends(get_db)
):
    """
    Submits a candidate's answer for validation.
    Updates the session current difficulty tier adaptively and returns feedback details.
    """
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    logger.info(f"Validating answer for question_id={payload.question_id}")

    question = db.query(DBQuestion).filter(DBQuestion.id == payload.question_id, DBQuestion.session_id == session_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
        
    # Check if this question is already answered
    existing_answer = db.query(DBAnswer).filter(DBAnswer.question_id == question.id).first()
    if existing_answer:
        raise HTTPException(status_code=400, detail="This question has already been answered.")
        
    # De-serialize expected points
    expected_points = json.loads(question.expected_answer_points_json) if question.expected_answer_points_json else []
    source_text = question.source_chunk_text or "No textbook passage (project transition question)."
    
    # Extract project description if checking a project or transition question
    project_description = None
    if question.question_type in ["transition", "project"]:
        try:
            profile = json.loads(session.resume_profile_json)
            plan = json.loads(session.planned_topics_json)
            skill_topics = plan.get("skill_topics", [])
            project_targets = plan.get("project_targets", [])
            
            if question.question_type == "transition" and project_targets:
                project_description = project_targets[0].get("description")
            elif question.question_type == "project" and project_targets:
                # project_idx is the sequence number minus the number of skill questions, minus the 1 transition question, minus 1 for 0-indexing
                project_idx = question.sequence_number - len(skill_topics) - 2
                if 0 <= project_idx < len(project_targets):
                    project_description = project_targets[project_idx].get("description")
        except Exception as e:
            print(f"[WARNING] Failed to extract project description for discrepancy checking: {e}")

    try:
        # Validate candidate answer against expected points, source passage, and resume description
        eval_res = answer_validator.validate_answer(
            question=question.question_text,
            candidate_answer=payload.answer_text,
            expected_answer_points=expected_points,
            source_chunk_text=source_text,
            project_description=project_description
        )
    except Exception as e:
        # Fallback eval response
        eval_res = {
            "quality_signal": "developing",
            "covered_points": [],
            "missed_points": expected_points,
            "brief_feedback": f"Coaching evaluation failed: {str(e)}"
        }
        
    quality_signal = eval_res.get("quality_signal", "developing")
    brief_feedback = eval_res.get("brief_feedback", "Response received.")
    covered_points = eval_res.get("covered_points", [])
    missed_points = eval_res.get("missed_points", [])
    
    # Shift session difficulty dynamically
    old_tier = session.current_difficulty_tier
    new_tier = adjust_difficulty(old_tier, quality_signal)
    session.current_difficulty_tier = new_tier
    
    # Store answer and evaluation results
    new_answer = DBAnswer(
        question_id=question.id,
        answer_text=payload.answer_text,
        quality_signal=quality_signal,
        covered_points_json=json.dumps(covered_points),
        missed_points_json=json.dumps(missed_points),
        brief_feedback=brief_feedback
    )
    db.add(new_answer)
    db.commit()
    
    return {
        "quality_signal": quality_signal,
        "brief_feedback": brief_feedback,
        "current_difficulty_tier": new_tier
    }

@app.get("/session/{session_id}/plan")
def get_session_plan(session_id: str, db: Session = Depends(get_db)):
    """
    Returns the planned interview topics, their buckets, reasons, priority scores,
    and whether a Job Description was utilized.
    """
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    topic_plan = session.topic_plan or []
    if not topic_plan and session.planned_topics_json:
        plan = json.loads(session.planned_topics_json)
        topic_plan = [
            {
                "topic": t,
                "bucket": "spot_check",
                "priority_score": 1.0,
                "reason": "Listed on your resume",
                "source": "skills",
                "resume_ref": None
            }
            for t in plan.get("skill_topics", [])
        ]

    jd_used = bool(session.jd_profile and session.jd_profile.get("usable"))
    return {
        "session_id": session.id,
        "company": session.company,
        "jd_used": jd_used,
        "starting_difficulty_tier": session.starting_difficulty_tier,
        "total_planned_questions": session.total_planned_questions,
        "plan": topic_plan
    }

@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str, db: Session = Depends(get_db)):
    """
    Summarizes all answers and returns full performance statistics.
    """
    session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    logger.info(f"Generating session summary for session_id={session_id}")

    validated_answers = []
    qa_pairs = []
    
    for q in session.questions:
        ans = q.answers[0] if q.answers else None
        
        covered_points = []
        missed_points = []
        if ans:
            covered_points = json.loads(ans.covered_points_json) if ans.covered_points_json else []
            missed_points = json.loads(ans.missed_points_json) if ans.missed_points_json else []
            
        validated_answers.append({
            "topic": q.topic,
            "quality_signal": ans.quality_signal if ans else "needs_review",
            "covered_points": covered_points,
            "missed_points": missed_points,
            "brief_feedback": ans.brief_feedback if ans else "No answer provided."
        })
        
        qa_pairs.append({
            "question_id": q.id,
            "question_text": q.question_text,
            "topic": q.topic,
            "difficulty_tier": q.difficulty_tier,
            "question_type": q.question_type,
            "sequence_number": q.sequence_number,
            "source": q.source,
            "resume_ref": q.resume_ref,
            "answer_text": ans.answer_text if ans else None,
            "quality_signal": ans.quality_signal if ans else None,
            "brief_feedback": ans.brief_feedback if ans else None,
            "covered_points": covered_points,
            "missed_points": missed_points
        })
        
    try:
        summary_data = answer_validator.summarize_session(validated_answers)
    except Exception as e:
        summary_data = {
            "total_questions": len(validated_answers),
            "topics_covered": list(set(q.topic for q in session.questions)),
            "quality_distribution": {"strong": 0, "developing": 0, "needs_review": len(validated_answers)},
            "per_topic_summary": {},
            "overall_impression": f"Could not generate summary report: {str(e)}"
        }
        
    return {
        "candidate_name": session.candidate_name,
        "role": session.role,
        "company": session.company,
        "jd_used": bool(session.jd_profile and session.jd_profile.get("usable")),
        "topic_plan": session.topic_plan,
        "total_planned_questions": session.total_planned_questions,
        "overall_impression": summary_data.get("overall_impression"),
        "quality_distribution": summary_data.get("quality_distribution"),
        "per_topic_summary": summary_data.get("per_topic_summary"),
        "topics_covered": summary_data.get("topics_covered"),
        "qa_pairs": qa_pairs
    }
