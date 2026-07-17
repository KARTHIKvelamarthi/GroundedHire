import os
import sys
import json
import time
import logging
import warnings
from typing import List, Dict

# Suppress all future and deprecation warnings to keep the terminal output clean
warnings.simplefilter("ignore")

# Suppress INFO logs to keep terminal interface clean
logging.getLogger().setLevel(logging.WARNING)
# Also set levels for common noisy libraries specifically
for log_name in ["httpx", "sentence_transformers", "urllib3", "retriever", "llm_client"]:
    logging.getLogger(log_name).setLevel(logging.WARNING)


# Ensure standard output uses UTF-8 to prevent Windows terminal printing encoding errors
sys.stdout.reconfigure(encoding='utf-8')

# Ensure core directory is in python search path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(_script_dir, "core")
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

# Import modules from the current directory
import config
import llm_client
from resume_parser import parse_resume
from question_generator import generate_question
from answer_validator import validate_answer, summarize_session

# Tiers of difficulty
DIFFICULTY_ORDER = ["beginner", "intermediate", "intermediate-dense"]

def adjust_difficulty(current_tier: str, quality_signal: str) -> str:
    """
    Shifts the difficulty level up or down based on candidate performance.
    """
    try:
        current_idx = DIFFICULTY_ORDER.index(current_tier)
    except ValueError:
        current_idx = 1  # Default to intermediate
        
    if quality_signal == "strong":
        new_idx = min(current_idx + 1, len(DIFFICULTY_ORDER) - 1)
        if new_idx != current_idx:
            print(f"\n[Adaptive Logic] Quality signal was 'strong'. Increasing difficulty to: '{DIFFICULTY_ORDER[new_idx]}'.")
        return DIFFICULTY_ORDER[new_idx]
    elif quality_signal == "needs_review":
        new_idx = max(current_idx - 1, 0)
        if new_idx != current_idx:
            print(f"\n[Adaptive Logic] Quality signal was 'needs_review'. Decreasing difficulty to: '{DIFFICULTY_ORDER[new_idx]}'.")
        return DIFFICULTY_ORDER[new_idx]
        
    # 'developing' keeps current difficulty
    print(f"\n[Adaptive Logic] Quality signal was 'developing'. Keeping difficulty at: '{current_tier}'.")
    return current_tier

def main():
    print("="*60)
    print("   GROUNDEDHIRE: E2E INTERACTIVE TECHNICAL INTERVIEW SESSION")
    print("="*60)
    
    # 1. Select Role
    print("\nSelect the role you are interviewing for:")
    print("1) AI/ML Engineer (ai_ml_engineer)")
    print("2) Data Science & Applied ML (data_science_applied_ml)")
    role_choice = input("Enter choice (1 or 2, default 1): ").strip()
    
    if role_choice == "2":
        role = "data_science_applied_ml"
    else:
        role = "ai_ml_engineer"
        
    # 2. Get Resume Path
    print(f"\nLet's load your resume.")
    print("Note: You can use 'test_resumes/j0_resume.pdf' or 'test_resumes/sample_resume.txt'.")
    resume_path = input("Enter path to resume file: ").strip()
    
    if not resume_path:
        resume_path = "test_resumes/sample_resume.txt"
        print(f"No path entered, defaulting to '{resume_path}'.")
        
    if not os.path.exists(resume_path):
        print(f"[ERROR] Resume file not found at: {resume_path}")
        sys.exit(1)
        
    # 3. Parse Resume
    print(f"\nParsing resume '{resume_path}' using local LLM seniority inference...")
    timing = {}
    try:
        candidate_context = parse_resume(resume_path, timing_dict=timing)
        print("Resume parsed successfully!")
        print(f"  Candidate Name:     {candidate_context.get('candidate_name', 'Unknown')}")
        print(f"  Starting Difficulty: {candidate_context.get('starting_difficulty_tier', 'intermediate')}")
        print(f"  Mapped Skills:       {', '.join(candidate_context.get('raw_skills', [])[:10])}...")
    except Exception as e:
        print(f"[ERROR] Failed to parse resume: {e}")
        sys.exit(1)
        
    # Set initial difficulty
    current_difficulty = candidate_context.get("starting_difficulty_tier", "intermediate")
    
    # 4. Determine Interview Topics
    # Pick a subset of topics based on role/resume
    standard_topics = [
        "supervised_learning",
        "overfitting_regularization",
        "model_selection",
        "evaluation_metrics",
        "ensemble_methods"
    ]
    
    # Prioritize any topics mapped from the resume if any exist, otherwise use standard ones
    mapped_topics = candidate_context.get("mapped_topics", [])
    topics_to_ask = []
    for t in mapped_topics:
        if t in standard_topics and t not in topics_to_ask:
            topics_to_ask.append(t)
            
    for t in standard_topics:
        if t not in topics_to_ask:
            topics_to_ask.append(t)
            
    # Let's cap the interview to 3 questions for a quick interactive test
    num_questions_input = input("\nHow many questions would you like to practice? (default 3): ").strip()
    try:
        num_questions = int(num_questions_input)
    except ValueError:
        num_questions = 3
        
    topics_to_ask = topics_to_ask[:num_questions]
    
    # 5. Interactive Interview Loop
    session_answers = []
    used_chunk_ids = set()
    
    print("\n" + "="*60)
    print("               STARTING INTERVIEW QUESTIONS")
    print("="*60)
    
    for idx, topic in enumerate(topics_to_ask, 1):
        print(f"\n--- Question {idx} of {num_questions} | Topic: {topic} | Target Difficulty: {current_difficulty} ---")
        print("Retrieving relevant textbook passage and generating question...")
        
        try:
            # Generate grounded question
            q_res = generate_question(
                role=role,
                topic=topic,
                difficulty_tier=current_difficulty,
                candidate_context=candidate_context,
                used_chunk_ids=used_chunk_ids
            )
            
            # Track used chunks
            used_chunk_ids.add(q_res["source_chunk_id"])
            
            print(f"\n[Question]: {q_res['question']}\n")
            print(f"(Grounding Source: {q_res['source_book']} | Section: {q_res['source_section_title']})")
            
            # Get Answer from candidate
            answer = input("\nYour Answer (type 'skip' to pass): ").strip()
            
            # Validate Answer
            print("\nEvaluating your answer against grounding material...")
            eval_res = validate_answer(
                question=q_res["question"],
                candidate_answer=answer,
                expected_answer_points=q_res["expected_answer_points"],
                source_chunk_text=q_res["source_chunk_text"]
            )
            
            # Display Coaching Feedback
            print(f"\n[Coaching Feedback]:")
            print(f"  Quality Signal:  {eval_res['quality_signal'].upper()}")
            print(f"  Feedback:        {eval_res['brief_feedback']}")
            print(f"  Covered Points:  {eval_res['covered_points']}")
            print(f"  Missed Points:   {eval_res['missed_points']}")
            
            # Append topic and tier for summary aggregation
            eval_res["topic"] = topic
            eval_res["difficulty_tier"] = current_difficulty
            session_answers.append(eval_res)
            
            # Adjust difficulty for next turn
            current_difficulty = adjust_difficulty(current_difficulty, eval_res["quality_signal"])
            
        except Exception as e:
            print(f"[ERROR] Failed to run Question {idx}: {e}")
            
    # 6. End of Session Summary
    print("\n" + "="*60)
    print("                    INTERVIEW COMPLETED")
    print("="*60)
    
    summary = summarize_session(session_answers)
    print(f"\nOverall Impression:\n{summary['overall_impression']}")
    print(f"\nPerformance Distribution: {summary['quality_distribution']}")
    
    # Save session summary to disk
    summary_path = "session_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "candidate_name": candidate_context.get("candidate_name", "Unknown"),
            "role": role,
            "summary": summary,
            "detailed_responses": session_answers
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\nDetailed session results saved to: {summary_path}")

if __name__ == "__main__":
    main()
