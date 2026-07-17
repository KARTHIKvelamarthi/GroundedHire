import os
import sys
import warnings

# Suppress all future and deprecation warnings to keep the terminal output clean
warnings.simplefilter("ignore")

# Add parent directory to sys.path so we can import from the root directory modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from question_generator import generate_question

# Reconfigure stdout to utf-8 to prevent Windows terminal printing encoding errors
sys.stdout.reconfigure(encoding='utf-8')

def run_test_case(case_num: int, role: str, topic: str, difficulty_tier: str, candidate_context: dict | None = None):
    print("\n" + "="*80)
    print(f"TEST CASE {case_num}: {role} | Topic: {topic} | Tier: {difficulty_tier}")
    if candidate_context:
        print(f"Candidate Context: {json.dumps(candidate_context)}")
    print("="*80)
    
    start_time = time.time()
    try:
        result = generate_question(
            role=role,
            topic=topic,
            difficulty_tier=difficulty_tier,
            candidate_context=candidate_context
        )
        elapsed = time.time() - start_time
        
        print(f"Generation successful! Latency: {elapsed:.2f} seconds\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n" + "-"*80)
        print("SUMMARY:")
        print(f"  - Question:              {result['question']}")
        print(f"  - Concept:               {result['concept_being_tested']}")
        print(f"  - Expected Answer:       {result['expected_answer_points']}")
        print(f"  - Source Book/Section:   {result['source_book']} / {result['source_section_title']}")
        print(f"  - Copy Overlap Ratio:    {result['copy_overlap_ratio']:.4f}")
        print(f"  - Copy Overlap Warning:  {result['copy_overlap_warning_triggered']}")
        print("-"*80)
    except Exception as e:
        print(f"[ERROR] Failed test case {case_num}: {e}")

def main():
    print("Starting Interview Question Generation Module Validation Runs...")
    
    # Case 1: AI/ML Engineer - Overfitting/Regularization - Beginner
    run_test_case(
        case_num=1,
        role="ai_ml_engineer",
        topic="overfitting_regularization",
        difficulty_tier="beginner"
    )
    
    # Case 2: AI/ML Engineer - Overfitting/Regularization - Intermediate-Dense (to compare same topic at different difficulty)
    run_test_case(
        case_num=2,
        role="ai_ml_engineer",
        topic="overfitting_regularization",
        difficulty_tier="intermediate-dense"
    )
    
    # Case 3: AI/ML Engineer - Classification - Intermediate (with candidate context)
    run_test_case(
        case_num=3,
        role="ai_ml_engineer",
        topic="classification",
        difficulty_tier="intermediate",
        candidate_context={
            "raw_skills": ["scikit-learn", "Random Forest"],
            "projects": [{"title": "Predictive Housing Web App", "description": "Created a web application that uses a pre-trained scikit-learn Random Forest Regressor to predict house prices based on user input.", "technologies_used": ["Python", "Flask", "Scikit-learn", "HTML", "Pandas"]}]
        }
    )
    
    # Case 4: Data Science / Applied ML - Model Selection - Intermediate
    run_test_case(
        case_num=4,
        role="data_science_applied_ml",
        topic="model_selection",
        difficulty_tier="intermediate"
    )

if __name__ == "__main__":
    main()
