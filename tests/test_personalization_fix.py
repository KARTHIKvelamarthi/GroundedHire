import os
import sys
import json

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resume_parser import parse_resume
from question_generator import generate_question

# Ensure standard output uses UTF-8 to prevent Windows terminal printing encoding errors
sys.stdout.reconfigure(encoding='utf-8')

def test_case(resume_path: str, role: str, topic: str, difficulty: str):
    print(f"\n======================================================================")
    print(f"TESTING RESUME: {os.path.basename(resume_path)}")
    print(f"Topic: {topic} | Difficulty: {difficulty}")
    print(f"======================================================================")
    
    # 1. Parse resume
    print("Parsing resume...")
    candidate_context = parse_resume(resume_path)
    print(f"Candidate Name: {candidate_context.get('candidate_name')}")
    
    # Print the projects to see what the model is given
    print("Parsed Projects in context:")
    for idx, p in enumerate(candidate_context.get("projects", []), 1):
        print(f"  Project {idx}: Title='{p.get('title')}'")
        print(f"    Techs: {p.get('technologies_used')}")
        # print snippet of desc
        desc = p.get('description', '')
        print(f"    Desc Snippet: {desc[:150]}...")

    # 2. Generate Question
    print("\nGenerating personalized question...")
    try:
        result = generate_question(
            role=role,
            topic=topic,
            difficulty_tier=difficulty,
            candidate_context=candidate_context
        )
        print("\nGenerated Question:")
        print(result["question"])
        print(f"\nSource Book: {result['source_book']}")
        print(f"Source Section: {result['source_section_title']}")
        print(f"Concept tested: {result['concept_being_tested']}")
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")

def main():
    # 1. Varshith's resume: GraphGuard project, evaluation_metrics topic
    test_case(
        resume_path="test_resumes/test_4.pdf",
        role="ai_ml_engineer",
        topic="evaluation_metrics",
        difficulty="intermediate-dense"
    )
    
    # 2. Karthik's resume: DocuMIND RAG project, evaluation_metrics topic
    test_case(
        resume_path="test_resumes/j0_resume.pdf",
        role="ai_ml_engineer",
        topic="evaluation_metrics",
        difficulty="intermediate-dense"
    )
    
    # 3. Suraj's resume: Sentiment Analysis project, classification topic
    test_case(
        resume_path="test_resumes/test_3.pdf",
        role="ai_ml_engineer",
        topic="classification",
        difficulty="intermediate-dense"
    )

if __name__ == "__main__":
    main()
