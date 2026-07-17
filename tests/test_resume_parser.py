import os
import sys
import warnings

# Suppress all future and deprecation warnings to keep the terminal output clean
warnings.simplefilter("ignore")

# Add parent directory to sys.path so we can import from the root directory modules
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

import time
import json
from resume_parser import parse_resume

# Ensure stdout uses UTF-8 to prevent Windows terminal printing encoding errors
sys.stdout.reconfigure(encoding='utf-8')

DUMMY_RESUME_CONTENT = """Jane Doe
Email: jane.doe@example.com
Phone: (555) 123-4567
Profile: Machine Learning Engineer with 3+ years of experience specializing in deep learning optimization and web-based predictor systems.

Work Experience:
Lead Machine Learning Engineer | TechCorp Inc. (2024 - Present)
- Led design and implementation of large-scale deep learning workflows.
- Mentored junior engineers and managed cloud compute resources.

Projects:
1. Distributed Training Optimization for Large Language Models
- Designed a custom training loop using PyTorch FSDP and Megatron-LM to scale model training across 64 A100 GPUs.
- Implemented FP8 mixed-precision training and gradient checkpointing, reducing memory footprint by 40% and increasing training throughput by 25%.
- Diagnosed and resolved neural network training bottlenecks related to backpropagation, gradient accumulation, and communications bandwidth.
- Technologies: PyTorch, FSDP, Megatron-LM, CUDA, Python, Docker.

2. Predictive Housing Web App
- Created a quick web application that allows users to input details about a house (e.g., number of rooms, square footage) and see a price prediction.
- Loaded a pre-trained scikit-learn Random Forest Regressor from a file, set up a simple Flask backend, and built a basic HTML landing page.
- Performed minor feature scaling using pandas before passing features to the classifier.
- Technologies: Python, Flask, Scikit-learn, HTML, Pandas.
"""

def create_sample_resume():
    file_path = os.path.join(PARENT_DIR, "test_resumes", "sample_resume.txt")
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(DUMMY_RESUME_CONTENT)
        print(f"Created dummy resume: {file_path}")
    return file_path

def test_file(file_path: str):
    print("\n" + "="*80)
    print(f"TESTING PARSER ON: {file_path}")
    print("="*80)
    
    timing = {}
    start_time = time.time()
    try:
        parsed_data = parse_resume(file_path, timing_dict=timing)
        total_elapsed = time.time() - start_time
        
        print("Parsing successful!\n")
        print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
        
        print("\n" + "-"*40)
        print("STAGE TIMING BREAKDOWN:")
        print(f"  - Text Extraction:     {timing.get('text_extraction', 0.0)*1000:.2f} ms")
        print(f"  - Section Splitting:   {timing.get('section_splitting', 0.0)*1000:.2f} ms")
        print(f"  - Skills Extraction:   {timing.get('skills_extraction', 0.0)*1000:.2f} ms")
        print(f"  - Topic Tagging:       {timing.get('topic_tagging', 0.0)*1000:.2f} ms")
        print(f"  - Project Parsing:     {timing.get('project_parsing', 0.0)*1000:.2f} ms")
        print(f"  - LLM Call (Seniority): {timing.get('llm_call', 0.0):.2f} seconds")
        print(f"  - Total Latency:       {total_elapsed:.2f} seconds")
        print("-"*40)
        
        if "projects" in parsed_data:
            print("\nProject Differentiation Check:")
            for p in parsed_data["projects"]:
                title = p.get("title", "Unknown")
                related = p.get("related_topics", [])
                print(f"  - Project: '{title}'")
                print(f"    Related Topics: {related}")
                print(f"    Technologies:   {p.get('technologies_used', [])}")
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path}: {e}")

def main():
    dummy_path = create_sample_resume()
    test_file(dummy_path)
    
    pdf_path = os.path.join(PARENT_DIR, "test_resumes", "j0_resume.pdf")
    if os.path.exists(pdf_path):
        test_file(pdf_path)
    else:
        print(f"\n[WARNING] Actual resume file '{pdf_path}' not found in test_resumes directory.")

if __name__ == "__main__":
    main()
