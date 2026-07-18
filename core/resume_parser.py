import os
import sys
import re
import json
import requests
import logging
import fitz  # PyMuPDF
import llm_client
from ingest import assign_topics  # Reuse exact same taxonomy tagger from ingest.py

logger = logging.getLogger("core.resume_parser")

# Fixed taxonomy for downstream retrieval consistency
FIXED_TAXONOMY = [
    "supervised_learning",
    "unsupervised_learning",
    "decision_trees",
    "neural_networks",
    "clustering",
    "regression",
    "classification",
    "overfitting_regularization",
    "evaluation_metrics",
    "feature_engineering",
    "optimization",
    "probability_statistics",
    "model_selection",
    "ensemble_methods",
    "deep_learning",
    "dimensionality_reduction"
]

# Broad list of common tools, frameworks, languages, and technologies (103 terms)
COMMON_SKILLS = [
    # Languages
    "Python", "SQL", "R", "C\\+\\+", "Java", "JavaScript", "TypeScript", "HTML", "CSS", "Go", "Rust", "Scala", "Julia", "Bash",
    # ML Frameworks / Libraries
    "PyTorch", "TensorFlow", "Keras", "scikit-learn", "sklearn", "XGBoost", "LightGBM", "CatBoost", "FastAI", "JAX", "MXNet",
    # NLP / LLM / GenAI
    "Transformers", "LangChain", "LlamaIndex", "HuggingFace", "Hugging Face", "Ollama", "SentenceTransformers", "Sentence-Transformers", "NLTK", "SpaCy", "Gensim",
    "RAG", "LLM", "VLM", "Generative AI", "GenAI", "BERT", "GPT", "Llama", "Mistral", "Claude", "OpenAI",
    # Computer Vision
    "OpenCV", "MediaPipe", "PaddleOCR", "YOLO", "Ultralytics",
    # MLOps / Vector DBs
    "FAISS", "Chroma", "Pinecone", "Milvus", "Qdrant", "Weaviate", "MLflow", "DVC", "W&B", "Weights & Biases", "Triton",
    # Data processing / Database
    "NumPy", "Pandas", "Polars", "SciPy", "Matplotlib", "Seaborn", "Spark", "PySpark", "Hadoop", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis",
    # Cloud / Infrastructure / DevOps
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Google Cloud", "Git", "GitHub", "GitLab", "CI/CD", "Terraform", "Ansible",
    # Backend / Web
    "Flask", "FastAPI", "Django", "Streamlit", "Gradio", "Node.js", "React",
    # Hardware/Accelerators/Methods
    "CUDA", "FSDP", "DeepSpeed", "Megatron-LM", "LoRA", "QLoRA", "PEFT"
]

DOMAIN_KEYWORDS = {
    "healthcare": [r"\bhealthcare\b", r"\bmedical\b", r"\bclinical\b", r"\bpatient\b", r"\bmedicine\b", r"\bhospital\b", r"\bpolyp\b", r"\bcamouflage\b"],
    "finance": [r"\bfinance\b", r"\bfinancial\b", r"\bbanking\b", r"\btrading\b", r"\bstock\b", r"\binvestment\b"],
    "nlp": [r"\bnlp\b", r"\bnatural\s+language\b", r"\btranslation\b", r"\bspeech\b", r"\basr\b", r"\btext\b", r"\bgloss\b"],
    "computer_vision": [r"\bcomputer\s+vision\b", r"\bimage\b", r"\bvideo\b", r"\bsegmentation\b", r"\bopencv\b", r"\bocr\b", r"\bvision\b", r"\bportrait\b", r"\bcamera\b"]
}

def extract_text(file_path: str) -> str:
    """
    Extracts text from a PDF or a text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Only PDF and TXT files are supported.")
        
    if len(text.strip()) < 50:
        raise ValueError(
            "Extracted text is too short or empty. "
            "The file might be empty, corrupted, or an unscanned/image-only PDF."
        )
        
    return text

def split_resume_into_sections(text: str) -> dict:
    """
    Splits the resume text into sections using regex-based heading matching.
    """
    lines = [line.strip() for line in text.split("\n")]
    
    section_patterns = {
        "summary": ["summary", "profile", "professional summary", "about me", "objective", "professional profile"],
        "skills": ["skills", "technical skills", "core competencies", "key skills", "technologies", "technical expertise", "skills & tools", "skills/tools"],
        "projects": ["projects", "personal projects", "academic projects", "key projects", "selected projects", "technical projects"],
        "experience": ["experience", "work experience", "professional experience", "employment history", "employment", "professional background"],
        "education": ["education", "academic history", "background", "academic qualifications"],
    }
    
    sections = {
        "candidate_info": [],
        "summary": [],
        "skills": [],
        "projects": [],
        "experience": [],
        "education": [],
    }
    
    current_section = "candidate_info"
    
    for line in lines:
        if not line:
            if current_section != "candidate_info":
                sections[current_section].append(line)
            continue
            
        clean_line = line.lower().strip(" \t:.-*#1234567890•⋄")
        
        found_section = None
        for sec_name, keywords in section_patterns.items():
            if clean_line in keywords:
                found_section = sec_name
                break
                
        if found_section:
            current_section = found_section
        else:
            sections[current_section].append(line)
            
    return {k: "\n".join(v).strip() for k, v in sections.items()}

def extract_skills_programmatically(text: str) -> list[str]:
    """
    Case-insensitive substring/boundary matching of technologies within a text snippet.
    """
    matched_skills = []
    sorted_skills = sorted(COMMON_SKILLS, key=len, reverse=True)
    
    spaced_text = " " + text.replace("\n", " ").replace("\r", " ") + " "
    
    for skill in sorted_skills:
        pattern = r"(?i)(?:^|[\s,.:;()'\"/\\⋄\-])" + re.escape(skill) + r"(?:$|[\s,.:;()'\"/\\⋄\-])"
        match = re.search(pattern, spaced_text)
        if match:
            start = match.start()
            end = match.end()
            matched_span = spaced_text[start:end].strip(" ,.:;()'\"/\\⋄\-")
            matched_skills.append(matched_span)
            
    seen = set()
    deduped = []
    for s in matched_skills:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            deduped.append(s)
    return deduped

def extract_candidate_name(text: str) -> str:
    """
    Regex heuristic to find candidate name (typically first short non-empty line at the top).
    """
    lines = [line.strip() for line in text.split("\n")]
    for line in lines:
        if not line:
            continue
        if "@" in line or "http" in line or "+" in line or "⋄" in line or "|" in line:
            continue
        if len(line) < 50:
            return line
    return "Unknown Candidate"

def extract_domain_exposure(text: str) -> list[str]:
    """
    Finds domain exposure using keyword trigger sets.
    """
    domains = []
    for domain, regexes in DOMAIN_KEYWORDS.items():
        for regex in regexes:
            if re.search(regex, text, re.IGNORECASE):
                domains.append(domain)
                break
    return domains

def split_projects_verbatim(project_text: str) -> list[dict]:
    """
    Heuristic-based splitting of project section text into individual projects.
    Handles standalone bullet characters and continuation lines correctly.
    """
    if not project_text.strip():
        return []
        
    raw_lines = [line.strip() for line in project_text.split("\n")]
    processed_lines = []
    
    # 1. Merge standalone bullets onto the start of their content line
    bullet_chars = ("•", "-", "*", "+", "o", "⋄")
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if not line:
            i += 1
            continue
        if line in bullet_chars and i + 1 < len(raw_lines):
            processed_lines.append(f"{line} {raw_lines[i+1]}")
            i += 2
        else:
            processed_lines.append(line)
            i += 1
            
    # 2. Extract projects
    projects = []
    current_project = None
    
    for line in processed_lines:
        is_bullet = line.startswith(bullet_chars)
        starts_with_num = re.match(r"^\d+[\s.)\]-]", line) is not None
        contains_pipe = "|" in line
        
        # A line is a title if it contains a pipe, a dash surrounded by spaces, starts with a number, or is the first project
        is_title = False
        if not is_bullet:
            if starts_with_num or contains_pipe or " - " in line or " – " in line or current_project is None:
                is_title = True
                
        if is_title:
            if current_project:
                projects.append(current_project)
            
            title_clean = line.split("|")[0].strip()
            title_clean = re.sub(r"^\d+[\s.)\]-]\s*", "", title_clean)
            
            current_project = {
                "title": title_clean,
                "bullets": [],
                "raw_lines": [line]
            }
        else:
            if current_project is not None:
                current_project["bullets"].append(line)
                current_project["raw_lines"].append(line)
                
    if current_project:
        projects.append(current_project)
        
    formatted_projects = []
    for proj in projects:
        description = "\n".join(proj["bullets"]).strip()
        proj_full_text = "\n".join(proj["raw_lines"])
        
        formatted_projects.append({
            "title": proj["title"],
            "description": description,
            "technologies_used": extract_skills_programmatically(proj_full_text),
            "related_topics": assign_topics(proj_full_text)
        })
        
    return formatted_projects

def query_ollama(prompt: str) -> str:
    """
    Sends a JSON generation request to local Ollama via llm_client.
    """
    return llm_client.call_llm(prompt, temperature=0.1, json_mode=True)

def infer_experience_level_and_justification_via_llm(project_text: str) -> tuple[str, str]:
    """
    Uses the local LLM to infer the experience level ("beginner", "intermediate", "advanced")
    along with a one-sentence justification.
    """
    prompt = f"""Given the following project/work experience text from a resume, classify the candidate's seniority level as "beginner", "intermediate", or "advanced".
Provide a one-sentence justification.

Return ONLY a JSON object that adheres strictly to this schema:
{{
  "inferred_experience_level": "beginner" | "intermediate" | "advanced",
  "justification": "<one-sentence justification>"
}}

Experience / Projects Text:
{project_text}
"""
    # Try once
    raw_output = query_ollama(prompt)
    try:
        data = json.loads(raw_output)
        level = str(data.get("inferred_experience_level", "")).strip().lower()
        justification = str(data.get("justification", "")).strip()
        if level in ["beginner", "intermediate", "advanced"]:
            return level, justification
    except Exception:
        pass
        
    # Retry once
    retry_prompt = f"""{prompt}

[WARNING] Your previous response was invalid. Return ONLY a JSON object with keys "inferred_experience_level" and "justification".
"""
    raw_output_retry = query_ollama(retry_prompt)
    try:
        data = json.loads(raw_output_retry)
        level = str(data.get("inferred_experience_level", "")).strip().lower()
        justification = str(data.get("justification", "")).strip()
        if level in ["beginner", "intermediate", "advanced"]:
            return level, justification
    except Exception as e:
        log_file = "debug_experience_inference.log"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- INFERENCE FAILURE ---\n")
            lf.write(f"Attempt 1: {raw_output}\n")
            lf.write(f"Attempt 2: {raw_output_retry}\n")
            lf.write(f"Error: {e}\n")
        raise ValueError(f"Failed to infer experience level via LLM after retry. Logs written to {log_file}.")
        
    return "intermediate", "Defaulted to intermediate due to parsing failure."

def parse_resume(file_path: str, timing_dict: dict = None) -> dict:
    """
    Parses a candidate's resume by combining pure python extraction and a single target LLM call.
    """
    import time
    logger.info(f"Parsing resume: {os.path.basename(file_path)}")
    
    # 0. Text extraction
    t_start = time.time()
    resume_text = extract_text(file_path)
    if timing_dict is not None:
        timing_dict["text_extraction"] = time.time() - t_start
        
    # 1. Section splitting
    t_sec = time.time()
    sections = split_resume_into_sections(resume_text)
    if timing_dict is not None:
        timing_dict["section_splitting"] = time.time() - t_sec
        
    # 2. Skills extraction
    t_skills = time.time()
    raw_skills_resume = extract_skills_programmatically(resume_text)
    if timing_dict is not None:
        timing_dict["skills_extraction"] = time.time() - t_skills
        
    # 3. Topic mapping
    t_topic = time.time()
    mapped_topics = assign_topics(resume_text)
    if timing_dict is not None:
        timing_dict["topic_tagging"] = time.time() - t_topic
        
    # 4. Project parsing
    t_proj = time.time()
    project_text = sections.get("projects") or sections.get("experience") or ""
    projects = split_projects_verbatim(project_text)
    if timing_dict is not None:
        timing_dict["project_parsing"] = time.time() - t_proj
        
    # Merge skills from full resume and projects
    project_techs = []
    for proj in projects:
        project_techs.extend(proj["technologies_used"])
    
    seen = set()
    raw_skills = []
    for s in raw_skills_resume + project_techs:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            raw_skills.append(s)
            
    # 5. LLM Call for experience level and justification
    t_llm = time.time()
    inferred_level = "intermediate"
    level_justification = "No projects or experience text found to evaluate."
    if project_text.strip():
        inferred_level, level_justification = infer_experience_level_and_justification_via_llm(project_text)
    if timing_dict is not None:
        timing_dict["llm_call"] = time.time() - t_llm
        
    # Map starting difficulty tier
    difficulty_map = {
        "beginner": "beginner",
        "intermediate": "intermediate",
        "advanced": "intermediate-dense"
    }
    starting_difficulty_tier = difficulty_map.get(inferred_level, "intermediate")
    
    # 6. Overall schema construction
    candidate_name = extract_candidate_name(resume_text)
    domain_exposure = extract_domain_exposure(resume_text)
    
    return {
        "candidate_name": candidate_name,
        "raw_skills": raw_skills,
        "mapped_topics": mapped_topics,
        "inferred_experience_level": inferred_level,
        "starting_difficulty_tier": starting_difficulty_tier,
        "level_source": "llm_inference_from_resume",
        "level_justification": level_justification,
        "note": "This is a starting point only; actual difficulty adjusts based on interview performance.",
        "projects": projects,
        "domain_exposure": domain_exposure
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <path_to_resume>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    try:
        result = parse_resume(file_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
