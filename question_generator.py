import os
import re
import json
import logging
import requests
from datetime import datetime
from typing import Any
import retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOPIC_QUERY_MAP = {
    "supervised_learning": "supervised learning algorithms, labeled training data, inputs and outputs",
    "unsupervised_learning": "unsupervised learning, clustering, clustering algorithms, unlabeled data",
    "decision_trees": "decision trees, splitting criteria, Gini impurity, entropy, decision tree training",
    "neural_networks": "neural networks, feedforward neural network, activation functions, backpropagation",
    "clustering": "clustering algorithms, k-means, hierarchical clustering, cluster similarity metrics",
    "regression": "regression analysis, linear regression, loss function, gradient descent, predicting values",
    "classification": "classification algorithms, logistic regression, support vector machines, multiclass classification",
    "overfitting_regularization": "overfitting and regularization, L1 L2 regularization, bias-variance tradeoff, dropout",
    "evaluation_metrics": "evaluation metrics, precision, recall, F1 score, ROC AUC curve, confusion matrix",
    "feature_engineering": "feature engineering, feature scaling, normalization, standardization, categorical encoding",
    "optimization": "optimization algorithms, gradient descent, Adam optimizer, loss minimization",
    "probability_statistics": "probability and statistics, probability distributions, Bayes theorem, expected values",
    "model_selection": "model selection, cross-validation, hyperparameter tuning, grid search",
    "ensemble_methods": "ensemble methods, bagging, boosting, random forest, AdaBoost, gradient boosting",
    "deep_learning": "deep learning, deep neural networks, convolutional neural networks, training deep networks",
    "dimensionality_reduction": "dimensionality reduction, principal component analysis PCA, t-SNE, feature selection"
}

def query_ollama_json(prompt: str, temperature: float = 0.7) -> str:
    """
    Sends a structured generation request to local Ollama running mistral:7b-instruct.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b-instruct",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to local Ollama server at {url}: {e}")

def validate_question_data(data: dict) -> bool:
    """
    Checks that the dictionary has required keys, is not empty, and the question ends with a question mark.
    """
    required_keys = {"question", "concept_being_tested", "expected_answer_points"}
    if not isinstance(data, dict):
        return False
    if not required_keys.issubset(data.keys()):
        return False
    q = data["question"]
    if not isinstance(q, str) or not q.strip().endswith("?"):
        return False
    if not isinstance(data["expected_answer_points"], list) or len(data["expected_answer_points"]) == 0:
        return False
    return True

def check_copy_overlap(question: str, chunk_text: str, n: int = 8) -> tuple[float, bool]:
    """
    Checks for plagiarism/copying by detecting long contiguous phrases of n words
    verbatim in the chunk.
    Also returns the old word-level overlap ratio for informational logging.
    
    Returns:
        (word_overlap_ratio, warning_triggered)
    """
    q_words = re.findall(r"\w+", question.lower())
    c_words = re.findall(r"\w+", chunk_text.lower())
    
    if not q_words or not c_words:
        return 0.0, False
        
    # Calculate raw word-level overlap ratio (for informational reporting)
    q_words_set = set(q_words)
    c_words_set = set(c_words)
    overlap = q_words_set.intersection(c_words_set)
    word_overlap_ratio = len(overlap) / len(q_words_set) if q_words_set else 0.0
    
    # Check for contiguous phrase overlap of length n
    n_adjusted = min(n, len(q_words))
    if n_adjusted <= 0:
        return word_overlap_ratio, False
        
    # Convert chunk words to a set of n-grams (as tuples) for fast lookup
    chunk_ngrams = set()
    for i in range(len(c_words) - n_adjusted + 1):
        chunk_ngrams.add(tuple(c_words[i : i + n_adjusted]))
        
    # Generate n-grams of the question
    question_ngrams = []
    for i in range(len(q_words) - n_adjusted + 1):
        question_ngrams.append(tuple(q_words[i : i + n_adjusted]))
        
    # Check if any question n-gram is in the chunk n-grams
    warning_triggered = False
    for ngram in question_ngrams:
        if ngram in chunk_ngrams:
            warning_triggered = True
            break
            
    return word_overlap_ratio, warning_triggered


def assess_project_chunk_relation(project: dict, chunk_text: str) -> dict:
    """
    Asks the local LLM to assess whether the candidate's project and the textbook chunk
    have matching/compatible ML task types and data domains.
    """
    title = project.get("title", "Unknown Project")
    desc = project.get("description", "")
    techs = project.get("technologies_used", [])
    
    prompt = f"""Analyze the candidate's project and the textbook chunk to determine if their machine learning task types and data domains genuinely match for a direct technical comparison.

CANDIDATE PROJECT:
Title: {title}
Description: {desc}
Technologies: {', '.join(techs)}

TEXTBOOK CHUNK:
{chunk_text}

JSON OUTPUT SCHEMA (Return ONLY a JSON object conforming to this schema, no markdown fences, no extra keys):
{{
  "project_task_type": "<classification, regression, clustering, generation, dimensionality reduction, or other>",
  "project_data_domain": "<tabular, text, image, audio, time-series, or other>",
  "chunk_task_type": "<classification, regression, clustering, generation, dimensionality reduction, or other>",
  "chunk_data_domain": "<tabular, text, image, audio, time-series, or other>",
  "genuinely_relate": <true or false. Set to true ONLY if they share BOTH the same task type and data domain, OR are directly comparable techniques (e.g. comparing two classification methods, or two regression methods). Set to false if one is tabular regression (e.g., house price prediction) and the other is text classification (e.g., naive Bayes text classification).>
}}
"""
    try:
        raw_output = query_ollama_json(prompt, temperature=0.1)
        data = json.loads(raw_output)
        if "genuinely_relate" in data:
            return data
    except Exception as e:
        logger.error(f"Error in assess_project_chunk_relation: {e}")
    
    return {"genuinely_relate": False}


def generate_question(
    role: str,
    topic: str,
    difficulty_tier: str,
    candidate_context: dict | None = None,
    used_chunk_ids: set[str] | None = None
) -> dict:
    """
    Retrieves a textbook chunk based on role, topic, and difficulty,
    and calls the local LLM to generate a personalized and grounded question.
    """
    # 1. Map difficulty tier for retrieval
    retriever_tier = difficulty_tier
    
    # 2. Get query text from topic
    query = TOPIC_QUERY_MAP.get(topic, topic.replace("_", " "))
    
    # Apply query biasing based on candidate context keywords
    if candidate_context:
        skills = [s.lower() for s in candidate_context.get("raw_skills", [])]
        projects = candidate_context.get("projects", [])
        
        bias_keywords = [
            "random forest", "decision tree", "neural network", "regression", "svm", "support vector machine",
            "linear regression", "logistic regression", "clustering", "k-means", "naive bayes", "deep learning",
            "cnn", "transformer", "pca", "dimensionality reduction", "regularization", "cross-validation",
            "gradient descent", "adam", "evaluation metric", "precision", "recall", "f1"
        ]
        
        found_keywords = []
        for kw in bias_keywords:
            if any(kw in s for s in skills):
                found_keywords.append(kw)
                
        for proj in projects:
            title = proj.get("title", "").lower()
            desc = proj.get("description", "").lower()
            for kw in bias_keywords:
                if (kw in title or kw in desc) and kw not in found_keywords:
                    found_keywords.append(kw)
                    
        if found_keywords:
            query_lower = query.lower()
            to_add = [kw for kw in found_keywords if kw not in query_lower]
            if to_add:
                query = f"{query} {to_add[0]}"
    
    # 3. Retrieve chunks (request up to 5 to allow deduplication against used_chunk_ids)
    chunks = retriever.retrieve(
        query=query,
        role=role,
        top_k=5,
        difficulty_tier=retriever_tier,
        topics=[topic]
    )
    
    if not chunks:
        raise ValueError(
            f"No source chunks found for role '{role}', topic '{topic}', "
            f"and retriever tier '{retriever_tier}'."
        )
        
    # 4. Deduplicate: pick the highest-ranked chunk that is not in used_chunk_ids
    selected_chunk = None
    if used_chunk_ids:
        for chunk in chunks:
            if chunk["chunk_id"] not in used_chunk_ids:
                selected_chunk = chunk
                break
                
    if selected_chunk is None:
        # Fallback to rank-1 chunk if all candidates are used or used_chunk_ids is empty
        selected_chunk = chunks[0]
        
    chunk_text = selected_chunk["text"]
    
    # 5. Formulate prompt based on difficulty tier
    difficulty_instructions = {
        "beginner": "Ask for a definition or simple explanation of a core concept in plain terms.",
        "intermediate": "Ask the candidate to explain a mechanism, compare two related concepts, or apply the idea to a scenario.",
        "intermediate-dense": "Ask a deeper applied or comparative question that requires genuine understanding, such as trade-offs, edge cases, or 'why' rather than 'what'."
    }
    difficulty_instruction = difficulty_instructions.get(difficulty_tier, difficulty_instructions["intermediate"])
    
    # Formulate candidate context instruction
    context_instruction = ""
    if candidate_context:
        skills = candidate_context.get("raw_skills", [])
        projects = candidate_context.get("projects", [])
        
        # Check compatibility for each project
        has_compatible_project = False
        compatible_project = None
        
        for p in projects:
            relation = assess_project_chunk_relation(p, chunk_text)
            logger.info(f"Compatibility check for project '{p.get('title')}': {relation}")
            relate_val = relation.get("genuinely_relate")
            if relate_val is True or str(relate_val).lower() == "true":
                has_compatible_project = True
                compatible_project = p
                break
                
        if has_compatible_project and compatible_project:
            proj_title = compatible_project.get("title", "Unknown Project")
            proj_desc = compatible_project.get("description", "")
            proj_techs = compatible_project.get("technologies_used", [])
            context_str = f"Skills: {', '.join(skills)}\nCompatible Project:\nProject '{proj_title}' using {', '.join(proj_techs)} (Description: {proj_desc})"
            
            context_instruction = f"""
Subtly frame/tailor the question in a way that aligns with the candidate's background (listed below), if natural.
Do NOT mention or invent facts about the candidate that are not provided in this background.

IMPORTANT RULES FOR PERSONALIZATION:
1. Reference the candidate's project '{proj_title}' and their experience with its techniques/skills as a natural jumping-off point.
2. Ensure the technical comparison is sound and logical (e.g. comparing classification methods to classification methods).

CANDIDATE BACKGROUND:
{context_str}
"""
        else:
            context_instruction = """
The candidate's background does NOT technically match the concepts in the textbook passage.
Therefore, do NOT make any technical comparison, do NOT name any specific candidate project, and do NOT name any specific candidate skills or algorithms (such as Random Forest, scikit-learn, etc.).
Instead, use a loose, high-level framing referencing their general machine learning background (e.g., 'Given your experience building machine learning applications...' or 'Given your background in machine learning systems...').
"""
    
    prompt = f"""You are a senior technical interviewer. Write ONE interview question that tests understanding of a concept described in the following textbook passage.

TEXTBOOK PASSAGE:
{chunk_text}

DIFFICULTY LEVEL INSTRUCTIONS:
{difficulty_instruction}
{context_instruction}

IMPORTANT FORMATTING RULES:
1. The question MUST end with a question mark (?).
2. Do NOT append any follow-up sentences ending with a period (e.g., do NOT write "Can you explain X? Describe its usage."). Instead, incorporate all parts into a single query: "Can you explain X and describe its usage?"

JSON OUTPUT SCHEMA (Return ONLY a JSON object conforming to this schema, with no markdown fences, no preamble, and no extra keys):
{{
  "question": "<the interview question text, MUST end with a question mark '?' and not contain trailing sentences ending with a period>",
  "concept_being_tested": "<short phrase, e.g. 'bias-variance tradeoff'>",
  "expected_answer_points": [
    "<bullet point 1>",
    "<bullet point 2>"
  ]
}}
"""

    # 6. Query Ollama and Validate
    raw_output = query_ollama_json(prompt)
    data = None
    try:
        data = json.loads(raw_output)
    except Exception:
        pass
        
    if data and validate_question_data(data):
        # Succeeded on first attempt
        pass
    else:
        # Retry once with warning
        retry_prompt = f"""{prompt}

[WARNING] Your previous response was invalid. Specifically, the "question" field MUST be non-empty and MUST end with a question mark (?). Ensure there are no sentences at the end of the question ending with a period.
Return ONLY a valid JSON object matching the schema.
"""
        raw_output_retry = query_ollama_json(retry_prompt)
        try:
            data = json.loads(raw_output_retry)
        except Exception:
            raise ValueError(f"Failed to parse question JSON from Ollama after retry. Raw output: {raw_output_retry}")
            
        if not validate_question_data(data):
            raise ValueError(f"Ollama output failed validation after retry. Data: {data}")
            
    # 7. Check copy overlap
    word_overlap_ratio, overlap_triggered = check_copy_overlap(data["question"], chunk_text)
    if overlap_triggered:
        logger.warning(
            f"Copy-overlap warning triggered for chunk_id '{selected_chunk['chunk_id']}': "
            f"question text contains a contiguous phrase of 8+ words from the textbook content."
        )
        
    # 8. Construct final response with full provenance
    generated_at = datetime.utcnow().isoformat() + "Z"
    
    return {
        "question": data["question"],
        "concept_being_tested": data["concept_being_tested"],
        "expected_answer_points": data["expected_answer_points"],
        "topic": topic,
        "difficulty_tier": difficulty_tier,
        "role": role,
        "source_chunk_id": selected_chunk["chunk_id"],
        "source_book": selected_chunk["source_book"],
        "source_page_start": selected_chunk["page_start"],
        "source_page_end": selected_chunk["page_end"],
        "source_section_title": selected_chunk["section_title"],
        "source_chunk_text": chunk_text,
        "copy_overlap_ratio": word_overlap_ratio,
        "copy_overlap_warning_triggered": overlap_triggered,
        "generated_at": generated_at
    }
