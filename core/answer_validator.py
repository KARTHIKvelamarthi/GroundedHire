import json
import logging
import requests
from typing import Any
import config
import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_ollama_json(prompt: str, temperature: float = 0.2) -> str:
    """
    Sends a structured evaluation request to local Ollama/Gemini via llm_client.
    """
    return llm_client.call_llm(prompt, temperature=temperature, json_mode=True)

def validate_json_response(data: dict) -> bool:
    """
    Checks that the dictionary has the required keys and that quality_signal
    is one of the 3 allowed values: 'strong', 'developing', 'needs_review'.
    """
    required_keys = {"quality_signal", "covered_points", "missed_points", "brief_feedback"}
    if not isinstance(data, dict):
        return False
    if not required_keys.issubset(data.keys()):
        return False
    
    sig = data["quality_signal"]
    if sig not in ["strong", "developing", "needs_review"]:
        return False
        
    if not isinstance(data["covered_points"], list) or not isinstance(data["missed_points"], list):
        return False
        
    if not isinstance(data["brief_feedback"], str):
        return False
        
    return True

def validate_answer(
    question: str,
    candidate_answer: str,
    expected_answer_points: list[str],
    source_chunk_text: str,
    project_description: str = None
) -> dict:
    """
    Evaluates the candidate's answer against the expected points and source passage.
    Generates coaching-oriented constructive feedback.
    """
    # Quick client-side check for extremely short/empty/nonsensical inputs to skip LLM or ensure it returns needs_review
    answer_clean = candidate_answer.strip()
    is_empty_or_too_short = len(answer_clean.split()) < 5 or answer_clean.lower() in ["skip", "i don't know", "dont know", "not sure", "no idea", "pass"]
    
    discrepancy_clause = ""
    discrepancy_schema = ""
    if project_description:
        discrepancy_clause = f"""
CANDIDATE'S RESUME PROJECT DESCRIPTION:
{project_description}

CRITICAL DISCREPANCY RULE:
Compare the candidate's answer against their Resume Project Description.
Check if the candidate's answer contains any details, metrics, values, or technologies that do not match, contradict, or are missing from the provided Resume Project Description.
If there is a mismatch or contradiction, identify it clearly.
"""
        discrepancy_schema = """,
  "discrepancy": "<discrepancy : metric/value/tech does not match/miss from resume OR null if no discrepancy is found>" """
    
    prompt = f"""You are a constructive technical interviewer and coach. Assess how well the candidate's answer aligns with the source material and expected answer points.

CRITICAL RULES:
1. Do NOT judge based on your own external knowledge. Only compare the candidate's answer against the provided textbook source passage and expected points.
2. Be generous, constructive, and coaching-oriented. This is NOT a punitive pass/fail exam.
3. If the candidate's answer is extremely short (e.g. less than 5 words, "skip", "not sure", or empty), set quality_signal to "needs_review" and note in brief_feedback that the answer was too brief or skipped to be assessed.
{discrepancy_clause}
TEXTBOOK SOURCE PASSAGE:
{source_chunk_text}

QUESTION:
{question}

EXPECTED ANSWER POINTS:
{json.dumps(expected_answer_points, indent=2)}

CANDIDATE'S ANSWER:
{candidate_answer}

JSON OUTPUT SCHEMA (Return ONLY a JSON object conforming to this schema, with no markdown fences, no preamble, and no extra keys):
{{
  "quality_signal": "<strong, developing, or needs_review>",
  "covered_points": [
    "<exact items from the EXPECTED ANSWER POINTS list that were addressed, paraphrased, or referenced by the candidate>"
  ],
  "missed_points": [
    "<exact items from the EXPECTED ANSWER POINTS list that were NOT addressed by the candidate>"
  ],
  "brief_feedback": "<1-2 sentences of constructive and specific feedback, referencing the source material where relevant (e.g. 'Good explanation of X; the source material also emphasizes Y, which could strengthen this answer.')>"{discrepancy_schema}
}}
"""

    # If we determine client-side it is a trivial/empty answer, we can still run it with a simplified prompt or force a direct mock to save time/LLM call.
    # However, to be 100% compliant with using the LLM for edge cases, we let the LLM handle it, but we can pass instructions clearly.
    
    # 1. First Attempt
    raw_output = query_ollama_json(prompt)
    data = None
    try:
        data = json.loads(raw_output)
    except Exception:
        pass
        
    if data and validate_json_response(data):
        if "discrepancy" in data and data["discrepancy"]:
            disc_text = str(data["discrepancy"])
            if "discrepancy" not in disc_text.lower():
                disc_text = f"discrepancy : {disc_text}"
            data["brief_feedback"] = data["brief_feedback"] + f"\n\n{disc_text}"
        return data
        
    # 2. Retry Attempt on Failure
    retry_prompt = f"""{prompt}

[WARNING] Your previous response was invalid. Ensure that quality_signal is exactly one of 'strong', 'developing', or 'needs_review'. covered_points and missed_points must be arrays containing exact strings from the EXPECTED ANSWER POINTS list. Return ONLY a valid JSON object matching the schema.
"""
    raw_output_retry = query_ollama_json(retry_prompt)
    try:
        data = json.loads(raw_output_retry)
    except Exception:
        raise ValueError(f"Failed to parse answer validation JSON from Ollama after retry. Raw: {raw_output_retry}")
        
    if not validate_json_response(data):
        raise ValueError(f"Ollama output failed validation after retry. Data: {data}")
        
    if "discrepancy" in data and data["discrepancy"]:
        disc_text = str(data["discrepancy"])
        if "discrepancy" not in disc_text.lower():
            disc_text = f"discrepancy : {disc_text}"
        data["brief_feedback"] = data["brief_feedback"] + f"\n\n{disc_text}"
        
    return data

def summarize_session(validated_answers: list[dict]) -> dict:
    """
    Aggregates the individual question validation results into session-level statistics
    and provides a rule-based overall summary.
    """
    total = len(validated_answers)
    topics = list(set(ans.get("topic", "unknown") for ans in validated_answers))
    
    # Calculate quality distribution
    quality_dist = {"strong": 0, "developing": 0, "needs_review": 0}
    per_topic = {}
    
    for ans in validated_answers:
        q_sig = ans.get("quality_signal", "needs_review")
        topic = ans.get("topic", "unknown")
        
        if q_sig in quality_dist:
            quality_dist[q_sig] += 1
            
        if topic not in per_topic:
            per_topic[topic] = {"strong": 0, "developing": 0, "needs_review": 0}
        if q_sig in per_topic[topic]:
            per_topic[topic][q_sig] += 1
            
    # Formulate a natural-language overall impression based on performance distribution
    strong_cnt = quality_dist["strong"]
    dev_cnt = quality_dist["developing"]
    needs_cnt = quality_dist["needs_review"]
    
    if total == 0:
        impression = "No questions were completed during this session."
    elif strong_cnt == total:
        impression = "The candidate demonstrated an exceptionally strong grasp of all technical concepts discussed during the session, addressing all expected core points clearly."
    elif strong_cnt + dev_cnt == total and strong_cnt > 0:
        impression = f"Overall, the candidate showed a good understanding of the topics (covering {strong_cnt} topics strongly and {dev_cnt} developing), with minor gaps in coverage or detail."
    elif needs_cnt > 0 and strong_cnt > 0:
        impression = f"The candidate showed solid understanding in some areas, but other sections need review (particularly where answers were too brief or missed the core concepts)."
    elif dev_cnt == total or (dev_cnt > 0 and strong_cnt == 0):
        impression = "The candidate is developing their understanding of these concepts; they covered some expected points but need to deepen their explanations of key details."
    else:
        impression = "Most topics covered in the session need a thorough review, as key concepts were not addressed or explanations were incomplete."

    return {
        "total_questions": total,
        "topics_covered": topics,
        "quality_distribution": quality_dist,
        "per_topic_summary": per_topic,
        "overall_impression": impression
    }
