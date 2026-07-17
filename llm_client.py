import os
import logging
import requests
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

# Tracker to see which backend served the last request
last_used_backend = None

def call_llm(prompt: str, temperature: float, json_mode: bool = True) -> str:
    """
    Calls the Groq API (model "llama-3.3-70b-versatile") as primary backend.
    If unavailable, missing API key, rate-limited, or fails for any reason,
    automatically falls back to local Ollama ("mistral:7b-instruct").
    """
    global last_used_backend
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
    
    # Try Groq if API key is present
    if api_key:
        try:
            logger.info("Attempting LLM call via Groq API (llama-3.3-70b-versatile)...")
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
                
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                res_json = response.json()
                content = res_json["choices"][0]["message"]["content"]
                if content:
                    last_used_backend = "Groq API (llama-3.3-70b-versatile)"
                    return content.strip()
                else:
                    raise ValueError("Received empty content from Groq API.")
            else:
                raise RuntimeError(f"Groq API returned error status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}. Falling back to local Ollama (mistral:7b-instruct)...")
    else:
        logger.info("GROQ_API_KEY environment variable is not set. Falling back to local Ollama (mistral:7b-instruct)...")

    # Fallback to local Ollama
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b-instruct",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    if json_mode:
        payload["format"] = "json"
        
    try:
        logger.info("Executing fallback call to local Ollama...")
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        last_used_backend = "Local Ollama (mistral:7b-instruct) fallback"
        return response.json().get("response", "").strip()
    except Exception as e:
        raise ConnectionError(f"Ollama fallback call failed: {e}")
