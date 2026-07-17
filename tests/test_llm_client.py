import os
import sys
import time
import warnings

# Suppress all future and deprecation warnings to keep the terminal output clean
warnings.simplefilter("ignore")

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_client import call_llm

def main():
    print("Testing LLM Client...")
    
    prompt = "Return a JSON object with a single key 'status' set to 'success' and 'message' set to 'Hello World'."
    
    start_time = time.time()
    try:
        response_text = call_llm(prompt, temperature=0.1, json_mode=True)
        latency = time.time() - start_time
        
        # Check which backend was used
        import llm_client
        backend = llm_client.last_used_backend
            
        print("\nTest completed successfully!")
        print(f"Backend Used: {backend}")
        print(f"Latency:      {latency:.4f} seconds")
        print(f"Response:     {response_text}")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")

if __name__ == "__main__":
    main()
