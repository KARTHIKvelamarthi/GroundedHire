import json
import time
import os
import sys
import numpy as np
from tqdm import tqdm
import config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] sentence-transformers is not installed in this environment.")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] faiss-cpu is not installed in this environment.")
    sys.exit(1)

def load_chunks(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        print(f"[ERROR] Chunk file not found: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_role_index(role: str, input_file: str, index_file: str, metadata_file: str, batch_size: int = 64):
    print(f"\n=== Processing Role: {role} ===")
    chunks = load_chunks(input_file)
    if not chunks:
        print(f"[WARNING] No chunks found for {role}. Skipping.")
        return

    print(f"Loaded {len(chunks)} chunks from {input_file}.")
    
    # Extract text fields
    texts = [c["text"] for c in chunks]
    
    # Load SentenceTransformer model
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"[WARNING] Failed to load '{model_name}' due to: {e}. Falling back to 'all-MiniLM-L6-v2'.")
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        
    print(f"Model loaded successfully. Generating embeddings using {model_name}...")
    
    start_time = time.time()
    embeddings = []
    
    # Batch processing with tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding chunks"):
        batch_texts = texts[i : i + batch_size]
        # normalize_embeddings=True ensures cosine similarity via inner product (IndexFlatIP)
        batch_embeddings = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        
    embeddings = np.vstack(embeddings).astype("float32")
    elapsed = time.time() - start_time
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} in {elapsed:.2f} seconds.")
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, index_file)
    print(f"Saved FAISS index to {index_file}.")
    
    # Save parallel metadata (the full original chunk object per index ID)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved parallel metadata to {metadata_file}.")
    
    return elapsed

def main():
    total_start = time.time()
    
    # Process AI/ML Engineer
    ai_time = build_role_index(
        role="ai_ml_engineer",
        input_file=os.path.join(config.CHUNKS_DIR, "chunks_ai_ml_engineer.json"),
        index_file=os.path.join(config.FAISS_DIR, "faiss_ai_ml_engineer.index"),
        metadata_file=os.path.join(config.METADATA_DIR, "metadata_ai_ml_engineer.json")
    )
    
    # Process Data Science / Applied ML
    ds_time = build_role_index(
        role="data_science_applied_ml",
        input_file=os.path.join(config.CHUNKS_DIR, "chunks_data_science_applied_ml.json"),
        index_file=os.path.join(config.FAISS_DIR, "faiss_data_science_applied_ml.index"),
        metadata_file=os.path.join(config.METADATA_DIR, "metadata_data_science_applied_ml.json")
    )
    
    total_elapsed = time.time() - total_start
    print("\n" + "="*40)
    print(f"Indexing completed in {total_elapsed:.2f} seconds.")
    if ai_time is not None and ds_time is not None:
        print(f"Total embedding generation time: {ai_time + ds_time:.2f} seconds.")
    print("="*40)

if __name__ == "__main__":
    main()
