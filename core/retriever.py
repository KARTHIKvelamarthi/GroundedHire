import os
import json
import logging
import numpy as np
from typing import Any
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy load model and index caches to make repeated queries fast
_MODEL_CACHE: dict[str, Any] = {}
_INDEX_CACHE: dict[str, Any] = {}
_METADATA_CACHE: dict[str, list[dict]] = {}

def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> Any:
    """
    Retrieves and caches the sentence-transformers model.
    """
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}...")
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            fallback_name = "all-MiniLM-L6-v2"
            logger.warning(f"Failed to load '{model_name}' due to: {e}. Falling back to '{fallback_name}'.")
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[model_name] = SentenceTransformer(fallback_name)
    return _MODEL_CACHE[model_name]

def get_index_and_metadata(role: str) -> tuple[Any, list[dict]]:
    """
    Retrieves and caches the FAISS index and parallel metadata for the given role.
    """
    if role not in _INDEX_CACHE:
        import faiss
        index_file = os.path.join(config.FAISS_DIR, f"faiss_{role}.index")
        metadata_file = os.path.join(config.METADATA_DIR, f"metadata_{role}.json")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file not found: {index_file}. Run build_index.py first.")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}. Run build_index.py first.")
            
        logger.info(f"Loading FAISS index from {index_file}...")
        _INDEX_CACHE[role] = faiss.read_index(index_file)
        
        logger.info(f"Loading metadata from {metadata_file}...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            _METADATA_CACHE[role] = json.load(f)
            
    return _INDEX_CACHE[role], _METADATA_CACHE[role]

def retrieve(
    query: str,
    role: str,
    top_k: int = 5,
    difficulty_tier: str | None = None,
    topics: list[str] | None = None
) -> list[dict]:
    """
    Embeds the query, runs FAISS search against the role's index, post-filters
    candidates by difficulty tier and/or topics, and returns the top_k matching chunks
    with similarity scores.
    
    Args:
        query (str): The search query.
        role (str): The role folder name (e.g. 'ai_ml_engineer' or 'data_science_applied_ml').
        top_k (int): Number of results to return.
        difficulty_tier (str | None): Exact match filter for difficulty ('beginner', 'intermediate', 'advanced').
        topics (list[str] | None): Overlapping topics filter. The chunk must have at least one overlapping topic.
        
    Returns:
        list[dict]: List of matched chunk dictionaries, each containing 'similarity_score'.
    """
    # 1. Load model and indices
    topic_str = topics[0] if topics else query
    logger.info(f"Retrieving chunks for topic='{topic_str}', difficulty='{difficulty_tier}', role='{role}'")
    
    model = get_embedding_model()
    try:
        index, metadata = get_index_and_metadata(role)
    except FileNotFoundError as e:
        logger.error(str(e))
        return []
        
    # 2. Embed the query (normalized for cosine similarity)
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    
    # 3. Retrieve a generous candidate pool to allow for post-filtering
    candidate_pool_size = max(50, top_k * 5)
    # Clamp candidate pool size to the total number of chunks in the index
    candidate_pool_size = min(candidate_pool_size, len(metadata))
    
    if candidate_pool_size == 0:
        logger.warning(f"Index for role '{role}' is empty.")
        return []
        
    scores, indices = index.search(query_vector, candidate_pool_size)
    
    # 4. Filter and score candidates
    filtered_results = []
    
    for score, idx in zip(scores[0], indices[0]):
        # FAISS returns -1 for empty/invalid slots
        if idx < 0 or idx >= len(metadata):
            continue
            
        chunk = metadata[idx].copy()
        chunk["similarity_score"] = float(score)
        
        # Apply difficulty tier filter (exact case-insensitive match)
        if difficulty_tier:
            chunk_tier = chunk.get("difficulty_tier", "")
            if chunk_tier.lower() != difficulty_tier.lower():
                continue
                
        # Apply topics filter (must have at least one overlapping topic, case-insensitive)
        if topics:
            chunk_topics = [t.lower() for t in chunk.get("topics", [])]
            query_topics = [t.lower() for t in topics]
            # Check for intersection
            if not any(qt in chunk_topics for qt in query_topics):
                continue
                
        filtered_results.append(chunk)
        
    # 5. Sort by similarity score descending and slice to top_k
    filtered_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_results = filtered_results[:top_k]
    
    if len(top_results) < top_k:
        logger.warning(
            f"Post-filtering left only {len(top_results)} results, "
            f"which is fewer than requested top_k={top_k}."
        )
        
    return top_results
