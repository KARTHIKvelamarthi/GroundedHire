import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_indices")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

# Ensure folders exist
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TESTS_DIR, exist_ok=True)
