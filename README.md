# AI/ML Interview Prep System (RAG-based Question Generator)

This repository implements a local, RAG-driven question generation and resume-parsing pipeline for technical interviews. It automatically matches textbook concept chunks (grounding) against parsed candidate resumes (personalization) to formulate sound, coherent, and difficulty-appropriate questions.

> [!IMPORTANT]
> **Source Textbook PDFs Required**: 
> The `role_pdfs/` folder is excluded from version control (`.gitignore`) due to size and copyright restrictions. Users must manually place their source textbook PDFs in the corresponding subdirectories before running the ingest pipeline:
> - `role_pdfs/ai_ml_engineer/`
> - `role_pdfs/data_science_applied_ml/`

---

## Folder Structure

```
├── chunks/                  # Generated book chunks JSON files
├── faiss_indices/           # Saved FAISS index files for each role
├── metadata/                # Parallel metadata JSON files (aligned with FAISS IDs)
├── role_pdfs/               # [Gitignored] User-provided textbook PDFs
├── tests/                   # Automation and validation test suite
│   ├── test_copy_detector.py
│   ├── test_question_generation.py
│   ├── test_resume_parser.py
│   └── test_retrieval.py
├── build_index.py           # Embeds textbook chunks & constructs FAISS indices
├── config.py                # Central directory and path configuration
├── ingest.py                # Extracts, cleans, and filters textbook passages into chunks
├── question_generator.py    # Local LLM generator pipeline for technical questions
├── resume_parser.py         # ATS resume section splitter and skills parser
├── retriever.py             # Similarity search interface with FAISS
├── requirements.txt         # Package dependencies
└── .gitignore               # Ignored folders and files
```

---

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Populate Textbooks**:
   Place PDF textbooks under `role_pdfs/ai_ml_engineer/` and `role_pdfs/data_science_applied_ml/`.

3. **Run Chunk Ingestion**:
   ```bash
   python ingest.py
   ```

4. **Build Vector Indexes**:
   ```bash
   python build_index.py
   ```

5. **Run Verification Suite**:
   ```bash
   python tests/test_retrieval.py
   python tests/test_copy_detector.py
   python tests/test_question_generation.py
   python tests/test_resume_parser.py
   ```
