# EvidenceDesk

An evidence-backed security questionnaire assistant with retrieval evaluation and human review.

**Problem:** Help reviewers find reliable evidence for security questionnaires without treating matching text as proof.

**Current implementation:** FastAPI, synthetic security policies, lexical retrieval baseline, embedding retrieval (primary), constrained LLM reasoning layer (Ollama) to classify evidence, retrieval evaluation, and human review interface.

**Evaluation:** 25 labeled questions covering direct, paraphrased, unanswerable, ambiguous, and conflicting cases, with expected evidence and behavior recorded for each.

**How to run:** 
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install fastapi uvicorn sentence-transformers ollama`
4. Make sure you have Ollama running locally with the `qwen2.5:3b` model.
5. Start the backend: `python evidencedesk.py`
6. Open your browser to: `http://127.0.0.1:8000`

**Retrieval & Reasoning:** 
- **Retrieval:** The primary method is Embedding retrieval (`all-MiniLM-L6-v2`), which finds semantically related policy sentences. A lexical (word-overlap) baseline is also available.
- **Reasoning:** After retrieval, a local LLM strictly classifies the evidence as ANSWERABLE, AMBIGUOUS, INSUFFICIENT_EVIDENCE, or CONFLICTING_EVIDENCE based *only* on the provided passages.

**Limitations:** Human review is still completely required. The AI does not generate conversational final answers. Not for production use.
