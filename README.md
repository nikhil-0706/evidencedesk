# EvidenceDesk

An evidence-backed security questionnaire assistant with retrieval evaluation and human review governance. EvidenceDesk provides a robust, deterministic, verifiable pipeline for answering compliance and security questions against ingested company policies.

## Problem
Security questionnaires require reviewers to find trustworthy evidence in policy documents. Traditional RAG systems often conflate semantic similarity with factual proof, or they invent facts when data is missing. Reviewers need to see the exact provenance of the evidence, and they need a system that safely abstains or warns when evidence is conflicting or insufficient.

## Solution
Evidence-backed retrieval + structured reasoning + human review. EvidenceDesk enforces a strict pipeline: it retrieves evidence using hybrid search, uses a local small language model strictly as a classifier rather than a generator, and mandates human reviewer sign-off for every answer.

## Architecture
User Question
→ Workspace Filter
→ BM25 + Dense Retrieval
→ RRF (Reciprocal Rank Fusion)
→ Cross-Encoder
→ Structured SLM Reasoning (Qwen 2.5 3B)
→ Human Review

## Features
- Multi-document PDF/DOCX/TXT ingestion with structure-aware chunking
- Provenance-aware evidence tracking (document, version, chunk, workspace)
- Workspace isolation
- Hybrid retrieval (BM25 + 384-dim dense embeddings)
- Reciprocal Rank Fusion (k=60)
- Cross-Encoder reranking (BAAI/bge-reranker-large)
- Structured SLM decisions (ANSWERABLE, AMBIGUOUS, CONFLICTING, INSUFFICIENT_EVIDENCE)
- Human review interface
- Audit receipts

## Local Setup
1. Clone the repository and enter the directory.
2. Create and activate a Python 3.10+ virtual environment:
   `python -m venv .venv`
   `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
3. Install dependencies:
   `pip install -r requirements.txt`
4. Make sure you have [Ollama](https://ollama.ai) installed and running locally.
5. Pull the SLM model:
   `ollama pull qwen2.5:3b`

## Configuration
EvidenceDesk supports the following environment variables (defaults shown):
- `QDRANT_PATH=qdrant_db`
- `QDRANT_COLLECTION=evidencedesk`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `RERANKER_MODEL=BAAI/bge-reranker-large`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_MODEL=qwen2.5:3b`
- `APP_HOST=127.0.0.1`
- `APP_PORT=8000`

See `.env.example` for details. You do not need to configure anything for standard local usage.

## Running
Start the application server:
`python evidencedesk.py`

Then open the application in your browser:
`http://127.0.0.1:8000`

## Uploading Documents
You can upload your own custom security policies.
Supported formats:
- **PDF**: Preserves page numbers and paragraph layout.
- **DOCX**: Extracts heading and hierarchical structure metadata.
- **TXT**: Standard blank-line splitting logic.

Each document is uniquely versioned and isolated into the specified workspace.

## Human Review and Audit Trail
**AI RECOMMENDS · HUMAN DECIDES.** 
The system does not generate conversational final answers. Instead, it extracts the most relevant passages and classifies the state of the evidence. A human reviewer must inspect the highlighted evidence, resolve conflicts, optionally edit the proposed response, provide audit notes, and formally approve or reject the final response.

Every human review decision is securely logged in a SQLite-backed Tamper-Evident Audit Trail (`audit_trail.db`). 
Features of the Audit Log:
- **Persistence across Restarts:** Decisions survive application restarts and are decoupled from the AI evaluation loop.
- **Immutable Historical Context:** Logs capture the precise `document_versions` and `original_ai_response` at the exact time of the review, preserving context even if source documents are updated later.
- **Cryptographic Hash Chaining:** Every log entry is SHA-256 hashed and chained to the previous entry. Using the `verify_chain()` validation, any unauthorized modification of the SQLite database is instantly detected.
- **Workspace Isolation Validation:** Review histories preserve multitenant workspace boundaries structurally.

## Persistence
All indexed documents and vectors are persistently stored in the `qdrant_db/` folder via QdrantLocal. 
**Note:** QdrantLocal uses strict process locking. The local Qdrant storage must not be opened by multiple `QdrantLocal` clients or processes simultaneously. The recommended workflow is to run one `evidencedesk.py` server process and reuse its Qdrant client. Stop the application cleanly before replacing or resetting local storage.

## Testing
Run the complete regression suite using the included test runner:
`python tests/run_all_tests.py`

This test runner correctly manages Qdrant local locks between consecutive test suites and isolates the database.
You can also run specific diagnostics like the CAIQ test:
`python tests/regression/test_day22_caiq.py`

## Evaluation
EvidenceDesk was evaluated against a frozen 25-question security compliance benchmark, split into Development (Q01-Q15) and Held-Out (Q16-Q25) sets.
**End-to-End Performance:**
- **Development (Q01-Q15):** 93.3% accuracy (14/15)
- **Held-Out (Q16-Q25):** 80.0% accuracy (8/10)
- **Overall:** 88.0% accuracy (22/25)
- **Zero Grounding Failures / Zero Malformed Outputs**

The retrieval pipeline achieved 100% correct passage retrieval using the RRF+Cross-Encoder setup. The three remaining E2E mismatches (Q08, Q18, Q23) are SLM classification boundary cases — the 3B model occasionally classifies ambiguous/paraphrased questions as `INSUFFICIENT_EVIDENCE` instead of `ANSWERABLE` or `AMBIGUOUS`. No hallucinations or fabricated evidence were observed. For full metrics, see [docs/final-evaluation.md](docs/final-evaluation.md).

## Security Limitations
- **Workspace Isolation:** Workspace isolation currently operates exclusively at the retrieval layer.
- **Authentication:** There is no authentication. Production deployments should derive workspace identity from authenticated user/session/JWT/OIDC claims instead of trusting an arbitrary client-supplied `workspace_id`.
- **Local Deployment Only:** This is a local-only baseline intended for demonstration and evaluation. Do not expose this server publicly.
