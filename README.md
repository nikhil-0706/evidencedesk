# EvidenceDesk 🛡️

An evidence-backed security questionnaire assistant with retrieval evaluation, provenance tracking, and human review governance. EvidenceDesk provides a robust, deterministic, verifiable pipeline for answering compliance and security questionnaires against company policies.

**AI RECOMMENDS · HUMAN DECIDES.**

---

## 🌐 Live Cloud Deployment

EvidenceDesk is deployed live on Netlify:
👉 **[https://evidencedeskk.netlify.app](https://evidencedeskk.netlify.app)**

- **Landing Page**: [https://evidencedeskk.netlify.app/](https://evidencedeskk.netlify.app/)
- **Human Review Workspace**: [https://evidencedeskk.netlify.app/app](https://evidencedeskk.netlify.app/app)

---

## 🛡️ Project Symbol: The Verifier Shield

The official emblem of EvidenceDesk is **The Verifier Shield**:
- **The Outer Emerald Hex-Shield**: Represents hardened enterprise policy boundaries and workspace isolation (policies never leak across company lines).
- **The Internal Refraction Prism**: Symbolizes decomposing unstructured policy documents into discrete, verifiable facts via Hybrid RRF & Cross-Encoder ranking.
- **The Ruby Beacon & Verification Seal**: Signifies final human sign-off—automated systems never publish compliance answers without a reviewer's cryptographic seal.

---

## Problem
Security questionnaires (SOC 2, ISO 27001, CAIQ) require reviewers to find trustworthy evidence in policy documents. Traditional RAG systems often conflate semantic similarity with factual proof, or they invent facts when data is missing. Reviewers need to see the exact provenance of the evidence, and they need a system that safely abstains or warns when evidence is conflicting or ambiguous.

## Solution
Evidence-backed retrieval + structured reasoning + human review. EvidenceDesk enforces a strict pipeline: it retrieves evidence using hybrid search, uses a local small language model strictly as an evidence classifier rather than an unconstrained generator, and mandates human reviewer sign-off for every answer.

---

## Architecture
```
User Question
→ Multi-Tenant Workspace Filter
→ BM25 (Lexical) + 384-dim Dense Retrieval (Qdrant)
→ Reciprocal Rank Fusion (k=60)
→ Cross-Encoder Deep Reranking (BAAI/bge-reranker-large)
→ Contiguous Adjacent Chunk Expansion
→ Structured SLM Evidence Reasoning (Qwen 2.5 3B via Ollama)
→ 8-Stage Output Validator & Anti-Contamination Guard
→ Human Review Governance (APPROVE / EDIT / REJECT)
→ Cryptographic SHA-256 Chained Audit Trail (audit_trail.db)
```

---

## Features
- **Live Netlify Cloud Deployment**: Instant access with interactive state simulation and review playground.
- **Multi-Document Ingestion**: PDF (page-aware), DOCX (heading hierarchy), and TXT structure-aware chunking.
- **Multi-Tenant Workspace Isolation**: Strict document and chunk boundaries by `workspace_id`.
- **Hybrid Retrieval**: BM25Okapi + 384-dimensional dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Reciprocal Rank Fusion (RRF)**: Deterministic score fusion ($k=60$).
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-large` for deep query-passage semantic attention.
- **Deterministic SLM Decisions**: Classifies evidence strictly into 4 states:
  - `ANSWERABLE`: Direct factual evidence found with verbatim quote extraction.
  - `AMBIGUOUS`: Underspecified question scope requiring clarification.
  - `CONFLICTING`: Two incompatible policy claims identified for human resolution.
  - `INSUFFICIENT_EVIDENCE`: Clean safe abstention (no hallucinations).
- **Tamper-Evident Audit Receipts**: Cryptographic SHA-256 hash chaining on all reviewer sign-offs.

---

## Evaluation Benchmark

EvidenceDesk was evaluated against a frozen 25-question security compliance benchmark:
- **Development Set (Q01–Q15)**: **93.3%** accuracy (14/15)
- **Held-Out Set (Q16–Q25)**: **80.0%** accuracy (8/10)
- **Overall Accuracy**: **88.0%** (22/25)
- **Retrieval Pipeline Accuracy**: **100%** correct passage retrieval
- **Hallucinations / Fabricated Evidence**: **0.0%**

For complete metrics, see [docs/final-evaluation.md](docs/final-evaluation.md).

---

## Local Setup & Engine Execution

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) (for the local SLM)

### 1. Clone & Enter Directory
```bash
git clone https://github.com/nikhil-0706/evidencedesk.git
cd evidencedesk
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Linux/Mac

# Install clean core requirements
pip install -r requirements.txt
```

### 3. Start Ollama and Pull SLM
```powershell
# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull qwen2.5:3b
```

### 4. Run EvidenceDesk
```powershell
python evidencedesk.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser:
- Landing Page: `http://127.0.0.1:8000/`
- Review Workspace: `http://127.0.0.1:8000/app`

---

## Running Tests
Run the complete regression suite:
```powershell
python tests/run_all_tests.py
```

Run smoke test evaluation:
```powershell
python evidencedesk.py --evaluate
```

---

## Deployment Configuration
EvidenceDesk includes pre-configured deployment manifests:
- **Netlify**: [`netlify.toml`](netlify.toml) configured for static hosting & clean SPA routing.
- **Vercel**: [`vercel.json`](vercel.json) configured for instant Vercel edge deployment.

---

## Security Limitations
- **Local Deployment Baseline**: The local Python API is intended for internal, private evaluation.
- **Workspace Authentication**: In production, workspace identity should be derived from authenticated JWT/OIDC claims rather than arbitrary client-supplied `workspace_id`.
