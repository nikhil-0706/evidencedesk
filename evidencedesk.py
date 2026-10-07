"""EvidenceDesk — local, deterministic evidence-retrieval baseline.
Python 3.10+
Install: python -m pip install fastapi uvicorn
Run:     python evidencedesk.py
Open:    http://127.0.0.1:8000
Evaluate: python evidencedesk.py --evaluate
No LLM, embeddings, external API, persistence, authentication, or PDF parsing yet.
Synthetic documents only. Do not expose this development server publicly.
"""
import json
import re
import sys
from collections import Counter
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import ollama
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
import uuid


# ---------------------------------------------------------------------------
# Day 15: Structured output schema for SLM evidence classification
# ---------------------------------------------------------------------------
class EvidenceDecision(BaseModel):
    status: Literal[
        "ANSWERABLE",
        "AMBIGUOUS",
        "CONFLICTING",
        "INSUFFICIENT_EVIDENCE",
    ]
    reason: str
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    evidence_quote: str = Field(default="")

DOCUMENTS = [
    {"id": "SEC-001", "title": "Access control policy", "version": "2026-01", "text": "Employees must use multi-factor authentication (MFA) to access production systems. Access permissions are reviewed quarterly. Administrative access requires manager approval.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-002", "title": "Encryption policy", "version": "2026-01", "text": "Customer data is encrypted at rest using AES-256. Data in transit is protected using TLS 1.2 or later. Encryption keys are managed in a dedicated key management service.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-003", "title": "Backup policy", "version": "2026-01", "text": "Database backups are created daily. Backup retention is 30 days. Restore procedures are tested quarterly.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-004", "title": "Incident response policy", "version": "2026-01", "text": "Security incidents are triaged by the on-call engineer. Confirmed incidents are escalated to the security lead. This policy does not specify a contractual customer notification deadline.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-005", "title": "Backup retention exception", "version": "2026-01",  "text": "Database backup retention is 90 days.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-G001", "title": "Globex Production Policy", "version": "2026-01", "text": "Production infrastructure is hosted on ExampleCloud.", "workspace_id": "ws_globex_corp"}
]
# Development fixtures, not a held-out benchmark or evidence of general accuracy.
CASES = [
    {"question": "What encryption protects customer data at rest?", "expected": "SEC-002"},
    {"question": "What is the backup retention period?", "expected": "SEC-003"},
    {"question": "Is multi-factor authentication required for production systems?", "expected": "SEC-001"},
    {"question": "Who triages security incidents?", "expected": "SEC-004"},
    {"question": "Are you SOC 2 certified?", "expected": None},
    {"question": "What is your annual revenue?", "expected": None},
]
STOP = set("a an the is are do does what which how who your you we our for to of in at and or with required".split())
THRESHOLD = 0.25  # Heuristic lexical coverage, NOT calibrated confidence.
app = FastAPI(title="EvidenceDesk baseline", version="0.1.0")

# --- EMBEDDING RETRIEVER (Day 4) ---
print("Loading embedding model (this may take a moment)...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Loading reranker model (this may take a moment)...")
reranker_model = CrossEncoder('BAAI/bge-reranker-large')

PASSAGES = []
for doc in DOCUMENTS:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", doc["text"]) if s.strip()]
    for idx, sentence in enumerate(sentences, start=1):
        PASSAGES.append({
            "chunk_id": f"chk_{doc['id'].lower().replace('-', '')}_{idx:03d}",
            "workspace_id": doc.get("workspace_id", "ws_acme_corp"),
            "document_id": doc["id"],
            "title": doc["title"],
            "version": doc["version"],
            "excerpt": sentence.strip()
        })

print("Initializing Qdrant and indexing policy passages...")
qdrant = QdrantClient(path="qdrant_db")

if not qdrant.collection_exists("evidencedesk"):
    qdrant.create_collection(
        collection_name="evidencedesk",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

points = []
for passage in PASSAGES:
    vector = embed_model.encode(passage["excerpt"]).tolist()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, passage["chunk_id"]))
    points.append(
        models.PointStruct(
            id=point_id,
            vector=vector,
            payload=passage
        )
    )

qdrant.upsert(
    collection_name="evidencedesk",
    points=points
)

def tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP]

passage_texts = [p["excerpt"] for p in PASSAGES]
tokenized_passages = [tokens(p) for p in passage_texts]
bm25_model = BM25Okapi(tokenized_passages)

def retrieve_by_bm25(question, workspace_id, top_k=20):
    query_tokens = tokens(question)
    scores = bm25_model.get_scores(query_tokens)
    ranked = []
    for i, score in enumerate(scores):
        if score > 0:
            passage = PASSAGES[i]
            if passage["workspace_id"] == workspace_id:
                p = passage.copy()
                p["bm25_score"] = round(float(score), 4)
                ranked.append(p)
    return sorted(ranked, key=lambda x: x["bm25_score"], reverse=True)[:top_k]

def retrieve_by_embedding(question, workspace_id, top_k=20):
    query_vector = embed_model.encode(question).tolist()
    search_result = qdrant.query_points(
        collection_name="evidencedesk",
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="workspace_id",
                    match=models.MatchValue(value=workspace_id),
                )
            ]
        ),
        limit=top_k
    )
    
    ranked = []
    for hit in search_result.points:
        passage = hit.payload.copy()
        passage["similarity_score"] = round(float(hit.score), 4)
        ranked.append(passage)
        
    return ranked

def rrf_fuse(bm25_results, embedding_results, k=60, top_k=10):
    rrf_scores = {}
    fused_passages = {}
    
    def make_key(p):
        return p["chunk_id"]
        
    for rank, p in enumerate(bm25_results, start=1):
        key = make_key(p)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        fused_passages[key] = p.copy()
        
    for rank, p in enumerate(embedding_results, start=1):
        key = make_key(p)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in fused_passages:
            fused_passages[key] = p.copy()
            
    fused_list = []
    for key, score in rrf_scores.items():
        passage = fused_passages[key]
        passage["rrf_score"] = round(score, 6)
        fused_list.append(passage)
        
    return sorted(fused_list, key=lambda x: x["rrf_score"], reverse=True)[:top_k]

def rerank_evidence(query, candidates, top_k=5):
    if not candidates:
        return []
    
    pairs = [(query, p["excerpt"]) for p in candidates]
    scores = reranker_model.predict(pairs)
    
    reranked = []
    for i, p in enumerate(candidates):
        passage = p.copy()
        passage["reranker_score"] = round(float(scores[i]), 4)
        reranked.append(passage)
        
    return sorted(reranked, key=lambda x: x["reranker_score"], reverse=True)[:top_k]

# ---------------------------------------------------------------------------
# Day 15: Improved prompt with semantic-equivalence guidance and evidence_quote
# ---------------------------------------------------------------------------
_REASONING_PROMPT = """\
You are an evidence classifier for a security-policy questionnaire.

Use ONLY the supplied evidence passages below.
Do not use outside knowledge.
Do not invent missing information.
Do not generate a compliance answer.

===========================================================
CLASSIFICATION RULES
===========================================================

ANSWERABLE
Return ANSWERABLE when at least one evidence passage directly establishes
the fact requested by the question.
Semantic equivalence is allowed — the question and evidence do NOT need to
use identical words. Focus on whether the evidence establishes the requested fact.

Before returning INSUFFICIENT_EVIDENCE, ask yourself:
"Does any supplied passage establish the core fact requested, even with
different terminology?"
If YES → return ANSWERABLE.

INSUFFICIENT_EVIDENCE
Return INSUFFICIENT_EVIDENCE only when the supplied evidence genuinely does
not establish the requested fact.

AMBIGUOUS
Return AMBIGUOUS when the question itself is underspecified and clarification
is required before determining which policy applies.

CONFLICTING
Return CONFLICTING when the question is specific but the supplied evidence
gives materially different answers for the same requested fact.

===========================================================
AMBIGUOUS vs CONFLICTING
===========================================================

AMBIGUOUS = the QUESTION is underspecified.
CONFLICTING = the QUESTION is specific but evidence disagrees.

===========================================================
SEMANTIC EQUIVALENCE — KEY EXAMPLES
===========================================================

These pairs of phrases are semantically equivalent and MUST be treated as ANSWERABLE:

  "Who takes the first look at a reported security incident?"
  + "Security incidents are triaged by the on-call engineer."
  → ANSWERABLE  ("first look" ≈ "triage"; "on-call engineer" is the role)

  "Where are encryption keys managed?"
  + "Encryption keys are managed in a dedicated key management service."
  → ANSWERABLE  (the evidence explicitly names the management location)

  "Who gives permission for administrative privileges?"
  + "Administrative access requires manager approval."
  → ANSWERABLE  ("gives permission" ≈ "requires approval")

Do not require exact keyword matching.
Do not reject an answer merely because the evidence uses a related operational
term rather than the exact wording of the question.

===========================================================
FEW-SHOT EXAMPLES
===========================================================

Example 1 — ANSWERABLE (direct)

Question: Who approves administrative access?
Evidence: [chk_sec001_003] Administrative access requires manager approval.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly establishes who approves administrative access.",
  "evidence_chunk_ids": ["chk_sec001_003"],
  "evidence_quote": "Administrative access requires manager approval."
}}


Example 2 — ANSWERABLE (paraphrase: 'first look' = 'triage')

Question: Who takes the first look at a reported security incident?
Evidence: [chk_sec004_001] Security incidents are triaged by the on-call engineer.
         [chk_sec004_002] Confirmed incidents are escalated to the security lead.

{{
  "status": "ANSWERABLE",
  "reason": "Triage is the initial handling of a security incident. The on-call engineer performs the first look.",
  "evidence_chunk_ids": ["chk_sec004_001"],
  "evidence_quote": "Security incidents are triaged by the on-call engineer."
}}


Example 3 — ANSWERABLE (paraphrase: 'managed' in KMS = location)

Question: Where are encryption keys managed?
Evidence: [chk_sec002_003] Encryption keys are managed in a dedicated key management service.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence explicitly identifies where encryption keys are managed.",
  "evidence_chunk_ids": ["chk_sec002_003"],
  "evidence_quote": "Encryption keys are managed in a dedicated key management service."
}}


Example 4 — ANSWERABLE (data in transit)

Question: How is data in transit protected?
Evidence: [chk_sec002_002] Data in transit is protected using TLS 1.2 or later.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly states how data in transit is protected.",
  "evidence_chunk_ids": ["chk_sec002_002"],
  "evidence_quote": "Data in transit is protected using TLS 1.2 or later."
}}


Example 5 — AMBIGUOUS (retention policy — unspecified scope)

Question: What is your retention policy?
Evidence: [chk_sec003_002] Backup retention is 30 days.
         [chk_sec005_001] Database backup retention is 90 days.

{{
  "status": "AMBIGUOUS",
  "reason": "The question does not specify which type of retention policy or which records are being asked about.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}


Example 6 — CONFLICTING (database backup retention — specific question, different values)

Question: How long are database backups retained?
Evidence: [chk_sec003_002] Backup retention is 30 days.
         [chk_sec005_001] Database backup retention is 90 days.

{{
  "status": "CONFLICTING",
  "reason": "The evidence gives two different retention periods for database backups.",
  "evidence_chunk_ids": ["chk_sec003_002", "chk_sec005_001"],
  "evidence_quote": ""
}}


Example 7 — INSUFFICIENT_EVIDENCE (SOC 2 not in corpus)

Question: Is the company SOC 2 certified?
Evidence: [chk_sec001_001] Employees must use multi-factor authentication.
         [chk_sec003_001] Database backups are created daily.

{{
  "status": "INSUFFICIENT_EVIDENCE",
  "reason": "The evidence does not establish whether the company is SOC 2 certified.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}


===========================================================
NOW CLASSIFY THE CURRENT REQUEST
===========================================================

Question:
{question}

Evidence:
{evidence_text}

Return ONLY valid JSON matching this schema exactly:
{{
  "status": "ANSWERABLE | AMBIGUOUS | INSUFFICIENT_EVIDENCE | CONFLICTING",
  "reason": "short explanation based only on the evidence",
  "evidence_chunk_ids": ["chunk_id_1"],
  "evidence_quote": "exact verbatim quote from supplied evidence if ANSWERABLE, else empty string"
}}
"""


def _validate_slm_output(
    raw: dict,
    retrieved_evidence: list,
) -> dict:
    """
    Validate and sanitise SLM output against EvidenceDecision schema.

    Returns a sanitised dict.  Never raises — always returns a safe dict
    whose 'status' is one of the four allowed values or 'VALIDATION_ERROR'.
    """
    ALLOWED_STATUSES = {"ANSWERABLE", "AMBIGUOUS", "CONFLICTING", "INSUFFICIENT_EVIDENCE"}
    valid_chunk_ids = {p["chunk_id"] for p in retrieved_evidence}
    all_excerpts = [p["excerpt"] for p in retrieved_evidence]

    # 1. Pydantic schema validation
    try:
        decision = EvidenceDecision(**raw)
    except (ValidationError, TypeError) as exc:
        return {
            "status": "VALIDATION_ERROR",
            "reason": f"Schema validation failed: {exc}",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "_validation_error": True,
        }

    # 2. Status must be one of the four allowed values (Pydantic already checks, but be explicit)
    if decision.status not in ALLOWED_STATUSES:
        return {
            "status": "VALIDATION_ERROR",
            "reason": f"Invalid status value: {decision.status!r}",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "_validation_error": True,
        }

    # 3. Filter chunk_ids to only those actually in retrieved evidence
    safe_chunk_ids = [cid for cid in decision.evidence_chunk_ids if cid in valid_chunk_ids]
    if decision.evidence_chunk_ids and not safe_chunk_ids:
        # All claimed chunk IDs were unknown — downgrade to safe state
        return {
            "status": "VALIDATION_ERROR",
            "reason": (
                f"All returned chunk_ids {decision.evidence_chunk_ids!r} "
                "are not in the retrieved evidence set."
            ),
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "_validation_error": True,
        }

    # 4. ANSWERABLE must have a non-empty evidence_quote
    if decision.status == "ANSWERABLE" and not decision.evidence_quote.strip():
        # Treat as validation failure — do not silently accept
        return {
            "status": "VALIDATION_ERROR",
            "reason": "ANSWERABLE classification requires a non-empty evidence_quote.",
            "evidence_chunk_ids": safe_chunk_ids,
            "evidence_quote": "",
            "_validation_error": True,
        }

    # 5. evidence_quote must appear in the supplied evidence (verbatim substring check)
    if decision.status == "ANSWERABLE" and decision.evidence_quote.strip():
        quote = decision.evidence_quote.strip()
        quote_found = any(quote in excerpt for excerpt in all_excerpts)
        if not quote_found:
            return {
                "status": "VALIDATION_ERROR",
                "reason": (
                    f"evidence_quote not found in supplied evidence: {quote!r}"
                ),
                "evidence_chunk_ids": safe_chunk_ids,
                "evidence_quote": "",
                "_validation_error": True,
            }

    return {
        "status": decision.status,
        "reason": decision.reason,
        "evidence_chunk_ids": safe_chunk_ids,
        "evidence_quote": decision.evidence_quote,
    }


def analyze_evidence(question: str, retrieved_evidence: list) -> dict:
    """
    Call the local SLM to classify whether the retrieved evidence answers the
    question, then validate and sanitise the structured JSON output.

    Returns a dict with keys: status, reason, evidence_chunk_ids, evidence_quote.
    Never raises — always returns a safe dict.
    """
    if not retrieved_evidence:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "No evidence was retrieved.",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
        }

    evidence_text = "\n".join(
        [f"[{doc['chunk_id']}] {doc['excerpt']}" for doc in retrieved_evidence]
    )
    prompt = _REASONING_PROMPT.format(question=question, evidence_text=evidence_text)

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        raw = json.loads(response["message"]["content"])
    except json.JSONDecodeError as exc:
        return {
            "status": "VALIDATION_ERROR",
            "reason": f"SLM returned malformed JSON: {exc}",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "_validation_error": True,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "reason": f"LLM call failed: {exc}",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
        }

    return _validate_slm_output(raw, retrieved_evidence)
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    workspace_id: str = Field(default="ws_acme_corp")
    ai_status: str
    decision: Literal["APPROVE", "EDIT", "REJECT"]
    edited_response: Optional[str] = None
    reviewer_notes: Optional[str] = None
    selected_chunk_ids: List[str] = Field(default_factory=list)

class ReviewRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    method: str = Field(default="hybrid")
    workspace_id: str = Field(default="ws_acme_corp")

def retrieve(question, workspace_id):
    query = set(tokens(question))
    ranked = []
    for passage in PASSAGES:
        if passage["workspace_id"] != workspace_id:
            continue
        words = Counter(tokens(passage["excerpt"]))
        matches = query.intersection(words)
        score = len(matches) / max(len(query), 1)
        if matches:
            p = passage.copy()
            p["lexical_score"] = round(score, 4)
            ranked.append(p)
    return sorted(ranked, key=lambda item: item["lexical_score"], reverse=True)[:3]

def review(question, method="hybrid", workspace_id="ws_acme_corp", debug=False):
    if method == "hybrid":
        bm25_results = retrieve_by_bm25(question, workspace_id, top_k=20)
        emb_results = retrieve_by_embedding(question, workspace_id, top_k=20)
        rrf_candidates = rrf_fuse(bm25_results, emb_results, k=60, top_k=10)
        evidence = rerank_evidence(question, rrf_candidates, top_k=5)
        
        if debug:
            print("\n=== RRF TOP-10 ===")
            for i, p in enumerate(rrf_candidates, 1):
                print(f"rank={i} chunk_id={p['chunk_id']} document_id={p['document_id']} rrf_score={p.get('rrf_score')} excerpt=\"{p['excerpt']}\"")
                
            print("\n=== CROSS-ENCODER TOP-5 ===")
            for i, p in enumerate(evidence, 1):
                print(f"rank={i} chunk_id={p['chunk_id']} document_id={p['document_id']} reranker_score={p.get('reranker_score')} rrf_score={p.get('rrf_score')} excerpt=\"{p['excerpt']}\"")
                
            evidence_text = "\n".join([f"[{doc['document_id']}] {doc['excerpt']}" for doc in evidence])
            print(f"\n=== FINAL REASONING INPUT ===\nQuestion: {question}\nEvidence:\n{evidence_text}")
            
        llm_analysis = analyze_evidence(question, evidence)
        
        if debug:
            print("\n=== FINAL CLASSIFICATION ===")
            print(json.dumps(llm_analysis, indent=2))
            
        mode = "hybrid_rrf_reranked_reasoning"
    elif method == "bm25":
        evidence = retrieve_by_bm25(question, workspace_id, top_k=3)
        llm_analysis = analyze_evidence(question, evidence)
        mode = "bm25_reasoning"
    elif method == "embedding":
        evidence = retrieve_by_embedding(question, workspace_id, top_k=3)
        llm_analysis = analyze_evidence(question, evidence)
        mode = "embedding_with_reasoning"
    else:
        evidence = retrieve(question, workspace_id)
        sufficient = bool(evidence and evidence[0]["lexical_score"] >= THRESHOLD)
        mode = "deterministic_lexical_baseline"

        return {
            "question": question,
            "workspace_id": workspace_id,
            "status": "review_required" if sufficient else "insufficient_evidence",
            "candidate_excerpt": evidence[0]["excerpt"] if sufficient else None,
            "evidence": evidence,
            "mode": mode,
            "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
        }

    return {
        "question": question,
        "workspace_id": workspace_id,
        "status": llm_analysis.get("status", "review_required"),
        "candidate_excerpt": evidence[0]["excerpt"] if evidence else None,
        "evidence": evidence,
        "mode": mode,
        "reason": llm_analysis.get("reason", ""),
        "evidence_chunk_ids": llm_analysis.get("evidence_chunk_ids", []),
        "evidence_quote": llm_analysis.get("evidence_quote", ""),
        "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
    }

def evaluate():
    rows = []
    for case in CASES:
        result = review(case["question"])
        if result.get("status") == "INSUFFICIENT_EVIDENCE":
            actual = None
        else:
            actual = result["evidence"][0]["document_id"] if (result.get("evidence") and result.get("candidate_excerpt")) else None
        rows.append({**case, "actual": actual, "pass": actual == case["expected"]})
    return {"suite": "synthetic development smoke tests", "passed": sum(row["pass"] for row in rows),
            "total": len(rows), "cases": rows,
            "limitation": "Checks top document or abstention only; does not measure semantic correctness, security, or production readiness."}

@app.get("/health")
def health():
    return {"status": "ok", "mode": "local_baseline"}

@app.get("/documents")
def documents():
    return DOCUMENTS

@app.post("/review")
def review_endpoint(body: ReviewRequest):
    if len(body.question.strip()) < 3:
        raise HTTPException(status_code=422, detail="Enter at least three non-whitespace characters.")
    return review(body.question.strip(), body.method, body.workspace_id)

@app.post("/review/decision")
def decision_endpoint(body: DecisionRequest):
    if body.decision == "EDIT" and not (body.edited_response or "").strip():
        raise HTTPException(status_code=422, detail="Edited response required when decision is EDIT.")
    import datetime
    return {
        "status": "DECISION_RECORDED",
        "decision": body.decision,
        "question": body.question,
        "workspace_id": body.workspace_id,
        "ai_status": body.ai_status,
        "final_response": body.edited_response.strip() if body.decision == "EDIT" else (body.edited_response or ""),
        "reviewer_notes": (body.reviewer_notes or "").strip(),
        "selected_chunk_ids": body.selected_chunk_ids,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.get("/evaluate")
def evaluation_endpoint():
    return evaluate()

PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EvidenceDesk — Human Review Workspace</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box}
body{margin:0;background:#0b0f19;color:#e2e8f0;font-family:'Inter',system-ui,sans-serif;line-height:1.5}
main{max-width:1200px;margin:auto;padding:32px 20px}
header{border-bottom:1px solid #1e293b;padding-bottom:20px;margin-bottom:28px;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px}
.brand-tag{color:#10b981;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase}
h1{font-size:32px;font-weight:700;margin:4px 0 0 0;color:#f8fafc;letter-spacing:-0.5px}
.subtitle{color:#94a3b8;font-size:14px;margin-top:4px}
.gov-banner{background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#34d399;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;letter-spacing:0.5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.panel{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:24px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.3)}
.panel h2{font-size:18px;font-weight:600;margin:0 0 16px 0;color:#f1f5f9;display:flex;align-items:center;gap:8px}
label{display:block;font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px}
select,textarea,input[type="text"]{width:100%;background:#090d16;color:#f8fafc;border:1px solid #334155;border-radius:8px;padding:10px 12px;font-family:inherit;font-size:14px}
select:focus,textarea:focus,input[type="text"]:focus{outline:none;border-color:#10b981}
textarea{min-height:100px;resize:vertical}
.presets{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.preset-btn{background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;transition:all 0.15s}
.preset-btn:hover{background:#334155;color:#fff}
.method-group{display:flex;gap:12px;margin:12px 0 16px 0;font-size:13px;color:#cbd5e1}
.btn-primary{background:#10b981;color:#042f2e;border:0;border-radius:8px;padding:12px 20px;font-weight:700;font-size:14px;cursor:pointer;transition:background 0.15s}
.btn-primary:hover{background:#34d399}
.btn-primary:disabled{opacity:0.5;cursor:not-allowed}
.btn-secondary{background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:12px 16px;font-weight:600;font-size:14px;cursor:pointer;margin-left:8px}
.btn-secondary:hover{background:#334155}

/* AI Status Badges */
.badge{display:inline-flex;align-items:center;padding:4px 12px;border-radius:6px;font-size:13px;font-weight:700;letter-spacing:0.5px}
.badge-ANSWERABLE{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
.badge-AMBIGUOUS{background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
.badge-INSUFFICIENT_EVIDENCE{background:rgba(139,92,246,0.15);color:#c084fc;border:1px solid rgba(139,92,246,0.3)}
.badge-CONFLICTING{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3)}
.badge-VALIDATION_ERROR{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3)}

.box{background:#090d16;border:1px solid #1e293b;border-radius:8px;padding:14px;margin-top:12px}
.box-title{font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
.quote-box{border-left:3px solid #10b981;background:rgba(16,185,129,0.05);color:#a7f3d0;font-style:italic}
.alert-box{border-left:3px solid #f59e0b;background:rgba(245,158,11,0.05);color:#fcd34d}
.conflict-box{border-left:3px solid #ef4444;background:rgba(239,68,68,0.05);color:#fca5a5}
.purple-box{border-left:3px solid #8b5cf6;background:rgba(139,92,246,0.05);color:#ddd6fe}

/* Evidence list cards */
.evidence-card{border-top:1px solid #1e293b;padding-top:14px;margin-top:14px}
.provenance-tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.tag-chip{background:#1e293b;color:#94a3b8;font-size:11px;padding:2px 8px;border-radius:4px;font-family:monospace}
.tag-chip.highlight{background:rgba(16,185,129,0.2);color:#34d399}
.score-signal{font-size:11px;color:#64748b;margin-top:6px;font-style:italic}

/* Human governance actions */
.decision-group{display:flex;gap:10px;margin:14px 0}
.dec-btn{flex:1;padding:10px;border-radius:8px;font-weight:700;font-size:13px;cursor:pointer;border:1px solid #334155;background:#1e293b;color:#94a3b8;transition:all 0.15s;text-align:center}
.dec-btn.active-approve{background:#065f46;color:#a7f3d0;border-color:#10b981}
.dec-btn.active-edit{background:#78350f;color:#fef3c7;border-color:#f59e0b}
.dec-btn.active-reject{background:#7f1d1d;color:#fecaca;border-color:#ef4444}

.receipt-card{background:rgba(15,23,42,0.8);border:1px solid #334155;border-radius:8px;padding:14px;margin-top:16px}
</style>
</head>
<body>
<main>
<header>
  <div>
    <div class="brand-tag">EVIDENCEDESK / BUILD 0.19</div>
    <h1>Human Review Workspace</h1>
    <div class="subtitle">Evidence-backed security questionnaire assistant with human governance</div>
  </div>
  <div class="gov-banner">AI RECOMMENDS · HUMAN DECIDES</div>
</header>

<div class="grid">
  <!-- LEFT COLUMN: INPUT & PIPELINE CONTROL -->
  <section class="panel">
    <h2>1. Questionnaire Input</h2>
    
    <label for="workspace_select">Target Workspace Context</label>
    <select id="workspace_select">
      <option value="ws_acme_corp" selected>ws_acme_corp (Acme Corporation)</option>
      <option value="ws_globex_corp">ws_globex_corp (Globex Corporation)</option>
    </select>

    <div style="margin-top:14px">
      <label for="question">Security Questionnaire Question</label>
      <textarea id="question">Who takes the first look at a reported security incident?</textarea>
      
      <div class="presets">
        <button class="preset-btn" onclick="setQ('Who takes the first look at a reported security incident?', 'ws_acme_corp')">Incident Triage (Q21)</button>
        <button class="preset-btn" onclick="setQ('What safeguards customer information while it moves between systems?', 'ws_acme_corp')">Data in Transit (Q02)</button>
        <button class="preset-btn" onclick="setQ('What is your retention policy?', 'ws_acme_corp')">Retention (Q13 - AMBIGUOUS)</button>
        <button class="preset-btn" onclick="setQ('How long are database backups retained?', 'ws_acme_corp')">DB Retention (Q15 - CONFLICTING)</button>
        <button class="preset-btn" onclick="setQ('What is your SLA for resolving high-severity security incidents?', 'ws_acme_corp')">SLA (Q12 - INSUFFICIENT)</button>
        <button class="preset-btn" onclick="setQ('Where is production infrastructure hosted?', 'ws_globex_corp')">Globex Infra (Cross-WS)</button>
      </div>
    </div>

    <div style="margin-top:16px">
      <label>Retrieval & Reranking Pipeline</label>
      <div class="method-group">
        <label><input type="radio" name="method" value="hybrid" checked> Hybrid (BM25 + Dense + RRF + Reranker)</label>
        <label><input type="radio" name="method" value="bm25"> BM25 Only</label>
        <label><input type="radio" name="method" value="embedding"> Dense Only</label>
      </div>
    </div>

    <button id="btn_run" class="btn-primary">Run Evidence Pipeline</button>
    <button id="btn_evaluate" class="btn-secondary">Run Development Tests</button>

    <div style="margin-top:20px; font-size:12px; color:#64748b">
      <strong>Architecture Status:</strong> Retrieval & Reranker FROZEN · SLM Reasoning FROZEN · Provenance Enabled
    </div>
  </section>

  <!-- RIGHT COLUMN: AI RECOMMENDATION & HUMAN DECISION -->
  <section class="panel" aria-live="polite">
    <h2>2. AI Recommendation & Human Governance</h2>
    <div id="result">
      <div style="color:#64748b; padding:20px 0; text-align:center">
        Submit a security questionnaire question to run retrieval, reranking, and SLM evidence reasoning.
      </div>
    </div>
  </section>
</div>

<section class="panel" style="margin-top:24px">
  <h2>Synthetic Corpus Documents (Workspace Provenance)</h2>
  <div id="sources" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:12px"></div>
</section>
</main>

<script>
const el = id => document.getElementById(id);

function setQ(q, ws) {
  el('question').value = q;
  if(ws) el('workspace_select').value = ws;
}

async function request(url, options) {
  const response = await fetch(url, options);
  if(!response.ok) throw new Error('API Error: ' + response.status + ' ' + await response.text());
  return response.json();
}

let currentReviewData = null;
let selectedDecision = 'APPROVE';

el('btn_run').onclick = async () => {
  const btn = el('btn_run');
  btn.disabled = true;
  el('result').innerHTML = '<div style="color:#34d399; font-weight:600; padding:20px 0">Executing retrieval, RRF fusion, Cross-Encoder reranking, and SLM evidence reasoning...</div>';

  const method = document.querySelector('input[name="method"]:checked').value;
  const workspace_id = el('workspace_select').value;
  const question = el('question').value.trim();

  try {
    const data = await request('/review', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, method, workspace_id})
    });
    currentReviewData = data;
    renderReviewWorkspace(data);
  } catch(err) {
    el('result').innerHTML = '<div style="color:#ef4444">Error: ' + err.message + '</div>';
  } finally {
    btn.disabled = false;
  }
};

function renderReviewWorkspace(data) {
  const target = el('result');
  target.replaceChildren();

  // 1. Status Badge
  const statusDiv = document.createElement('div');
  statusDiv.style.display = 'flex';
  statusDiv.style.alignItems = 'center';
  statusDiv.style.justifyContent = 'space-between';
  statusDiv.style.marginBottom = '12px';

  const badge = document.createElement('span');
  badge.className = 'badge badge-' + data.status;
  badge.textContent = data.status.replace(/_/g, ' ');
  statusDiv.appendChild(badge);

  const wsLabel = document.createElement('span');
  wsLabel.style.fontSize = '12px';
  wsLabel.style.color = '#94a3b8';
  wsLabel.textContent = 'Workspace: ' + (data.workspace_id || 'ws_acme_corp');
  statusDiv.appendChild(wsLabel);
  target.appendChild(statusDiv);

  // 2. AI Reasoning Box
  if (data.reason) {
    const reasonBox = document.createElement('div');
    reasonBox.className = 'box';
    reasonBox.innerHTML = '<div class="box-title">AI Evidence Reasoning</div><div>' + escapeHtml(data.reason) + '</div>';
    target.appendChild(reasonBox);
  }

  // 3. Verbatim Evidence Quote Box (if ANSWERABLE)
  if (data.status === 'ANSWERABLE' && data.evidence_quote) {
    const quoteBox = document.createElement('div');
    quoteBox.className = 'box quote-box';
    quoteBox.innerHTML = '<div class="box-title" style="color:#34d399">Verbatim Supporting Quote</div><div>"' + escapeHtml(data.evidence_quote) + '"</div>';
    target.appendChild(quoteBox);
  }

  // 4. Special State Notice Cards
  if (data.status === 'AMBIGUOUS') {
    const alertBox = document.createElement('div');
    alertBox.className = 'box alert-box';
    alertBox.innerHTML = '<div class="box-title" style="color:#fbbf24">Underspecified Question Notice</div><div>Question scope is ambiguous. Clarification required before selecting policy response.</div>';
    target.appendChild(alertBox);
  } else if (data.status === 'INSUFFICIENT_EVIDENCE') {
    const purpleBox = document.createElement('div');
    purpleBox.className = 'box purple-box';
    purpleBox.innerHTML = '<div class="box-title" style="color:#c084fc">Safe Abstention Notice</div><div>The corpus contains no evidence to establish this claim. AI has abstained. Do not invent missing facts.</div>';
    target.appendChild(purpleBox);
  } else if (data.status === 'CONFLICTING') {
    const conflictBox = document.createElement('div');
    conflictBox.className = 'box conflict-box';
    conflictBox.innerHTML = '<div class="box-title" style="color:#f87171">Conflicting Evidence Warning</div><div>Multiple policy statements give differing values for this fact. Human reviewer must resolve conflict.</div>';
    target.appendChild(conflictBox);
  }

  // 5. Supporting Evidence List (Provenance Explorer)
  const evHeader = document.createElement('div');
  evHeader.style.fontSize = '14px';
  evHeader.style.fontWeight = '700';
  evHeader.style.marginTop = '20px';
  evHeader.style.marginBottom = '8px';
  evHeader.style.color = '#f1f5f9';
  evHeader.textContent = 'Supporting Evidence Chunks (' + (data.evidence ? data.evidence.length : 0) + ')';
  target.appendChild(evHeader);

  if (data.evidence && data.evidence.length > 0) {
    for (const item of data.evidence) {
      const card = document.createElement('div');
      card.className = 'evidence-card';

      const isQuoteMatch = data.evidence_quote && item.excerpt.includes(data.evidence_quote);
      
      let tagsHtml = `
        <div class="provenance-tags">
          <span class="tag-chip ${isQuoteMatch ? 'highlight' : ''}">${escapeHtml(item.chunk_id)}</span>
          <span class="tag-chip">Doc: ${escapeHtml(item.document_id)}</span>
          <span class="tag-chip">Version: ${escapeHtml(item.version)}</span>
          <span class="tag-chip">Workspace: ${escapeHtml(item.workspace_id)}</span>
        </div>
      `;

      let scoreText = '';
      if (item.rrf_score !== undefined && item.reranker_score !== undefined) {
        scoreText = `RRF score: ${item.rrf_score} · Reranker score: ${item.reranker_score}`;
      } else if (item.bm25_score !== undefined) {
        scoreText = `BM25 score: ${item.bm25_score}`;
      } else if (item.similarity_score !== undefined) {
        scoreText = `Embedding similarity: ${item.similarity_score}`;
      } else if (item.lexical_score !== undefined) {
        scoreText = `Lexical score: ${item.lexical_score}`;
      }

      card.innerHTML = tagsHtml +
        '<div style="color:#e2e8f0; font-size:13px; margin:4px 0">' + escapeHtml(item.excerpt) + '</div>' +
        '<div class="score-signal">' + scoreText + ' — ranking signals (not confidence scores)</div>';

      target.appendChild(card);
    }
  }

  // 6. Human Review Governance Controls
  const govPanel = document.createElement('div');
  govPanel.style.marginTop = '24px';
  govPanel.style.paddingTop = '16px';
  govPanel.style.borderTop = '2px solid #1e293b';

  govPanel.innerHTML = `
    <div style="font-size:14px; font-weight:700; color:#f1f5f9; margin-bottom:10px">
      3. Human Reviewer Governance Action
    </div>
    
    <div class="decision-group">
      <div id="btn_dec_approve" class="dec-btn active-approve" onclick="selectDecision('APPROVE')">APPROVE</div>
      <div id="btn_dec_edit" class="dec-btn" onclick="selectDecision('EDIT')">EDIT RESPONSE</div>
      <div id="btn_dec_reject" class="dec-btn" onclick="selectDecision('REJECT')">REJECT</div>
    </div>

    <div id="edit_box" style="display:none; margin-bottom:12px">
      <label for="edited_text">Customized Compliance Answer / Proposed Text</label>
      <textarea id="edited_text">${escapeHtml(data.evidence_quote || data.candidate_excerpt || '')}</textarea>
    </div>

    <div style="margin-bottom:14px">
      <label for="reviewer_notes">Compliance Reviewer Audit Notes</label>
      <textarea id="reviewer_notes" placeholder="Enter compliance reviewer justification, policy notes, or manual override reason..."></textarea>
    </div>

    <button id="btn_submit_dec" class="btn-primary" style="width:100%" onclick="submitDecision()">Record Final Reviewer Decision</button>
    <div id="receipt_container"></div>
  `;

  target.appendChild(govPanel);
}

function selectDecision(type) {
  selectedDecision = type;
  const btnApprove = el('btn_dec_approve');
  const btnEdit = el('btn_dec_edit');
  const btnReject = el('btn_dec_reject');
  const editBox = el('edit_box');

  btnApprove.className = 'dec-btn' + (type === 'APPROVE' ? ' active-approve' : '');
  btnEdit.className = 'dec-btn' + (type === 'EDIT' ? ' active-edit' : '');
  btnReject.className = 'dec-btn' + (type === 'REJECT' ? ' active-reject' : '');

  if (editBox) {
    editBox.style.display = type === 'EDIT' ? 'block' : 'none';
  }
}

async function submitDecision() {
  if (!currentReviewData) return;

  const notes = el('reviewer_notes') ? el('reviewer_notes').value : '';
  const editedText = el('edited_text') ? el('edited_text').value : '';

  const payload = {
    question: currentReviewData.question,
    workspace_id: currentReviewData.workspace_id || el('workspace_select').value,
    ai_status: currentReviewData.status,
    decision: selectedDecision,
    edited_response: editedText,
    reviewer_notes: notes,
    selected_chunk_ids: currentReviewData.evidence_chunk_ids || []
  };

  try {
    const receipt = await request('/review/decision', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    const container = el('receipt_container');
    container.innerHTML = `
      <div class="receipt-card">
        <div style="color:#10b981; font-weight:700; font-size:13px; margin-bottom:6px">
          DECISION RECORDED · ${receipt.decision}
        </div>
        <div style="font-size:12px; color:#94a3b8">
          Timestamp: ${receipt.timestamp}<br>
          Question: ${escapeHtml(receipt.question)}<br>
          AI Status: ${receipt.ai_status}<br>
          Reviewer Notes: ${escapeHtml(receipt.reviewer_notes || 'None')}<br>
          ${receipt.decision === 'EDIT' ? 'Approved Text: "' + escapeHtml(receipt.final_response) + '"' : ''}
        </div>
      </div>
    `;
  } catch(err) {
    alert('Failed to record decision: ' + err.message);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

el('btn_evaluate').onclick = async () => {
  const btn = el('btn_evaluate');
  btn.disabled = true;
  try {
    const data = await request('/evaluate');
    el('result').innerHTML = '<pre style="background:#090d16; padding:16px; border-radius:8px; color:#34d399">' + JSON.stringify(data, null, 2) + '</pre>';
  } catch(err) {
    el('result').innerHTML = '<div style="color:#ef4444">' + err.message + '</div>';
  } finally {
    btn.disabled = false;
  }
};

request('/documents').then(docs => {
  const container = el('sources');
  container.replaceChildren();
  for(const doc of docs) {
    const art = document.createElement('article');
    art.style.background = '#090d16';
    art.style.padding = '12px';
    art.style.borderRadius = '8px';
    art.style.border = '1px solid #1e293b';
    art.innerHTML = '<div style="font-size:11px; color:#10b981; font-weight:700">' + escapeHtml(doc.id) + ' · ' + escapeHtml(doc.title) + ' (v' + escapeHtml(doc.version) + ') [' + escapeHtml(doc.workspace_id) + ']</div><div style="font-size:12px; color:#cbd5e1; margin-top:4px">' + escapeHtml(doc.text) + '</div>';
    container.appendChild(art);
  }
}).catch(err => {
  el('sources').textContent = err.message;
});
</script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        print(json.dumps(evaluate(), indent=2))
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

