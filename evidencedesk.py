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
            "status": "review_required" if sufficient else "insufficient_evidence",
            "candidate_excerpt": evidence[0]["excerpt"] if sufficient else None,
            "evidence": evidence,
            "mode": mode,
            "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
        }

    return {
        "question": question,
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
        actual = result["evidence"][0]["document_id"] if result["candidate_excerpt"] else None
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

@app.get("/evaluate")
def evaluation_endpoint():
    return evaluate()

PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EvidenceDesk baseline</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101725;color:#edf2fc;font:16px/1.6 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:40px 22px}header{border-bottom:1px solid #344257;padding-bottom:24px;margin-bottom:24px}.tag{color:#87d9c3;font-size:13px;letter-spacing:2px}h1{font-size:38px;margin:8px 0}p{color:#b8c4d8}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:#192337;border:1px solid #344257;border-radius:16px;padding:24px}textarea{width:100%;min-height:130px;background:#101725;color:white;border:1px solid #5a6c84;border-radius:8px;padding:12px;font:inherit}button{background:#9ae5cf;border:0;border-radius:8px;padding:12px 17px;margin:12px 8px 0 0;font-weight:700;cursor:pointer}button:disabled{opacity:.5}article{border-top:1px solid #344257;padding-top:12px;margin-top:16px}small{color:#9ae5cf}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.note{font-size:13px}.status{font-weight:700;color:#9ae5cf}a{color:#9ae5cf}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:30px}}
</style></head><body><main><header><div class="tag">EVIDENCEDESK / BUILD 0.1</div><h1>Evidence before answers.</h1><p>A security-questionnaire review baseline using synthetic policies.</p><small>Local demo · lexical retrieval · no LLM · no authentication</small></header>
<div class="grid"><section class="panel"><h2>Ask a policy question</h2><label for="question">Question</label><textarea id="question">What encryption protects customer data at rest?</textarea><div style="margin-top:12px; margin-bottom:12px"><strong>Retrieval method:</strong> <label style="margin-right:12px"><input type="radio" name="method" value="lexical"> Lexical</label> <label style="margin-right:12px"><input type="radio" name="method" value="bm25"> BM25</label> <label style="margin-right:12px"><input type="radio" name="method" value="embedding"> Embedding</label> <label><input type="radio" name="method" value="hybrid" checked> Hybrid (RRF)</label></div><button id="review">Find evidence</button><button id="evaluate">Run smoke tests</button><p class="note">Try: “Are you SOC 2 certified?” No generated answer is produced. Relevant excerpts always require human review.</p><a href="/docs">API documentation</a></section><section class="panel" aria-live="polite"><h2>Review workspace</h2><div id="result">Submit a question to inspect the retrieved evidence.</div></section></div>
<section class="panel" style="margin-top:20px"><h2>Synthetic source documents</h2><div id="sources"></div></section></main>
<script>
const el=id=>document.getElementById(id);
function text(parent,tag,value){const node=document.createElement(tag);node.textContent=value;parent.appendChild(node);return node;}
async function request(url,options){const response=await fetch(url,options);if(!response.ok)throw new Error('Request failed: '+response.status+' '+await response.text());return response.json();}
el('review').onclick=async()=>{const button=el('review');button.disabled=true;el('result').textContent='Retrieving…';const method=document.querySelector('input[name="method"]:checked').value;try{const data=await request('/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:el('question').value,method:method})});const target=el('result');target.replaceChildren();text(target,'div',data.status.replaceAll('_',' ')).className='status';if(data.reason){text(target,'p','Reason: '+data.reason).style.fontWeight='bold';}if(data.clarification_question){text(target,'p','Clarification needed: '+data.clarification_question).style.color='#ffb86c';}text(target,'p',data.warning);for(const item of data.evidence){const block=document.createElement('article');text(block,'small','Workspace: '+item.workspace_id+' · Document: '+item.document_id+' · Version: '+item.version+' · Chunk: '+item.chunk_id);text(block,'p',item.excerpt);let scoreText=item.rrf_score!==undefined?'RRF score: '+item.rrf_score:item.bm25_score!==undefined?'BM25 score: '+item.bm25_score:item.lexical_score!==undefined?'Lexical overlap: '+item.lexical_score:'Embedding similarity: '+item.similarity_score;text(block,'small',scoreText+' — not confidence');target.appendChild(block);}}catch(error){el('result').textContent=error.message;}finally{button.disabled=false;}};
el('evaluate').onclick=async()=>{el('evaluate').disabled=true;try{const data=await request('/evaluate');el('result').replaceChildren();text(el('result'),'pre',JSON.stringify(data,null,2));}catch(error){el('result').textContent=error.message;}finally{el('evaluate').disabled=false;}};
request('/documents').then(docs=>{for(const doc of docs){const article=document.createElement('article');text(article,'small',doc.id+' · '+doc.title);text(article,'p',doc.text);el('sources').appendChild(article);}}).catch(error=>{el('sources').textContent=error.message;});
</script></body></html>'''

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        print(json.dumps(evaluate(), indent=2))
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
