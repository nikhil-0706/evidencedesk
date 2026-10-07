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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import ollama
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
import uuid

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

def analyze_evidence(question, retrieved_evidence):
    evidence_text = "\n".join([f"[{doc['document_id']}] {doc['excerpt']}" for doc in retrieved_evidence])
    
    prompt = f"""You are a security questionnaire review assistant. 
Your job is to strictly classify the interaction based ONLY on the user question and the retrieved evidence.
DO NOT use any external knowledge to generate facts. You MUST use your knowledge to recognize technical synonyms (e.g., 'in transit' means 'moves between systems'). DO NOT generate unsupported answers.

Classify the status as ONE of the following:
1. ANSWERABLE: At least one of the retrieved passages contains the information requested by the user. If the passage explicitly addresses the topic of the question (e.g., using synonyms like 'in transit' for 'moves between systems', or 'restore procedures' for 'verify backups'), it is ANSWERABLE. (Note: if there are multiple passages with contradictory answers to the question, choose CONFLICTING_EVIDENCE instead).
2. AMBIGUOUS: The question itself is underspecified or vague (e.g., asking "What is your retention policy?" when the evidence lists multiple types). Do NOT output AMBIGUOUS if one of the passages provides a direct answer to a reasonable interpretation of the user's question.
3. INSUFFICIENT_EVIDENCE: None of the retrieved passages contain information that addresses the question. Before choosing this, carefully consider if the evidence answers the question using different phrasing (e.g., 'in transit' means 'moves between systems').
4. CONFLICTING_EVIDENCE: Two or more retrieved policy passages specify materially different facts that could both answer the question (e.g., if one says 'backup retention is 30 days' and another says 'database backup retention is 90 days', and the user asks about database backup retention, you MUST choose CONFLICTING_EVIDENCE because the general policy contradicts the specific one).

Output your classification in strict JSON format:
{{
  "status": "<STATUS>",
  "reason": "<Explain briefly why based ONLY on the evidence>",
  "clarification_question": "<If AMBIGUOUS, ask a specific question to clarify. Otherwise omit this field>"
}}

Example 1:
Question: "What is your retention policy?"
Evidence: [SEC-003] Backup retention is 30 days. [SEC-005] Database backup retention is 90 days.
JSON: {{"status": "AMBIGUOUS", "reason": "The question does not specify which type of retention policy is intended.", "clarification_question": "Do you mean database backup retention, customer data retention, or another type of record?"}}

Example 2:
Question: "Are backups retained for 30 days or 90 days?"
Evidence: [SEC-003] Backup retention is 30 days. [SEC-005] Database backup retention is 90 days.
JSON: {{"status": "CONFLICTING_EVIDENCE", "reason": "Two retrieved policy passages specify different backup retention periods."}}

Example 3:
Question: "How often are database backups created?"
Evidence: 
[SEC-003] Database backups are created daily.
[SEC-001] Access permissions are reviewed quarterly.
[SEC-004] Security incidents are triaged by the on-call engineer.
JSON: {{"status": "ANSWERABLE", "reason": "The evidence directly states that database backups are created daily."}}

Example 4:
Question: "How is network traffic secured?"
Evidence: 
[SEC-010] Network traffic is secured using IPsec.
JSON: {{"status": "ANSWERABLE", "reason": "The evidence explicitly states that network traffic is secured using IPsec."}}


Now, classify this interaction:

User Question:
{question}

Retrieved Evidence:
{evidence_text}
"""
    try:
        response = ollama.chat(
            model='qwen2.5:3b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={"temperature": 0.0}
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {
            "status": "ERROR",
            "reason": f"LLM reasoning failed: {str(e)}"
        }
# -----------------------------------

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
        "clarification_question": llm_analysis.get("clarification_question", None),
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
