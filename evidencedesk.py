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
from sentence_transformers import SentenceTransformer, util
import ollama

DOCUMENTS = [
    {"id": "SEC-001", "title": "Access control policy", "version": "2026-01", "text": "Employees must use multi-factor authentication (MFA) to access production systems. Access permissions are reviewed quarterly. Administrative access requires manager approval."},
    {"id": "SEC-002", "title": "Encryption policy", "version": "2026-01", "text": "Customer data is encrypted at rest using AES-256. Data in transit is protected using TLS 1.2 or later. Encryption keys are managed in a dedicated key management service."},
    {"id": "SEC-003", "title": "Backup policy", "version": "2026-01", "text": "Database backups are created daily. Backup retention is 30 days. Restore procedures are tested quarterly."},
    {"id": "SEC-004", "title": "Incident response policy", "version": "2026-01", "text": "Security incidents are triaged by the on-call engineer. Confirmed incidents are escalated to the security lead. This policy does not specify a contractual customer notification deadline."},
    {"id": "SEC-005", "title": "Backup retention exception", "version": "2026-01",  "text": "Database backup retention is 90 days." },
       
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

PASSAGES = []
for doc in DOCUMENTS:
    for sentence in re.split(r"(?<=[.!?])\s+", doc["text"]):
        if sentence.strip():
            PASSAGES.append({
                "document_id": doc["id"],
                "title": doc["title"],
                "version": doc["version"],
                "excerpt": sentence.strip()
            })

print("Encoding policy passages...")
passage_texts = [p["excerpt"] for p in PASSAGES]
passage_embeddings = embed_model.encode(passage_texts, convert_to_tensor=True)

def retrieve_by_embedding(question):
    query_embedding = embed_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, passage_embeddings)[0]
    
    ranked = []
    for i, score in enumerate(scores):
        passage = PASSAGES[i].copy()
        passage["similarity_score"] = round(float(score), 4)
        ranked.append(passage)
        
    return sorted(ranked, key=lambda x: x["similarity_score"], reverse=True)[:3]

def analyze_evidence(question, retrieved_evidence):
    evidence_text = "\n".join([f"[{doc['document_id']}] {doc['excerpt']}" for doc in retrieved_evidence])
    
    prompt = f"""You are a security questionnaire review assistant. 
Your job is to strictly classify the interaction based ONLY on the user question and the retrieved evidence.
DO NOT use any external knowledge. DO NOT generate unsupported answers.

Classify the status as ONE of the following:
1. ANSWERABLE: The retrieved evidence directly and clearly addresses the question. (Note: If the evidence contains the answer AND some extra information, it is still ANSWERABLE, not ambiguous.)
2. AMBIGUOUS: The question itself is underspecified or vague (e.g. if the user asks "What is your retention policy?" but there are multiple types of retention, or it's not clear which they mean).
3. INSUFFICIENT_EVIDENCE: The retrieved passages do not contain evidence supporting the requested fact.
4. CONFLICTING_EVIDENCE: Two or more retrieved policy passages specify contradictory facts (e.g. one says 30 days, another says 90 days).

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
Evidence: [SEC-003] Database backups are created daily.
JSON: {{"status": "ANSWERABLE", "reason": "The evidence directly states that database backups are created daily."}}

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
    method: str = Field(default="embedding")

def tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP]

def retrieve(question):
    query = set(tokens(question))
    ranked = []
    for doc in DOCUMENTS:
        for sentence in re.split(r"(?<=[.!?])\s+", doc["text"]):
            words = Counter(tokens(sentence))
            matches = query.intersection(words)
            score = len(matches) / max(len(query), 1)
            if matches:
                ranked.append({"document_id": doc["id"], "title": doc["title"],
                               "version": doc["version"], "excerpt": sentence,
                               "lexical_score": round(score, 4)})
    return sorted(ranked, key=lambda item: item["lexical_score"], reverse=True)[:3]

def review(question, method="embedding"):
    if method == "embedding":
        evidence = retrieve_by_embedding(question)
        llm_analysis = analyze_evidence(question, evidence)
        
        return {
            "question": question,
            "status": llm_analysis.get("status", "review_required"),
            "candidate_excerpt": evidence[0]["excerpt"] if evidence else None,
            "evidence": evidence,
            "mode": "embedding_with_reasoning",
            "reason": llm_analysis.get("reason", ""),
            "clarification_question": llm_analysis.get("clarification_question", None),
            "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
        }
    else:
        evidence = retrieve(question)
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
    return review(body.question.strip(), body.method)

@app.get("/evaluate")
def evaluation_endpoint():
    return evaluate()

PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EvidenceDesk baseline</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101725;color:#edf2fc;font:16px/1.6 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:40px 22px}header{border-bottom:1px solid #344257;padding-bottom:24px;margin-bottom:24px}.tag{color:#87d9c3;font-size:13px;letter-spacing:2px}h1{font-size:38px;margin:8px 0}p{color:#b8c4d8}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:#192337;border:1px solid #344257;border-radius:16px;padding:24px}textarea{width:100%;min-height:130px;background:#101725;color:white;border:1px solid #5a6c84;border-radius:8px;padding:12px;font:inherit}button{background:#9ae5cf;border:0;border-radius:8px;padding:12px 17px;margin:12px 8px 0 0;font-weight:700;cursor:pointer}button:disabled{opacity:.5}article{border-top:1px solid #344257;padding-top:12px;margin-top:16px}small{color:#9ae5cf}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.note{font-size:13px}.status{font-weight:700;color:#9ae5cf}a{color:#9ae5cf}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:30px}}
</style></head><body><main><header><div class="tag">EVIDENCEDESK / BUILD 0.1</div><h1>Evidence before answers.</h1><p>A security-questionnaire review baseline using synthetic policies.</p><small>Local demo · lexical retrieval · no LLM · no authentication</small></header>
<div class="grid"><section class="panel"><h2>Ask a policy question</h2><label for="question">Question</label><textarea id="question">What encryption protects customer data at rest?</textarea><div style="margin-top:12px; margin-bottom:12px"><strong>Retrieval method:</strong> <label style="margin-right:12px"><input type="radio" name="method" value="lexical"> Lexical</label> <label><input type="radio" name="method" value="embedding" checked> Embedding</label></div><button id="review">Find evidence</button><button id="evaluate">Run smoke tests</button><p class="note">Try: “Are you SOC 2 certified?” No generated answer is produced. Relevant excerpts always require human review.</p><a href="/docs">API documentation</a></section><section class="panel" aria-live="polite"><h2>Review workspace</h2><div id="result">Submit a question to inspect the retrieved evidence.</div></section></div>
<section class="panel" style="margin-top:20px"><h2>Synthetic source documents</h2><div id="sources"></div></section></main>
<script>
const el=id=>document.getElementById(id);
function text(parent,tag,value){const node=document.createElement(tag);node.textContent=value;parent.appendChild(node);return node;}
async function request(url,options){const response=await fetch(url,options);if(!response.ok)throw new Error('Request failed: '+response.status+' '+await response.text());return response.json();}
el('review').onclick=async()=>{const button=el('review');button.disabled=true;el('result').textContent='Retrieving…';const method=document.querySelector('input[name="method"]:checked').value;try{const data=await request('/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:el('question').value,method:method})});const target=el('result');target.replaceChildren();text(target,'div',data.status.replaceAll('_',' ')).className='status';if(data.reason){text(target,'p','Reason: '+data.reason).style.fontWeight='bold';}if(data.clarification_question){text(target,'p','Clarification needed: '+data.clarification_question).style.color='#ffb86c';}text(target,'p',data.warning);for(const item of data.evidence){const block=document.createElement('article');text(block,'small',item.document_id+' · '+item.title+' · '+item.version);text(block,'p',item.excerpt);let scoreText=item.lexical_score!==undefined?'Lexical overlap: '+item.lexical_score:'Embedding similarity: '+item.similarity_score;text(block,'small',scoreText+' — not confidence');target.appendChild(block);}}catch(error){el('result').textContent=error.message;}finally{button.disabled=false;}};
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
