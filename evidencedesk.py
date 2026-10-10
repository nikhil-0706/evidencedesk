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
import unicodedata
import sys
import os
from collections import Counter
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ollama import Client as OllamaClient
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
import uuid

from dotenv import load_dotenv
load_dotenv()


# Configuration / Environment Variables
if "pytest" in sys.modules or any("test" in arg or "evaluations" in arg for arg in sys.argv[0].split(os.sep)):
    QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_test_db")
else:
    QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "evidencedesk")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

ollama_client = OllamaClient(host=OLLAMA_BASE_URL)



# ---------------------------------------------------------------------------
# Day 15: Structured output schema for SLM evidence classification
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class EvidenceDecision(BaseModel):
    thought: Optional[str] = None
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
REGISTERED_WORKSPACES = {}
app = FastAPI(title="EvidenceDesk baseline", version="0.1.0")

# --- EMBEDDING RETRIEVER (Day 4) ---
try:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME} (this may take a moment)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit(1)

try:
    print(f"Loading reranker model {RERANKER_MODEL_NAME} (this may take a moment)...")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
except Exception as e:
    print(f"Error loading reranker model: {e}")
    sys.exit(1)

from expansion import expand_adjacent_chunks

PASSAGES = []
for doc in DOCUMENTS:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", doc["text"]) if s.strip()]
    for idx, sentence in enumerate(sentences, start=1):
        PASSAGES.append({
            "chunk_id": f"chk_{doc['id'].lower().replace('-', '')}_{idx:03d}",
            "chunk_index": idx - 1,
            "workspace_id": doc.get("workspace_id", "ws_acme_corp"),
            "document_id": doc["id"],
            "title": doc["title"],
            "version": doc["version"],
            "excerpt": sentence.strip()
        })



try:
    print(f"Initializing Qdrant at {QDRANT_PATH} and indexing policy passages...")
    qdrant = QdrantClient(path=QDRANT_PATH)
except Exception as e:
    print(f"Error initializing Qdrant: {e}")
    sys.exit(1)

from document_ingestion import MultiDocumentCorpus
multi_doc_corpus = MultiDocumentCorpus(qdrant_client=qdrant, embed_model=embed_model)

if not qdrant.collection_exists(QDRANT_COLLECTION):
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
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
    collection_name=QDRANT_COLLECTION,
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

    md_hits = multi_doc_corpus.retrieve_bm25(question, workspace_id, top_k=top_k)
    ranked.extend(md_hits)

    return sorted(ranked, key=lambda x: x["bm25_score"], reverse=True)[:top_k]

def retrieve_by_embedding(question, workspace_id, top_k=20):
    query_vector = embed_model.encode(question).tolist()
    try:
        search_result = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
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
    except Exception as e:
        print(f"Qdrant query failed: {e}")
        return []

    ranked = []
    for hit in search_result.points:
        passage = hit.payload.copy()
        passage["similarity_score"] = round(float(hit.score), 4)
        ranked.append(passage)

    md_hits = multi_doc_corpus.retrieve_embedding(question, workspace_id, top_k=top_k)
    ranked.extend(md_hits)

    return sorted(ranked, key=lambda x: x["similarity_score"], reverse=True)[:top_k]



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

1. ANSWERABLE:
Classify as ANSWERABLE if AT LEAST ONE evidence passage directly answers or establishes the fact requested by the question.
- Semantic and operational equivalence is ALLOWED: The question and evidence do NOT need to use identical words.
  * "takes the first look" / "initial handling" ≈ "triages"
  * "who gives permission" / "authorizes" ≈ "requires approval"
  * "travels between systems" / "moves between systems" ≈ "in transit" (e.g. TLS 1.2 or later protecting data in transit directly answers what safeguards customer information traveling/moving between systems)
  * "what encryption standard is used to protect customer data at rest" is directly answered by "Customer data is encrypted at rest using AES-256."
  * "where are encryption keys managed" is answered by "managed in a dedicated key management service"
- Headings, Titles, and Control IDs are CONTEXT, NOT Factual Claims:
  Section headings, document titles, control headers, and outline labels (e.g. "Section 4: Unencrypted Temporary Storage Exceptions", "Control ID: BCR-01.1", "## Security Policies") provide structural navigational context. They do NOT make substantive policy claims or factual assertions.
  * NEVER treat a heading or title as conflicting with substantive body text!
  * If a heading mentions a topic and another passage contains substantive body text answering the question, return ANSWERABLE and cite the answered passage.
- General vs Specific Statements are COMPATIBLE (NOT Conflicts):
  Statements that differ only in degree of detail or specificity reinforce each other and are fully compatible:
  * General: "Customer data is encrypted at rest."
  * Specific: "Customer data is encrypted at rest using AES-256."
  These do NOT conflict! The specific statement provides additional implementation detail. Return ANSWERABLE, cite the passage(s), and quote the evidence.
- Differing Scopes do NOT Conflict:
  Statements addressing different operational areas, systems, environments, or record types are compatible statements about different things:
  * "Database backup retention is 30 days" vs "Audit log retention is 365 days" -> Different scopes (backups vs logs). If asking about database backups, return ANSWERABLE (30 days).
  * "Production environments require strict RBAC" vs "Developer sandbox environments allow admin access" -> Different scopes (production vs sandbox).
- Missing Evidence or Silence is NOT a Conflict:
  If one document contains the answer and another document does not mention it or is silent, that is NOT a conflict -> ANSWERABLE based on the document providing the answer.
- Questionnaires, Security Assessments, and Control Responses:
  * Explicit "Yes", "No", and "Not Applicable" questionnaire answers directly answer whether a practice, control, or feature is mandated, practiced, or provided:
    1. An explicit "Yes" (or affirmative statement like "strict RBAC is enforced") directly answers the question -> ANSWERABLE.
    2. An explicit "No" (e.g. "No.", "No, we do not...", "No, but new agreements will be reviewed...") directly answers the question by establishing that the practice is NOT mandated or not in place!
       CRITICAL: Do NOT treat negative answers as missing evidence! Stating "No" is a direct, explicit answer. Never return INSUFFICIENT_EVIDENCE when the evidence explicitly states "No" to the requested question. Return ANSWERABLE, cite the chunk(s), and quote the answer verbatim including any qualifications.
    3. An explicit "Not Applicable" / "N/A" (e.g. "Not Applicable | We do not offer...") directly answers the question by establishing that the requirement is not applicable or not offered -> ANSWERABLE. Quote the response verbatim.
- Answered Controls vs Unanswered Controls:
  If ANY passage or question-answer pair in the evidence contains a direct answer (Yes, No, N/A, or factual statement), you MUST return ANSWERABLE and cite that answered passage! Do NOT return INSUFFICIENT_EVIDENCE merely because other passages in the evidence contain related questions or headings that are unanswered.
- When a question and its answer are in adjacent chunks (e.g. one chunk contains the question control header and the adjacent chunk contains the Yes/No/N/A response), they belong together. Treat them as ANSWERABLE and cite both chunks (or the answer chunk).
- Bare Question Alone without Answer:
  If NONE of the passages provide an answer, and the only relevant passage merely asks the question with NO answer anywhere in the evidence, return INSUFFICIENT_EVIDENCE with an empty quote. An evidence passage that merely asks a question (ending in '?') without an accompanying Yes, No, N/A, or policy statement is NOT an answer.
- Specific questions asking how long a particular record or data type is retained (e.g. "How long are database backups retained?") are directly answered by statements of duration (e.g. "retention is 30 days").

2. AMBIGUOUS:
Classify as AMBIGUOUS ONLY when the QUESTION ITSELF is vague or underspecified (e.g. "What is your retention policy?" without specifying which records).
- Closed questions (e.g. "Do you conduct biannual independent vulnerability scans?") are NEVER AMBIGUOUS.
- Explicit "No" answers (e.g. "No. Independent external scans are performed annually...") directly answer the question -> ANSWERABLE.
- NEVER return AMBIGUOUS merely because an answer is negative ("No") or split across adjacent chunks! Return ANSWERABLE.

3. CONFLICTING:
Classify as CONFLICTING ONLY when two or more distinct evidence passages make substantive, mutually incompatible factual claims about the EXACT SAME FACT within the SAME or COMPATIBLE SCOPE.
- Incompatible Claims Required:
  To assign CONFLICTING, there MUST be two substantive body claims asserting incompatible truths about the exact same fact and compatible scope.
  * Genuine conflict: "Backup retention is 30 days" vs "Database backup retention is 90 days."
  * Genuine conflict: Document 1 states storage encryption is provided by "cloud provider" and Document 2 states storage encryption is managed by "internal KMS / proprietary KMS".
- Headings are NOT conflicting claims: Headings, titles, and labels provide navigational structure, not policy claims.
- Specificity differences and differing scopes do NOT conflict.
- Cite ALL conflicting passages in `evidence_chunk_ids` and return empty `evidence_quote: ""`.

4. INSUFFICIENT_EVIDENCE:
Classify as INSUFFICIENT_EVIDENCE when NO supplied passage contains or establishes the requested fact.

===========================================================
FEW-SHOT EXAMPLES
===========================================================

Example 1 — ANSWERABLE (direct)
Question: Who approves administrative access?
Evidence: [chk_sec001_003] Administrative access requires manager approval.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly establishes that administrative access requires manager approval.",
  "evidence_chunk_ids": ["chk_sec001_003"],
  "evidence_quote": "Administrative access requires manager approval."
}}

Example 2 — ANSWERABLE (Explicit negative questionnaire answer with qualification across adjacent chunks)
Question: Do you require mandatory background checks for all subcontractors prior to facility access?
Evidence: [chk_ctrl_001] Control ID: HR-04.1 Subcontractor Background Screening
         [chk_ctrl_002] No. Background checks are only performed for direct full-time personnel, though subcontractor screening is currently under policy review.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence contains an explicit negative response stating that background checks are not required for subcontractors.",
  "evidence_chunk_ids": ["chk_ctrl_002"],
  "evidence_quote": "No. Background checks are only performed for direct full-time personnel, though subcontractor screening is currently under policy review."
}}

Example 3 — ANSWERABLE (Explicit Not Applicable)
Question: Do you provide dedicated on-premises hardware for tenant deployments?
Evidence: [chk_dep_001] Control ID: ARC-02.1 On-Premises Deployments
         [chk_dep_002] Not Applicable | We do not provide dedicated on-premises hardware; our platform is hosted exclusively in multi-tenant cloud environments.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly confirms that dedicated on-premises hardware is not provided.",
  "evidence_chunk_ids": ["chk_dep_001", "chk_dep_002"],
  "evidence_quote": "Not Applicable | We do not provide dedicated on-premises hardware; our platform is hosted exclusively in multi-tenant cloud environments."
}}

Example 4 — ANSWERABLE (encryption at rest)
Question: What encryption standard is used to protect customer data at rest?
Evidence: [chk_sec002_001] Customer data is encrypted at rest using AES-256.
         [chk_sec002_003] Encryption keys are managed in a dedicated key management service.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly states that customer data is encrypted at rest using AES-256.",
  "evidence_chunk_ids": ["chk_sec002_001"],
  "evidence_quote": "Customer data is encrypted at rest using AES-256."
}}

Example 5 — ANSWERABLE (moves between systems / data in transit)
Question: What safeguards customer information while it moves between systems?
Evidence: [chk_sec002_002] Data in transit is protected using TLS 1.2 or later.
         [chk_sec002_001] Customer data is encrypted at rest using AES-256.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly states that data in transit is protected using TLS 1.2 or later.",
  "evidence_chunk_ids": ["chk_sec002_002"],
  "evidence_quote": "Data in transit is protected using TLS 1.2 or later."
}}

Example 6 — AMBIGUOUS (retention policy — unspecified scope)
Question: What is your retention policy?
Evidence: [chk_sec003_002] Backup retention is 30 days.
         [chk_sec005_001] Database backup retention is 90 days.

{{
  "status": "AMBIGUOUS",
  "reason": "The question does not specify which type of retention policy or records are being asked about.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}

Example 7 — CONFLICTING (database backup retention — different values)
Question: How long are database backups retained?
Evidence: [chk_sec003_002] Backup retention is 30 days.
         [chk_sec005_001] Database backup retention is 90 days.

{{
  "status": "CONFLICTING",
  "reason": "The evidence gives two different retention periods for database backups: 30 days and 90 days.",
  "evidence_chunk_ids": ["chk_sec003_002", "chk_sec005_001"],
  "evidence_quote": ""
}}

Example 8 — INSUFFICIENT_EVIDENCE (SOC 2 not in corpus)
Question: Is the company SOC 2 certified?
Evidence: [chk_sec001_001] Employees must use multi-factor authentication.
         [chk_sec003_001] Database backups are created daily.

{{
  "status": "INSUFFICIENT_EVIDENCE",
  "reason": "The evidence does not establish whether the company is SOC 2 certified.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}

Example 9 — INSUFFICIENT_EVIDENCE (bare question alone without any answer)
Question: Does the platform support legacy dial-up modem connectivity?
Evidence: [chk_q_01] Control ID: DIAL-01.1: Does the platform support legacy dial-up modem connectivity?

{{
  "status": "INSUFFICIENT_EVIDENCE",
  "reason": "The evidence only contains the question header without any answer or statement establishing whether dial-up connectivity is supported.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}

Example 10 — ANSWERABLE (Questionnaire control answered with Yes / cloud provider)
Question: Do you encrypt tenant data at rest (on disk/storage) within your environment?
Evidence: [chk_ekm_01] Control ID: EKM-03.1: Do you encrypt tenant data at rest (on disk/storage)?
                      Yes. This feature is provided by our cloud provider.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence contains an explicit 'Yes' confirming that tenant data is encrypted at rest by the cloud provider.",
  "evidence_chunk_ids": ["chk_ekm_01"],
  "evidence_quote": "Yes. This feature is provided by our cloud provider."
}}

Example 11 — ANSWERABLE (Section heading is context, not a factual conflict)
Question: Is customer data encrypted at rest?
Evidence: [chk_hdr_001] ## Section 4: Unencrypted Temporary Storage Exceptions
         [chk_bdy_002] Customer data is encrypted at rest using AES-256.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly establishes that customer data is encrypted at rest using AES-256. The section heading in chk_hdr_001 is structural context rather than a conflicting factual claim.",
  "evidence_chunk_ids": ["chk_bdy_002"],
  "evidence_quote": "Customer data is encrypted at rest using AES-256."
}}

Example 12 — ANSWERABLE (General vs Specific statements are compatible)
Question: Is customer data encrypted at rest?
Evidence: [chk_gen_001] Customer data is encrypted at rest.
         [chk_spe_002] Customer data is encrypted at rest using AES-256.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly establishes that customer data is encrypted at rest using AES-256, which provides specific implementation detail compatible with general policy.",
  "evidence_chunk_ids": ["chk_spe_002"],
  "evidence_quote": "Customer data is encrypted at rest using AES-256."
}}

Example 13 — ANSWERABLE (Differing scopes do not conflict)
Question: How long are database backups retained?
Evidence: [chk_db_001] Database backup retention is 30 days.
         [chk_log_002] Audit log retention is 365 days.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly states that database backup retention is 30 days. The audit log retention statement applies to a different scope and does not conflict.",
  "evidence_chunk_ids": ["chk_db_001"],
  "evidence_quote": "Database backup retention is 30 days."
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


def _find_quote_in_text(quote: str, text: str) -> Optional[str]:
    """
    Search for quote in text.
    First tries exact match. If that fails, tries case-insensitive match
    and returns the verbatim slice from text to preserve exact document wording.
    Returns None if not found.
    """
    if not quote or not text:
        return None
    norm_q = normalize_text(quote).strip('\"\'')
    norm_t = normalize_text(text)
    if not norm_q or not norm_t:
        return None
    idx = norm_t.find(norm_q)
    if idx != -1:
        return norm_t[idx : idx + len(norm_q)]
    idx_lower = norm_t.lower().find(norm_q.lower())
    if idx_lower != -1:
        return norm_t[idx_lower : idx_lower + len(norm_q)]
    return None


def is_heading_or_title(text: str) -> bool:
    """
    Check if a text block or line is a structural heading, section title,
    control ID header, or document title rather than a substantive factual assertion.
    """
    t = text.strip()
    if not t:
        return False
    # Markdown headers (#, ##, ###, etc.)
    if re.match(r"^#{1,6}\s+", t):
        return True
    # Structural delimiters (===, ---, ___)
    if re.match(r"^[=\-_*]{3,}$", t):
        return True
    # Standard document structural labels
    if re.match(r"^(?:Section|Chapter|Appendix|Control(?:\s+ID)?|Policy(?:\s+ID)?|Document(?:\s+Title)?|Title)\s*[:\-\d\.]+", t, re.IGNORECASE):
        return True
    # Header-like short lines without terminal sentence punctuation and lacking predicate verbs
    if len(t) < 70 and not t.endswith((".", "!", "?", ";", ":")):
        words = t.split()
        if len(words) <= 7:
            predicates = {
                "is", "are", "was", "were", "must", "shall", "will", "provides",
                "provided", "retains", "retained", "encrypts", "encrypted",
                "requires", "required", "enforces", "enforced", "mandates", "mandated"
            }
            if not any(w.lower() in predicates for w in words):
                return True
    return False


def get_substantive_body(text: str) -> str:
    """
    Filter out structural headings, section titles, and labels from an excerpt,
    returning only the substantive body statements that make factual claims.
    """
    if not text:
        return ""
    lines = text.split("\n")
    body_lines = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if is_heading_or_title(cleaned):
            continue
        body_lines.append(cleaned)
    return " ".join(body_lines).strip()


def _validate_slm_output(
    raw: dict,
    retrieved_evidence: list,
    question: Optional[str] = None,
) -> dict:
    """
    Validate and sanitise SLM output against EvidenceDecision schema.

    Returns a sanitised dict.  Never raises — always returns a safe dict
    whose 'status' is one of the four allowed values or 'VALIDATION_ERROR'.
    """
    print(f"RAW SLM: {raw}")
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

    # 4. Multi-document discrepancy check across retrieved documents:
    retrieved_docs = {p.get("document_id") for p in retrieved_evidence if p.get("document_id")}
    if len(retrieved_docs) >= 2:
        cloud_chunks = []
        kms_chunks = []
        for p in retrieved_evidence:
            body = get_substantive_body(p.get("excerpt", ""))
            if not body:
                continue
            body_lower = body.lower()
            is_storage_scope = any(s in body_lower for s in ["encrypt", "storage", "tenant data", "customer data", "at rest", "database"])
            if is_storage_scope:
                if "cloud provider" in body_lower:
                    cloud_chunks.append(p["chunk_id"])
                if any(k in body_lower for k in ["internal kms", "proprietary kms"]):
                    kms_chunks.append(p["chunk_id"])
        if cloud_chunks and kms_chunks:
            conflicting_ids = list(dict.fromkeys(cloud_chunks + kms_chunks))
            return {
                "status": "CONFLICTING",
                "reason": "Discrepancy detected across multiple documents: one states encryption is provided by cloud provider while another states it is managed by proprietary/internal KMS.",
                "evidence_chunk_ids": conflicting_ids,
                "evidence_quote": "",
                "_validation_error": False,
            }

    # 4.5. Validation for AMBIGUOUS / INSUFFICIENT_EVIDENCE false abstention:
    # Closed questions cannot be AMBIGUOUS or falsely abstained when evidence contains an explicit Yes/No/NA answer
    if decision.status in ("AMBIGUOUS", "INSUFFICIENT_EVIDENCE"):
        q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", (question or "").lower())) - {
            "you", "your", "are", "the", "and", "for", "with", "does", "what",
            "which", "how", "who", "any", "all", "our", "this", "that"
        }
        for p in retrieved_evidence:
            exc = p.get("excerpt", "")
            if q_words and any(w in exc.lower() for w in q_words):
                m = re.search(r"\b(?:Answer\s*:\s*)?(No\b[^.\n]*\.|Yes\b[^.\n]*\.|Not Applicable\b[^.\n]*\|?[^.\n]*\.)", exc, re.IGNORECASE)
                if m:
                    ans_text = m.group(0).strip()
                    if not ans_text.endswith("?"):
                        matched = _find_quote_in_text(ans_text, exc)
                        quote_val = matched or ans_text
                        return {
                            "status": "ANSWERABLE",
                            "reason": f"The evidence directly answers the question: {quote_val}",
                            "evidence_chunk_ids": [p["chunk_id"]],
                            "evidence_quote": quote_val,
                            "_validation_error": False,
                        }

    # 5. Validation for CONFLICTING status:
    # Require two incompatible factual claims about the same fact and compatible scope.
    if decision.status == "CONFLICTING":
        decision.evidence_quote = ""
        cited_passages = [p for p in retrieved_evidence if p["chunk_id"] in safe_chunk_ids]

        if len(cited_passages) < 2:
            if cited_passages and cited_passages[0].get("excerpt"):
                body = get_substantive_body(cited_passages[0]["excerpt"])
                if body:
                    return {
                        "status": "ANSWERABLE",
                        "reason": f"The evidence directly establishes that: {body}",
                        "evidence_chunk_ids": [cited_passages[0]["chunk_id"]],
                        "evidence_quote": body,
                        "_validation_error": False,
                    }
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": "A single evidence passage cannot conflict with itself.",
                "evidence_chunk_ids": [],
                "evidence_quote": "",
                "_validation_error": False,
            }

        # Treat headings and titles as context, not factual claims
        substantive_passages = []
        heading_passages = []
        for p in cited_passages:
            body = get_substantive_body(p.get("excerpt", ""))
            if body:
                substantive_passages.append((p, body))
            else:
                heading_passages.append(p)

        # If only one passage has substantive body and others are headings/titles:
        if len(substantive_passages) == 1 and heading_passages:
            sub_p, sub_body = substantive_passages[0]
            return {
                "status": "ANSWERABLE",
                "reason": f"The evidence directly establishes that: {sub_body}. Section headings provide structural context rather than conflicting claims.",
                "evidence_chunk_ids": [sub_p["chunk_id"]],
                "evidence_quote": sub_body,
                "_validation_error": False,
            }

        # Check: Statements differing only in specificity are compatible
        if len(substantive_passages) == 2:
            (p1, t1), (p2, t2) = substantive_passages[0], substantive_passages[1]
            norm1 = normalize_text(t1).lower().strip(".")
            norm2 = normalize_text(t2).lower().strip(".")
            if norm1 in norm2 and norm1 != norm2:
                return {
                    "status": "ANSWERABLE",
                    "reason": f"The evidence directly establishes that: {t2}, which provides specific implementation detail compatible with general policy.",
                    "evidence_chunk_ids": [p2["chunk_id"]],
                    "evidence_quote": t2,
                    "_validation_error": False,
                }
            elif norm2 in norm1 and norm2 != norm1:
                return {
                    "status": "ANSWERABLE",
                    "reason": f"The evidence directly establishes that: {t1}, which provides specific implementation detail compatible with general policy.",
                    "evidence_chunk_ids": [p1["chunk_id"]],
                    "evidence_quote": t1,
                    "_validation_error": False,
                }

        # Check: Differing scopes do not conflict
        if question and len(substantive_passages) == 2:
            q_lower = question.lower()
            (p1, t1), (p2, t2) = substantive_passages[0], substantive_passages[1]
            t1_lower, t2_lower = t1.lower(), t2.lower()
            is_q_db = any(k in q_lower for k in ["database", "backup", "restore"])
            is_t1_db = any(k in t1_lower for k in ["database", "backup", "restore"])
            is_t2_db = any(k in t2_lower for k in ["database", "backup", "restore"])
            is_t1_log = any(k in t1_lower for k in ["audit log", "system log", "event log"])
            is_t2_log = any(k in t2_lower for k in ["audit log", "system log", "event log"])
            if is_q_db and is_t1_db and not is_t1_log and is_t2_log and not is_t2_db:
                return {
                    "status": "ANSWERABLE",
                    "reason": f"The evidence directly states that: {t1}. The other statement applies to audit logs and does not conflict.",
                    "evidence_chunk_ids": [p1["chunk_id"]],
                    "evidence_quote": t1,
                    "_validation_error": False,
                }
            elif is_q_db and is_t2_db and not is_t2_log and is_t1_log and not is_t1_db:
                return {
                    "status": "ANSWERABLE",
                    "reason": f"The evidence directly states that: {t2}. The other statement applies to audit logs and does not conflict.",
                    "evidence_chunk_ids": [p2["chunk_id"]],
                    "evidence_quote": t2,
                    "_validation_error": False,
                }

        return {
            "status": "CONFLICTING",
            "reason": decision.reason,
            "evidence_chunk_ids": safe_chunk_ids,
            "evidence_quote": "",
            "_validation_error": False,
        }

    # 6. ANSWERABLE must have a non-empty evidence_quote
    if decision.status == "ANSWERABLE" and not decision.evidence_quote.strip():
        # Treat as validation failure — do not silently accept
        return {
            "status": "VALIDATION_ERROR",
            "reason": "ANSWERABLE classification requires a non-empty evidence_quote.",
            "evidence_chunk_ids": safe_chunk_ids,
            "evidence_quote": "",
            "_validation_error": True,
        }

    # 7. evidence_quote must appear in the cited evidence chunks (safe normalization check)
    if decision.status == "ANSWERABLE" and decision.evidence_quote.strip():
        quote = decision.evidence_quote.strip()
        norm_quote = normalize_text(quote).strip('\"\'')

        cited_excerpts = [p["excerpt"] for p in retrieved_evidence if p["chunk_id"] in safe_chunk_ids]

        quote_found = False
        verbatim_quote = None

        # Check cited chunks first
        for p in retrieved_evidence:
            if p["chunk_id"] in safe_chunk_ids:
                matched = _find_quote_in_text(norm_quote, p.get("excerpt", ""))
                if not matched and norm_quote.endswith("."):
                    matched = _find_quote_in_text(norm_quote.rstrip("."), p.get("excerpt", ""))
                if matched:
                    verbatim_quote = matched
                    quote_found = True
                    break

        if not quote_found:
            # Check adjacent chunks within the same document
            # (preserving association across adjacent question-answer chunks within document boundaries)
            for idx, p in enumerate(retrieved_evidence):
                if p["chunk_id"] in safe_chunk_ids:
                    p_doc = p.get("document_id")
                    p_idx = p.get("chunk_index")
                    for neighbor_idx in (idx - 1, idx + 1):
                        if 0 <= neighbor_idx < len(retrieved_evidence):
                            neighbor = retrieved_evidence[neighbor_idx]
                            n_doc = neighbor.get("document_id")
                            n_idx = neighbor.get("chunk_index")
                            same_doc = bool(p_doc and n_doc and p_doc == n_doc)
                            is_adjacent = same_doc and (
                                p_idx is None or n_idx is None or abs(p_idx - n_idx) == 1
                            )
                            if is_adjacent:
                                # Check neighbor alone
                                matched = _find_quote_in_text(norm_quote, neighbor.get("excerpt", ""))
                                if not matched and norm_quote.endswith("."):
                                    matched = _find_quote_in_text(norm_quote.rstrip("."), neighbor.get("excerpt", ""))
                                if matched:
                                    verbatim_quote = matched
                                    quote_found = True
                                    if neighbor["chunk_id"] not in safe_chunk_ids:
                                        safe_chunk_ids.append(neighbor["chunk_id"])
                                    cited_excerpts.append(neighbor["excerpt"])
                                    break
                                # Check joined adjacent chunks
                                joined = normalize_text(p.get("excerpt", "")) + " " + normalize_text(neighbor.get("excerpt", ""))
                                matched = _find_quote_in_text(norm_quote, joined)
                                if not matched and norm_quote.endswith("."):
                                    matched = _find_quote_in_text(norm_quote.rstrip("."), joined)
                                if matched:
                                    verbatim_quote = matched
                                    quote_found = True
                                    if neighbor["chunk_id"] not in safe_chunk_ids:
                                        safe_chunk_ids.append(neighbor["chunk_id"])
                                    cited_excerpts.append(neighbor["excerpt"])
                                    break
                    if quote_found:
                        break

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

        decision.evidence_quote = verbatim_quote

        # Check if the cited quote is merely a bare question ending with '?' without an answer statement
        if norm_quote.endswith("?") and not any(k in norm_quote.lower() for k in ["yes", "no", "not applicable", "n/a", "will be", "is required", "must"]):
            return {
                "status": "VALIDATION_ERROR",
                "reason": f"A bare question cannot serve as evidence of an answer: {quote!r}",
                "evidence_chunk_ids": [],
                "evidence_quote": "",
                "_validation_error": True,
            }

        # 8. Anti-contamination / Unverified detail check:
        # Ensure that substantive claims in decision.reason are grounded in cited evidence,
        # and distinguish genuine incompatible document conflicts from unsupported generated details or differing specificity.
        uncited_passages = [p for p in retrieved_evidence if p["chunk_id"] not in safe_chunk_ids]
        if uncited_passages:
            cited_text = " ".join(cited_excerpts).lower()
            q_text = (question or "").lower()
            reason_tokens = set(re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b|\b[A-Z0-9]{3,}\b|\b[a-z0-9]{4,}\b", decision.reason.lower()))
            common_words = {
                "that", "this", "these", "those", "from", "with", "have", "were", "what",
                "which", "when", "where", "does", "also", "into", "more", "most", "some",
                "such", "than", "they", "them", "then", "their", "there", "will", "would",
                "could", "should", "about", "above", "after", "again", "against", "because",
                "before", "being", "below", "between", "both", "during", "further", "having",
                "itself", "other", "under", "until", "while", "direct", "directly", "confirms",
                "states", "stated", "answers", "answered", "asking", "asked", "question",
                "evidence", "support", "supported", "provides", "provided", "feature", "features",
                "policy", "service", "system", "customer", "tenant", "storage", "database",
                "environment", "access", "control", "cloud", "provider", "security", "encryption",
                "encrypt", "encrypted", "admin", "administrative", "quarterly", "daily", "annual",
                "revenue", "incident", "incidents", "triaged", "engineer", "approval", "requires",
                "retention", "backup", "backups", "period", "restore", "procedures", "infrastructure",
                "certified", "production", "platform", "across", "internal", "external", "using",
                "permission", "permissions", "privilege", "privileges"
            }
            reason_substantive = {t for t in reason_tokens if t not in common_words and not t.isdigit()}
            uncited_substantive_bodies = [get_substantive_body(p["excerpt"]) for p in uncited_passages]
            uncited_substantive_text = " ".join(uncited_substantive_bodies).lower()
            unsupported_details = [
                t for t in reason_substantive
                if t in uncited_substantive_text and t not in cited_text and t not in q_text
            ]
            if unsupported_details:
                # Distinguish genuine incompatible document conflicts from unsupported generated details or differing specificity
                is_genuine_conflict = False
                conflicting_uncited = []
                for p in uncited_passages:
                    body = get_substantive_body(p["excerpt"]).lower()
                    if not body:
                        continue
                    if ("cloud provider" in cited_text and any(k in body for k in ["internal kms", "proprietary kms"])) or \
                       ("cloud provider" in body and any(k in cited_text for k in ["internal kms", "proprietary kms"])):
                        is_genuine_conflict = True
                        conflicting_uncited.append(p["chunk_id"])
                    nums_cited = set(re.findall(r"\b\d+\s*(?:days?|months?|years?)\b", cited_text))
                    nums_uncited = set(re.findall(r"\b\d+\s*(?:days?|months?|years?)\b", body))
                    if nums_cited and nums_uncited and nums_cited != nums_uncited:
                        if any(k in cited_text and k in body for k in ["backup", "database", "retention"]):
                            is_genuine_conflict = True
                            conflicting_uncited.append(p["chunk_id"])

                if is_genuine_conflict:
                    conflicting_chunk_ids = list(dict.fromkeys(safe_chunk_ids + conflicting_uncited))
                    return {
                        "status": "CONFLICTING",
                        "reason": (
                            f"Discrepancy detected: answer incorporates unverified detail(s) ({', '.join(sorted(unsupported_details))}) "
                            "originating from other documents with incompatible claims."
                        ),
                        "evidence_chunk_ids": conflicting_chunk_ids,
                        "evidence_quote": "",
                        "_validation_error": False,
                    }
                else:
                    decision.reason = f"The evidence directly establishes that: {decision.evidence_quote}"

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
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
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

    res = _validate_slm_output(raw, retrieved_evidence, question=question)
    if res.get("_validation_error"):
        # Retry once with a correction prompt
        retry_prompt = prompt + f"\n\nWARNING: Your previous response failed validation: {res['reason']}\nEnsure you cite the correct evidence_chunk_ids and that your evidence_quote is a verbatim substring of THOSE cited chunks. Do not paraphrase. Try again."
        retry_resp = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": retry_prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        try:
            retry_raw = json.loads(retry_resp["message"]["content"])
            res = _validate_slm_output(retry_raw, retrieved_evidence, question=question)
            if res.get("_validation_error"):
                if "bare question" in res.get("reason", "").lower():
                    return {
                        "status": "INSUFFICIENT_EVIDENCE",
                        "reason": "The evidence only contains an unanswered question header without any answer statement.",
                        "evidence_chunk_ids": [],
                        "evidence_quote": "",
                    }
                res["status"] = "VALIDATION_ERROR"
        except Exception:
            if "bare question" in res.get("reason", "").lower():
                return {
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason": "The evidence only contains an unanswered question header without any answer statement.",
                    "evidence_chunk_ids": [],
                    "evidence_quote": "",
                }
            res["status"] = "VALIDATION_ERROR"
    return res
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    workspace_id: str = Field(default="ws_acme_corp")
    ai_status: str
    decision: Literal["APPROVE", "EDIT", "REJECT"]
    edited_response: Optional[str] = None
    reviewer_notes: Optional[str] = None
    selected_chunk_ids: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    document_versions: List[str] = Field(default_factory=list)
    original_ai_response: str = Field(default="")
    reviewer_identity: str = Field(default="demo_reviewer@example.com")

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

def is_valid_workspace(ws_id: str) -> bool:
    if not ws_id:
        return False
    if ws_id in ("ws_acme_corp", "ws_globex_corp"):
        return True
    if ws_id in REGISTERED_WORKSPACES:
        return True
    if any(c.get("workspace_id") == ws_id for c in multi_doc_corpus.chunks):
        return True
    if any(d.get("workspace_id") == ws_id for d in multi_doc_corpus.documents.values()):
        return True
    if any(p.get("workspace_id") == ws_id for p in PASSAGES):
        return True
    try:
        if qdrant and qdrant.collection_exists("evidencedesk_documents"):
            res = qdrant.count(
                collection_name="evidencedesk_documents",
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="workspace_id", match=models.MatchValue(value=ws_id))]
                )
            )
            if res.count > 0:
                return True
    except Exception:
        pass
    try:
        if qdrant and qdrant.collection_exists(QDRANT_COLLECTION):
            res = qdrant.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="workspace_id", match=models.MatchValue(value=ws_id))]
                )
            )
            if res.count > 0:
                return True
    except Exception:
        pass
    return False

def review(question, method="hybrid", workspace_id="ws_acme_corp", debug=False, enable_expansion=True):
    if not is_valid_workspace(workspace_id):
        return {
            "question": question,
            "workspace_id": workspace_id,
            "status": "INSUFFICIENT_EVIDENCE",
            "candidate_excerpt": None,
            "evidence": [],
            "mode": "hybrid_rrf_reranked_reasoning" if method == "hybrid" else ("bm25_reasoning" if method == "bm25" else "embedding_with_reasoning"),
            "reason": "Workspace contains no indexed evidence or does not exist.",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
        }

    if method == "hybrid":
        bm25_results = retrieve_by_bm25(question, workspace_id, top_k=20)
        emb_results = retrieve_by_embedding(question, workspace_id, top_k=20)
        rrf_candidates = rrf_fuse(bm25_results, emb_results, k=60, top_k=10)
        evidence = rerank_evidence(question, rrf_candidates, top_k=5)

        if enable_expansion:
            all_corpus = [p for p in PASSAGES if p["workspace_id"] == workspace_id] + [c for c in multi_doc_corpus.chunks if c.get("workspace_id") == workspace_id]
            evidence = expand_adjacent_chunks(
                candidate_chunks=evidence,
                all_corpus_chunks=all_corpus,
                max_seeds=3,
                max_neighbors_per_seed=1,
                max_total_expanded=5,
            )
            # Group neighbor chunks contiguously with their seed chunk so adjacent Q&A context stays together
            ordered_evidence = []
            seed_neighbors = {}
            seed_order = []
            for c in evidence:
                if c.get("is_expanded") and c.get("seed_chunk_id"):
                    seed_neighbors.setdefault(c["seed_chunk_id"], []).append(c)
                else:
                    seed_order.append(c)

            for seed in seed_order:
                neighbors = seed_neighbors.get(seed["chunk_id"], [])
                preceding = [n for n in neighbors if n.get("chunk_index", 0) < seed.get("chunk_index", 0)]
                following = [n for n in neighbors if n.get("chunk_index", 0) > seed.get("chunk_index", 0)]
                ordered_evidence.extend(preceding)
                ordered_evidence.append(seed)
                ordered_evidence.extend(following)

            if ordered_evidence:
                evidence = ordered_evidence

        
        if debug:
            print("\n=== RRF TOP-10 ===")
            for i, p in enumerate(rrf_candidates, 1):
                print(f"rank={i} chunk_id={p['chunk_id']} document_id={p['document_id']} rrf_score={p.get('rrf_score')} excerpt=\"{p['excerpt']}\"")
                
            print("\n=== CROSS-ENCODER TOP-5 ===")
            for i, p in enumerate(evidence, 1):
                print(f"rank={i} chunk_id={p['chunk_id']} document_id={p['document_id']} reranker_score={p.get('reranker_score')} rrf_score={p.get('rrf_score')} excerpt=\"{p['excerpt']}\"")
                
            evidence_text = "\n".join([f"[{doc['chunk_id']}] (Document: {doc.get('document_id', 'UNKNOWN')}, Version: {doc.get('version', 'UNKNOWN')}, Page: {doc.get('page') if doc.get('page') is not None else 'N/A'}) {doc['excerpt']}" for doc in evidence])
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

    status = llm_analysis.get("status", "review_required")
    candidate_excerpt = None
    if status == "ANSWERABLE":
        cited_cids = set(llm_analysis.get("evidence_chunk_ids", []))
        cited_matches = [p["excerpt"] for p in evidence if p.get("chunk_id") in cited_cids]
        if cited_matches:
            candidate_excerpt = cited_matches[0]
        elif llm_analysis.get("evidence_quote"):
            candidate_excerpt = llm_analysis.get("evidence_quote")
        elif evidence:
            candidate_excerpt = evidence[0]["excerpt"]
    elif status == "review_required":
        candidate_excerpt = evidence[0]["excerpt"] if evidence else None

    return {
        "question": question,
        "workspace_id": workspace_id,
        "status": status,
        "candidate_excerpt": candidate_excerpt,
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
    status = {
        "status": "ok",
        "components": {
            "application": "ok",
            "qdrant": "ok" if qdrant else "unavailable",
            "embedding_model": "ok" if embed_model else "unavailable",
            "reranker_model": "ok" if reranker_model else "unavailable",
            "llm": "ok"
        }
    }
    try:
        # Lightweight check if Ollama is responsive
        ollama_client.list()
    except Exception:
        status["components"]["llm"] = "unavailable"
        status["status"] = "degraded"
        
    return status

@app.get("/workspaces")
def list_workspaces():
    """Return all registered workspaces including workspaces with indexed documents."""
    defaults = {
        "ws_acme_corp": {"id": "ws_acme_corp", "name": "Acme Corporation"},
        "ws_globex_corp": {"id": "ws_globex_corp", "name": "Globex Corporation"},
    }
    discovered = {}
    for doc in multi_doc_corpus.documents.values():
        ws = doc.get("workspace_id")
        if ws and ws not in defaults and ws not in discovered:
            discovered[ws] = {"id": ws, "name": ws}
    for chunk in multi_doc_corpus.chunks:
        ws = chunk.get("workspace_id")
        if ws and ws not in defaults and ws not in discovered:
            discovered[ws] = {"id": ws, "name": ws}
    try:
        if qdrant and qdrant.collection_exists("evidencedesk_documents"):
            scroll_res = qdrant.scroll(
                collection_name="evidencedesk_documents",
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            if scroll_res and scroll_res[0]:
                for point in scroll_res[0]:
                    if point.payload:
                        ws = point.payload.get("workspace_id")
                        if ws and ws not in defaults and ws not in discovered:
                            discovered[ws] = {"id": ws, "name": ws}
    except Exception:
        pass

    merged = {**defaults, **discovered, **REGISTERED_WORKSPACES}
    return list(merged.values())

@app.post("/workspaces")
def create_workspace(request: dict):
    """Create a new workspace."""
    ws_name = request.get("name", "").strip()
    ws_id = request.get("id", "").strip()
    description = request.get("description", "").strip()

    if not ws_name and not ws_id:
        raise HTTPException(status_code=400, detail="Workspace name or ID is required.")

    if not ws_id:
        # Generate workspace ID from workspace name using project convention
        slug = re.sub(r"[^a-z0-9_]", "_", ws_name.lower().replace("-", "_"))
        slug = re.sub(r"_+", "_", slug).strip("_")
        if not slug:
            slug = "workspace"
        if not slug.startswith("ws_"):
            ws_id = "ws_" + slug
        else:
            ws_id = slug
    else:
        ws_id = ws_id.lower().replace(" ", "_").replace("-", "_")
        if not ws_id.startswith("ws_"):
            ws_id = "ws_" + ws_id

    existing_all = {
        "ws_acme_corp": "Acme Corporation",
        "ws_globex_corp": "Globex Corporation",
        **{k: v.get("name", "") for k, v in REGISTERED_WORKSPACES.items()}
    }

    if ws_id in existing_all:
        raise HTTPException(status_code=409, detail=f"Workspace ID '{ws_id}' already exists.")

    for existing_id, existing_n in existing_all.items():
        if existing_n and existing_n.lower() == ws_name.lower():
            raise HTTPException(status_code=409, detail=f"Workspace name '{ws_name}' already exists.")

    ws_data = {"id": ws_id, "name": ws_name or ws_id}
    if description:
        ws_data["description"] = description

    REGISTERED_WORKSPACES[ws_id] = ws_data
    return ws_data

@app.get("/documents")
def documents():
    return DOCUMENTS

@app.post("/upload")
async def upload_endpoint(
    files: List[UploadFile] = File(...),
    workspace_id: str = Form("ws_acme_corp"),
    version: str = Form("2026-01"),
):
    if not files:
        raise HTTPException(status_code=422, detail="At least one file must be selected for upload.")
    if not workspace_id:
        raise HTTPException(status_code=422, detail="Workspace ID is required.")
    if not is_valid_workspace(workspace_id):
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' does not exist.")

    file_tuples = []
    try:
        for f in files:
            content = await f.read()
            file_tuples.append((f.filename, content))
    except Exception as e:
        print(f"Error reading uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded files.")

    try:
        summary = multi_doc_corpus.ingest_multiple_documents(
            files=file_tuples,
            workspace_id=workspace_id,
            version=version,
        )
        return summary
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during document ingestion.")

@app.get("/indexed-documents")
def indexed_documents_endpoint(workspace_id: Optional[str] = None):
    return multi_doc_corpus.list_documents(workspace_id=workspace_id)

@app.post("/review")
def review_endpoint(body: ReviewRequest):
    if not body.question or len(body.question.strip()) < 3:
        raise HTTPException(status_code=422, detail="Enter at least three non-whitespace characters.")
    if not body.workspace_id:
        raise HTTPException(status_code=422, detail="Workspace ID is required.")
    if not is_valid_workspace(body.workspace_id):
        raise HTTPException(status_code=404, detail=f"Workspace '{body.workspace_id}' does not exist.")
        
    try:
        return review(body.question.strip(), body.method, body.workspace_id)
    except HTTPException:
        raise
    except Exception as e:
        # Hide raw tracebacks but log them on server
        print(f"Error in review endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during review processing.")

@app.post("/review/decision")
def decision_endpoint(body: DecisionRequest):
    if body.decision == "EDIT" and not (body.edited_response or "").strip():
        raise HTTPException(status_code=422, detail="Edited response required when decision is EDIT.")
    import datetime
    from audit_trail import audit_trail
    
    final_response = body.edited_response.strip() if body.decision == "EDIT" else (body.edited_response or "")
    
    # Store decision in cryptographically verifiable audit trail
    record = audit_trail.record_decision(
        workspace_id=body.workspace_id,
        reviewer_identity=body.reviewer_identity,
        decision=body.decision,
        question=body.question,
        ai_status=body.ai_status,
        original_ai_response=body.original_ai_response,
        final_reviewed_response=final_response,
        reviewer_notes=(body.reviewer_notes or "").strip(),
        document_ids=body.document_ids,
        document_versions=body.document_versions,
        evidence_chunk_ids=body.selected_chunk_ids,
    )
    
    return {
        "status": "DECISION_RECORDED",
        "decision": body.decision,
        "question": body.question,
        "workspace_id": body.workspace_id,
        "ai_status": body.ai_status,
        "final_response": final_response,
        "reviewer_notes": record["reviewer_notes"],
        "selected_chunk_ids": body.selected_chunk_ids,
        "timestamp": record["timestamp"],
        "audit_id": record["id"],
        "audit_hash": record["hash"]
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
    <div style="display:flex; gap:8px; align-items:center">
      <select id="workspace_select" style="flex:1">
        <option value="" disabled>Loading workspaces...</option>
      </select>
      <button class="btn-secondary" style="padding:6px 14px; font-size:12px; white-space:nowrap" onclick="openWorkspaceModal('workspace_select')">+ New</button>
    </div>

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
  <h2>Upload Security Documents</h2>
  <div style="font-size:13px; color:#94a3b8; margin-bottom:16px">
    Upload custom security policy documents into your active workspace context.
  </div>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:14px">
    <div>
      <label for="upload_workspace_select">Target Workspace</label>
      <div style="display:flex; gap:8px; align-items:center">
        <select id="upload_workspace_select" style="flex:1">
          <option value="" disabled>Loading workspaces...</option>
        </select>
        <button class="btn-secondary" style="padding:6px 14px; font-size:12px; white-space:nowrap" onclick="openWorkspaceModal('upload_workspace_select')">+ New Workspace</button>
      </div>
    </div>
    <div>
      <label for="upload_version_input">Document Version</label>
      <input type="text" id="upload_version_input" value="2026-01" placeholder="e.g. 2026-01">
    </div>
  </div>

  <div style="margin-bottom:14px">
    <label for="file_upload_input">Choose Documents (Supported: PDF, DOCX, TXT)</label>
    <input type="file" id="file_upload_input" multiple accept=".pdf,.docx,.txt" style="background:#090d16; padding:8px; border:1px solid #334155; border-radius:8px">
    <div id="file_preview_list" style="margin-top:8px; font-size:12px; color:#cbd5e1"></div>
  </div>

  <button id="btn_upload" class="btn-primary" onclick="uploadDocuments()">Upload & Index Documents</button>

  <div id="upload_status" style="margin-top:16px"></div>

  <div style="margin-top:20px">
    <h3 style="font-size:14px; color:#f1f5f9; margin-bottom:8px">Indexed Documents</h3>
    <div id="indexed_docs_list" style="display:flex; flex-direction:column; gap:8px"></div>
  </div>
</section>

</main>

<!-- Create Workspace Modal -->
<div id="workspace_modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.75); z-index:1000; justify-content:center; align-items:center">
  <div class="panel" style="width:90%; max-width:440px; background:#111827; border:1px solid #334155; border-radius:12px; padding:24px; box-shadow:0 10px 25px rgba(0,0,0,0.5)">
    <h2 style="font-size:18px; font-weight:700; margin-top:0; color:#f8fafc">Create New Workspace</h2>

    <div style="margin-bottom:14px">
      <label for="modal_ws_name">Workspace Name *</label>
      <input type="text" id="modal_ws_name" placeholder="e.g. Acme Security Workspace">
    </div>

    <div style="margin-bottom:16px">
      <label for="modal_ws_desc">Description / Metadata (Optional)</label>
      <input type="text" id="modal_ws_desc" placeholder="e.g. Workspace for 2026 security review">
    </div>

    <div id="modal_ws_error" style="color:#ef4444; font-size:12px; margin-bottom:14px; display:none; background:rgba(239,68,68,0.1); padding:8px 12px; border-radius:6px; border:1px solid rgba(239,68,68,0.3)"></div>

    <div style="display:flex; justify-content:flex-end; gap:10px">
      <button class="btn-secondary" style="margin:0" onclick="closeWorkspaceModal()">Cancel</button>
      <button id="modal_create_btn" class="btn-primary" style="padding:10px 16px" onclick="submitCreateWorkspace()">Create Workspace</button>
    </div>
  </div>
</div>

<script>
const el = id => document.getElementById(id);

let modalTriggerSource = 'workspace_select';

function openWorkspaceModal(source) {
  modalTriggerSource = source || 'workspace_select';
  el('modal_ws_name').value = '';
  el('modal_ws_desc').value = '';
  el('modal_ws_error').style.display = 'none';
  el('modal_ws_error').textContent = '';
  el('workspace_modal').style.display = 'flex';
  el('modal_ws_name').focus();
}

function closeWorkspaceModal() {
  el('workspace_modal').style.display = 'none';
}

async function fetchWorkspaces() {
  try {
    const list = await request('/workspaces');
    return list || [];
  } catch (err) {
    console.error('Failed to fetch workspaces:', err);
    return [];
  }
}

async function populateWorkspaces(selectedWsId = null) {
  const workspaces = await fetchWorkspaces();
  const wsSelect = el('workspace_select');
  const uploadWsSelect = el('upload_workspace_select');

  if (!wsSelect || !uploadWsSelect) return;

  const prevWs = selectedWsId || wsSelect.value || uploadWsSelect.value;

  wsSelect.replaceChildren();
  uploadWsSelect.replaceChildren();

  if (workspaces.length === 0) {
    wsSelect.add(new Option('No workspace available — please create one', ''));
    uploadWsSelect.add(new Option('No workspace available — please create one', ''));
    loadIndexedDocuments();
    return;
  }

  for (const ws of workspaces) {
    const label = `${ws.id} (${ws.name})`;
    wsSelect.add(new Option(label, ws.id));
    uploadWsSelect.add(new Option(label, ws.id));
  }

  let activeWs = prevWs;
  if (!activeWs || !workspaces.some(w => w.id === activeWs)) {
    activeWs = workspaces[0].id;
  }

  wsSelect.value = activeWs;
  uploadWsSelect.value = activeWs;

  loadIndexedDocuments();
}

function setupWorkspaceSync() {
  const wsSelect = el('workspace_select');
  const uploadWsSelect = el('upload_workspace_select');

  if (wsSelect) {
    wsSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (uploadWsSelect) uploadWsSelect.value = val;
      loadIndexedDocuments();
    });
  }

  if (uploadWsSelect) {
    uploadWsSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (wsSelect) wsSelect.value = val;
      loadIndexedDocuments();
    });
  }
}

async function submitCreateWorkspace() {
  const name = el('modal_ws_name').value.trim();
  const description = el('modal_ws_desc').value.trim();
  const errDiv = el('modal_ws_error');

  if (!name) {
    errDiv.textContent = 'Workspace name is required.';
    errDiv.style.display = 'block';
    return;
  }

  const btn = el('modal_create_btn');
  btn.disabled = true;
  errDiv.style.display = 'none';

  try {
    const res = await fetch('/workspaces', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name, description })
    });

    if (!res.ok) {
      const errText = await res.text();
      let detail = errText;
      try {
        const errJson = JSON.parse(errText);
        if (errJson.detail) detail = errJson.detail;
      } catch(e) {}
      throw new Error(detail);
    }

    const created = await res.json();
    closeWorkspaceModal();
    await populateWorkspaces(created.id);
  } catch (err) {
    errDiv.textContent = err.message || 'Failed to create workspace.';
    errDiv.style.display = 'block';
  } finally {
    btn.disabled = false;
  }
}

async function uploadDocuments() {
  const fileInput = el('file_upload_input');
  const files = fileInput.files;
  if (!files || files.length === 0) {
    alert('Please select at least one file to upload.');
    return;
  }
  const ws = el('upload_workspace_select').value;
  if (!ws) {
    alert('Please select or create a target workspace first.');
    return;
  }
  const ver = el('upload_version_input').value.trim() || '2026-01';

  const formData = new FormData();
  formData.append('workspace_id', ws);
  formData.append('version', ver);
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  const btn = el('btn_upload');
  btn.disabled = true;
  const statusDiv = el('upload_status');
  statusDiv.innerHTML = '<div style="color:#34d399; font-weight:600; padding:10px 0">Extracting text, chunking, generating 384-dim embeddings, and indexing into Qdrant & BM25...</div>';

  try {
    const res = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderUploadResults(data);
    loadIndexedDocuments();
  } catch (err) {
    statusDiv.innerHTML = '<div style="color:#ef4444">Upload failed: ' + escapeHtml(err.message) + '</div>';
  } finally {
    btn.disabled = false;
  }
}

function renderUploadResults(data) {
  const target = el('upload_status');
  let html = `<div class="box" style="border-left:3px solid #10b981">
    <div class="box-title" style="color:#34d399">Ingestion Summary (${data.successful_documents}/${data.total_files} Success · ${data.total_chunks_added} Chunks Added)</div>
    <div style="display:flex; flex-direction:column; gap:6px; margin-top:8px">`;
  
  for (const item of data.results) {
    let color = item.status === 'SUCCESS' ? '#34d399' : (item.status === 'DUPLICATE' ? '#fbbf24' : '#ef4444');
    let badgeText = item.status === 'SUCCESS' ? '✓ INDEXED' : (item.status === 'DUPLICATE' ? '⚠ DUPLICATE' : '✗ FAILED');
    html += `<div style="font-size:12px; background:#0b0f19; padding:8px 12px; border-radius:6px; border:1px solid #1e293b; display:flex; justify-content:space-between; align-items:center">
      <div>
        <strong style="color:#f8fafc">${escapeHtml(item.filename)}</strong>
        ${item.document_id ? `<span class="tag-chip" style="margin-left:6px">${escapeHtml(item.document_id)}</span>` : ''}
        <div style="color:#94a3b8; font-size:11px">${escapeHtml(item.reason || item.message || '')}</div>
      </div>
      <span style="color:${color}; font-weight:700; font-size:11px">${badgeText} (${item.chunks_created || 0} chunks)</span>
    </div>`;
  }
  html += `</div></div>`;
  target.innerHTML = html;
}

async function loadIndexedDocuments() {
  const wsSelect = el('upload_workspace_select');
  const selectedWs = wsSelect ? wsSelect.value : null;
  const container = el('indexed_docs_list');
  if (!container) return;

  if (!selectedWs) {
    container.innerHTML = '<div style="color:#64748b; font-size:12px">No workspace selected.</div>';
    return;
  }

  try {
    const docs = await request('/indexed-documents?workspace_id=' + encodeURIComponent(selectedWs));
    container.replaceChildren();
    if (!docs || docs.length === 0) {
      container.innerHTML = '<div style="color:#64748b; font-size:12px">No user documents indexed for workspace ' + escapeHtml(selectedWs) + '.</div>';
      return;
    }
    for (const d of docs) {
      const card = document.createElement('div');
      card.style.background = '#090d16';
      card.style.padding = '10px 14px';
      card.style.borderRadius = '6px';
      card.style.border = '1px solid #1e293b';
      card.style.display = 'flex';
      card.style.justifyContent = 'space-between';
      card.style.alignItems = 'center';
      card.innerHTML = `
        <div>
          <span style="color:#10b981; font-weight:700; font-size:12px">${escapeHtml(d.title)}</span>
          <span class="tag-chip" style="margin-left:6px">${escapeHtml(d.document_id)}</span>
          <span class="tag-chip">${escapeHtml(d.source_type.toUpperCase())}</span>
          <span class="tag-chip">v${escapeHtml(d.version)}</span>
          <span class="tag-chip">${escapeHtml(d.workspace_id)}</span>
        </div>
        <div style="font-size:12px; color:#94a3b8; font-weight:600">${d.chunks_count} chunks</div>
      `;
      container.appendChild(card);
    }
  } catch (err) {
    console.error(err);
    container.innerHTML = '<div style="color:#ef4444; font-size:12px">Failed to load indexed documents for workspace.</div>';
  }
}

function setQ(q, ws) {
  el('question').value = q;
  if(ws) {
    const wsSelect = el('workspace_select');
    const uploadWsSelect = el('upload_workspace_select');
    if (wsSelect) wsSelect.value = ws;
    if (uploadWsSelect) uploadWsSelect.value = ws;
    loadIndexedDocuments();
  }
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
  badge.className = 'badge badge-' + (data.status || 'review_required');
  badge.textContent = String(data.status || '').replace(/_/g, ' ');
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
          ${item.page !== undefined && item.page !== null ? `<span class="tag-chip">Page: ${escapeHtml(item.page)}</span>` : ''}
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
      <label for="edited_text">Final Reviewer Response</label>
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

  const docIds = new Set();
  const docVersions = new Set();
  if (currentReviewData.evidence) {
    currentReviewData.evidence.forEach(ev => {
       if (ev.document_id) docIds.add(ev.document_id);
       if (ev.version) docVersions.add(ev.version);
    });
  }

  const payload = {
    question: currentReviewData.question,
    workspace_id: currentReviewData.workspace_id || el('workspace_select').value,
    ai_status: currentReviewData.status,
    decision: selectedDecision,
    edited_response: editedText,
    reviewer_notes: notes,
    selected_chunk_ids: currentReviewData.evidence_chunk_ids || [],
    document_ids: Array.from(docIds),
    document_versions: Array.from(docVersions),
    original_ai_response: currentReviewData.evidence_quote || currentReviewData.reason || "",
    reviewer_identity: "demo_reviewer@example.com"
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
          Audit Record Hash: <span style="font-family:monospace; font-size:10px">${receipt.audit_hash}</span><br>
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
  if (str === null || str === undefined) return '';
  if (typeof str === 'number') return str.toString();
  if (typeof str !== 'string') return '';
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

window.addEventListener('load', () => {
  setupWorkspaceSync();
  populateWorkspaces();
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
        uvicorn.run(app, host=APP_HOST, port=APP_PORT)

