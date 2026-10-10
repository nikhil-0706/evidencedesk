import re
from uuid import uuid4

DOCUMENTS = [
    {"id": "SEC-001", "title": "Access control policy", "version": "2026-01", "text": "Employees must use multi-factor authentication (MFA) to access production systems. Access permissions are reviewed quarterly. Administrative access requires manager approval.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-002", "title": "Encryption policy", "version": "2026-01", "text": "Customer data is encrypted at rest using AES-256. Data in transit is protected using TLS 1.2 or later. Encryption keys are managed in a dedicated key management service.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-003", "title": "Backup policy", "version": "2026-01", "text": "Database backups are created daily. Backup retention is 30 days. Restore procedures are tested quarterly.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-004", "title": "Incident response policy", "version": "2026-01", "text": "Security incidents are triaged by the on-call engineer. Confirmed incidents are escalated to the security lead. This policy does not specify a contractual customer notification deadline.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-005", "title": "Backup retention exception", "version": "2026-01",  "text": "Database backup retention is 90 days.", "workspace_id": "ws_acme_corp"},
    {"id": "SEC-G001", "title": "Globex Production Policy", "version": "2026-01", "text": "Production infrastructure is hosted on ExampleCloud.", "workspace_id": "ws_globex_corp"}
]

CASES = [
    {"question": "What encryption protects customer data at rest?", "expected": "SEC-002"},
    {"question": "What is the backup retention period?", "expected": "SEC-003"},
    {"question": "Is multi-factor authentication required for production systems?", "expected": "SEC-001"},
    {"question": "Who triages security incidents?", "expected": "SEC-004"},
    {"question": "Are you SOC 2 certified?", "expected": None},
    {"question": "What is your annual revenue?", "expected": None},
]

PASSAGES = []
for doc in DOCUMENTS:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", doc["text"]) if s.strip()]
    for idx, sentence in enumerate(sentences, start=1):
        PASSAGES.append({
            "chunk_id": f"chk_{doc['id'].lower().replace('-', '')}_{idx:03d}",
            "workspace_id": doc["workspace_id"],
            "document_id": doc["id"],
            "version": doc["version"],
            "title": doc["title"],
            "excerpt": sentence,
            "chunk_index": idx - 1,
            "source_type": "legacy_fixture"
        })

def inject_legacy_fixtures(corpus, qdrant_client, embed_model, collection_name):
    # Avoid duplicate injection
    if any(c.get("source_type") == "legacy_fixture" for c in corpus.chunks):
        return

    # Add to corpus chunks
    corpus.chunks.extend(PASSAGES)
    
    # Rebuild BM25 for corpus
    from rank_bm25 import BM25Okapi
    from document_ingestion import _tokenize
    if corpus.chunks:
        tokenized = [_tokenize(c.get("excerpt", "")) for c in corpus.chunks]
        corpus.bm25_model = BM25Okapi(tokenized)
    
    # Add to Qdrant
    from qdrant_client.http import models
    points = []
    for p in PASSAGES:
        vec = embed_model.encode(p["excerpt"]).tolist()
        points.append(
            models.PointStruct(
                id=uuid4().hex,
                vector=vec,
                payload=p
            )
        )
    import evidencedesk
    evidencedesk.REGISTERED_WORKSPACES["ws_acme_corp"] = {"id": "ws_acme_corp", "name": "Acme Corporation"}
    evidencedesk.REGISTERED_WORKSPACES["ws_globex_corp"] = {"id": "ws_globex_corp", "name": "Globex Corporation"}

    if points:
        corpus._ensure_collection()
        qdrant_client.upsert(collection_name=collection_name, points=points)
