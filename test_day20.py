import os
import sys
import json

from fastapi.testclient import TestClient

# Add current dir to path to import evidencedesk
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from evidencedesk import app, qdrant, multi_doc_corpus
from test_day19_75 import create_sample_pdf, create_sample_docx, create_sample_txt

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "application" in data["components"]
    assert data["components"]["qdrant"] == "ok"
    assert data["components"]["embedding_model"] == "ok"
    assert data["components"]["reranker_model"] == "ok"

def test_empty_upload():
    response = client.post("/upload", data={"workspace_id": "ws_test", "version": "1.0"})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_unsupported_upload():
    response = client.post("/upload", 
        data={"workspace_id": "ws_test", "version": "1.0"},
        files=[("files", ("test.png", b"fake png data", "image/png"))]
    )
    assert response.status_code == 200 # It returns 200 with FAILED status inside
    data = response.json()
    assert data["total_files"] == 1
    assert data["successful_documents"] == 0
    assert data["results"][0]["status"] == "FAILED"
    assert "Unsupported" in data["results"][0]["reason"]

def test_invalid_document_handling():
    response = client.post("/upload", 
        data={"workspace_id": "ws_test", "version": "1.0"},
        files=[("files", ("test.pdf", b"fake pdf data", "application/pdf"))]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["successful_documents"] == 0
    assert data["results"][0]["status"] == "FAILED"
    assert "Corrupt or invalid PDF" in data["results"][0]["reason"]

def test_valid_uploads_and_multi():
    pdf_bytes = create_sample_pdf("v_test")
    docx_bytes = create_sample_docx("v_test")
    txt_bytes = create_sample_txt("v_test")

    # Upload multiple
    response = client.post("/upload",
        data={"workspace_id": "ws_day20", "version": "2026-02"},
        files=[
            ("files", ("doc1.pdf", pdf_bytes, "application/pdf")),
            ("files", ("doc2.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("doc3.txt", txt_bytes, "text/plain"))
        ]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] == 3
    assert data["successful_documents"] == 3
    assert data["total_chunks_added"] > 0
    
    # Test Workspace Isolation through endpoint
    res_docs = client.get("/indexed-documents?workspace_id=ws_day20")
    docs = res_docs.json()
    assert len(docs) >= 3

    res_other = client.get("/indexed-documents?workspace_id=ws_empty")
    assert len(res_other.json()) == 0

def test_existing_review_workflow():
    response = client.post("/review", json={
        "question": "What is the backup retention period?",
        "workspace_id": "ws_acme_corp",
        "method": "hybrid"
    })
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ANSWERABLE", "AMBIGUOUS", "CONFLICTING", "INSUFFICIENT_EVIDENCE"]
    assert "evidence" in data

def test_existing_review_decision_workflow():
    response = client.post("/review/decision", json={
        "question": "What is the backup retention period?",
        "workspace_id": "ws_acme_corp",
        "ai_status": "ANSWERABLE",
        "decision": "APPROVE",
        "edited_response": "",
        "reviewer_notes": "Looks good",
        "selected_chunk_ids": ["chk_sec003_002"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DECISION_RECORDED"
    assert data["decision"] == "APPROVE"
    assert data["reviewer_notes"] == "Looks good"

if __name__ == "__main__":
    print("Running Day 20 Tests...")
    test_health_endpoint()
    test_empty_upload()
    test_unsupported_upload()
    test_invalid_document_handling()
    test_valid_uploads_and_multi()
    test_existing_review_workflow()
    test_existing_review_decision_workflow()
    print("All Day 20 tests passed!")
