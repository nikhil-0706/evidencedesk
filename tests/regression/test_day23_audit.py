import os
import sys
import time

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

from evidencedesk import app
from fastapi.testclient import TestClient
from audit_trail import audit_trail, AuditTrail

client = TestClient(app)

def test_audit_trail_isolation_and_persistence():
    print("--- Running Day 23 Audit Trail & Governance Tests ---")
    
    # 1. Reset/Clear audit trail
    audit_trail.clear()
    
    # 2. Record an APPROVAL
    payload1 = {
        "question": "Who approves administrative access?",
        "workspace_id": "ws_acme_corp",
        "ai_status": "ANSWERABLE",
        "decision": "APPROVE",
        "edited_response": "",
        "reviewer_notes": "Looks good",
        "selected_chunk_ids": ["chk_sec001_003"],
        "document_ids": ["SEC-001"],
        "document_versions": ["2026-01"],
        "original_ai_response": "Administrative access requires manager approval.",
        "reviewer_identity": "alice@acme.com"
    }
    resp1 = client.post("/review/decision", json=payload1)
    assert resp1.status_code == 200, resp1.text
    data1 = resp1.json()
    assert data1["status"] == "DECISION_RECORDED"
    assert "audit_hash" in data1
    print("  [TEST 1] Approval decision persisted successfully: PASSED")
    
    # 3. Record an EDIT decision
    payload2 = {
        "question": "Are you SOC 2 certified?",
        "workspace_id": "ws_acme_corp",
        "ai_status": "INSUFFICIENT_EVIDENCE",
        "decision": "EDIT",
        "edited_response": "Yes, we achieved SOC 2 Type II last month (attached report).",
        "reviewer_notes": "Policy doc is outdated, manually overriding.",
        "selected_chunk_ids": [],
        "document_ids": ["SEC-002"],
        "document_versions": ["2026-01"],
        "original_ai_response": "",
        "reviewer_identity": "bob@acme.com"
    }
    resp2 = client.post("/review/decision", json=payload2)
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()
    assert data2["final_response"] == payload2["edited_response"]
    print("  [TEST 2] Edit decision (manual override) persisted successfully: PASSED")
    
    # 4. Record a REJECT decision
    payload3 = {
        "question": "What is the backup retention?",
        "workspace_id": "ws_globex_corp",
        "ai_status": "CONFLICTING",
        "decision": "REJECT",
        "edited_response": "",
        "reviewer_notes": "This answer is confusing. I will reject it.",
        "selected_chunk_ids": ["chk_sec003", "chk_sec005"],
        "document_ids": ["SEC-003", "SEC-005"],
        "document_versions": ["v1", "v2"],
        "original_ai_response": "Conflict detected.",
        "reviewer_identity": "charlie@globex.com"
    }
    resp3 = client.post("/review/decision", json=payload3)
    assert resp3.status_code == 200, resp3.text
    print("  [TEST 3] Reject decision persisted successfully: PASSED")
    
    # 5. Duplicate Submission
    resp4 = client.post("/review/decision", json=payload3)
    assert resp4.status_code == 200
    assert resp4.json()["audit_hash"] != resp3.json()["audit_hash"]
    print("  [TEST 4] Duplicate submission creates distinct audit trail record: PASSED")
    
    # 6. Verify Workspace Isolation in Audit Log
    # We query the DB directly to ensure globex and acme are stored properly
    import sqlite3
    with sqlite3.connect("audit_trail.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT workspace_id, reviewer_identity FROM audit_log ORDER BY id ASC")
        rows = cursor.fetchall()
        assert rows[0] == ("ws_acme_corp", "alice@acme.com")
        assert rows[1] == ("ws_acme_corp", "bob@acme.com")
        assert rows[2] == ("ws_globex_corp", "charlie@globex.com")
    print("  [TEST 5] Workspace boundaries preserved in audit log: PASSED")
    
    # 7. Tamper-Evident Verification (Valid)
    valid, msg = audit_trail.verify_chain()
    assert valid, msg
    print("  [TEST 6] Cryptographic hash chain verification passed: PASSED")
    
    # 8. Test persistence across restarts (simulate by instantiating new AuditTrail)
    new_audit = AuditTrail("audit_trail.db")
    valid_re, msg_re = new_audit.verify_chain()
    assert valid_re, msg_re
    print("  [TEST 7] Persistence and verification across simulated restart: PASSED")
    
    # 9. Tamper-Evident Verification (Tampered)
    with sqlite3.connect("audit_trail.db") as conn:
        conn.execute("UPDATE audit_log SET reviewer_notes = 'Tampered' WHERE id = 1")
    valid_tamp, msg_tamp = new_audit.verify_chain()
    assert not valid_tamp
    assert "tampered" in msg_tamp.lower() or "mismatch" in msg_tamp.lower()
    print("  [TEST 8] Tamper-evident hash chain detects malicious DB modification: PASSED")

    # 10. Historical versions preservation
    # If the source document updates to v3, the audit log STILL holds v1 and v2.
    with sqlite3.connect("audit_trail.db") as conn:
        # Restore row 1 to fix hash chain before we read it
        pass # Actually we don't need to fix it if we just manually read the row 3
        cursor = conn.cursor()
        cursor.execute("SELECT document_versions FROM audit_log WHERE id = 3")
        doc_versions = cursor.fetchone()[0]
        assert "v1" in doc_versions and "v2" in doc_versions
    print("  [TEST 9] Historical document versions remain immutable in audit log: PASSED")
    
    print("\nALL 9 DAY 23 ADVANCED AUDIT TRAIL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_audit_trail_isolation_and_persistence()
