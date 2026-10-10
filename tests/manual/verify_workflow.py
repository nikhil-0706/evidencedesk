"""
verify_workflow.py — Full manual workflow validation script for EvidenceDesk.
Tests:
1. Workspace creation via API.
2. Dropdown population via GET /workspaces.
3. Uploading document into newly created workspace.
4. Scoping GET /indexed-documents to the workspace.
5. Asking question supported by uploaded document (ANSWERABLE + source quote).
6. Asking question absent from document (INSUFFICIENT_EVIDENCE + zero evidence leakage, no Acme fallback).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient
from evidencedesk import app
import json

client = TestClient(app)

import uuid
unique_suffix = uuid.uuid4().hex[:6]
ws_name_input = f"Fintech Security {unique_suffix}"

print("=== STEP 1 & 2 & 3: Create Workspace ===")
ws_res = client.post("/workspaces", json={"name": ws_name_input, "description": "Fintech Core workspace"})
assert ws_res.status_code == 200, f"Workspace creation failed: {ws_res.text}"
ws_data = ws_res.json()
ws_id = ws_data["id"]
ws_name = ws_data["name"]
print(f"Created workspace: {ws_id} ({ws_name})")

print("=== STEP 4: Confirm both dropdowns have it via GET /workspaces ===")
list_res = client.get("/workspaces")
workspaces = list_res.json()
ws_ids = [w["id"] for w in workspaces]
assert ws_id in ws_ids, f"{ws_id} not in workspace list"
print(f"Confirmed workspace {ws_id} in {len(workspaces)} registered workspaces.")

print("=== STEP 5: Upload document into workspace ===")
file_content = b"Patient Health Policy: Patient medical records must be retained in active storage for at least 7 years. Access logs are reviewed annually."
upload_res = client.post(
    "/upload",
    data={"workspace_id": ws_id, "version": "2026-Q1"},
    files={"files": (f"policy_{unique_suffix}.txt", file_content, "text/plain")}
)
assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
upload_summary = upload_res.json()
print(f"Upload summary: {upload_summary['successful_documents']} docs indexed, {upload_summary['total_chunks_added']} chunks added.")

# Verify indexed documents list is scoped to workspace
docs_res = client.get(f"/indexed-documents?workspace_id={ws_id}")
docs = docs_res.json()
assert len(docs) == 1, "Expected 1 document in workspace"
print(f"Indexed documents for {ws_id}: {[d['title'] for d in docs]}")

print("=== STEP 6 & 7: Ask question supported by document ===")
q_supported = "How long must patient medical records be retained?"
rev_res = client.post("/review", json={"question": q_supported, "workspace_id": ws_id, "method": "hybrid"})
assert rev_res.status_code == 200, f"Review failed: {rev_res.text}"
rev_data = rev_res.json()
print(f"Status: {rev_data['status']}")
print(f"Evidence Quote: {rev_data.get('evidence_quote')}")
print(f"Reason: {rev_data.get('reason')}")
print(f"Evidence Chunks: {[e['excerpt'] for e in rev_data.get('evidence', [])]}")
assert rev_data["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {rev_data['status']}"
assert "7 years" in rev_data["evidence_quote"].lower(), "Expected 7 years in evidence quote"
assert all(e["workspace_id"] == ws_id for e in rev_data.get("evidence", [])), "Cross-workspace leakage in evidence chunks!"

print("=== STEP 8 & 9: Ask question absent from document (No Acme fallback) ===")
q_absent = "What is your annual revenue?"
absent_res = client.post("/review", json={"question": q_absent, "workspace_id": ws_id, "method": "hybrid"})
assert absent_res.status_code == 200, f"Review failed: {absent_res.text}"
absent_data = absent_res.json()
print(f"Status for absent query: {absent_data['status']}")
print(f"Evidence Chunks: {len(absent_data.get('evidence', []))}")
assert absent_data["status"] == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {absent_data['status']}"
assert len(absent_data.get("evidence", [])) == 0 or all(e["workspace_id"] == ws_id for e in absent_data.get("evidence", [])), "Leaked foreign evidence!"

print("\nALL VERIFICATION STEPS PASSED PERFECTLY!")
