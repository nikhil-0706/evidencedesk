"""
test_workspace_upload_isolation.py — Regression tests for dynamic workspace creation,
document upload routing, workspace-scoped indexing, SLM context isolation, and cross-workspace security.
"""

import sys
import os
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evidencedesk import (
    app,
    create_workspace,
    list_workspaces,
    review,
    multi_doc_corpus,
    REGISTERED_WORKSPACES,
)
from fastapi.testclient import TestClient

client = TestClient(app)


class TestWorkspaceUploadIsolation(unittest.TestCase):

    def setUp(self):
        # Clear extra registered workspaces for test isolation
        REGISTERED_WORKSPACES.clear()

    def test_01_workspace_creation_api(self):
        """Test workspace creation from API with server-generated workspace ID."""
        # 1. Create workspace with name
        payload = {"name": "Security Audit 2026", "description": "Audit team workspace"}
        response = client.post("/workspaces", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertTrue(data["id"].startswith("ws_"))
        self.assertEqual(data["name"], "Security Audit 2026")
        self.assertEqual(data["description"], "Audit team workspace")

        # 2. Verify workspace appears in GET /workspaces list
        res_list = client.get("/workspaces")
        self.assertEqual(res_list.status_code, 200)
        workspaces = res_list.json()
        ws_ids = [w["id"] for w in workspaces]
        self.assertIn(data["id"], ws_ids)
        self.assertIn("ws_acme_corp", ws_ids)
        self.assertIn("ws_globex_corp", ws_ids)

        # 3. Verify duplicate creation produces clear 409 error
        dup_res = client.post("/workspaces", json={"name": "Security Audit 2026"})
        self.assertEqual(dup_res.status_code, 409)

        dup_id_res = client.post("/workspaces", json={"id": data["id"], "name": "Different Name"})
        self.assertEqual(dup_id_res.status_code, 409)

    def test_02_upload_routing_and_indexed_document_scoping(self):
        """Test document upload routing and scoped indexed documents list."""
        # Create fresh workspace
        ws_res = client.post("/workspaces", json={"name": "Finance Vault"})
        self.assertEqual(ws_res.status_code, 200)
        ws_id = ws_res.json()["id"]

        # Ingest document into Finance Vault
        doc_content = "Finance Policy: All wire transfers exceeding 50,000 USD require CFO approval.".encode("utf-8")
        ingest_res = multi_doc_corpus.ingest_single_document(
            filename="fin_policy.txt",
            file_bytes=doc_content,
            workspace_id=ws_id,
            version="2026-01"
        )
        self.assertEqual(ingest_res["status"], "SUCCESS")
        self.assertEqual(ingest_res["workspace_id"], ws_id)

        # Verify indexed documents list for Finance Vault contains 1 document
        fin_docs = multi_doc_corpus.list_documents(workspace_id=ws_id)
        self.assertEqual(len(fin_docs), 1)
        self.assertEqual(fin_docs[0]["workspace_id"], ws_id)

        # Verify Acme workspace does NOT list this Finance document
        acme_docs = multi_doc_corpus.list_documents(workspace_id="ws_acme_corp")
        acme_doc_ids = [d["document_id"] for d in acme_docs]
        self.assertNotIn(ingest_res["document_id"], acme_doc_ids)

    def test_03_end_to_end_reasoning_in_new_workspace(self):
        """Upload a synthetic fact into fresh workspace, query it, and test answer vs abstention."""
        ws_res = client.post("/workspaces", json={"name": "Healthcare Policy"})
        ws_id = ws_res.json()["id"]

        txt_data = "HIPAA Compliance Excerpt: Patient health records must be retained in encrypted storage for at least 7 years.".encode("utf-8")
        multi_doc_corpus.ingest_single_document(
            filename="hipaa.txt",
            file_bytes=txt_data,
            workspace_id=ws_id,
            version="2026-01"
        )

        # Query supported fact
        res_answer = review("How long must patient health records be retained?", method="hybrid", workspace_id=ws_id)
        self.assertEqual(res_answer["status"], "ANSWERABLE")
        self.assertIn("7 years", res_answer["evidence_quote"].lower())
        self.assertEqual(res_answer["workspace_id"], ws_id)
        for ev in res_answer["evidence"]:
            self.assertEqual(ev["workspace_id"], ws_id)

        # Query absent fact — must trigger safe abstention (INSUFFICIENT_EVIDENCE), no Acme fallback
        res_absent = review("What is the penalty for HIPAA violations?", method="hybrid", workspace_id=ws_id)
        self.assertEqual(res_absent["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res_absent["workspace_id"], ws_id)

    def test_04_cross_workspace_isolation(self):
        """Create two workspaces with conflicting facts and verify zero evidence leakage."""
        ws_a = client.post("/workspaces", json={"name": "Workspace Alpha"}).json()["id"]
        ws_b = client.post("/workspaces", json={"name": "Workspace Beta"}).json()["id"]

        doc_a = "Cloud Infra: Primary production servers are hosted on AWS us-east-1.".encode("utf-8")
        doc_b = "Cloud Infra: Primary production servers are hosted on Azure westeurope.".encode("utf-8")

        multi_doc_corpus.ingest_single_document("cloud_a.txt", doc_a, workspace_id=ws_a)
        multi_doc_corpus.ingest_single_document("cloud_b.txt", doc_b, workspace_id=ws_b)

        # Query Workspace Alpha
        res_a = review("Where are primary production servers hosted?", method="hybrid", workspace_id=ws_a)
        self.assertEqual(res_a["status"], "ANSWERABLE")
        self.assertIn("aws us-east-1", res_a["evidence_quote"].lower())
        for ev in res_a["evidence"]:
            self.assertEqual(ev["workspace_id"], ws_a)
            self.assertNotIn("Azure", ev["excerpt"])

        # Query Workspace Beta
        res_b = review("Where are primary production servers hosted?", method="hybrid", workspace_id=ws_b)
        self.assertEqual(res_b["status"], "ANSWERABLE")
        self.assertIn("azure westeurope", res_b["evidence_quote"].lower())
        for ev in res_b["evidence"]:
            self.assertEqual(ev["workspace_id"], ws_b)
            self.assertNotIn("AWS", ev["excerpt"])

        # Query invalid workspace
        err_res = client.post("/review", json={"question": "Where are production servers?", "workspace_id": "ws_nonexistent"})
        self.assertEqual(err_res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
