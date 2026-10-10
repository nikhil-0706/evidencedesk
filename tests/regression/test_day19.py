"""
test_day19.py — Day 19 Human Review Workspace & Governance Verification

Verifies:
1. TEST 1: ANSWERABLE state (Question: "Who takes the first look at a reported security incident?")
2. TEST 2: PARAPHRASED ANSWERABLE state ("What safeguards customer information while it moves between systems?")
3. TEST 3: AMBIGUOUS state ("What is your retention policy?")
4. TEST 4: CONFLICTING state ("How long are database backups retained?" — both 30d & 90d evidence visible)
5. TEST 5: INSUFFICIENT_EVIDENCE state ("What is your SLA for resolving high-severity security incidents?")
6. TEST 6: PROVENANCE payload completeness (workspace_id, document_id, title, version, chunk_id, excerpt)
7. TEST 7: WORKSPACE ISOLATION in review payload (Acme review excludes Globex data)
8. TEST 8: HUMAN REVIEW ACTIONS (Approve, Edit, Reject via /review/decision endpoint)
"""

import sys
import json

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

from evidencedesk import review, decision_endpoint, DecisionRequest

def test_01_answerable_state():
    """TEST 1: ANSWERABLE state."""
    question = "Who takes the first look at a reported security incident?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    assert res["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {res['status']}"
    assert "reason" in res and res["reason"]
    assert "evidence_quote" in res and res["evidence_quote"]
    assert "on-call engineer" in res["evidence_quote"].lower() or "triaged" in res["evidence_quote"].lower()
    assert len(res["evidence"]) > 0

    print("  [TEST 1] ANSWERABLE state: PASSED")


def test_02_paraphrased_answerable_state():
    """TEST 2: PARAPHRASED ANSWERABLE state."""
    question = "What safeguards customer information while it moves between systems?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    assert res["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {res['status']}"
    assert "TLS 1.2" in res["evidence_quote"] or "data in transit" in res["reason"].lower() or "tls" in res["reason"].lower()
    
    print("  [TEST 2] PARAPHRASED ANSWERABLE state: PASSED")


def test_03_ambiguous_state():
    """TEST 3: AMBIGUOUS state."""
    question = "What is your retention policy?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    assert res["status"] == "AMBIGUOUS", f"Expected AMBIGUOUS, got {res['status']}"
    assert "retention" in res["reason"].lower()
    assert res["evidence_quote"] == "", "AMBIGUOUS status must not generate an arbitrary quote"

    print("  [TEST 3] AMBIGUOUS state: PASSED")


def test_04_conflicting_state():
    """TEST 4: CONFLICTING state — both 30-day and 90-day evidence visible."""
    question = "How long are database backups retained?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    assert res["status"] == "CONFLICTING", f"Expected CONFLICTING, got {res['status']}"
    
    # Assert both conflicting evidence pieces are present in retrieved evidence list
    excerpts = [item["excerpt"] for item in res["evidence"]]
    has_30d = any("30 days" in exc for exc in excerpts)
    has_90d = any("90 days" in exc for exc in excerpts)

    assert has_30d, "CONFLICTING review must retain 30-day evidence piece"
    assert has_90d, "CONFLICTING review must retain 90-day evidence piece"

    print("  [TEST 4] CONFLICTING state: PASSED")


def test_05_insufficient_evidence_state():
    """TEST 5: INSUFFICIENT_EVIDENCE state."""
    question = "What is your SLA for resolving high-severity security incidents?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    assert res["status"] == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {res['status']}"
    assert res["evidence_quote"] == "", "INSUFFICIENT_EVIDENCE must have empty evidence quote"

    print("  [TEST 5] INSUFFICIENT_EVIDENCE state: PASSED")


def test_06_provenance_completeness():
    """TEST 6: PROVENANCE payload completeness."""
    question = "Who triages security incidents?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    assert "workspace_id" in res
    assert res["workspace_id"] == "ws_acme_corp"
    assert "evidence" in res and len(res["evidence"]) > 0

    for item in res["evidence"]:
        assert "workspace_id" in item and item["workspace_id"]
        assert "document_id" in item and item["document_id"]
        assert "title" in item and item["title"]
        assert "version" in item and item["version"]
        assert "chunk_id" in item and item["chunk_id"]
        assert "excerpt" in item and item["excerpt"]

    print("  [TEST 6] PROVENANCE payload completeness: PASSED")


def test_07_workspace_isolation_review():
    """TEST 7: WORKSPACE ISOLATION in review payload."""
    question = "Where is production infrastructure hosted?"
    res = review(question, workspace_id="ws_acme_corp", method="hybrid")

    for item in res["evidence"]:
        assert item["workspace_id"] == "ws_acme_corp"
        assert item["document_id"] != "SEC-G001"

    print("  [TEST 7] WORKSPACE ISOLATION in review payload: PASSED")


def test_08_human_review_actions():
    """TEST 8: HUMAN REVIEW ACTIONS (Approve, Edit, Reject)."""
    # 1. Approve Decision
    req_approve = DecisionRequest(
        question="Who takes the first look at a reported security incident?",
        workspace_id="ws_acme_corp",
        ai_status="ANSWERABLE",
        decision="APPROVE",
        reviewer_notes="Verified against SEC-004 policy."
    )
    rec_approve = decision_endpoint(req_approve)
    assert rec_approve["status"] == "DECISION_RECORDED"
    assert rec_approve["decision"] == "APPROVE"
    assert rec_approve["reviewer_notes"] == "Verified against SEC-004 policy."

    # 2. Edit Decision
    req_edit = DecisionRequest(
        question="Who takes the first look at a reported security incident?",
        workspace_id="ws_acme_corp",
        ai_status="ANSWERABLE",
        decision="EDIT",
        edited_response="Security incidents are triaged by on-call engineers 24/7.",
        reviewer_notes="Clarified operational schedule."
    )
    rec_edit = decision_endpoint(req_edit)
    assert rec_edit["status"] == "DECISION_RECORDED"
    assert rec_edit["decision"] == "EDIT"
    assert rec_edit["final_response"] == "Security incidents are triaged by on-call engineers 24/7."

    # 3. Reject Decision
    req_reject = DecisionRequest(
        question="Is the company SOC 2 certified?",
        workspace_id="ws_acme_corp",
        ai_status="INSUFFICIENT_EVIDENCE",
        decision="REJECT",
        reviewer_notes="Rejected AI recommendation — requested manual proof attachment."
    )
    rec_reject = decision_endpoint(req_reject)
    assert rec_reject["status"] == "DECISION_RECORDED"
    assert rec_reject["decision"] == "REJECT"

    print("  [TEST 8] HUMAN REVIEW ACTIONS (Approve, Edit, Reject): PASSED")


def main():
    print("============================================================")
    print("RUNNING DAY 19 HUMAN REVIEW WORKSPACE & GOVERNANCE TESTS")
    print("============================================================\n")

    test_01_answerable_state()
    test_02_paraphrased_answerable_state()
    test_03_ambiguous_state()
    test_04_conflicting_state()
    test_05_insufficient_evidence_state()
    test_06_provenance_completeness()
    test_07_workspace_isolation_review()
    test_08_human_review_actions()

    print("\n============================================================")
    print("ALL 8 DAY 19 HUMAN REVIEW WORKSPACE TESTS PASSED!")
    print("============================================================")

if __name__ == "__main__":
    main()
