import json
from evidencedesk import review, rerank_evidence, PASSAGES, rrf_fuse, retrieve_by_bm25, retrieve_by_embedding

def test_pipeline_and_debug():
    print("\n--- DIAGNOSTIC MODE TEST ---")
    result = review("How is data in transit protected?", method="hybrid", debug=True)
    assert result["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {result['status']}"
    assert result["evidence"][0]["document_id"] == "SEC-002"

def test_all_smoke_tests():
    print("\n--- RUNNING SMOKE TESTS ---")
    tests = [
        {
            "name": "TEST 1",
            "question": "What encryption standard is used to protect customer data at rest?",
            "expected_status": "ANSWERABLE",
            "expected_docs": ["SEC-002"]
        },
        {
            "name": "TEST 2",
            "question": "How is data in transit protected?",
            "expected_status": "ANSWERABLE",
            "expected_docs": ["SEC-002"]
        },
        {
            "name": "TEST 3",
            "question": "What safeguards customer information while it moves between systems?",
            "expected_status": "ANSWERABLE",
            "expected_docs": ["SEC-002"]
        },
        {
            "name": "TEST 4",
            "question": "How regularly do you verify that stored backups can actually be restored?",
            "expected_status": "INSUFFICIENT_EVIDENCE",
            "expected_docs": []
        },
        {
            "name": "TEST 5",
            "question": "What is your retention policy?",
            "expected_status": "AMBIGUOUS",
            "expected_docs": []
        },
        {
            "name": "TEST 6",
            "question": "How long are database backups retained?",
            "expected_status": "CONFLICTING",
            "expected_docs": ["SEC-003", "SEC-005"]
        },
        {
            "name": "TEST 7",
            "question": "Is the company SOC 2 certified?",
            "expected_status": "INSUFFICIENT_EVIDENCE",
            "expected_docs": []
        }
    ]
    
    for t in tests:
        print(f"\nRunning {t['name']}: {t['question']}")
        result = review(t["question"], method="hybrid")
        assert result["status"] == t["expected_status"], f"Expected {t['expected_status']}, got {result['status']} ({result.get('reason')})"
        
        doc_ids = [e["document_id"] for e in result["evidence"]]
        for ed in t["expected_docs"]:
            assert ed in doc_ids, f"Expected document {ed} not found in top 5"

def test_workspace_isolation():
    print("\n--- RUNNING WORKSPACE ISOLATION TESTS ---")
    question = "Where is production infrastructure hosted?"
    
    res_acme = review(question, workspace_id="ws_acme_corp", method="hybrid")
    assert res_acme["status"] == "INSUFFICIENT_EVIDENCE"
    assert not any(e["document_id"] == "SEC-G001" for e in res_acme["evidence"])
    
    res_globex = review(question, workspace_id="ws_globex_corp", method="hybrid")
    assert res_globex["status"] == "ANSWERABLE"
    assert any(e["document_id"] == "SEC-G001" for e in res_globex["evidence"])

if __name__ == "__main__":
    test_pipeline_and_debug()
    test_all_smoke_tests()
    test_workspace_isolation()
    print("\nAll new Day 12 tests passed successfully!")
