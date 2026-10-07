import json
from evidencedesk import review

tests = [
    {
        "name": "TEST 1 - DIRECT",
        "question": "What encryption protects customer data at rest?",
        "expected_doc": "SEC-002",
        "expected_status": "ANSWERABLE"
    },
    {
        "name": "TEST 2 - PARAPHRASED",
        "question": "What safeguards information while it travels between systems?",
        "expected_doc": "SEC-002",
        "expected_status": "ANSWERABLE"
    },
    {
        "name": "TEST 3 - CONFLICT",
        "question": "Are backups retained for 30 days or 90 days?",
        "expected_doc_1": "SEC-003",
        "expected_doc_2": "SEC-005",
        "expected_status": "CONFLICTING_EVIDENCE"
    },
    {
        "name": "TEST 4 - AMBIGUOUS",
        "question": "What is your retention policy?",
        "expected_status": "AMBIGUOUS"
    }
]

def run_tests():
    print("--- RUNNING DAY 8 SMOKE TESTS (HYBRID) ---\n")
    for t in tests:
        print(f"{t['name']}")
        print(f"Q: {t['question']}")
        
        result = review(t['question'], method="hybrid")
        status = result.get("status")
        reason = result.get("reason", "")
        evidence = result.get("evidence", [])
        
        print(f"Status: {status}")
        print(f"Reason: {reason}")
        print("Evidence Ranks:")
        for i, e in enumerate(evidence):
            print(f"  {i+1}. [{e['document_id']}] {e['excerpt']} (RRF: {e.get('rrf_score')})")
            
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    run_tests()
