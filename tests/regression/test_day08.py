import json

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

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
        
        result = review(t['question'], workspace_id='ws_acme_corp', method="hybrid")
        status = result.get("status")
        reason = result.get("reason", "")
        evidence = result.get("evidence", [])
        
        print(f"Status: {status}")
        print(f"Reason: {reason}")
        print("Evidence Ranks:")
        for i, e in enumerate(evidence):
            rrf = e.get('rrf_score', 'N/A')
            rerank = e.get('reranker_score')
            if rerank is not None:
                print(f"  {i+1}. [{e['document_id']}] {e['excerpt']} (RRF: {rrf}, Reranker: {rerank})")
            else:
                print(f"  {i+1}. [{e['document_id']}] {e['excerpt']} (RRF: {rrf})")
            
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    run_tests()
