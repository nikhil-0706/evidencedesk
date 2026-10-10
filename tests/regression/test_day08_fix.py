import json

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

from evidencedesk import review

tests = [
    {
        "name": "TEST 1 - ENCRYPTION (ANSWERABLE)",
        "question": "What encryption protects customer data at rest?"
    },
    {
        "name": "TEST 2 - SOC 2 (INSUFFICIENT)",
        "question": "Are you SOC 2 certified?"
    },
    {
        "name": "TEST 3 - AMBIGUOUS",
        "question": "What is your retention policy?"
    },
    {
        "name": "TEST 4 - CONFLICTING",
        "question": "Are backups retained for 30 days or 90 days?"
    }
]

def run_tests():
    print("--- RUNNING DAY 8 FIX TESTS ---\n")
    for t in tests:
        print(f"{t['name']}")
        print(f"Q: {t['question']}")
        
        result = review(t['question'], workspace_id='ws_acme_corp', method="hybrid")
        status = result.get("status")
        reason = result.get("reason", "")
        
        print(f"Status: {status}")
        print(f"Reason: {reason}")
        print("-" * 40)

if __name__ == "__main__":
    run_tests()
