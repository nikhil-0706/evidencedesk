import os
from tests.fixtures.synthetic_data import PASSAGES, DOCUMENTS, CASES, inject_legacy_fixtures

import shutil
from evidencedesk import retrieve_by_embedding, review, qdrant, embed_model
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)


def test_qdrant_payload():
    results = retrieve_by_embedding("What encryption protects customer data at rest?", workspace_id="ws_acme_corp", top_k=3)
    assert len(results) > 0
    top_hit = results[0]
    
    assert "chunk_id" in top_hit
    assert "workspace_id" in top_hit
    assert "document_id" in top_hit
    assert "title" in top_hit
    assert "version" in top_hit
    assert "excerpt" in top_hit
    assert "similarity_score" in top_hit
    assert top_hit["workspace_id"] == "ws_acme_corp"

def test_cross_workspace_leak():
    # ws_globex_corp has "Production infrastructure is hosted on ExampleCloud."
    # Query ws_acme_corp for this exact info. It should return NO relevant matching documents,
    # or at least NOT the SEC-G001 document.
    results = retrieve_by_embedding("Which cloud provider hosts production?", workspace_id="ws_acme_corp", top_k=5)
    for r in results:
        assert r["workspace_id"] == "ws_acme_corp"
        assert r["document_id"] != "SEC-G001"
        assert "ExampleCloud" not in r["excerpt"]

    # Also test via hybrid review endpoint
    response = review("Which cloud provider hosts production?", workspace_id="ws_acme_corp", method="hybrid")
    for r in response["evidence"]:
        assert r["document_id"] != "SEC-G001"
        assert r["workspace_id"] == "ws_acme_corp"

def test_positive_workspace_test():
    # Query ws_globex_corp
    results = retrieve_by_embedding("Which cloud provider hosts production?", workspace_id="ws_globex_corp", top_k=5)
    assert len(results) > 0
    found = any(r["document_id"] == "SEC-G001" for r in results)
    assert found, "Expected SEC-G001 to be retrieved for ws_globex_corp"
    
    # Via hybrid review
    response = review("Which cloud provider hosts production?", method="hybrid", workspace_id="ws_globex_corp")
    assert any(r["document_id"] == "SEC-G001" for r in response["evidence"])

def test_reindex_is_deterministic():
    # Qdrant DB points count should match PASSAGES count
    count = qdrant.count(multi_doc_corpus.collection_name).count
    assert count == len(PASSAGES)

if __name__ == "__main__":
    test_qdrant_payload()
    test_cross_workspace_leak()
    test_positive_workspace_test()
    test_reindex_is_deterministic()
    print("All Day 11 tests passed!")
