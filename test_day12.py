import os
from evidencedesk import review, rerank_evidence, PASSAGES, rrf_fuse, retrieve_by_bm25, retrieve_by_embedding

def test_reranker_returns_top5():
    # Trigger the hybrid pipeline
    question = "What encryption protects customer data at rest?"
    result = review(question, method="hybrid")
    # Result evidence should be Top 5
    assert len(result["evidence"]) <= 5

def test_reranker_ordering_and_metadata():
    question = "Are backups retained for 30 days or 90 days?"
    
    # We can fetch candidates manually to verify metadata
    bm25_res = retrieve_by_bm25(question, "ws_acme_corp", 20)
    emb_res = retrieve_by_embedding(question, "ws_acme_corp", 20)
    rrf_cands = rrf_fuse(bm25_res, emb_res, 60, 10)
    
    reranked = rerank_evidence(question, rrf_cands, top_k=5)
    
    # Test 2: Ordered descending by reranker_score
    for i in range(len(reranked) - 1):
        assert reranked[i]["reranker_score"] >= reranked[i+1]["reranker_score"]
        
    # Test 4, 5, 6: Metadata survives
    for doc in reranked:
        assert "chunk_id" in doc
        assert "workspace_id" in doc
        assert "document_id" in doc
        assert "title" in doc
        assert "version" in doc
        assert "excerpt" in doc
        assert "rrf_score" in doc
        assert "reranker_score" in doc
        
    # Test 7: No new candidates
    rrf_chunk_ids = {c["chunk_id"] for c in rrf_cands}
    for doc in reranked:
        assert doc["chunk_id"] in rrf_chunk_ids

def test_conflict_candidates_coexist():
    question = "Are backups retained for 30 days or 90 days?"
    result = review(question, method="hybrid")
    
    # Test 8: Conflict candidates can coexist
    docs_returned = [r["document_id"] for r in result["evidence"]]
    assert "SEC-003" in docs_returned
    assert "SEC-005" in docs_returned
    
    # Ambiguous test case 
    # Just verifying it returns AMBIGUOUS
    amb_res = review("What is your retention policy?", method="hybrid")
    assert amb_res["status"] == "AMBIGUOUS"
    
    # Unanswerable test case
    unans_res = review("Are you SOC 2 certified?", method="hybrid")
    assert unans_res["status"] == "INSUFFICIENT_EVIDENCE"
    
def test_workspace_isolation_survives():
    question = "Production infrastructure is hosted on?"
    
    # Under Acme, SEC-G001 must not appear
    acme_res = review(question, method="hybrid", workspace_id="ws_acme_corp")
    for ev in acme_res["evidence"]:
        assert ev["document_id"] != "SEC-G001"
        assert ev["workspace_id"] == "ws_acme_corp"
        
    # Under Globex, it must appear
    globex_res = review(question, method="hybrid", workspace_id="ws_globex_corp")
    assert any(ev["document_id"] == "SEC-G001" for ev in globex_res["evidence"])

if __name__ == "__main__":
    test_reranker_returns_top5()
    test_reranker_ordering_and_metadata()
    test_conflict_candidates_coexist()
    test_workspace_isolation_survives()
    print("All Day 12 reranker tests passed!")
