from evidencedesk import retrieve, retrieve_by_bm25, retrieve_by_embedding, rrf_fuse, PASSAGES

def check_metadata(passage):
    assert "chunk_id" in passage, "Missing chunk_id"
    assert "workspace_id" in passage, "Missing workspace_id"
    assert "document_id" in passage, "Missing document_id"
    assert "title" in passage, "Missing title"
    assert "version" in passage, "Missing version"
    assert "excerpt" in passage, "Missing excerpt"
    assert passage["workspace_id"] == "ws_acme_corp", "Wrong workspace_id"
    assert passage["chunk_id"].startswith("chk_"), "Wrong chunk_id format"

def test_lexical_preserves_metadata():
    results = retrieve("encryption")
    assert len(results) > 0
    for r in results:
        check_metadata(r)
        assert "lexical_score" in r
    print("test_lexical_preserves_metadata passed")

def test_bm25_preserves_metadata():
    results = retrieve_by_bm25("encryption", top_k=3)
    assert len(results) > 0
    for r in results:
        check_metadata(r)
        assert "bm25_score" in r
    print("test_bm25_preserves_metadata passed")

def test_embedding_preserves_metadata():
    results = retrieve_by_embedding("encryption", top_k=3)
    assert len(results) > 0
    for r in results:
        check_metadata(r)
        assert "similarity_score" in r
    print("test_embedding_preserves_metadata passed")

def test_hybrid_preserves_metadata():
    bm25 = retrieve_by_bm25("encryption", top_k=5)
    emb = retrieve_by_embedding("encryption", top_k=5)
    fused = rrf_fuse(bm25, emb, top_k=5)
    assert len(fused) > 0
    for r in fused:
        check_metadata(r)
        assert "rrf_score" in r
    print("test_hybrid_preserves_metadata passed")

def test_rrf_no_duplicates():
    bm25 = retrieve_by_bm25("encryption", top_k=5)
    emb = retrieve_by_embedding("encryption", top_k=5)
    fused = rrf_fuse(bm25, emb, top_k=10)
    
    chunk_ids = [r["chunk_id"] for r in fused]
    assert len(chunk_ids) == len(set(chunk_ids)), "RRF merged results contain duplicate chunks!"
    print("test_rrf_no_duplicates passed")

def test_rrf_version_survives():
    bm25 = retrieve_by_bm25("encryption", top_k=2)
    emb = retrieve_by_embedding("encryption", top_k=2)
    fused = rrf_fuse(bm25, emb, top_k=2)
    assert fused[0]["version"] == "2026-01"
    print("test_rrf_version_survives passed")

def test_rrf_workspace_survives():
    bm25 = retrieve_by_bm25("encryption", top_k=2)
    emb = retrieve_by_embedding("encryption", top_k=2)
    fused = rrf_fuse(bm25, emb, top_k=2)
    assert fused[0]["workspace_id"] == "ws_acme_corp"
    print("test_rrf_workspace_survives passed")

def test_chunk_identity_stable():
    lex = retrieve("customer data at rest")
    bm = retrieve_by_bm25("customer data at rest")
    emb = retrieve_by_embedding("customer data at rest")
    
    lex_id = next(r["chunk_id"] for r in lex if "AES-256" in r["excerpt"])
    bm_id = next(r["chunk_id"] for r in bm if "AES-256" in r["excerpt"])
    emb_id = next(r["chunk_id"] for r in emb if "AES-256" in r["excerpt"])
    
    assert lex_id == bm_id == emb_id
    print("test_chunk_identity_stable passed")

if __name__ == "__main__":
    print("Running metadata tests...")
    test_lexical_preserves_metadata()
    test_bm25_preserves_metadata()
    test_embedding_preserves_metadata()
    test_hybrid_preserves_metadata()
    test_rrf_no_duplicates()
    test_rrf_version_survives()
    test_rrf_workspace_survives()
    test_chunk_identity_stable()
    print("All metadata tests passed!")
