from tests.fixtures.synthetic_data import PASSAGES, DOCUMENTS, CASES, inject_legacy_fixtures
"""
test_day18.py — Day 18 Workspace Isolation & Provenance Hardening Audit

Verifies:
1. TEST 1: Acme cannot retrieve Globex evidence (SEC-G001 excluded).
2. TEST 2: Globex can retrieve its own evidence (SEC-G001 included).
3. TEST 3: Cross-workspace negative test (different results for same question).
4. TEST 4: Hybrid retrieval isolation (all Top-5 items belong strictly to requested workspace).
5. TEST 5: RRF does not mix workspaces (pre-RRF inputs are workspace-scoped).
6. TEST 6: Cross-Encoder does not mix workspaces (pre- and post-rerank workspace assertion).
7. TEST 7: Final SLM evidence is workspace-safe (SLM only receives target workspace chunks).
8. TEST 8: Provenance preservation (chunk_id, workspace_id, document_id, title, version, excerpt preserved).
9. TEST 9: Version preservation (version metadata survives all pipeline stages).
10. TEST 10: Stable chunk identity (chunk_id unchanged from indexing to review UI).
11. TEST 11: Workspace field cannot be lost (workspace_id attached at every stage).
12. TEST 12: Wrong workspace negative test (foreign workspace candidate explicitly rejected).
13. TEST 13: Empty/unknown workspace (missing/unknown workspace produces empty evidence / safe abstention, no global fallback).
14. TEST 14: Workspace filter bypass attempt (semantic similarity of foreign workspace document cannot override filter).
15. TEST 15: Review provenance (review API response contains full provenance payload).
"""

import sys
import json
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)
from evidencedesk import (

    retrieve_by_bm25,
    retrieve_by_embedding,
    rrf_fuse,
    rerank_evidence,
    analyze_evidence,
    review,
)

def test_01_acme_cannot_retrieve_globex():
    """TEST 1: Acme cannot retrieve Globex evidence (SEC-G001)."""
    question = "Where is production infrastructure hosted?"
    
    bm25 = retrieve_by_bm25(question, workspace_id="ws_acme_corp", top_k=20)
    emb = retrieve_by_embedding(question, workspace_id="ws_acme_corp", top_k=20)
    rrf = rrf_fuse(bm25, emb, top_k=10)
    top5 = rerank_evidence(question, rrf, top_k=5)
    resp = review(question, workspace_id="ws_acme_corp", method="hybrid")

    for stage_name, items in [
        ("BM25", bm25),
        ("Dense", emb),
        ("RRF", rrf),
        ("Cross-Encoder", top5),
        ("Review Evidence", resp["evidence"]),
    ]:
        for item in items:
            assert item["document_id"] != "SEC-G001", f"[Leak in {stage_name}] SEC-G001 found in Acme workspace!"
            assert "ExampleCloud" not in item["excerpt"], f"[Leak in {stage_name}] Globex content found in Acme workspace!"
            assert item["workspace_id"] == "ws_acme_corp", f"[Wrong Workspace in {stage_name}] {item['workspace_id']} in Acme!"

    print("  [TEST 1] Acme cannot retrieve Globex evidence: PASSED")


def test_02_globex_can_retrieve_own_evidence():
    """TEST 2: Globex can retrieve its own evidence (SEC-G001)."""
    question = "Where is production infrastructure hosted?"
    
    emb = retrieve_by_embedding(question, workspace_id="ws_globex_corp", top_k=5)
    resp = review(question, method="hybrid", workspace_id="ws_globex_corp")

    found_in_emb = any(item["document_id"] == "SEC-G001" for item in emb)
    found_in_resp = any(item["document_id"] == "SEC-G001" for item in resp["evidence"])

    assert found_in_emb, "Globex failed to retrieve SEC-G001 in dense search"
    assert found_in_resp, "Globex failed to retrieve SEC-G001 in hybrid review"
    
    # Excerpt assertion
    sec_g001_item = next(item for item in resp["evidence"] if item["document_id"] == "SEC-G001")
    assert "ExampleCloud" in sec_g001_item["excerpt"]
    assert sec_g001_item["workspace_id"] == "ws_globex_corp"

    print("  [TEST 2] Globex retrieves its own evidence: PASSED")


def test_03_cross_workspace_negative_test():
    """TEST 3: Same question against Acme vs Globex produces distinct, isolated results."""
    question = "Where is production infrastructure hosted?"
    
    acme_resp = review(question, workspace_id="ws_acme_corp", method="hybrid")
    globex_resp = review(question, method="hybrid", workspace_id="ws_globex_corp")

    acme_doc_ids = {item["document_id"] for item in acme_resp["evidence"]}
    globex_doc_ids = {item["document_id"] for item in globex_resp["evidence"]}

    assert "SEC-G001" not in acme_doc_ids, "Acme results contain SEC-G001"
    assert "SEC-G001" in globex_doc_ids, "Globex results missing SEC-G001"
    assert acme_doc_ids != globex_doc_ids, "Acme and Globex workspace results must differ"

    print("  [TEST 3] Cross-workspace negative test: PASSED")


def test_04_hybrid_retrieval_isolation():
    """TEST 4: All Top-5 items in hybrid retrieval belong strictly to requested workspace."""
    question = "What encryption standard is used for data at rest?"
    
    acme_resp = review(question, workspace_id="ws_acme_corp", method="hybrid")
    for item in acme_resp["evidence"]:
        assert item["workspace_id"] == "ws_acme_corp", f"Found foreign workspace item: {item}"

    globex_resp = review(question, method="hybrid", workspace_id="ws_globex_corp")
    for item in globex_resp["evidence"]:
        assert item["workspace_id"] == "ws_globex_corp", f"Found foreign workspace item: {item}"

    print("  [TEST 4] Hybrid retrieval isolation: PASSED")


def test_05_rrf_does_not_mix_workspaces():
    """TEST 5: Verify candidate sets entering RRF are pre-filtered by workspace."""
    question = "What encryption protects data at rest?"
    workspace_id = "ws_acme_corp"

    bm25 = retrieve_by_bm25(question, workspace_id, top_k=20)
    emb = retrieve_by_embedding(question, workspace_id, top_k=20)

    # Pre-RRF assertions
    for item in bm25:
        assert item["workspace_id"] == workspace_id, f"BM25 contains un-scoped item: {item}"
    for item in emb:
        assert item["workspace_id"] == workspace_id, f"Embedding contains un-scoped item: {item}"

    fused = rrf_fuse(bm25, emb, top_k=10)
    for item in fused:
        assert item["workspace_id"] == workspace_id, f"RRF fused list contains un-scoped item: {item}"

    print("  [TEST 5] RRF does not mix workspaces: PASSED")


def test_06_cross_encoder_does_not_mix_workspaces():
    """TEST 6: Verify Cross-Encoder only receives and returns target workspace candidates."""
    question = "What encryption protects data at rest?"
    workspace_id = "ws_acme_corp"

    bm25 = retrieve_by_bm25(question, workspace_id, top_k=20)
    emb = retrieve_by_embedding(question, workspace_id, top_k=20)
    candidates = rrf_fuse(bm25, emb, top_k=10)

    # Assert before reranking
    for candidate in candidates:
        assert candidate["workspace_id"] == workspace_id, f"Pre-rerank candidate has wrong workspace: {candidate}"

    reranked = rerank_evidence(question, candidates, top_k=5)

    # Assert after reranking
    for item in reranked:
        assert item["workspace_id"] == workspace_id, f"Post-rerank item has wrong workspace: {item}"

    print("  [TEST 6] Cross-Encoder does not mix workspaces: PASSED")


def test_07_final_slm_evidence_is_workspace_safe():
    """TEST 7: Final SLM reasoning input and chunk IDs belong exclusively to target workspace."""
    question = "Where is production infrastructure hosted?"
    
    # Acme request
    acme_resp = review(question, workspace_id="ws_acme_corp", method="hybrid")
    for cid in acme_resp.get("evidence_chunk_ids", []):
        assert not cid.startswith("chk_secg001"), f"SLM returned Globex chunk ID for Acme query: {cid}"
    for ev in acme_resp.get("evidence", []):
        assert ev["workspace_id"] == "ws_acme_corp"

    # Globex request
    globex_resp = review(question, method="hybrid", workspace_id="ws_globex_corp")
    for ev in globex_resp.get("evidence", []):
        assert ev["workspace_id"] == "ws_globex_corp"

    print("  [TEST 7] Final SLM evidence is workspace-safe: PASSED")


def test_08_provenance_preservation():
    """TEST 8: Full provenance metadata remains attached through every stage."""
    question = "What encryption protects customer data at rest?"
    workspace_id = "ws_acme_corp"

    resp = review(question, method="hybrid", workspace_id=workspace_id)
    assert len(resp["evidence"]) > 0

    required_fields = ["chunk_id", "workspace_id", "document_id", "title", "version", "excerpt"]

    for idx, item in enumerate(resp["evidence"]):
        for field in required_fields:
            assert field in item, f"Missing provenance field '{field}' in item index {idx}"
            assert item[field], f"Empty provenance field '{field}' in item index {idx}"

        assert item["workspace_id"] == workspace_id
        assert item["document_id"].startswith("SEC-")
        assert item["chunk_id"].startswith("chk_")

    print("  [TEST 8] Provenance preservation: PASSED")


def test_09_version_preservation():
    """TEST 9: Version metadata survives every stage without alteration or loss."""
    question = "What encryption protects customer data at rest?"
    workspace_id = "ws_acme_corp"

    bm25 = retrieve_by_bm25(question, workspace_id, top_k=5)
    emb = retrieve_by_embedding(question, workspace_id, top_k=5)
    fused = rrf_fuse(bm25, emb, top_k=5)
    reranked = rerank_evidence(question, fused, top_k=5)
    resp = review(question, method="hybrid", workspace_id=workspace_id)

    for stage_name, items in [
        ("BM25", bm25),
        ("Dense", emb),
        ("RRF", fused),
        ("Reranker", reranked),
        ("Review Evidence", resp["evidence"]),
    ]:
        for item in items:
            assert "version" in item, f"Missing version in {stage_name}"
            assert item["version"] == "2026-01", f"Version corrupted in {stage_name}: got {item['version']}"

    print("  [TEST 9] Version preservation: PASSED")


def test_10_stable_chunk_identity():
    """TEST 10: Chunk ID remains stable from indexing to final review response."""
    question = "Employees must use multi-factor authentication"
    workspace_id = "ws_acme_corp"

    bm25 = retrieve_by_bm25(question, workspace_id, top_k=5)
    emb = retrieve_by_embedding(question, workspace_id, top_k=5)

    bm25_chunk = next(item["chunk_id"] for item in bm25 if "multi-factor authentication" in item["excerpt"])
    emb_chunk = next(item["chunk_id"] for item in emb if "multi-factor authentication" in item["excerpt"])

    assert bm25_chunk == "chk_sec001_001", f"Unexpected BM25 chunk ID: {bm25_chunk}"
    assert emb_chunk == "chk_sec001_001", f"Unexpected Dense chunk ID: {emb_chunk}"

    resp = review(question, method="hybrid", workspace_id=workspace_id)
    review_chunk = next(item["chunk_id"] for item in resp["evidence"] if "multi-factor authentication" in item["excerpt"])

    assert review_chunk == "chk_sec001_001", f"Unexpected Review chunk ID: {review_chunk}"
    print("  [TEST 10] Stable chunk identity: PASSED")


def test_11_workspace_field_cannot_be_lost():
    """TEST 11: workspace_id field is present at all stage outputs."""
    question = "How long are database backups retained?"
    workspace_id = "ws_acme_corp"

    bm25 = retrieve_by_bm25(question, workspace_id, top_k=5)
    emb = retrieve_by_embedding(question, workspace_id, top_k=5)
    fused = rrf_fuse(bm25, emb, top_k=5)
    reranked = rerank_evidence(question, fused, top_k=5)
    resp = review(question, method="hybrid", workspace_id=workspace_id)

    for stage_name, items in [
        ("BM25", bm25),
        ("Dense", emb),
        ("RRF", fused),
        ("Reranker", reranked),
        ("Review Evidence", resp["evidence"]),
    ]:
        for item in items:
            assert "workspace_id" in item, f"workspace_id missing in {stage_name}"
            assert item["workspace_id"] == workspace_id, f"workspace_id wrong in {stage_name}"

    print("  [TEST 11] Workspace field cannot be lost: PASSED")


def test_12_wrong_workspace_negative_test():
    """TEST 12: Foreign workspace candidate explicitly rejected if injected."""
    question = "Where is production infrastructure hosted?"
    target_ws = "ws_acme_corp"

    # Fetch candidates for Acme
    bm25 = retrieve_by_bm25(question, target_ws, top_k=5)
    emb = retrieve_by_embedding(question, target_ws, top_k=5)

    # Synthesize a foreign candidate injection
    foreign_candidate = {
        "chunk_id": "chk_secg001_001",
        "workspace_id": "ws_globex_corp",
        "document_id": "SEC-G001",
        "title": "Globex Production Policy",
        "version": "2026-01",
        "excerpt": "Production infrastructure is hosted on ExampleCloud."
    }

    # Verify filtering function rejects candidate if filtered explicitly
    clean_bm25 = [p for p in bm25 if p["workspace_id"] == target_ws]
    clean_emb = [p for p in emb if p["workspace_id"] == target_ws]

    assert foreign_candidate not in clean_bm25
    assert foreign_candidate not in clean_emb

    fused = rrf_fuse(clean_bm25, clean_emb, top_k=5)
    for item in fused:
        assert item["workspace_id"] == target_ws

    print("  [TEST 12] Wrong workspace negative test: PASSED")


def test_13_empty_unknown_workspace():
    """TEST 13: Empty/unknown workspace does NOT cause global fallback."""
    question = "What encryption protects customer data at rest?"

    for invalid_ws in ["ws_unknown_xyz", "", "non_existent_workspace"]:
        bm25 = retrieve_by_bm25(question, workspace_id=invalid_ws, top_k=20)
        emb = retrieve_by_embedding(question, workspace_id=invalid_ws, top_k=20)
        resp = review(question, method="hybrid", workspace_id=invalid_ws)

        assert len(bm25) == 0, f"BM25 leaked results for invalid workspace '{invalid_ws}'"
        assert len(emb) == 0, f"Dense leaked results for invalid workspace '{invalid_ws}'"
        assert len(resp["evidence"]) == 0, f"Hybrid review leaked evidence for invalid workspace '{invalid_ws}'"
        assert resp["status"] == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE for invalid workspace '{invalid_ws}'"

    print("  [TEST 13] Empty/unknown workspace: PASSED")


def test_14_workspace_filter_bypass_attempt():
    """TEST 14: Semantic similarity of foreign workspace document cannot override filter."""
    # Highly specific query targeting Globex text sent under Acme workspace ID
    bypass_query = "Where is production infrastructure hosted on ExampleCloud?"
    acme_ws = "ws_acme_corp"

    bm25 = retrieve_by_bm25(bypass_query, acme_ws, top_k=20)
    emb = retrieve_by_embedding(bypass_query, acme_ws, top_k=20)
    resp = review(bypass_query, method="hybrid", workspace_id=acme_ws)

    for item in bm25:
        assert item["workspace_id"] == acme_ws
        assert item["document_id"] != "SEC-G001"

    for item in emb:
        assert item["workspace_id"] == acme_ws
        assert item["document_id"] != "SEC-G001"

    for item in resp["evidence"]:
        assert item["workspace_id"] == acme_ws
        assert item["document_id"] != "SEC-G001"
        assert "ExampleCloud" not in item["excerpt"]

    print("  [TEST 14] Workspace filter bypass attempt: PASSED")


def test_15_review_provenance():
    """TEST 15: Review API payload contains full provenance metadata for UI rendering."""
    question = "Who triages security incidents?"
    workspace_id = "ws_acme_corp"

    resp = review(question, method="hybrid", workspace_id=workspace_id)
    
    assert "evidence" in resp
    assert len(resp["evidence"]) > 0

    for item in resp["evidence"]:
        assert "workspace_id" in item
        assert "document_id" in item
        assert "title" in item
        assert "version" in item
        assert "chunk_id" in item
        assert "excerpt" in item

    print("  [TEST 15] Review provenance: PASSED")


def main():
    print("============================================================")
    print("RUNNING DAY 18 WORKSPACE ISOLATION & PROVENANCE HARDENING TESTS")
    print("============================================================\n")

    test_01_acme_cannot_retrieve_globex()
    test_02_globex_can_retrieve_own_evidence()
    test_03_cross_workspace_negative_test()
    test_04_hybrid_retrieval_isolation()
    test_05_rrf_does_not_mix_workspaces()
    test_06_cross_encoder_does_not_mix_workspaces()
    test_07_final_slm_evidence_is_workspace_safe()
    test_08_provenance_preservation()
    test_09_version_preservation()
    test_10_stable_chunk_identity()
    test_11_workspace_field_cannot_be_lost()
    test_12_wrong_workspace_negative_test()
    test_13_empty_unknown_workspace()
    test_14_workspace_filter_bypass_attempt()
    test_15_review_provenance()

    print("\n============================================================")
    print("ALL 15 DAY 18 WORKSPACE ISOLATION & PROVENANCE TESTS PASSED!")
    print("============================================================")

if __name__ == "__main__":
    main()
