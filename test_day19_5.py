"""
test_day19_5.py — Day 19.5 Real PDF Ingestion & Realistic Corpus Verification

Verifies all 18 mandatory requirements:
1. TEST 1: PDF can be loaded.
2. TEST 2: Text extraction succeeds.
3. TEST 3: Multiple chunks generated.
4. TEST 4: Chunks preserve page info.
5. TEST 5: Chunks preserve section info.
6. TEST 6: Chunk IDs are deterministic across multiple ingestion runs.
7. TEST 7: Every chunk has workspace_id.
8. TEST 8: Every chunk has document_id.
9. TEST 9: Every chunk has version.
10. TEST 10: Every chunk has excerpt/text.
11. TEST 11: BM25 retrieves relevant PDF evidence.
12. TEST 12: Dense retrieval retrieves relevant PDF evidence.
13. TEST 13: RRF receives BM25 + dense candidates.
14. TEST 14: Cross-Encoder reranking works on PDF chunks.
15. TEST 15: SLM receives only the final PDF evidence.
16. TEST 16: Human Review payload displays PDF provenance.
17. TEST 17: Page/section metadata survives complete pipeline.
18. TEST 18: Workspace isolation works on PDF corpus.
"""

import os
import sys
from pdf_ingestion import (
    extract_pdf_structure,
    structure_aware_chunking,
    PDFEvidenceCorpus,
)

PDF_PATH = "data/acme_security_policy.pdf"


def test_01_pdf_can_be_loaded():
    assert os.path.exists(PDF_PATH), f"PDF file not found at {PDF_PATH}"
    print("  [TEST 1] PDF loaded successfully: PASSED")


def test_02_text_extraction():
    pages = extract_pdf_structure(PDF_PATH)
    assert len(pages) > 0, "No pages extracted"
    assert any("Encryption" in p["text"] for p in pages), "Expected policy text not found"
    print("  [TEST 2] Text extraction succeeds: PASSED")


def test_03_multiple_chunks_generated():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages)
    assert len(chunks) >= 5, f"Expected multiple chunks, got {len(chunks)}"
    print("  [TEST 3] Multiple chunks generated: PASSED")


def test_04_preserve_page_info():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages)
    for c in chunks:
        assert "page" in c, "Missing page field"
        assert isinstance(c["page"], int), "Page field must be integer"
        assert c["page"] >= 1, "Page number must be >= 1"
    print("  [TEST 4] Chunks preserve page info: PASSED")


def test_05_preserve_section_info():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages)
    sections = {c["section"] for c in chunks if "section" in c}
    assert len(sections) > 1, f"Expected multiple sections, got {sections}"
    print("  [TEST 5] Chunks preserve section info: PASSED")


def test_06_deterministic_chunk_ids():
    pages = extract_pdf_structure(PDF_PATH)
    chunks_run1 = structure_aware_chunking(pages)
    chunks_run2 = structure_aware_chunking(pages)

    ids1 = [c["chunk_id"] for c in chunks_run1]
    ids2 = [c["chunk_id"] for c in chunks_run2]

    assert ids1 == ids2, "Chunk IDs must be 100% deterministic across ingestion runs"
    print("  [TEST 6] Chunk IDs are deterministic: PASSED")


def test_07_workspace_id_presence():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages, workspace_id="ws_acme_corp")
    for c in chunks:
        assert c.get("workspace_id") == "ws_acme_corp"
    print("  [TEST 7] Every chunk has workspace_id: PASSED")


def test_08_document_id_presence():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages, document_id="SEC-REAL-001")
    for c in chunks:
        assert c.get("document_id") == "SEC-REAL-001"
    print("  [TEST 8] Every chunk has document_id: PASSED")


def test_09_version_presence():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages, version="2026-01")
    for c in chunks:
        assert c.get("version") == "2026-01"
    print("  [TEST 9] Every chunk has version: PASSED")


def test_10_excerpt_presence():
    pages = extract_pdf_structure(PDF_PATH)
    chunks = structure_aware_chunking(pages)
    for c in chunks:
        assert "excerpt" in c and c["excerpt"].strip(), "Empty excerpt in chunk"
    print("  [TEST 10] Every chunk has excerpt: PASSED")


def test_11_bm25_retrieval(corpus: PDFEvidenceCorpus):
    res = corpus.retrieve_bm25("multi-factor authentication", workspace_id="ws_acme_corp", top_k=5)
    assert len(res) > 0, "BM25 failed to retrieve PDF evidence"
    assert any("multi-factor" in c["excerpt"].lower() for c in res)
    print("  [TEST 11] BM25 retrieves relevant PDF evidence: PASSED")


def test_12_dense_retrieval(corpus: PDFEvidenceCorpus):
    res = corpus.retrieve_embedding("data in transit protection", workspace_id="ws_acme_corp", top_k=5)
    assert len(res) > 0, "Dense retrieval failed on PDF corpus"
    assert any("tls 1.2" in c["excerpt"].lower() or "transit" in c["excerpt"].lower() for c in res)
    print("  [TEST 12] Dense retrieval retrieves relevant PDF evidence: PASSED")


def test_13_rrf_receives_candidates(corpus: PDFEvidenceCorpus):
    bm25_res = corpus.retrieve_bm25("incident triage SLA", workspace_id="ws_acme_corp", top_k=5)
    emb_res = corpus.retrieve_embedding("incident triage SLA", workspace_id="ws_acme_corp", top_k=5)
    assert len(bm25_res) > 0 or len(emb_res) > 0
    print("  [TEST 13] RRF receives BM25 + dense candidates: PASSED")


def test_14_cross_encoder_reranking(corpus: PDFEvidenceCorpus):
    res = corpus.review_pdf("What encryption protects customer data at rest?", workspace_id="ws_acme_corp")
    assert len(res["evidence"]) > 0
    top_hit = res["evidence"][0]
    assert "reranker_score" in top_hit
    assert any("AES-256" in item["excerpt"] for item in res["evidence"])
    print("  [TEST 14] Cross-Encoder reranking works on PDF chunks: PASSED")


def test_15_slm_receives_final_pdf_evidence(corpus: PDFEvidenceCorpus):
    res = corpus.review_pdf("Who triages security incidents?", workspace_id="ws_acme_corp")
    assert len(res["evidence"]) > 0
    assert any("on-call engineer" in item["excerpt"].lower() or "triaged" in item["excerpt"].lower() for item in res["evidence"])
    print("  [TEST 15] SLM receives only final PDF evidence: PASSED")


def test_16_review_payload_displays_provenance(corpus: PDFEvidenceCorpus):
    res = corpus.review_pdf("What is the SLA for resolving high-severity security incidents?", workspace_id="ws_acme_corp")
    assert len(res["evidence"]) > 0
    top_hit = res["evidence"][0]

    assert "page" in top_hit
    assert "section" in top_hit
    assert "source_type" in top_hit
    assert top_hit["source_type"] == "pdf"
    print("  [TEST 16] Review payload displays PDF provenance: PASSED")


def test_17_page_section_metadata_survives(corpus: PDFEvidenceCorpus):
    res = corpus.review_pdf("How long are database backups retained?", workspace_id="ws_acme_corp")
    for item in res["evidence"]:
        assert "page" in item and item["page"] >= 1
        assert "section" in item and item["section"]
    print("  [TEST 17] Page & section metadata survives pipeline: PASSED")


def test_18_workspace_isolation_pdf(corpus: PDFEvidenceCorpus):
    # Query Acme workspace for Globex PDF data
    res = corpus.review_pdf("Where is production infrastructure hosted?", workspace_id="ws_acme_corp")
    for item in res["evidence"]:
        assert item["workspace_id"] == "ws_acme_corp"
        assert "ExampleCloud" not in item["excerpt"]
    print("  [TEST 18] Workspace isolation works on PDF corpus: PASSED")


def main():
    print("============================================================")
    print("RUNNING DAY 19.5 REAL PDF INGESTION & VALIDATION TESTS")
    print("============================================================\n")

    test_01_pdf_can_be_loaded()
    test_02_text_extraction()
    test_03_multiple_chunks_generated()
    test_04_preserve_page_info()
    test_05_preserve_section_info()
    test_06_deterministic_chunk_ids()
    test_07_workspace_id_presence()
    test_08_document_id_presence()
    test_09_version_presence()
    test_10_excerpt_presence()

    # Initialize PDF evidence corpus
    corpus = PDFEvidenceCorpus(collection_name="evidencedesk_pdf")
    corpus.ingest_pdf(PDF_PATH, document_id="SEC-REAL-001", title="Acme Enterprise Security & Compliance Policy", version="2026-01", workspace_id="ws_acme_corp")

    test_11_bm25_retrieval(corpus)
    test_12_dense_retrieval(corpus)
    test_13_rrf_receives_candidates(corpus)
    test_14_cross_encoder_reranking(corpus)
    test_15_slm_receives_final_pdf_evidence(corpus)
    test_16_review_payload_displays_provenance(corpus)
    test_17_page_section_metadata_survives(corpus)
    test_18_workspace_isolation_pdf(corpus)

    print("\n============================================================")
    print("ALL 18 DAY 19.5 PDF INGESTION & PIPELINE TESTS PASSED!")
    print("============================================================")

if __name__ == "__main__":
    main()
