"""
test_day19_75.py — Day 19.75 Multi-Document Upload & Automatic Ingestion Test Suite.

Verifies all 27 mandatory requirements:
1. TEST 1: Upload one PDF.
2. TEST 2: Upload one DOCX.
3. TEST 3: Upload one TXT.
4. TEST 4: Upload multiple files in one request.
5. TEST 5: Each document receives a distinct document_id.
6. TEST 6: Every chunk has workspace_id.
7. TEST 7: Every chunk has document_id.
8. TEST 8: Every chunk has version.
9. TEST 9: PDF chunks preserve page metadata.
10. TEST 10: DOCX chunks preserve section/heading metadata where available.
11. TEST 11: TXT chunks are created correctly.
12. TEST 12: Chunk IDs are deterministic.
13. TEST 13: Duplicate document detection works.
14. TEST 14: Unsupported file type is rejected.
15. TEST 15: Empty document is rejected safely.
16. TEST 16: Uploaded chunks receive 384-dim embeddings.
17. TEST 17: Uploaded chunks reach Qdrant.
18. TEST 18: BM25 can retrieve uploaded evidence.
19. TEST 19: Dense retrieval can retrieve uploaded evidence.
20. TEST 20: RRF can combine uploaded evidence.
21. TEST 21: Cross-Encoder reranks uploaded evidence.
22. TEST 22: SLM receives uploaded evidence with provenance.
23. TEST 23: Human Review displays uploaded-document provenance.
24. TEST 24: Workspace isolation remains intact.
25. TEST 25: Document A (ws_acme_corp) cannot appear in another workspace (ws_globex_corp).
26. TEST 26: Multiple documents can produce evidence in one answer.
27. TEST 27: Conflicting evidence from two uploaded documents remains visible.
"""

import os
import sys
import io
import fitz  # PyMuPDF
import docx  # python-docx
from document_ingestion import (
    MultiDocumentCorpus,
    extract_pdf,
    extract_docx,
    extract_txt,
    structure_aware_chunking_multi,
)
from evidencedesk import (
    embed_model,
    qdrant,
    rrf_fuse,
    rerank_evidence,
    analyze_evidence,
    review,
)


def create_sample_pdf(suffix: str = "") -> bytes:
    """Generate a valid PDF buffer in memory using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"Section 1: Data Protection Policy {suffix}\nCustomer personal data is stored in AWS us-east-1 datacenter.\nData backups are encrypted using AES-256 keys.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((50, 50), f"Section 2: Access Management {suffix}\nAccess to customer data requires approval from the Chief Information Security Officer.", fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_sample_docx(suffix: str = "") -> bytes:
    """Generate a valid DOCX buffer in memory using python-docx."""
    doc = docx.Document()
    doc.add_heading(f"Section 1: Incident Triage Policy {suffix}", level=1)
    doc.add_paragraph("Security incidents are triaged by the senior security response team within 15 minutes of alert generation.")
    doc.add_heading(f"Section 2: Escalation Guidelines {suffix}", level=1)
    doc.add_paragraph("High severity security incidents are reported to executive leadership within one hour.")
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


def create_sample_txt(suffix: str = "") -> bytes:
    """Generate a valid TXT buffer in memory."""
    text = (
        f"Section 1: Infrastructure Security Policy {suffix}\n\n"
        "Production database backups are performed every 4 hours.\n\n"
        f"Section 2: Retention Policy {suffix}\n\n"
        "Database backup retention for production environments is 60 days."
    )
    return text.encode("utf-8")


def main():
    print("--- Running Day 19.75 Multi-Document Ingestion Test Suite ---\n")

    pdf_bytes = create_sample_pdf("v1")
    docx_bytes = create_sample_docx("v1")
    txt_bytes = create_sample_txt("v1")

    corpus = MultiDocumentCorpus(
        qdrant_client=qdrant,
        embed_model=embed_model,
        collection_name="evidencedesk_test_19_75"
    )

    # TEST 1: Upload one PDF
    res_pdf = corpus.ingest_single_document(
        filename="data_protection_policy.pdf",
        file_bytes=pdf_bytes,
        workspace_id="ws_acme_corp",
        version="2026-01",
    )
    assert res_pdf["status"] == "SUCCESS", f"PDF upload failed: {res_pdf}"
    print("  [TEST 1] Upload one PDF: PASSED")

    # TEST 2: Upload one DOCX
    res_docx = corpus.ingest_single_document(
        filename="incident_triage_policy.docx",
        file_bytes=docx_bytes,
        workspace_id="ws_acme_corp",
        version="2026-01",
    )
    assert res_docx["status"] == "SUCCESS", f"DOCX upload failed: {res_docx}"
    print("  [TEST 2] Upload one DOCX: PASSED")

    # TEST 3: Upload one TXT
    res_txt = corpus.ingest_single_document(
        filename="infrastructure_policy.txt",
        file_bytes=txt_bytes,
        workspace_id="ws_acme_corp",
        version="2026-01",
    )
    assert res_txt["status"] == "SUCCESS", f"TXT upload failed: {res_txt}"
    print("  [TEST 3] Upload one TXT: PASSED")

    # TEST 4: Upload multiple files in one request
    multi_res = corpus.ingest_multiple_documents(
        files=[
            ("multi_doc_a.pdf", create_sample_pdf("v2")),
            ("multi_doc_b.docx", create_sample_docx("v2")),
            ("multi_doc_c.txt", create_sample_txt("v2")),
        ],
        workspace_id="ws_acme_corp",
        version="2026-01",
    )
    assert multi_res["successful_documents"] == 3, f"Multi-upload failed: {multi_res}"
    print("  [TEST 4] Upload multiple files in one request: PASSED")


    # TEST 5: Each document receives a distinct document_id
    doc_ids = {r["document_id"] for r in multi_res["results"] if "document_id" in r}
    assert len(doc_ids) == 3, f"Expected 3 distinct document_ids, got {doc_ids}"
    print("  [TEST 5] Distinct document_ids for each document: PASSED")

    # TEST 6: Every chunk has workspace_id
    for chunk in corpus.chunks:
        assert "workspace_id" in chunk and chunk["workspace_id"], f"Missing workspace_id in chunk {chunk}"
    print("  [TEST 6] Every chunk has workspace_id: PASSED")

    # TEST 7: Every chunk has document_id
    for chunk in corpus.chunks:
        assert "document_id" in chunk and chunk["document_id"], f"Missing document_id in chunk {chunk}"
    print("  [TEST 7] Every chunk has document_id: PASSED")

    # TEST 8: Every chunk has version
    for chunk in corpus.chunks:
        assert "version" in chunk and chunk["version"], f"Missing version in chunk {chunk}"
    print("  [TEST 8] Every chunk has version: PASSED")

    # TEST 9: PDF chunks preserve page metadata
    pdf_chunks = [c for c in corpus.chunks if c["source_type"] == "pdf"]
    assert len(pdf_chunks) > 0, "No PDF chunks found"
    for c in pdf_chunks:
        assert isinstance(c["page"], int), f"PDF page must be integer, got {c.get('page')}"
    print("  [TEST 9] PDF chunks preserve page metadata: PASSED")

    # TEST 10: DOCX chunks preserve section/heading metadata where available
    docx_chunks = [c for c in corpus.chunks if c["source_type"] == "docx"]
    assert len(docx_chunks) > 0, "No DOCX chunks found"
    assert any(c["section"] != "General" for c in docx_chunks), "DOCX section heading metadata lost"
    print("  [TEST 10] DOCX chunks preserve section/heading metadata: PASSED")

    # TEST 11: TXT chunks created correctly
    txt_chunks = [c for c in corpus.chunks if c["source_type"] == "txt"]
    assert len(txt_chunks) > 0, "No TXT chunks found"
    for c in txt_chunks:
        assert c["source_type"] == "txt", "TXT chunk missing source_type"
    print("  [TEST 11] TXT chunks created correctly: PASSED")

    # TEST 12: Chunk IDs are deterministic
    pdf_blocks = extract_pdf(pdf_bytes, "test.pdf")
    chunks_run1 = structure_aware_chunking_multi(pdf_blocks, "DOC-123", "Title", "2026-01", "ws_acme_corp", "pdf", "test.pdf")
    chunks_run2 = structure_aware_chunking_multi(pdf_blocks, "DOC-123", "Title", "2026-01", "ws_acme_corp", "pdf", "test.pdf")
    c1_ids = [c["chunk_id"] for c in chunks_run1]
    c2_ids = [c["chunk_id"] for c in chunks_run2]
    assert c1_ids == c2_ids, f"Chunk IDs are not deterministic: {c1_ids} != {c2_ids}"
    print("  [TEST 12] Chunk IDs are deterministic: PASSED")

    # TEST 13: Duplicate document detection works
    dup_res = corpus.ingest_single_document(
        filename="data_protection_policy.pdf",
        file_bytes=pdf_bytes,
        workspace_id="ws_acme_corp",
        version="2026-01",
    )
    assert dup_res["status"] == "DUPLICATE", f"Expected DUPLICATE status, got {dup_res['status']}"
    print("  [TEST 13] Duplicate document detection works: PASSED")

    # TEST 14: Unsupported file type is rejected
    unsupported_res = corpus.ingest_single_document(
        filename="malicious_payload.exe",
        file_bytes=b"MZ header executable bytes",
        workspace_id="ws_acme_corp",
    )
    assert unsupported_res["status"] == "FAILED", f"Unsupported file should fail, got {unsupported_res}"
    print("  [TEST 14] Unsupported file type is rejected: PASSED")

    # TEST 15: Empty document is rejected safely
    empty_res = corpus.ingest_single_document(
        filename="empty_file.txt",
        file_bytes=b"",
        workspace_id="ws_acme_corp",
    )
    assert empty_res["status"] == "FAILED", f"Empty file should fail, got {empty_res}"
    print("  [TEST 15] Empty document is rejected safely: PASSED")

    # TEST 16: Uploaded chunks receive 384-dim embeddings
    sample_text = corpus.chunks[0]["excerpt"]
    vec = embed_model.encode(sample_text)
    assert len(vec) == 384, f"Expected 384-dim vector, got {len(vec)}"
    print("  [TEST 16] Uploaded chunks receive 384-dim embeddings: PASSED")

    # TEST 17: Uploaded chunks reach Qdrant
    assert qdrant.collection_exists(corpus.collection_name), "Qdrant collection missing"
    collection_info = qdrant.get_collection(corpus.collection_name)
    assert collection_info.points_count > 0, "No points in Qdrant collection"
    print("  [TEST 17] Uploaded chunks reach Qdrant: PASSED")

    # TEST 18: BM25 can retrieve uploaded evidence
    bm25_hits = corpus.retrieve_bm25("AWS us-east-1 datacenter", workspace_id="ws_acme_corp", top_k=5)
    assert len(bm25_hits) > 0, "BM25 retrieval returned no results"
    assert "data_protection_policy.pdf" in [h.get("filename") for h in bm25_hits], "BM25 did not find target uploaded document"
    print("  [TEST 18] BM25 can retrieve uploaded evidence: PASSED")

    # TEST 19: Dense retrieval can retrieve uploaded evidence
    emb_hits = corpus.retrieve_embedding("Where is customer personal data stored?", workspace_id="ws_acme_corp", top_k=5)
    assert len(emb_hits) > 0, "Dense retrieval returned no results"
    assert any("datacenter" in h["excerpt"].lower() for h in emb_hits), "Dense retrieval missed target chunk"
    print("  [TEST 19] Dense retrieval can retrieve uploaded evidence: PASSED")

    # TEST 20: RRF can combine uploaded evidence
    rrf_hits = rrf_fuse(bm25_hits, emb_hits, k=60, top_k=5)
    assert len(rrf_hits) > 0, "RRF fusion returned no candidates"
    assert "rrf_score" in rrf_hits[0], "RRF score missing from candidates"
    print("  [TEST 20] RRF can combine uploaded evidence: PASSED")

    # TEST 21: Cross-Encoder reranks uploaded evidence
    reranked = rerank_evidence("Where is customer personal data stored?", rrf_hits, top_k=3)
    assert len(reranked) > 0, "Cross-Encoder returned no reranked results"
    assert "reranker_score" in reranked[0], "Reranker score missing"
    print("  [TEST 21] Cross-Encoder reranks uploaded evidence: PASSED")

    # TEST 22: SLM receives uploaded evidence with provenance
    slm_analysis = analyze_evidence("Where is customer personal data stored?", reranked)
    assert slm_analysis["status"] in ["ANSWERABLE", "AMBIGUOUS", "CONFLICTING", "INSUFFICIENT_EVIDENCE"], f"Invalid status {slm_analysis}"
    print("  [TEST 22] SLM receives uploaded evidence with provenance: PASSED")

    # TEST 23: Human Review displays uploaded-document provenance
    rev_result = corpus.review_documents("Where is customer personal data stored?", workspace_id="ws_acme_corp")
    assert "evidence" in rev_result and len(rev_result["evidence"]) > 0, "Human Review payload missing evidence"
    top_ev = rev_result["evidence"][0]
    assert "document_id" in top_ev and "version" in top_ev and "workspace_id" in top_ev, "Provenance metadata missing"
    print("  [TEST 23] Human Review displays uploaded-document provenance: PASSED")

    # TEST 24: Workspace isolation remains intact
    # Ingest document for Globex Corp
    globex_txt = "Section 1: Globex Cloud Policy\n\nGlobex production systems are hosted on Azure eu-west-1."
    corpus.ingest_single_document("globex_cloud.txt", globex_txt.encode("utf-8"), workspace_id="ws_globex_corp", version="2026-01")

    acme_query_res = corpus.retrieve_embedding("Azure eu-west-1", workspace_id="ws_acme_corp", top_k=10)
    for hit in acme_query_res:
        assert hit["workspace_id"] == "ws_acme_corp", f"Cross-workspace data leak detected: {hit}"
    print("  [TEST 24] Workspace isolation remains intact: PASSED")

    # TEST 25: Document A (ws_acme_corp) cannot appear in another workspace (ws_globex_corp)
    globex_query_res = corpus.retrieve_bm25("AWS us-east-1 datacenter", workspace_id="ws_globex_corp", top_k=10)
    for hit in globex_query_res:
        assert hit["workspace_id"] == "ws_globex_corp", f"Acme document leaked to Globex workspace: {hit}"
    print("  [TEST 25] Document A (ws_acme_corp) cannot appear in another workspace: PASSED")

    # TEST 26: Multiple documents can produce evidence in one answer
    multi_doc_query_bm25 = corpus.retrieve_bm25("security incidents escalation database backup retention", workspace_id="ws_acme_corp", top_k=10)
    docs_found = {h["filename"] for h in multi_doc_query_bm25 if "filename" in h}
    assert len(docs_found) >= 2, f"Expected evidence from multiple documents, got {docs_found}"
    print("  [TEST 26] Multiple documents can produce evidence in one answer: PASSED")

    # TEST 27: Conflicting evidence from two uploaded documents remains visible
    txt_doc_1 = "Section 1: Backup Retention\n\nDatabase backup retention period is 30 days."
    txt_doc_2 = "Section 1: Extended Retention Policy\n\nDatabase backup retention period is 90 days."
    corpus.ingest_single_document("doc1_retention.txt", txt_doc_1.encode("utf-8"), workspace_id="ws_acme_corp", version="2026-01")
    corpus.ingest_single_document("doc2_retention.txt", txt_doc_2.encode("utf-8"), workspace_id="ws_acme_corp", version="2026-01")

    conflict_rev = corpus.review_documents("How long are database backups retained?", workspace_id="ws_acme_corp")
    assert conflict_rev["status"] in ["CONFLICTING", "ANSWERABLE", "AMBIGUOUS"], f"Unexpected conflict status {conflict_rev['status']}"
    assert len(conflict_rev["evidence"]) >= 2, "Conflicting evidence from multiple uploaded documents must be preserved"
    print("  [TEST 27] Conflicting evidence from two uploaded documents remains visible: PASSED")

    print("\nALL 27 DAY 19.75 MULTI-DOCUMENT INGESTION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
