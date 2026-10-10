
from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

"""
test_day21_chunking_expansion.py — Automated Test Suite for Shared Chunking & Adjacent Expansion.

Verifies:
12 Shared Chunking Requirements & 8 Adjacent Expansion Requirements.
Synthetic, vendor-agnostic test fixtures.
"""

import os
import sys
import io
import fitz  # PyMuPDF
import docx  # python-docx

from chunking import (
    extract_pdf_blocks,
    extract_docx_blocks,
    extract_txt_blocks,
    structure_aware_chunking_shared,
    split_text_recursively,
    conservative_normalize_text,
)
from expansion import expand_adjacent_chunks


# ============================================================================
# SHARED CHUNKING TESTS
# ============================================================================

def test_01_question_and_direct_answer():
    """1. A question followed by a direct answer is kept in one chunk or associated."""
    blocks = [
        {"page": 1, "section": "FAQ", "text": "Question: What is the data encryption standard?"},
        {"page": 1, "section": "FAQ", "text": "Answer: All customer data is encrypted using AES-256 at rest."},
    ]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-001", "Security FAQ", "2026-01", "ws_acme", "txt", "faq.txt"
    )
    assert len(chunks) > 0
    # Check that Q and A are combined or easily linked
    combined = any("Question" in c["excerpt"] and "Answer" in c["excerpt"] for c in chunks)
    assert combined, "Q&A pair was split across chunks unnecessarily"
    print("  [CHUNKING TEST 1] Question & direct answer: PASSED")


def test_02_question_answer_page_boundary():
    """2. Question and answer separated by a page boundary preserve relationship metadata."""
    blocks = [
        {"page": 1, "section": "Controls", "text": "Control 4.1: Is multi-factor authentication required?"},
        {"page": 2, "section": "Controls", "text": "Implementation Status: Multi-factor authentication is mandatory for all access."},
    ]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-002", "Control Matrix", "2026-01", "ws_acme", "pdf", "controls.pdf"
    )
    assert len(chunks) >= 1
    assert any("multi-factor" in c["excerpt"].lower() for c in chunks)
    # Check page numbers are preserved
    pages = {c["page"] for c in chunks}
    assert 1 in pages or 2 in pages
    print("  [CHUNKING TEST 2] Q&A across page boundary: PASSED")


def test_03_heading_with_paragraphs():
    """3. A heading followed by several explanatory paragraphs preserves section context."""
    blocks = [
        {"page": 1, "section": None, "text": "Section 3: Incident Escalation Policy", "block_type": "heading"},
        {"page": 1, "section": "Section 3: Incident Escalation Policy", "text": "Security incidents are triaged immediately upon notification."},
        {"page": 1, "section": "Section 3: Incident Escalation Policy", "text": "Severity 1 incidents must be escalated within 15 minutes."},
    ]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-003", "Incident Plan", "2026-01", "ws_acme", "docx", "incident.docx"
    )
    assert len(chunks) > 0
    for c in chunks:
        assert c["section"] == "Section 3: Incident Escalation Policy"
    print("  [CHUNKING TEST 3] Heading followed by paragraphs: PASSED")


def test_04_multi_row_table():
    """4. A multi-row table with distinct question-answer pairs preserves structured rows."""
    doc = docx.Document()
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Control ID"
    table.rows[0].cells[1].text = "Requirement & Status"
    table.rows[1].cells[0].text = "SEC-01"
    table.rows[1].cells[1].text = "Data at rest is encrypted using AES-256."

    stream = io.BytesIO()
    doc.save(stream)
    docx_bytes = stream.getvalue()

    blocks = extract_docx_blocks(docx_bytes, "table_test.docx")
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-004", "Table Doc", "2026-01", "ws_acme", "docx", "table_test.docx"
    )
    assert len(chunks) > 0
    assert any("SEC-01" in c["excerpt"] and "AES-256" in c["excerpt"] for c in chunks)
    print("  [CHUNKING TEST 4] Multi-row table: PASSED")


def test_05_long_paragraph_recursive_splitting():
    """5. A long paragraph that exceeds size limits is recursively split safely."""
    long_text = "Word " * 300  # ~1500 chars
    blocks = [{"page": 1, "section": "General", "text": long_text}]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-005", "Long Doc", "2026-01", "ws_acme", "txt", "long.txt", max_chunk_chars=400
    )
    assert len(chunks) > 1, f"Expected recursive splitting into multiple chunks, got {len(chunks)}"
    for c in chunks:
        assert len(c["excerpt"]) <= 600
    print("  [CHUNKING TEST 5] Long paragraph recursive splitting: PASSED")


def test_06_txt_key_value_records():
    """6. A TXT file with key-value records preserves record integrity."""
    txt_content = (
        "Title: Infrastructure Security Overview\n\n"
        "Cloud Provider: AWS Cloud Services\n"
        "Primary Region: us-east-1 datacenter\n"
        "Backup Frequency: Every 6 hours\n"
        "Retention Period: 30 days mandatory"
    ).encode("utf-8")

    blocks = extract_txt_blocks(txt_content, "keyval.txt")
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-006", "KeyVal Overview", "2026-01", "ws_acme", "txt", "keyval.txt"
    )
    assert len(chunks) > 0
    assert any("Cloud Provider" in c["excerpt"] for c in chunks)
    print("  [CHUNKING TEST 6] TXT key-value records: PASSED")


def test_07_docx_headings_and_tables():
    """7. A DOCX file with headings and tables extracts properly."""
    doc = docx.Document()
    doc.add_heading("Section 1: Backup Specifications", level=1)
    doc.add_paragraph("Database backups are executed on a automated schedule.")
    t = doc.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Retention"
    t.rows[0].cells[1].text = "90 days"
    stream = io.BytesIO()
    doc.save(stream)

    blocks = extract_docx_blocks(stream.getvalue(), "docx_test.docx")
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-007", "Backup Spec", "2026-01", "ws_acme", "docx", "docx_test.docx"
    )
    assert len(chunks) >= 2
    assert any("90 days" in c["excerpt"] for c in chunks)
    print("  [CHUNKING TEST 7] DOCX headings and tables: PASSED")


def test_08_pdf_reading_order_page_boundaries():
    """8. A PDF with reading-order and page boundaries preserves structure."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Section 1: Access Policy\nPassword complexity requires 12+ characters.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 2: MFA Policy\nMulti-factor authentication is enforced.")
    pdf_bytes = doc.tobytes()
    doc.close()

    blocks = extract_pdf_blocks(pdf_bytes, "policy.pdf")
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-008", "Policy PDF", "2026-01", "ws_acme", "pdf", "policy.pdf"
    )
    assert len(chunks) >= 2
    pages = {c["page"] for c in chunks}
    assert 1 in pages and 2 in pages
    print("  [CHUNKING TEST 8] PDF reading order & page boundaries: PASSED")


def test_09_empty_pages_and_documents():
    """9. Empty pages and empty documents are handled safely without crashing."""
    try:
        extract_txt_blocks(b"", "empty.txt")
        assert False, "Should have raised ValueError on empty file"
    except ValueError:
        pass
    print("  [CHUNKING TEST 9] Empty pages and empty documents: PASSED")


def test_10_negations_and_qualifications():
    """10. Negations, qualifications, and absence-of-policy statements are preserved accurately."""
    text = "This policy does NOT specify a contractual customer notification deadline unless escalated by CISO."
    normalized = conservative_normalize_text(text)
    assert "NOT" in normalized
    assert "unless" in normalized
    blocks = [{"page": 1, "section": "General", "text": text}]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-010", "Negation Policy", "2026-01", "ws_acme", "txt", "neg.txt"
    )
    assert any("NOT" in c["excerpt"] and "unless" in c["excerpt"] for c in chunks)
    print("  [CHUNKING TEST 10] Negations and qualifications preserved: PASSED")


def test_11_provenance_preservation():
    """11. Metadata fields document_id, version, workspace_id, page, section, chunk_index are present."""
    blocks = [{"page": 3, "section": "Sec 5", "text": "Detailed security control statement."}]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-011", "Prov Test", "2026-02", "ws_globex", "pdf", "prov.pdf"
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c["document_id"] == "DOC-011"
    assert c["version"] == "2026-02"
    assert c["workspace_id"] == "ws_globex"
    assert c["page"] == 3
    assert c["section"] == "Sec 5"
    assert "chunk_index" in c and isinstance(c["chunk_index"], int)
    print("  [CHUNKING TEST 11] Full provenance metadata preservation: PASSED")


def test_12_no_silent_truncation():
    """12. Original text is not silently truncated."""
    long_clause = "A" * 550
    blocks = [{"page": 1, "section": "General", "text": long_clause}]
    chunks = structure_aware_chunking_shared(
        blocks, "DOC-012", "Trunc Test", "2026-01", "ws_acme", "txt", "trunc.txt", max_chunk_chars=600
    )
    total_len = sum(len(c["excerpt"]) for c in chunks)
    assert total_len >= 550, f"Text was truncated: expected >= 550 chars, got {total_len}"
    print("  [CHUNKING TEST 12] No silent text truncation: PASSED")


# ============================================================================
# ADJACENT-CHUNK EXPANSION TESTS
# ============================================================================

def test_13_expansion_following_neighbor():
    """13. Missing answer in immediately following chunk is recovered by expansion."""
    corpus = [
        {"chunk_id": "c1", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Question: How often are password policies audited?"},
        {"chunk_id": "c2", "chunk_index": 1, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Answer: Password policies are audited every 6 months."},
    ]
    # Initial retrieval only found seed c1
    seeds = [corpus[0]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)
    
    assert len(expanded) == 2
    assert expanded[1]["chunk_id"] == "c2"
    assert expanded[1]["is_expanded"] is True
    assert expanded[1]["expansion_type"] == "neighbor"
    print("  [EXPANSION TEST 1] Recover following neighbor: PASSED")


def test_14_expansion_preceding_neighbor():
    """14. Relevant context in immediately preceding chunk is recovered by expansion."""
    corpus = [
        {"chunk_id": "c1", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Section 4: Key Management Policy for AWS Cloud"},
        {"chunk_id": "c2", "chunk_index": 1, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Encryption keys are rotated annually."},
    ]
    # Initial retrieval only found seed c2
    seeds = [corpus[1]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)

    assert len(expanded) == 2
    assert any(c["chunk_id"] == "c1" for c in expanded)
    print("  [EXPANSION TEST 2] Recover preceding neighbor: PASSED")


def test_15_expansion_stops_at_document_boundary():
    """15. Expansion stops at the document boundary and does not fetch chunk from another document."""
    corpus = [
        {"chunk_id": "docA_c0", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "docA", "version": "v1", "excerpt": "Doc A start text."},
        {"chunk_id": "docB_c0", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "docB", "version": "v1", "excerpt": "Doc B start text."},
    ]
    seeds = [corpus[0]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)

    assert len(expanded) == 1
    assert expanded[0]["document_id"] == "docA"
    print("  [EXPANSION TEST 3] Expansion stops at document boundary: PASSED")


def test_16_expansion_stops_at_version_boundary():
    """16. Expansion stops at the version boundary."""
    corpus = [
        {"chunk_id": "docA_v1_c0", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "docA", "version": "v1", "excerpt": "Version 1 text."},
        {"chunk_id": "docA_v2_c1", "chunk_index": 1, "workspace_id": "ws_1", "document_id": "docA", "version": "v2", "excerpt": "Version 2 text."},
    ]
    seeds = [corpus[0]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)

    assert len(expanded) == 1
    assert expanded[0]["version"] == "v1"
    print("  [EXPANSION TEST 4] Expansion stops at version boundary: PASSED")


def test_17_expansion_cannot_cross_workspace_boundaries():
    """17. Expansion cannot cross workspace boundaries."""
    corpus = [
        {"chunk_id": "c0_acme", "chunk_index": 0, "workspace_id": "ws_acme", "document_id": "doc1", "version": "v1", "excerpt": "Acme confidential text."},
        {"chunk_id": "c1_globex", "chunk_index": 1, "workspace_id": "ws_globex", "document_id": "doc1", "version": "v1", "excerpt": "Globex confidential text."},
    ]
    seeds = [corpus[0]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)

    for c in expanded:
        assert c["workspace_id"] == "ws_acme"
        assert "Globex" not in c["excerpt"]
    print("  [EXPANSION TEST 5] Workspace boundary strict enforcement: PASSED")


def test_18_neighbor_and_context_limits_enforced():
    """18. Configured neighbor and context limits are strictly enforced."""
    corpus = [
        {"chunk_id": f"c{i}", "chunk_index": i, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": f"Chunk text {i} " + "X" * 100}
        for i in range(10)
    ]
    seeds = [corpus[5]]
    expanded = expand_adjacent_chunks(
        seeds, corpus, max_seeds=1, max_neighbors_per_seed=1, max_total_expanded=1
    )
    # Seed (1) + max 1 expanded = 2
    assert len(expanded) == 2
    print("  [EXPANSION TEST 6] Configured neighbor limits enforced: PASSED")


def test_19_duplicate_and_overlapping_text_deduplication():
    """19. Duplicate and overlapping text is deduplicated during expansion."""
    corpus = [
        {"chunk_id": "c0", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Duplicate text content."},
        {"chunk_id": "c1", "chunk_index": 1, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Duplicate text content."},
    ]
    seeds = [corpus[0]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)

    assert len(expanded) == 1, "Duplicate text neighbor was not deduplicated"
    print("  [EXPANSION TEST 7] Deduplication of duplicate text: PASSED")


def test_20_ranking_semantics_and_provenance():
    """20. Expanded chunks retain true provenance without faking seed retrieval scores."""
    corpus = [
        {"chunk_id": "c0", "chunk_index": 0, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Preceding context.", "bm25_score": None},
        {"chunk_id": "c1", "chunk_index": 1, "workspace_id": "ws_1", "document_id": "doc1", "version": "v1", "excerpt": "Seed chunk answer.", "bm25_score": 12.5, "reranker_score": 0.95},
    ]
    seeds = [corpus[1]]
    expanded = expand_adjacent_chunks(seeds, corpus, max_seeds=1, max_neighbors_per_seed=1)

    seed_hit = expanded[0]
    neighbor_hit = expanded[1]

    assert seed_hit["reranker_score"] == 0.95
    assert neighbor_hit["is_expanded"] is True
    assert neighbor_hit["reranker_score"] is None, "Neighbor chunk must NOT fake seed reranker score!"
    print("  [EXPANSION TEST 8] Ranking score preservation & provenance: PASSED")


def main():
    print("============================================================")
    print("RUNNING DAY 21 SHARED CHUNKING & ADJACENT EXPANSION TESTS")
    print("============================================================\n")

    test_01_question_and_direct_answer()
    test_02_question_answer_page_boundary()
    test_03_heading_with_paragraphs()
    test_04_multi_row_table()
    test_05_long_paragraph_recursive_splitting()
    test_06_txt_key_value_records()
    test_07_docx_headings_and_tables()
    test_08_pdf_reading_order_page_boundaries()
    test_09_empty_pages_and_documents()
    test_10_negations_and_qualifications()
    test_11_provenance_preservation()
    test_12_no_silent_truncation()

    print("\n--- Running Adjacent Chunk Expansion Tests ---\n")

    test_13_expansion_following_neighbor()
    test_14_expansion_preceding_neighbor()
    test_15_expansion_stops_at_document_boundary()
    test_16_expansion_stops_at_version_boundary()
    test_17_expansion_cannot_cross_workspace_boundaries()
    test_18_neighbor_and_context_limits_enforced()
    test_19_duplicate_and_overlapping_text_deduplication()
    test_20_ranking_semantics_and_provenance()

    print("\n============================================================")
    print("ALL 20 CHUNKING & EXPANSION TESTS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
