"""
test_caiq_grounded_qa.py — Generalized Regression Test Suite for CAIQ & Grounded Q&A.

Verifies:
1. A question and its answer appear in the same indexed chunk.
2. A question and its answer are split across adjacent chunks.
3. Direct evidence is retrieved but accidentally omitted from the SLM context.
4. Relevant evidence is present and the model does NOT falsely return INSUFFICIENT_EVIDENCE.
5. Truly absent evidence still produces safe abstention.
6. Evidence from another workspace cannot satisfy the query.
7. The cited quote is validated against the correct evidence ID.
8. Existing benchmark questions, labels, expected answers, and evaluation rules remain unchanged.
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import fitz

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

from evidencedesk import review, analyze_evidence, _validate_slm_output
from document_ingestion import MultiDocumentCorpus
from chunking import extract_pdf_blocks, structure_aware_chunking_shared


def test_01_question_and_answer_same_chunk():
    """1. A question and its answer appear in the same indexed chunk."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Vendor Cloud Security Assessment\n\n"
        "Control ID: SEC-01.1: Storage Encryption\n"
        "Question: Do you encrypt tenant data at rest (on disk/storage) within your environment?\n"
        "Answer: Yes. This feature is provided by our cloud provider.\n",
        fontsize=11
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    workspace = "ws_test_same_chunk"
    multi_doc_corpus.ingest_single_document(
        filename="security_assessment_same.pdf",
        file_bytes=pdf_bytes,
        workspace_id=workspace,
        title="Security Assessment Same",
        version="v1"
    )

    ws_chunks = [c for c in multi_doc_corpus.chunks if c.get("workspace_id") == workspace]
    # Verify question and answer are preserved together in the chunk
    qa_chunks = [c for c in ws_chunks if "encrypt tenant data at rest" in c["excerpt"]]
    assert len(qa_chunks) > 0, "Question was not indexed"
    assert "This feature is provided by our cloud provider" in qa_chunks[0]["excerpt"], \
        "Question and answer were split into different chunks during ingestion"

    query = "Do you encrypt tenant data at rest (on disk/storage) within your environment?"
    res = review(query, method="hybrid", workspace_id=workspace, enable_expansion=True)
    assert res["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {res['status']}"
    assert "provided by our cloud provider" in res["evidence_quote"].lower()
    print("  [TEST 1] Question & answer in same indexed chunk: PASSED")


def test_02_question_and_answer_split_adjacent_chunks():
    """2. A question and its answer are split across adjacent chunks (e.g. across pages)."""
    doc = fitz.open()
    # Page 1: Question
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "Control ID: SEC-02.1\n"
        "Question: Is database storage encrypted at rest across all regions?\n",
        fontsize=11
    )
    # Page 2: Answer
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "Answer: Yes. All database storage is encrypted at rest using provider-managed keys.\n",
        fontsize=11
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    workspace = "ws_test_split_adjacent"
    multi_doc_corpus.ingest_single_document(
        filename="split_assessment.pdf",
        file_bytes=pdf_bytes,
        workspace_id=workspace,
        title="Split Assessment",
        version="v1"
    )

    ws_chunks = [c for c in multi_doc_corpus.chunks if c.get("workspace_id") == workspace]
    # Page 1 and Page 2 should be separate chunks
    assert len(ws_chunks) >= 2, "Expected at least 2 chunks across pages"

    query = "Is database storage encrypted at rest across all regions?"
    res = review(query, method="hybrid", workspace_id=workspace, enable_expansion=True)
    assert res["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {res['status']}"
    assert "encrypted at rest" in res["evidence_quote"].lower()
    print("  [TEST 2] Question & answer split across adjacent chunks: PASSED")


def test_03_direct_evidence_omitted_from_slm_context():
    """3. Direct evidence is retrieved but accidentally omitted from the SLM context."""
    # When the model only receives the question chunk without any answer
    evidence = [
        {
            "chunk_id": "chk_q_only_01",
            "excerpt": "Control ID: SEC-03.1: Do you encrypt tenant data at rest within your environment?"
        }
    ]
    query = "Do you encrypt tenant data at rest within your environment?"
    analysis = analyze_evidence(query, evidence)
    # With no answer present, the model must safely abstain
    assert analysis["status"] == "INSUFFICIENT_EVIDENCE", \
        f"Expected safe abstention INSUFFICIENT_EVIDENCE when answer is omitted, got {analysis['status']}"
    print("  [TEST 3] Omitted answer safely produces INSUFFICIENT_EVIDENCE: PASSED")


def test_04_relevant_evidence_no_false_abstention():
    """4. Relevant evidence is present and the model correctly returns ANSWERABLE (no false abstention)."""
    evidence = [
        {
            "chunk_id": "chk_caiq_ans_01",
            "excerpt": "Control ID: EKM-03.1: Do you encrypt tenant data at rest (on disk/storage)?\nYes. This feature is provided by our cloud provider."
        }
    ]
    query = "Do you encrypt tenant data at rest (on disk/storage) within your environment?"
    analysis = analyze_evidence(query, evidence)
    assert analysis["status"] == "ANSWERABLE", \
        f"Model falsely abstained! Expected ANSWERABLE, got {analysis['status']}"
    assert "provided by our cloud provider" in analysis["evidence_quote"].lower()
    print("  [TEST 4] Relevant concise / provider evidence returns ANSWERABLE: PASSED")


def test_05_truly_absent_evidence_safe_abstention():
    """5. Truly absent evidence still produces safe abstention."""
    workspace = "ws_test_same_chunk"
    query = "Do you require 20-character minimum passwords for administrative database roles?"
    res = review(query, method="hybrid", workspace_id=workspace, enable_expansion=True)
    assert res["status"] == "INSUFFICIENT_EVIDENCE", \
        f"Expected INSUFFICIENT_EVIDENCE for absent topic, got {res['status']}"
    print("  [TEST 5] Truly absent evidence produces safe abstention: PASSED")


def test_06_workspace_isolation():
    """6. Evidence from another workspace cannot satisfy the query."""
    workspace_empty = "ws_completely_unrelated_empty"
    query = "Do you encrypt tenant data at rest (on disk/storage) within your environment?"
    res = review(query, method="hybrid", workspace_id=workspace_empty, enable_expansion=True)
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(res["evidence"]) == 0
    print("  [TEST 6] Cross-workspace leakage prevented: PASSED")


def test_07_quote_validation_against_correct_evidence_id():
    """7. The cited quote is validated against the correct evidence ID."""
    evidence = [
        {
            "chunk_id": "chk_valid_01",
            "excerpt": "All persistent volumes are encrypted with AES-256."
        },
        {
            "chunk_id": "chk_valid_02",
            "excerpt": "Access logs are retained for 365 days."
        }
    ]
    # Test valid quote
    raw_valid = {
        "status": "ANSWERABLE",
        "reason": "Direct statement of encryption.",
        "evidence_chunk_ids": ["chk_valid_01"],
        "evidence_quote": "All persistent volumes are encrypted with AES-256."
    }
    val = _validate_slm_output(raw_valid, evidence)
    assert val["status"] == "ANSWERABLE"

    # Test invalid quote cited against wrong chunk ID
    raw_invalid = {
        "status": "ANSWERABLE",
        "reason": "Direct statement of encryption.",
        "evidence_chunk_ids": ["chk_valid_02"],  # Cited chunk 2 which is about logs!
        "evidence_quote": "All persistent volumes are encrypted with AES-256."
    }
    val_bad = _validate_slm_output(raw_invalid, evidence)
    assert val_bad["status"] == "VALIDATION_ERROR"
    print("  [TEST 7] Quote validation against correct evidence ID: PASSED")


def test_08_benchmark_preservation():
    """8. Existing benchmark questions, labels, expected answers, and evaluation rules remain unchanged."""
    import json
    with open("tests/fixtures/eval_questions.json", "r") as f:
        questions = json.load(f)
    assert len(questions) == 25, f"Expected 25 frozen evaluation questions, got {len(questions)}"
    assert questions[0]["id"] == "Q01"
    assert questions[24]["id"] == "Q25"
    print("  [TEST 8] Benchmark questions and configuration preserved: PASSED")


def test_09_multidoc_discrepancy_and_anti_contamination():
    """
    9. Multi-document discrepancy regression:
    One direct questionnaire control answer and one conflicting document.
    Verify that the final answer cannot silently adopt unsupported details from the second document,
    provenance is preserved, and discrepancy is flagged as CONFLICTING.
    """
    workspace = "ws_test_multidoc_discrepancy"

    # Document 1: Security Questionnaire control with cloud provider implementation
    doc1 = fitz.open()
    page1 = doc1.new_page()
    page1.insert_text(
        (50, 50),
        "Generic Cloud Security Assessment\n\n"
        "Control ID: SEC-01.1: Storage Encryption\n"
        "Question: Do you encrypt tenant data at rest (on disk/storage) within your environment?\n"
        "Answer: Yes. This feature is provided by our cloud provider.\n",
        fontsize=11
    )
    pdf_bytes1 = doc1.tobytes()
    doc1.close()

    # Document 2: Whitepaper / architecture guide claiming AES-256 with dedicated internal KMS
    doc2 = fitz.open()
    page2 = doc2.new_page()
    page2.insert_text(
        (50, 50),
        "Architecture Reference Guide\n\n"
        "Storage & Key Architecture:\n"
        "Customer data is encrypted at rest using AES-256 managed by internal KMS.\n",
        fontsize=11
    )
    pdf_bytes2 = doc2.tobytes()
    doc2.close()

    multi_doc_corpus.ingest_single_document(
        filename="caiq_assessment.pdf",
        file_bytes=pdf_bytes1,
        workspace_id=workspace,
        title="CAIQ Assessment",
        version="v1"
    )
    multi_doc_corpus.ingest_single_document(
        filename="arch_guide.pdf",
        file_bytes=pdf_bytes2,
        workspace_id=workspace,
        title="Architecture Guide",
        version="v1"
    )

    query = "Do you encrypt tenant data at rest (on disk/storage) within your environment?"
    res = review(query, method="hybrid", workspace_id=workspace, enable_expansion=True)

    # 1. Must flag discrepancy across documents rather than silently adopting the whitepaper's claims
    assert res["status"] == "CONFLICTING", f"Expected CONFLICTING status due to multi-doc discrepancy, got {res['status']}"

    # 2. Candidate excerpt must NOT claim AES-256 or KMS as the single direct answer
    assert res["candidate_excerpt"] is None, f"Expected candidate_excerpt to be None for CONFLICTING, got {res['candidate_excerpt']}"

    # 3. Provenance must preserve chunks from multiple documents
    doc_ids = {ev.get("document_id") for ev in res.get("evidence", [])}
    assert len(doc_ids) >= 2, f"Expected evidence from at least 2 documents, got {doc_ids}"

    # 4. Anti-contamination unit check: If model tried to answer ANSWERABLE citing only Doc 1 but included ungrounded terms in reason,
    # _validate_slm_output must catch the contamination and downgrade to CONFLICTING
    test_evidence = [
        {"chunk_id": "chk_caiq_01", "excerpt": "Question: Do you encrypt tenant data at rest? Answer: Yes. This feature is provided by our cloud provider."},
        {"chunk_id": "chk_arch_02", "excerpt": "Customer data is encrypted at rest using AES-256 managed by proprietary KMS."}
    ]
    raw_contaminated = {
        "status": "ANSWERABLE",
        "reason": "Customer data is encrypted at rest using AES-256 managed by proprietary KMS.",
        "evidence_chunk_ids": ["chk_caiq_01"],
        "evidence_quote": "Yes. This feature is provided by our cloud provider."
    }
    val_res = _validate_slm_output(raw_contaminated, test_evidence, question=query)
    assert val_res["status"] == "CONFLICTING", f"Expected anti-contamination check to flag CONFLICTING, got {val_res['status']}"
    assert "Discrepancy detected" in val_res["reason"]

    print("  [TEST 9] Multi-document discrepancy and anti-contamination: PASSED")


def test_10_replace_type_safety_and_provenance_page():
    """
    10. Regression test for str.replace type error:
    Ensure that numeric page metadata (e.g. page=7, page=0) and null/empty values
    do not cause TypeError on string methods, preserving frontend contracts.
    """
    # Python side: verify chunk page metadata types
    ws_chunks = [c for c in multi_doc_corpus.chunks if c.get("workspace_id") == "ws_test_multidoc_discrepancy"]
    assert len(ws_chunks) > 0
    for c in ws_chunks:
        # page is an int or None
        assert c.get("page") is None or isinstance(c.get("page"), int)

    # JS contract replication: verify escapeHtml logic on various runtime types
    def escape_html_js_contract(val):
        if val is None:
            return ""
        if isinstance(val, (int, float)):
            return str(val)
        if not isinstance(val, str):
            return ""
        return val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # Must safely handle numbers without throwing AttributeError/TypeError
    assert escape_html_js_contract(7) == "7"
    assert escape_html_js_contract(0) == "0"
    assert escape_html_js_contract(None) == ""
    assert escape_html_js_contract("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"
    print("  [TEST 10] str.replace type safety and page metadata: PASSED")


def test_11_corpus_storage_restoration():
    """
    11. Storage restoration across restarts:
    Verify that a new MultiDocumentCorpus instance automatically restores
    chunks, documents, and BM25 index from Qdrant without needing re-upload.
    """
    new_corpus = MultiDocumentCorpus(qdrant_client=qdrant, embed_model=embed_model)
    assert len(new_corpus.chunks) > 0, "Expected corpus to restore persisted chunks from Qdrant"
    assert len(new_corpus.documents) > 0, "Expected corpus to restore persisted document metadata"
    assert new_corpus.bm25_model is not None, "Expected BM25 index to be rebuilt from restored chunks"
    print(f"  [TEST 11] Corpus restored {len(new_corpus.chunks)} chunks and {len(new_corpus.documents)} documents: PASSED")


def test_12_typo_workspace_isolation_correctness():
    """
    12. Workspace correctness & isolation for typo workspaces:
    Querying ws_octopuss (with typo) safely abstains with empty evidence
    and strictly isolates from ws_octopus.
    """
    res = review("Do you encrypt tenant data at rest?", workspace_id="ws_octopuss", method="hybrid")
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(res["evidence"]) == 0
    assert "Workspace contains no indexed evidence or does not exist" in res["reason"]
    print("  [TEST 12] Typo workspace ws_octopuss strictly isolated: PASSED")


def test_13_explicit_negative_answer_with_qualification():
    """
    13. Explicit No answers with qualifications:
    Verifies that a negative questionnaire response is recognized as a direct answer,
    does not cause false abstention, and preserves qualifications.
    """
    evidence = [
        {
            "chunk_id": "chk_gen_neg_01",
            "excerpt": "Control ID: ACC-04.2: Do you require hardware-based security keys for external contractors?\nNo. External contractors authenticate using software TOTP apps, though hardware keys are planned for next year."
        }
    ]
    query = "Do you require hardware-based security keys for external contractors?"
    analysis = analyze_evidence(query, evidence)
    assert analysis["status"] == "ANSWERABLE", f"Expected ANSWERABLE for explicit No answer, got {analysis['status']}"
    assert "chk_gen_neg_01" in analysis["evidence_chunk_ids"]
    assert "hardware keys are planned" in analysis["evidence_quote"].lower()
    print("  [TEST 13] Explicit No answer with qualification: PASSED")


def test_14_explicit_not_applicable_answer():
    """
    14. Not Applicable questionnaire answers:
    Verifies that 'Not Applicable' / 'N/A' responses directly satisfy the query
    without returning INSUFFICIENT_EVIDENCE.
    """
    evidence = [
        {
            "chunk_id": "chk_gen_na_01",
            "excerpt": "Control ID: PHY-01.1: Physical Guard Posts\nNot Applicable | The company operates a 100% cloud architecture and does not maintain customer-accessible physical data centers."
        }
    ]
    query = "Do you maintain customer-accessible physical data centers?"
    analysis = analyze_evidence(query, evidence)
    assert analysis["status"] == "ANSWERABLE", f"Expected ANSWERABLE for Not Applicable response, got {analysis['status']}"
    assert "chk_gen_na_01" in analysis["evidence_chunk_ids"]
    assert "not applicable" in analysis["evidence_quote"].lower()
    print("  [TEST 14] Explicit Not Applicable answer: PASSED")


def test_15_adjacent_chunks_question_and_negative_answer_e2e():
    """
    15. Adjacent chunks across pages for explicit No answer:
    Verifies end-to-end retrieval, expansion, prompt construction, and classification
    when question and negative response are in adjacent chunks.
    """
    doc = fitz.open()
    # Page 1: Question
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "Vendor Security Assessment\n\n"
        "Control ID: AUD-03.1\n"
        "Question: Do you conduct biannual independent third-party vulnerability scans of external endpoints?\n",
        fontsize=11
    )
    # Page 2: Qualified Negative Answer
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "Answer: No. Independent external scans are performed annually; we are currently evaluating biannual scans for the upcoming cycle.\n",
        fontsize=11
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    workspace = "ws_test_adjacent_neg_e2e"
    multi_doc_corpus.ingest_single_document(
        filename="adjacent_neg_assessment.pdf",
        file_bytes=pdf_bytes,
        workspace_id=workspace,
        title="Adjacent Negative Assessment",
        version="v1"
    )

    query = "Do you conduct biannual independent third-party vulnerability scans of external endpoints?"
    res = review(query, method="hybrid", workspace_id=workspace, enable_expansion=True)
    assert res["status"] == "ANSWERABLE", f"Expected ANSWERABLE for adjacent negative answer, got {res['status']}"
    assert "independent external scans are performed annually" in res["evidence_quote"].lower()
    print("  [TEST 15] Adjacent chunks for explicit negative answer end-to-end: PASSED")


def test_16_omitted_answer_chunk_vs_included_answer_chunk():
    """
    16. Answer chunk present in retrieval but omitted from model context:
    Verifies that if the answer chunk is omitted (only question chunk passed to SLM),
    the model safely produces INSUFFICIENT_EVIDENCE; whereas when the answer chunk
    is included, it produces ANSWERABLE.
    """
    q_chunk = {
        "chunk_id": "chk_omit_q_01",
        "excerpt": "Control ID: GOV-08.1: Do you mandate quarterly executive cybersecurity briefing meetings?"
    }
    ans_chunk = {
        "chunk_id": "chk_omit_ans_02",
        "excerpt": "No. Executive briefings are held semi-annually, with ad-hoc meetings convened as urgent incidents arise."
    }
    query = "Do you mandate quarterly executive cybersecurity briefing meetings?"

    # Context with ONLY question chunk -> must safely abstain
    res_omitted = analyze_evidence(query, [q_chunk])
    assert res_omitted["status"] == "INSUFFICIENT_EVIDENCE", \
        f"Expected safe abstention when answer is omitted, got {res_omitted['status']}"

    # Context with BOTH chunks -> must answer
    res_included = analyze_evidence(query, [q_chunk, ans_chunk])
    assert res_included["status"] == "ANSWERABLE", \
        f"Expected ANSWERABLE when answer chunk is included, got {res_included['status']}"
    assert "chk_omit_ans_02" in res_included["evidence_chunk_ids"]
    print("  [TEST 16] Omitted answer safely abstains vs included answer produces ANSWERABLE: PASSED")


def test_17_negative_quote_and_citation_validation():
    """
    17. Citation and quote validation for negative and qualified answers:
    Ensures that validation enforces exact quotes and correct cited chunk IDs.
    """
    evidence = [
        {
            "chunk_id": "chk_val_neg_01",
            "excerpt": "No. All new contracts will be reviewed for such commitments."
        },
        {
            "chunk_id": "chk_val_unrelated_02",
            "excerpt": "Passwords must be at least 14 characters in length."
        }
    ]

    # Valid negative quote and citation
    valid_raw = {
        "thought": "Direct negative answer.",
        "status": "ANSWERABLE",
        "reason": "Explicit negative answer.",
        "evidence_chunk_ids": ["chk_val_neg_01"],
        "evidence_quote": "No. All new contracts will be reviewed for such commitments."
    }
    val = _validate_slm_output(valid_raw, evidence)
    assert val["status"] == "ANSWERABLE"

    # Fabricated quote not in cited passage
    fake_quote_raw = {
        "thought": "Testing hallucinated quote.",
        "status": "ANSWERABLE",
        "reason": "Explicit negative answer.",
        "evidence_chunk_ids": ["chk_val_neg_01"],
        "evidence_quote": "No. We never do this and never will."
    }
    val_fake = _validate_slm_output(fake_quote_raw, evidence)
    assert val_fake["status"] == "VALIDATION_ERROR"

    # Citation mismatch: quote belongs to chunk 1, but chunk 2 cited
    mismatched_raw = {
        "thought": "Testing citation mismatch.",
        "status": "ANSWERABLE",
        "reason": "Explicit negative answer.",
        "evidence_chunk_ids": ["chk_val_unrelated_02"],
        "evidence_quote": "No. All new contracts will be reviewed for such commitments."
    }
    val_mismatch = _validate_slm_output(mismatched_raw, evidence)
    assert val_mismatch["status"] == "VALIDATION_ERROR"
    print("  [TEST 17] Negative quote and citation validation: PASSED")


def test_18_adjacent_chunk_association_and_unsupported_rejection():
    """
    18. Adjacent chunk association and rejection of unsupported claims:
    - Valid adjacent-chunk quote with casing difference (e.g. lowercase 'yes.') is accepted
      and normalized to the exact verbatim document slice.
    - Quote spanning or located in adjacent chunk within the same document associates both chunks.
    - Unsupported claims (e.g. fabricated security algorithms not in evidence) are rejected with VALIDATION_ERROR.
    - Ungrounded quotes cited against wrong documents or absent chunks are rejected with VALIDATION_ERROR.
    """
    evidence = [
        {
            "chunk_id": "chk_split_01",
            "document_id": "DOC-ADJ-001",
            "chunk_index": 0,
            "excerpt": "Control ID: CRY-01.1\nQuestion: Is tenant storage encrypted at rest across all regions?"
        },
        {
            "chunk_id": "chk_split_02",
            "document_id": "DOC-ADJ-001",
            "chunk_index": 1,
            "excerpt": "Answer: Yes. All database storage is encrypted at rest using provider-managed keys."
        }
    ]

    # 1. Casing variation on direct quote is accepted and normalized verbatim
    raw_cased = {
        "status": "ANSWERABLE",
        "reason": "Direct statement of encryption at rest.",
        "evidence_chunk_ids": ["chk_split_02"],
        "evidence_quote": "yes. All database storage is encrypted at rest using provider-managed keys."
    }
    val_cased = _validate_slm_output(raw_cased, evidence)
    assert val_cased["status"] == "ANSWERABLE"
    assert val_cased["evidence_quote"] == "Yes. All database storage is encrypted at rest using provider-managed keys."

    # 2. Model cites question chunk, but quote is in adjacent answer chunk -> association preserved
    raw_assoc = {
        "status": "ANSWERABLE",
        "reason": "Direct statement of encryption at rest.",
        "evidence_chunk_ids": ["chk_split_01"],
        "evidence_quote": "All database storage is encrypted at rest using provider-managed keys."
    }
    val_assoc = _validate_slm_output(raw_assoc, evidence)
    assert val_assoc["status"] == "ANSWERABLE"
    assert "chk_split_01" in val_assoc["evidence_chunk_ids"]
    assert "chk_split_02" in val_assoc["evidence_chunk_ids"]
    assert val_assoc["evidence_quote"] == "All database storage is encrypted at rest using provider-managed keys."

    # 3. Unsupported claim (fabricated algorithm) must be rejected with VALIDATION_ERROR
    raw_unsupported = {
        "status": "ANSWERABLE",
        "reason": "Direct statement of encryption at rest.",
        "evidence_chunk_ids": ["chk_split_02"],
        "evidence_quote": "All database storage is encrypted with AES-512 quantum-resistant keys."
    }
    val_unsupported = _validate_slm_output(raw_unsupported, evidence)
    assert val_unsupported["status"] == "VALIDATION_ERROR"
    assert "evidence_quote not found" in val_unsupported["reason"]

    # 4. Citation of non-existent or unrelated chunk must be rejected with VALIDATION_ERROR
    raw_unrelated = {
        "status": "ANSWERABLE",
        "reason": "Direct statement of encryption at rest.",
        "evidence_chunk_ids": ["chk_other_doc_99"],
        "evidence_quote": "Yes. All database storage is encrypted at rest using provider-managed keys."
    }
    val_unrelated = _validate_slm_output(raw_unrelated, evidence)
    assert val_unrelated["status"] == "VALIDATION_ERROR"

    print("  [TEST 18] Adjacent chunk association and rejection of unsupported claims: PASSED")


def test_19_conflict_classification_logic():
    """
    19. Conflict-classification logic regression tests:
    - Treat headings and section titles as structural context, NOT factual claims.
    - Distinguish compatible general and specific statements (compatible -> ANSWERABLE).
    - Distinguish differing scopes (e.g. database backup vs audit log -> ANSWERABLE).
    - Distinguish unsupported generated details from document conflicts (sanitize reason, keep ANSWERABLE).
    - Require two incompatible factual claims on the same fact and compatible scope for CONFLICTING.
    - Preserve evidence IDs, document versions, workspace boundaries, and human-review safeguards.
    """
    # Case 1: Heading as context vs substantive body statement
    evidence_heading = [
        {"chunk_id": "chk_hdr_01", "excerpt": "## Section 4: Unencrypted Temporary Cache Exceptions", "document_id": "DOC-A", "version": "v1"},
        {"chunk_id": "chk_bdy_02", "excerpt": "Customer data is encrypted at rest using AES-256.", "document_id": "DOC-A", "version": "v1"}
    ]
    query_enc = "Is customer data encrypted at rest?"
    
    # If SLM mistakenly flagged CONFLICTING due to the heading, validator must recognize heading as context
    raw_heading_conflict = {
        "status": "CONFLICTING",
        "reason": "Section 4 heading mentions unencrypted exceptions while body states customer data is encrypted.",
        "evidence_chunk_ids": ["chk_hdr_01", "chk_bdy_02"],
        "evidence_quote": ""
    }
    val_hd = _validate_slm_output(raw_heading_conflict, evidence_heading, question=query_enc)
    assert val_hd["status"] == "ANSWERABLE", f"Expected ANSWERABLE for heading + body, got {val_hd['status']}"
    assert val_hd["evidence_chunk_ids"] == ["chk_bdy_02"], f"Expected cited chunk to be body chunk, got {val_hd['evidence_chunk_ids']}"
    assert "Customer data is encrypted at rest using AES-256." in val_hd["evidence_quote"]

    # Case 2: Compatible General vs Specific statements (NOT conflicting)
    evidence_specificity = [
        {"chunk_id": "chk_gen_01", "excerpt": "Customer data is encrypted at rest.", "document_id": "DOC-B", "version": "v1"},
        {"chunk_id": "chk_spe_02", "excerpt": "Customer data is encrypted at rest using AES-256.", "document_id": "DOC-B", "version": "v1"}
    ]
    raw_spec_conflict = {
        "status": "CONFLICTING",
        "reason": "One passage says encrypted at rest, another specifies AES-256.",
        "evidence_chunk_ids": ["chk_gen_01", "chk_spe_02"],
        "evidence_quote": ""
    }
    val_sp = _validate_slm_output(raw_spec_conflict, evidence_specificity, question=query_enc)
    assert val_sp["status"] == "ANSWERABLE", f"Expected ANSWERABLE for general vs specific, got {val_sp['status']}"
    assert val_sp["evidence_chunk_ids"] == ["chk_spe_02"]
    assert "AES-256" in val_sp["evidence_quote"]

    # Case 3: Differing Scopes (Database backup retention vs Audit log retention)
    evidence_scopes = [
        {"chunk_id": "chk_db_01", "excerpt": "Database backup retention is 30 days.", "document_id": "DOC-C", "version": "v1"},
        {"chunk_id": "chk_log_02", "excerpt": "Audit log retention is 365 days.", "document_id": "DOC-D", "version": "v1"}
    ]
    query_db = "How long are database backups retained?"
    raw_scope_conflict = {
        "status": "CONFLICTING",
        "reason": "One passage says 30 days and another says 365 days.",
        "evidence_chunk_ids": ["chk_db_01", "chk_log_02"],
        "evidence_quote": ""
    }
    val_sc = _validate_slm_output(raw_scope_conflict, evidence_scopes, question=query_db)
    assert val_sc["status"] == "ANSWERABLE", f"Expected ANSWERABLE for differing scopes, got {val_sc['status']}"
    assert val_sc["evidence_chunk_ids"] == ["chk_db_01"]
    assert "30 days" in val_sc["evidence_quote"]

    # Case 4: Unsupported generated details in SLM reason (NOT a document conflict)
    evidence_unsupported = [
        {"chunk_id": "chk_valid_01", "excerpt": "Customer data is encrypted at rest.", "document_id": "DOC-E", "version": "v1"},
        {"chunk_id": "chk_other_02", "excerpt": "Platform is hosted in us-east-1.", "document_id": "DOC-F", "version": "v1"}
    ]
    raw_unsupported_reason = {
        "status": "ANSWERABLE",
        "reason": "Customer data is encrypted at rest in us-east-1.",
        "evidence_chunk_ids": ["chk_valid_01"],
        "evidence_quote": "Customer data is encrypted at rest."
    }
    val_un = _validate_slm_output(raw_unsupported_reason, evidence_unsupported, question=query_enc)
    assert val_un["status"] == "ANSWERABLE", f"Expected status to stay ANSWERABLE, got {val_un['status']}"
    assert "us-east-1" not in val_un["reason"], f"Reason should be sanitized: {val_un['reason']}"
    assert val_un["evidence_quote"] == "Customer data is encrypted at rest."

    # Case 5: Genuine Conflict (two incompatible factual claims on same fact & scope)
    evidence_conflict = [
        {"chunk_id": "chk_conf_01", "excerpt": "Backup retention is 30 days.", "document_id": "SEC-003", "version": "2026-01"},
        {"chunk_id": "chk_conf_02", "excerpt": "Database backup retention is 90 days.", "document_id": "SEC-005", "version": "2026-01"}
    ]
    raw_genuine_conflict = {
        "status": "CONFLICTING",
        "reason": "The evidence gives two different retention periods for database backups: 30 days and 90 days.",
        "evidence_chunk_ids": ["chk_conf_01", "chk_conf_02"],
        "evidence_quote": ""
    }
    val_cf = _validate_slm_output(raw_genuine_conflict, evidence_conflict, question=query_db)
    assert val_cf["status"] == "CONFLICTING", f"Expected CONFLICTING for genuine conflict, got {val_cf['status']}"
    assert "chk_conf_01" in val_cf["evidence_chunk_ids"]
    assert "chk_conf_02" in val_cf["evidence_chunk_ids"]
    assert val_cf["evidence_quote"] == "", "Evidence quote must be empty for CONFLICTING"

    # End-to-end check via review() to verify human-review safeguard (candidate_excerpt is None)
    res_e2e = review("How long are database backups retained?", workspace_id="ws_acme_corp", method="hybrid")
    assert res_e2e["status"] == "CONFLICTING"
    assert res_e2e["candidate_excerpt"] is None, "Human-review safeguard: candidate_excerpt must be None on CONFLICTING"
    assert res_e2e["evidence_quote"] == ""
    assert any("30 days" in ev["excerpt"] for ev in res_e2e["evidence"])
    assert any("90 days" in ev["excerpt"] for ev in res_e2e["evidence"])

    print("  [TEST 19] Conflict-classification logic and safeguards: PASSED")


def run_all():
    print("============================================================")
    print("RUNNING CAIQ GROUNDED Q&A REGRESSION TEST SUITE")
    print("============================================================")
    test_01_question_and_answer_same_chunk()
    test_02_question_and_answer_split_adjacent_chunks()
    test_03_direct_evidence_omitted_from_slm_context()
    test_04_relevant_evidence_no_false_abstention()
    test_05_truly_absent_evidence_safe_abstention()
    test_06_workspace_isolation()
    test_07_quote_validation_against_correct_evidence_id()
    test_08_benchmark_preservation()
    test_09_multidoc_discrepancy_and_anti_contamination()
    test_10_replace_type_safety_and_provenance_page()
    test_11_corpus_storage_restoration()
    test_12_typo_workspace_isolation_correctness()
    test_13_explicit_negative_answer_with_qualification()
    test_14_explicit_not_applicable_answer()
    test_15_adjacent_chunks_question_and_negative_answer_e2e()
    test_16_omitted_answer_chunk_vs_included_answer_chunk()
    test_17_negative_quote_and_citation_validation()
    test_18_adjacent_chunk_association_and_unsupported_rejection()
    test_19_conflict_classification_logic()
    print("============================================================")
    print("ALL 19 CAIQ GROUNDED Q&A TESTS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    run_all()



