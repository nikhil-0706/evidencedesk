"""
test_day17.py — Day 17 Final Stabilization & Verification Smoke Tests
======================================================================

Verifies the 7 mandatory Day 17 smoke test cases:
1. "Who takes the first look at a reported security incident?" -> ANSWERABLE
2. "What is your retention policy?" -> AMBIGUOUS
3. "How long are database backups retained?" -> CONFLICTING
4. "Are you SOC 2 certified?" -> INSUFFICIENT_EVIDENCE
5. "What is your SLA for resolving high-severity security incidents?" -> INSUFFICIENT_EVIDENCE
6. "What safeguards customer information while it moves between systems?" -> ANSWERABLE
7. "Who gives permission for an employee to receive administrative privileges?" -> ANSWERABLE
"""

import sys
import os
from tests.fixtures.synthetic_data import PASSAGES, DOCUMENTS, CASES, inject_legacy_fixtures


sys.path.append("d:/Final_lap/EvidenceDesk/evidencedesk")

from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)
from evidencedesk import (

    retrieve_by_bm25,
    retrieve_by_embedding,
    rrf_fuse,
    rerank_evidence,
)
from day16_reasoning import analyze_evidence_day16

WORKSPACE = "ws_acme_corp"

SMOKE_CASES = [
    {
        "id": "SMOKE_1",
        "question": "Who takes the first look at a reported security incident?",
        "expected_status": "ANSWERABLE",
    },
    {
        "id": "SMOKE_2",
        "question": "What is your retention policy?",
        "expected_status": "AMBIGUOUS",
    },
    {
        "id": "SMOKE_3",
        "question": "How long are database backups retained?",
        "expected_status": "CONFLICTING",
    },
    {
        "id": "SMOKE_4",
        "question": "Are you SOC 2 certified?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
    },
    {
        "id": "SMOKE_5",
        "question": "What is your SLA for resolving high-severity security incidents?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
    },
    {
        "id": "SMOKE_6",
        "question": "What safeguards customer information while it moves between systems?",
        "expected_status": "ANSWERABLE",
    },
    {
        "id": "SMOKE_7",
        "question": "Who gives permission for an employee to receive administrative privileges?",
        "expected_status": "ANSWERABLE",
    },
]


def run_smoke_tests():
    print("\n" + "=" * 60)
    print("RUNNING DAY 17 MANDATORY SMOKE TESTS")
    print("=" * 60)

    all_passed = True
    for case in SMOKE_CASES:
        q = case["question"]
        exp = case["expected_status"]

        bm25 = retrieve_by_bm25(q, WORKSPACE, top_k=20)
        emb = retrieve_by_embedding(q, WORKSPACE, top_k=20)
        rrf = rrf_fuse(bm25, emb, k=60, top_k=10)
        top5 = rerank_evidence(q, rrf, top_k=5)

        res = analyze_evidence_day16(q, top5)
        status = res.get("status", "")
        match = status == exp

        if not match:
            all_passed = False

        print(f"\n  [{case['id']}] Question: '{q}'")
        print(f"    Expected: {exp:<22} | Got: {status:<22} | Match: {'PASS' if match else 'FAIL'}")
        print(f"    Reason:   {res.get('reason', '')}")
        if status == "ANSWERABLE":
            print(f"    Quote:    '{res.get('evidence_quote', '')}'")

        # Grounding & schema assertions
        assert "status" in res, "Missing 'status' key"
        assert "reason" in res, "Missing 'reason' key"
        assert "evidence_chunk_ids" in res, "Missing 'evidence_chunk_ids' key"
        assert "evidence_quote" in res, "Missing 'evidence_quote' key"

        if status == "ANSWERABLE":
            assert res["evidence_quote"].strip(), "ANSWERABLE must have non-empty quote"
            excerpts = [p["excerpt"] for p in top5]
            assert any(
                res["evidence_quote"] in exc for exc in excerpts
            ), f"Quote '{res['evidence_quote']}' not found in evidence!"

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL DAY 17 SMOKE TESTS PASSED PERFECTLY!")
    else:
        print("SOME SMOKE TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_tests()
