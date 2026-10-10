
from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

"""
test_day16.py — Unit test suite for Day 16 semantic reasoning layer
"""

import sys
import os

sys.path.append("d:/Final_lap/EvidenceDesk/evidencedesk")

from day16_reasoning import analyze_evidence_day16, _REASONING_PROMPT_DAY16


def test_semantic_reasoning_triage():
    question = "Who takes the first look at a reported security incident?"
    retrieved_evidence = [
        {
            "chunk_id": "chk_sec004_001",
            "workspace_id": "ws_acme_corp",
            "document_id": "SEC-004",
            "title": "Incident response policy",
            "version": "2026-01",
            "excerpt": "Security incidents are triaged by the on-call engineer.",
        }
    ]

    result = analyze_evidence_day16(question, retrieved_evidence)

    assert result["status"] == "ANSWERABLE", f"Expected ANSWERABLE, got {result['status']}"
    assert "evidence_chunk_ids" in result
    assert "chk_sec004_001" in result["evidence_chunk_ids"]
    assert result["evidence_quote"] == "Security incidents are triaged by the on-call engineer."
    print("  [test_semantic_reasoning_triage] PASSED")


def test_unanswerable_grounding():
    question = "Is the company SOC 2 certified?"
    retrieved_evidence = [
        {
            "chunk_id": "chk_sec001_001",
            "workspace_id": "ws_acme_corp",
            "document_id": "SEC-001",
            "title": "Access control policy",
            "version": "2026-01",
            "excerpt": "Employees must use multi-factor authentication (MFA) to access production systems.",
        }
    ]

    result = analyze_evidence_day16(question, retrieved_evidence)

    assert result["status"] == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {result['status']}"
    assert result["evidence_chunk_ids"] == []
    assert result["evidence_quote"] == ""
    print("  [test_unanswerable_grounding] PASSED")


def main():
    print("Running Day 16 unit tests...")
    test_semantic_reasoning_triage()
    test_unanswerable_grounding()
    print("Day 16 unit tests passed successfully!")


if __name__ == "__main__":
    main()
