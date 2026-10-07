"""
Day 15 — Structured Reasoning Tests
====================================

Covers:

A. Seven smoke-test questions (correct SLM classification expected).
B. Eight structured-output validation scenarios (schema / safety tests).

All tests run against the *live* pipeline (SLM call included) except the
validation tests, which inject synthetic raw dicts directly into
_validate_slm_output() to avoid flakiness from the SLM.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from evidencedesk import (
    analyze_evidence,
    _validate_slm_output,
    EvidenceDecision,
    review,
    retrieve_by_bm25,
    retrieve_by_embedding,
    rrf_fuse,
    rerank_evidence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKSPACE = "ws_acme_corp"


def run_hybrid(question: str) -> dict:
    """Full pipeline: BM25+Emb → RRF → Cross-Encoder → SLM."""
    bm25 = retrieve_by_bm25(question, workspace_id=WORKSPACE, top_k=20)
    emb = retrieve_by_embedding(question, workspace_id=WORKSPACE, top_k=20)
    rrf = rrf_fuse(bm25, emb, k=60, top_k=10)
    evidence = rerank_evidence(question, rrf, top_k=5)
    return analyze_evidence(question, evidence)


def assert_status(result: dict, expected: str, label: str):
    actual = result.get("status")
    if actual != expected:
        raise AssertionError(
            f"[{label}] Expected status={expected!r} but got {actual!r}\n"
            f"  reason: {result.get('reason')}\n"
            f"  evidence_quote: {result.get('evidence_quote')}"
        )
    print(f"  PASS  [{label}] → {actual}")


# ---------------------------------------------------------------------------
# Section A — Smoke tests (7 questions)
# ---------------------------------------------------------------------------

def _smoke_tests():
    print("\n=== Section A: Smoke Tests (7 questions) ===\n")

    cases = [
        (
            "TEST-1 [Q16 historical failure] — Where are encryption keys managed?",
            "Where are encryption keys managed?",
            "ANSWERABLE",
        ),
        (
            "TEST-2 [Q21 historical failure] — Who takes the first look at a reported security incident?",
            "Who takes the first look at a reported security incident?",
            "ANSWERABLE",
        ),
        (
            "TEST-3 — Who approves administrative access?",
            "Who approves administrative access?",
            "ANSWERABLE",
        ),
        (
            "TEST-4 — How is data in transit protected?",
            "How is data in transit protected?",
            "ANSWERABLE",
        ),
        (
            "TEST-5 — What is your retention policy?",
            "What is your retention policy?",
            "AMBIGUOUS",
        ),
        (
            "TEST-6 — How long are database backups retained?",
            "How long are database backups retained?",
            "CONFLICTING",
        ),
        (
            "TEST-7 — Is the company SOC 2 certified?",
            "Is the company SOC 2 certified?",
            "INSUFFICIENT_EVIDENCE",
        ),
    ]

    passed = 0
    failed = 0

    for label, question, expected in cases:
        print(f"Running: {label}")
        try:
            result = run_hybrid(question)
            assert_status(result, expected, label)

            # Extra assertion: ANSWERABLE must have evidence_quote
            if expected == "ANSWERABLE":
                quote = result.get("evidence_quote", "")
                if not quote.strip():
                    raise AssertionError(
                        f"[{label}] ANSWERABLE but evidence_quote is empty!"
                    )
                print(f"         evidence_quote: {quote!r}")

            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {exc}")
            failed += 1

    return passed, failed


# ---------------------------------------------------------------------------
# Section B — Structured-output validation tests (8 scenarios)
# ---------------------------------------------------------------------------

# A minimal synthetic evidence list for validation tests
_FAKE_EVIDENCE = [
    {
        "chunk_id": "chk_fake_001",
        "document_id": "FAKE-001",
        "workspace_id": "ws_acme_corp",
        "excerpt": "The sky is blue.",
        "title": "Fake Doc",
        "version": "2026-01",
    }
]


def _validate_test(label: str, raw: dict, expect_error: bool, evidence=None):
    ev = evidence if evidence is not None else _FAKE_EVIDENCE
    result = _validate_slm_output(raw, ev)
    is_error = result.get("_validation_error") or result["status"] in (
        "VALIDATION_ERROR",
        "ERROR",
    )
    if expect_error and is_error:
        print(f"  PASS  [{label}] → correctly rejected: {result['status']} | {result['reason'][:80]}")
        return True
    elif not expect_error and not is_error:
        print(f"  PASS  [{label}] → correctly accepted: {result['status']}")
        return True
    else:
        direction = "rejected" if is_error else "accepted"
        expected_direction = "rejected" if expect_error else "accepted"
        print(
            f"  FAIL  [{label}] → was {direction} but expected to be {expected_direction}\n"
            f"          status={result['status']} | reason={result.get('reason','')[:80]}"
        )
        return False


def _validation_tests():
    print("\n=== Section B: Structured-Output Validation Tests (8 scenarios) ===\n")

    passed = 0
    failed = 0

    def run(label, raw, expect_error, evidence=None):
        nonlocal passed, failed
        ok = _validate_test(label, raw, expect_error, evidence)
        if ok:
            passed += 1
        else:
            failed += 1

    # A — Valid JSON with correct schema → accepted
    run(
        "A: Valid ANSWERABLE JSON",
        {
            "status": "ANSWERABLE",
            "reason": "The evidence says so.",
            "evidence_chunk_ids": ["chk_fake_001"],
            "evidence_quote": "The sky is blue.",
        },
        expect_error=False,
        evidence=_FAKE_EVIDENCE,
    )

    # B — Correct schema (INSUFFICIENT_EVIDENCE, no quote needed) → accepted
    run(
        "B: Valid INSUFFICIENT_EVIDENCE JSON",
        {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "Nothing relevant found.",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
        },
        expect_error=False,
    )

    # C — Invalid status value → rejected
    run(
        "C: Invalid status value",
        {
            "status": "MAYBE",
            "reason": "Not sure.",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
        },
        expect_error=True,
    )

    # D — Missing required field (reason) → rejected
    run(
        "D: Missing 'reason' field",
        {
            "status": "ANSWERABLE",
            "evidence_chunk_ids": ["chk_fake_001"],
            "evidence_quote": "The sky is blue.",
            # 'reason' intentionally omitted
        },
        expect_error=True,
    )

    # E — Simulate malformed JSON path (non-dict input to validator) → rejected
    #     (We test with a raw string — Python catches TypeError inside _validate_slm_output)
    run(
        "E: Non-dict input (malformed JSON object)",
        "this is not a dict",  # type: ignore[arg-type]
        expect_error=True,
    )

    # F — ANSWERABLE without evidence_quote → rejected
    run(
        "F: ANSWERABLE without evidence_quote",
        {
            "status": "ANSWERABLE",
            "reason": "The evidence says so.",
            "evidence_chunk_ids": ["chk_fake_001"],
            "evidence_quote": "",
        },
        expect_error=True,
        evidence=_FAKE_EVIDENCE,
    )

    # G — Unknown chunk_id only → rejected
    run(
        "G: Unknown chunk_id only",
        {
            "status": "ANSWERABLE",
            "reason": "The evidence says so.",
            "evidence_chunk_ids": ["chk_does_not_exist"],
            "evidence_quote": "The sky is blue.",
        },
        expect_error=True,
        evidence=_FAKE_EVIDENCE,
    )

    # H — evidence_quote not in supplied evidence → rejected
    run(
        "H: evidence_quote not in supplied evidence",
        {
            "status": "ANSWERABLE",
            "reason": "The evidence says so.",
            "evidence_chunk_ids": ["chk_fake_001"],
            "evidence_quote": "This sentence was never in the corpus.",
        },
        expect_error=True,
        evidence=_FAKE_EVIDENCE,
    )

    return passed, failed


# ---------------------------------------------------------------------------
# Section C — Workspace isolation regression
# ---------------------------------------------------------------------------

def _workspace_tests():
    print("\n=== Section C: Workspace Isolation Regression ===\n")
    passed = 0
    failed = 0

    question = "Where is production infrastructure hosted?"

    # Acme — SEC-G001 must NOT appear, status must be INSUFFICIENT_EVIDENCE
    acme = review(question, method="hybrid", workspace_id="ws_acme_corp")
    acme_docs = [e["document_id"] for e in acme.get("evidence", [])]
    if acme["status"] == "INSUFFICIENT_EVIDENCE" and "SEC-G001" not in acme_docs:
        print(f"  PASS  [Acme Corp] status={acme['status']} docs={acme_docs}")
        passed += 1
    else:
        print(
            f"  FAIL  [Acme Corp] status={acme['status']} docs={acme_docs} "
            f"(expected INSUFFICIENT_EVIDENCE without SEC-G001)"
        )
        failed += 1

    # Globex — SEC-G001 must appear, status must be ANSWERABLE
    globex = review(question, method="hybrid", workspace_id="ws_globex_corp")
    globex_docs = [e["document_id"] for e in globex.get("evidence", [])]
    if globex["status"] == "ANSWERABLE" and "SEC-G001" in globex_docs:
        print(f"  PASS  [Globex Corp] status={globex['status']} docs={globex_docs}")
        passed += 1
    else:
        print(
            f"  FAIL  [Globex Corp] status={globex['status']} docs={globex_docs} "
            f"(expected ANSWERABLE with SEC-G001)"
        )
        failed += 1

    return passed, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DAY 15 — STRUCTURED REASONING TESTS")
    print("=" * 60)

    smoke_pass, smoke_fail = _smoke_tests()
    valid_pass, valid_fail = _validation_tests()
    ws_pass, ws_fail = _workspace_tests()

    total_pass = smoke_pass + valid_pass + ws_pass
    total_fail = smoke_fail + valid_fail + ws_fail
    total = total_pass + total_fail

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Smoke tests:      {smoke_pass} PASS / {smoke_fail} FAIL")
    print(f"  Validation tests: {valid_pass} PASS / {valid_fail} FAIL")
    print(f"  Workspace tests:  {ws_pass} PASS / {ws_fail} FAIL")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:            {total_pass}/{total} PASS")

    if total_fail > 0:
        print(f"\n  {total_fail} test(s) FAILED.")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED.")


if __name__ == "__main__":
    main()
