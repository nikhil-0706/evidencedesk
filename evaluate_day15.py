"""
evaluate_day15.py — Day 15 held-out evaluation
================================================

Runs the hybrid pipeline (BM25 + Embedding → RRF k=60 → Cross-Encoder →
Structured SLM) on Q16–Q25 and prints Layer-1 (retrieval) and Layer-2
(SLM reasoning) accuracy, comparing against the Day-14 baseline.

IMPORTANT:
- No tuning is performed using this data.
- This is a pure measurement of system generalization after Day-15
  prompt / structured-output improvements.
- Q16–Q25 labels are NOT modified.
"""

import json
import os
import sys

sys.path.append(os.getcwd())

from evidencedesk import (
    review,
    retrieve_by_bm25,
    retrieve_by_embedding,
    rrf_fuse,
    rerank_evidence,
    analyze_evidence,
)

# ---------------------------------------------------------------------------
# Ground-truth labels for Q16–Q25 (held-out, unmodified)
# ---------------------------------------------------------------------------
HELD_OUT_CASES = [
    {
        "id": "Q16",
        "category": "direct",
        "question": "Who triages security incidents?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Security incidents are triaged by the on-call engineer."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q17",
        "category": "direct",
        "question": "Who receives escalated security incidents?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Confirmed incidents are escalated to the security lead."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q18",
        "category": "direct",
        "question": "What is the contractual customer notification deadline for security incidents?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": [
            "This policy does not specify a contractual customer notification deadline."
        ],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q19",
        "category": "paraphrased",
        "question": "Which role performs initial assessment of security events?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Security incidents are triaged by the on-call engineer."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q20",
        "category": "paraphrased",
        "question": "To whom are verified security incidents escalated?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Confirmed incidents are escalated to the security lead."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q21",
        "category": "unanswerable",
        "question": "What is your SLA for resolving high-severity security incidents?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "Q22",
        "category": "unanswerable",
        "question": "How frequently are incident response tabletop exercises conducted?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "Q23",
        "category": "ambiguous",
        "question": "How quickly do you respond to incidents?",
        "expected_status": "AMBIGUOUS",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "ask_clarification",
    },
    {
        "id": "Q24",
        "category": "unanswerable",
        "question": "Are security incidents escalated immediately or within 24 hours?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "Q25",
        "category": "conflicting",
        "question": "What single database backup retention period should I put in the questionnaire?",
        "expected_status": "CONFLICTING",
        "expected_document_ids": ["SEC-003", "SEC-005"],
        "expected_evidence": [
            "Backup retention is 30 days.",
            "Database backup retention is 90 days.",
        ],
        "expected_behavior": "flag_conflict",
    },
]


# ---------------------------------------------------------------------------
# Layer-1 retrieval scoring (unchanged from Day-14)
# ---------------------------------------------------------------------------

def score_retrieval(evidence_list: list, case: dict) -> tuple[str, str]:
    expected_docs = case.get("expected_document_ids", [])
    expected_evidence = case.get("expected_evidence", [])
    expected_behavior = case.get("expected_behavior", "")

    retrieved_docs = [e["document_id"] for e in evidence_list]

    if expected_behavior == "abstain":
        return "PASS", "No expected docs required for unanswerable"

    if expected_behavior == "flag_conflict":
        if all(doc in retrieved_docs for doc in expected_docs):
            return "PASS", f"Found all conflicting docs: {expected_docs}"
        elif any(doc in retrieved_docs for doc in expected_docs):
            return "PARTIAL", f"Found some conflicting docs: {retrieved_docs}"
        return "FAIL", f"Found no conflicting docs: {retrieved_docs}"

    if expected_behavior == "ask_clarification":
        if any(doc in retrieved_docs for doc in expected_docs):
            return "PASS", f"Surfaced ambiguity docs: {expected_docs}"
        if not expected_docs:
            return "PASS", "No expected docs for clarification case"
        return "FAIL", f"Missed ambiguity docs: {retrieved_docs}"

    if not expected_docs:
        return "FAIL", "No expected docs specified"

    found_doc = False
    found_exact = False
    for e in evidence_list:
        if e["document_id"] in expected_docs:
            found_doc = True
            if e["excerpt"] in expected_evidence:
                found_exact = True
                break

    if found_exact:
        return "PASS", "Found exact expected passage"
    elif found_doc:
        return "PARTIAL", "Found correct document but wrong passage"
    return "FAIL", f"Failed to retrieve expected doc; got {retrieved_docs}"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    results = []
    WORKSPACE = "ws_acme_corp"

    print("Starting Day 15 Held-Out Evaluation (Q16–Q25)…\n")

    for case in HELD_OUT_CASES:
        qid = case["id"]
        q = case["question"]
        print(f"  [{qid}] {q}")

        # Retrieve
        bm25 = retrieve_by_bm25(q, workspace_id=WORKSPACE, top_k=20)
        emb = retrieve_by_embedding(q, workspace_id=WORKSPACE, top_k=20)
        rrf = rrf_fuse(bm25, emb, k=60, top_k=10)
        ce_top5 = rerank_evidence(q, rrf, top_k=5)

        # Classify
        slm_result = analyze_evidence(q, ce_top5)

        # Score retrieval
        retrieval_score, retrieval_note = score_retrieval(ce_top5, case)

        # Score reasoning
        slm_status = slm_result.get("status", "")
        slm_match = slm_status == case["expected_status"]

        # Final
        if retrieval_score == "PASS" and slm_match:
            final = "PASS"
        elif retrieval_score == "PASS" or slm_match:
            final = "PARTIAL"
        else:
            final = "FAIL"

        results.append(
            {
                "id": qid,
                "category": case["category"],
                "question": q,
                "expected_status": case["expected_status"],
                "expected_docs": case["expected_document_ids"],
                "retrieval_score": retrieval_score,
                "retrieval_note": retrieval_note,
                "slm_status": slm_status,
                "slm_reason": slm_result.get("reason", ""),
                "evidence_quote": slm_result.get("evidence_quote", ""),
                "slm_match": slm_match,
                "final": final,
            }
        )

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DAY 15 HELD-OUT RESULTS TABLE (Q16–Q25)")
    print("=" * 70)
    print(f"{'QID':<5} {'Category':<14} {'Expected':<24} {'Retrieval':<10} {'SLM':<24} {'Final'}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['id']:<5} {r['category'].capitalize():<14} {r['expected_status']:<24} "
            f"{r['retrieval_score']:<10} {r['slm_status']:<24} {r['final']}"
        )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    total = len(results)
    ret_pass = sum(1 for r in results if r["retrieval_score"] == "PASS")
    slm_pass = sum(1 for r in results if r["slm_match"])
    e2e_pass = sum(1 for r in results if r["final"] == "PASS")

    print("\n" + "=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)
    print(f"  Retrieval Pass Rate (Layer 1): {ret_pass}/{total} ({ret_pass/total*100:.1f}%)")
    print(f"  SLM Accuracy       (Layer 2): {slm_pass}/{total} ({slm_pass/total*100:.1f}%)")
    print(f"  End-to-End Pass Rate:          {e2e_pass}/{total} ({e2e_pass/total*100:.1f}%)")

    # Day 14 baseline comparison
    print("\n" + "=" * 50)
    print("COMPARISON: Day 14 -> Day 15")
    print("=" * 50)
    print(f"  Day 14 Retrieval:  100.0%  | Day 15 Retrieval:  {ret_pass/total*100:.1f}%")
    print(f"  Day 14 SLM Acc:     80.0%  | Day 15 SLM Acc:    {slm_pass/total*100:.1f}%")
    print(f"  Day 14 E2E:         80.0%  | Day 15 E2E:        {e2e_pass/total*100:.1f}%")

    # Category breakdown
    print("\n" + "=" * 50)
    print("CATEGORY BREAKDOWN")
    print("=" * 50)
    categories = sorted(set(r["category"] for r in results))
    print(f"{'Category':<15} {'Count':<7} {'Retrieval':<12} {'SLM Acc':<12} {'E2E Pass'}")
    print("-" * 55)
    for cat in categories:
        items = [r for r in results if r["category"] == cat]
        n = len(items)
        r_p = sum(1 for r in items if r["retrieval_score"] == "PASS")
        s_p = sum(1 for r in items if r["slm_match"])
        e_p = sum(1 for r in items if r["final"] == "PASS")
        print(f"  {cat.capitalize():<13} {n:<7} {r_p}/{n:<11} {s_p}/{n:<11} {e_p}/{n}")

    # Failure analysis
    failures = [r for r in results if r["final"] != "PASS"]
    if failures:
        print("\n" + "=" * 50)
        print("FAILURE ANALYSIS")
        print("=" * 50)
        for r in failures:
            print(f"\n  {r['id']} ({r['final']})")
            print(f"    Question:       {r['question']}")
            print(f"    Expected:       {r['expected_status']}")
            print(f"    SLM returned:   {r['slm_status']}")
            print(f"    SLM reason:     {r['slm_reason']}")
            print(f"    Retrieval:      {r['retrieval_score']} — {r['retrieval_note']}")
    else:
        print("\n  No failures. All Q16–Q25 cases passed end-to-end.")

    # Save raw results
    os.makedirs("docs/evaluation", exist_ok=True)
    with open("docs/evaluation/day-15-raw-results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: docs/evaluation/day-15-raw-results.json")


if __name__ == "__main__":
    main()
