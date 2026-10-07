"""
evaluate_day16.py — Day 16 Evaluation Script
==============================================

Evaluates the Day 16 improved semantic reasoning configuration vs Day 15 baseline
using a strict two-phase protocol:
  Phase 1: Development Regression (Q01–Q15)
  Phase 2: Held-Out Evaluation (Q16–Q25)

Retrieval architecture is FROZEN.
Day 15 baseline is FROZEN.
"""

import json
import os
import sys

sys.path.append("d:/Final_lap/EvidenceDesk/evidencedesk")

from evidencedesk import (
    retrieve_by_bm25,
    retrieve_by_embedding,
    rrf_fuse,
    rerank_evidence,
    analyze_evidence as analyze_evidence_day15,
)
from day16_reasoning import analyze_evidence_day16

WORKSPACE = "ws_acme_corp"

# Development cases Q01-Q15
with open("data/eval_questions.json", "r") as f:
    all_questions = json.load(f)

DEV_CASES = [q for q in all_questions if q["split"] == "development"]

# Held-Out cases Q16-Q25 (Day 14 / Day 16 spec)
HELD_OUT_CASES = [
    {
        "id": "Q16",
        "category": "direct",
        "question": "Where are encryption keys managed?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-002"],
        "expected_evidence": [
            "Encryption keys are managed in a dedicated key management service."
        ],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q17",
        "category": "direct",
        "question": "How often are restore procedures tested?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-003"],
        "expected_evidence": ["Restore procedures are tested quarterly."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q18",
        "category": "direct",
        "question": "Who triages security incidents?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Security incidents are triaged by the on-call engineer."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q19",
        "category": "direct",
        "question": "To whom are confirmed incidents escalated?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Confirmed incidents are escalated to the security lead."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q20",
        "category": "paraphrased",
        "question": "Who gives permission for an employee to receive admin privileges?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-001"],
        "expected_evidence": ["Administrative access requires manager approval."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q21",
        "category": "paraphrased",
        "question": "Who takes the first look at a reported security incident?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Security incidents are triaged by the on-call engineer."],
        "expected_behavior": "show_evidence",
    },
    {
        "id": "Q22",
        "category": "unanswerable",
        "question": "What is the guaranteed uptime percentage?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "Q23",
        "category": "unanswerable",
        "question": "Are customer-managed encryption keys supported?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "Q24",
        "category": "ambiguous",
        "question": "How often is access reviewed for customers and employees?",
        "expected_status": "AMBIGUOUS",
        "expected_document_ids": ["SEC-001"],
        "expected_evidence": ["Access permissions are reviewed quarterly."],
        "expected_behavior": "ask_clarification",
    },
    {
        "id": "Q25",
        "category": "conflicting",
        "question": "What single backup retention period should I put in questionnaire?",
        "expected_status": "CONFLICTING",
        "expected_document_ids": ["SEC-003", "SEC-005"],
        "expected_evidence": [
            "Backup retention is 30 days.",
            "Database backup retention is 90 days.",
        ],
        "expected_behavior": "flag_conflict",
    },
]


def score_retrieval(evidence_list: list, case: dict) -> tuple[str, str]:
    expected_docs = case.get("expected_document_ids", [])
    expected_evidence = case.get("expected_evidence", [])
    expected_behavior = case.get("expected_behavior", "")

    retrieved_docs = [e["document_id"] for e in evidence_list]

    if expected_behavior == "abstain":
        return "PASS", "No expected docs required for abstain"

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


def get_expected_status(case: dict) -> str:
    if "expected_status" in case:
        return case["expected_status"]
    exp_b = case.get("expected_behavior", "")
    if exp_b == "show_evidence":
        return "ANSWERABLE"
    elif exp_b == "abstain":
        return "INSUFFICIENT_EVIDENCE"
    elif exp_b == "ask_clarification":
        return "AMBIGUOUS"
    elif exp_b == "flag_conflict":
        return "CONFLICTING"
    return "UNKNOWN"


def evaluate_suite(cases: list, suite_name: str):
    print(f"\n" + "=" * 70)
    print(f"RUNNING EVALUATION: {suite_name}")
    print("=" * 70)

    results = []
    malformed_count_day15 = 0
    malformed_count_day16 = 0
    grounding_failures_day15 = 0
    grounding_failures_day16 = 0
    regressions = []

    for case in cases:
        qid = case["id"]
        cat = case["category"]
        q = case["question"]
        expected = get_expected_status(case)

        # Retrieval (Frozen pipeline)
        bm25 = retrieve_by_bm25(q, WORKSPACE, top_k=20)
        emb = retrieve_by_embedding(q, WORKSPACE, top_k=20)
        rrf = rrf_fuse(bm25, emb, k=60, top_k=10)
        ce_top5 = rerank_evidence(q, rrf, top_k=5)
        ret_score, ret_note = score_retrieval(ce_top5, case)

        # Day 15 SLM
        res_d15 = analyze_evidence_day15(q, ce_top5)
        d15_status = res_d15.get("status", "")
        d15_match = d15_status == expected
        d15_e2e = (ret_score == "PASS") and d15_match
        if res_d15.get("_validation_error"):
            malformed_count_day15 += 1
            if "evidence_quote" in res_d15.get("reason", ""):
                grounding_failures_day15 += 1

        # Day 16 SLM
        res_d16 = analyze_evidence_day16(q, ce_top5)
        d16_status = res_d16.get("status", "")
        d16_match = d16_status == expected
        d16_e2e = (ret_score == "PASS") and d16_match
        if res_d16.get("_validation_error"):
            malformed_count_day16 += 1
            if "evidence_quote" in res_d16.get("reason", ""):
                grounding_failures_day16 += 1

        # Regression check: Day 15 was correct, but Day 16 failed
        is_regression = d15_match and not d16_match
        if is_regression:
            regressions.append(
                {
                    "id": qid,
                    "question": q,
                    "expected": expected,
                    "day15_got": d15_status,
                    "day16_got": d16_status,
                    "day16_reason": res_d16.get("reason", ""),
                }
            )

        results.append(
            {
                "id": qid,
                "category": cat,
                "question": q,
                "expected": expected,
                "retrieval_score": ret_score,
                "day15_status": d15_status,
                "day15_match": d15_match,
                "day15_e2e": d15_e2e,
                "day15_reason": res_d15.get("reason", ""),
                "day15_quote": res_d15.get("evidence_quote", ""),
                "day16_status": d16_status,
                "day16_match": d16_match,
                "day16_e2e": d16_e2e,
                "day16_reason": res_d16.get("reason", ""),
                "day16_quote": res_d16.get("evidence_quote", ""),
                "is_regression": is_regression,
            }
        )

        print(
            f"[{qid}] Expected={expected:<22} | Day15={d15_status:<22} ({'PASS' if d15_match else 'FAIL'}) | Day16={d16_status:<22} ({'PASS' if d16_match else 'FAIL'})"
        )

    total = len(results)
    ret_pass = sum(1 for r in results if r["retrieval_score"] == "PASS")
    d15_slm_pass = sum(1 for r in results if r["day15_match"])
    d15_e2e_pass = sum(1 for r in results if r["day15_e2e"])
    d16_slm_pass = sum(1 for r in results if r["day16_match"])
    d16_e2e_pass = sum(1 for r in results if r["day16_e2e"])

    print("\n" + "=" * 70)
    print(f"SUMMARY METRICS ({suite_name})")
    print("=" * 70)
    print(
        f"{'Metric':<25} {'Day 15 Baseline':<20} {'Day 16 Improved':<20} {'Delta':<10}"
    )
    print("-" * 75)
    print(
        f"{'Retrieval Pass Rate':<25} {ret_pass}/{total} ({ret_pass/total*100:.1f}%)     {ret_pass}/{total} ({ret_pass/total*100:.1f}%)     0.0%"
    )
    print(
        f"{'SLM Classification Acc':<25} {d15_slm_pass}/{total} ({d15_slm_pass/total*100:.1f}%)     {d16_slm_pass}/{total} ({d16_slm_pass/total*100:.1f}%)     {(d16_slm_pass-d15_slm_pass)/total*100:+.1f}%"
    )
    print(
        f"{'End-to-End Pass Rate':<25} {d15_e2e_pass}/{total} ({d15_e2e_pass/total*100:.1f}%)     {d16_e2e_pass}/{total} ({d16_e2e_pass/total*100:.1f}%)     {(d16_e2e_pass-d15_e2e_pass)/total*100:+.1f}%"
    )
    print(
        f"{'Malformed SLM Outputs':<25} {malformed_count_day15:<20} {malformed_count_day16:<20}"
    )
    print(
        f"{'Grounding Failures':<25} {grounding_failures_day15:<20} {grounding_failures_day16:<20}"
    )
    print(f"{'Regression Count':<25} {'-':<20} {len(regressions):<20}")

    if regressions:
        print("\n" + "!" * 50)
        print("REGRESSIONS DETECTED:")
        print("!" * 50)
        for reg in regressions:
            print(f"  {reg['id']}: {reg['question']}")
            print(f"    Expected: {reg['expected']}")
            print(f"    Day 15:   {reg['day15_got']} (Correct)")
            print(f"    Day 16:   {reg['day16_got']} (Failed — {reg['day16_reason']})")
    else:
        print(f"\nZero regressions detected in {suite_name}.")

    return {
        "suite": suite_name,
        "total": total,
        "retrieval_pass": ret_pass,
        "d15_slm_pass": d15_slm_pass,
        "d15_e2e_pass": d15_e2e_pass,
        "d16_slm_pass": d16_slm_pass,
        "d16_e2e_pass": d16_e2e_pass,
        "malformed_day15": malformed_count_day15,
        "malformed_day16": malformed_count_day16,
        "grounding_failures_day15": grounding_failures_day15,
        "grounding_failures_day16": grounding_failures_day16,
        "regressions": regressions,
        "results": results,
    }


def main():
    print("======================================================================")
    print("DAY 16 EVALUATION PROTOCOL")
    print("======================================================================")

    # Phase 1: Development Regression (Q01-Q15)
    phase1_summary = evaluate_suite(DEV_CASES, "PHASE 1 - DEVELOPMENT (Q01-Q15)")

    # Phase 2: Held-Out Evaluation (Q16-Q25)
    phase2_summary = evaluate_suite(HELD_OUT_CASES, "PHASE 2 - HELD-OUT (Q16-Q25)")

    # Save complete evaluation output
    os.makedirs("docs/evaluation", exist_ok=True)
    report_data = {
        "phase1_development": phase1_summary,
        "phase2_held_out": phase2_summary,
    }
    with open("docs/evaluation/day-16-raw-results.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("\nSaved raw evaluation report to docs/evaluation/day-16-raw-results.json")


if __name__ == "__main__":
    main()
