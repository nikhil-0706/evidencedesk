import json
import sys
import os
sys.path.append(os.getcwd())

from evidencedesk import review

def score_retrieval(result, case):
    expected_docs = case.get("expected_document_ids", [])
    expected_evidence = case.get("expected_evidence", [])
    expected_behavior = case.get("expected_behavior", "")
    
    status = result.get("status", "")
    
    # Check reasoning
    reasoning_match = False
    if expected_behavior == "show_evidence" and status == "ANSWERABLE":
        reasoning_match = True
    elif expected_behavior == "abstain" and status == "INSUFFICIENT_EVIDENCE":
        reasoning_match = True
    elif expected_behavior == "ask_clarification" and status == "AMBIGUOUS":
        reasoning_match = True
    elif expected_behavior == "flag_conflict" and status == "CONFLICTING":
        reasoning_match = True
        
    # Check retrieval evidence
    top_doc = ""
    top_excerpt = ""
    if result.get("evidence"):
        top_doc = result["evidence"][0]["document_id"]
        top_excerpt = result["evidence"][0]["excerpt"]
    
    retrieval_match = False
    if expected_behavior == "show_evidence" or expected_behavior == "flag_conflict":
        if top_doc in expected_docs and top_excerpt in expected_evidence:
            retrieval_match = True
    else:
        # if abstain or clarify, retrieval doesn't need to perfectly match expected facts since there aren't any
        retrieval_match = True 

    if reasoning_match and retrieval_match:
        return "PASS", f"Matched ({status})"
    elif reasoning_match or retrieval_match:
        return "PARTIAL", f"Reasoning: {status}, Retrieval doc: {top_doc}"
    else:
        return "FAIL", f"Reasoning: {status}, Retrieval doc: {top_doc}"

def main():
    with open("data/eval_questions.json", "r") as f:
        cases = json.load(f)
        
    held_out_cases = [c for c in cases if c["split"] == "held_out"]
    
    results = []
    
    for case in held_out_cases:
        res = review(case["question"], method="hybrid")
        score, note = score_retrieval(res, case)
            
        results.append({
            "id": case["id"],
            "category": case["category"],
            "expected_behavior": case["expected_behavior"],
            "status": res.get("status"),
            "score": score,
            "note": note
        })

    print("## Summary")
    pass_cnt = sum(1 for r in results if r["score"] == "PASS")
    partial_cnt = sum(1 for r in results if r["score"] == "PARTIAL")
    fail_cnt = sum(1 for r in results if r["score"] == "FAIL")
    
    print(f"Held-Out Evaluation: PASS: {pass_cnt}, PARTIAL: {partial_cnt}, FAIL: {fail_cnt} (Pass Rate: {pass_cnt/len(held_out_cases)*100:.1f}%)")
    
    categories = set(c["category"] for c in held_out_cases)
    print("\n## By Category (PASS count)")
    for cat in categories:
        cat_cases = [r for r in results if r["category"] == cat]
        cat_pass = sum(1 for r in cat_cases if r["score"] == "PASS")
        print(f"- {cat.capitalize()}: {cat_pass}/{len(cat_cases)}")
        
    print("\n## Detailed Table")
    print("| ID | Category | Expected behavior | Actual Status | Score | Notes |")
    print("|---|---|---|---|---|---|")
    
    for r in results:
        print(f"| {r['id']} | {r['category']} | {r['expected_behavior']} | {r['status']} | {r['score']} | {r['note']} |")

if __name__ == "__main__":
    main()
