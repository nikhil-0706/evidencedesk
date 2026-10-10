import json
import sys
import os
from tests.fixtures.synthetic_data import PASSAGES, DOCUMENTS, CASES, inject_legacy_fixtures

sys.path.append(os.getcwd())

from evidencedesk import review, retrieve_by_bm25, retrieve_by_embedding, rrf_fuse, rerank_evidence, analyze_evidence
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)


HELD_OUT_CASES = [
    {
        "id": "Q16",
        "category": "direct",
        "question": "Where are encryption keys managed?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-002"],
        "expected_evidence": ["Encryption keys are managed in a dedicated key management service."],
        "expected_behavior": "show_evidence"
    },
    {
        "id": "Q17",
        "category": "direct",
        "question": "How often are restore procedures tested?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-003"],
        "expected_evidence": ["Restore procedures are tested quarterly."],
        "expected_behavior": "show_evidence"
    },
    {
        "id": "Q18",
        "category": "direct",
        "question": "Who triages security incidents?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Security incidents are triaged by the on-call engineer."],
        "expected_behavior": "show_evidence"
    },
    {
        "id": "Q19",
        "category": "direct",
        "question": "To whom are confirmed incidents escalated?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Confirmed incidents are escalated to the security lead."],
        "expected_behavior": "show_evidence"
    },
    {
        "id": "Q20",
        "category": "paraphrased",
        "question": "Who gives permission for an employee to receive admin privileges?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-001"],
        "expected_evidence": ["Administrative access requires manager approval."],
        "expected_behavior": "show_evidence"
    },
    {
        "id": "Q21",
        "category": "paraphrased",
        "question": "Who takes the first look at a reported security incident?",
        "expected_status": "ANSWERABLE",
        "expected_document_ids": ["SEC-004"],
        "expected_evidence": ["Security incidents are triaged by the on-call engineer."],
        "expected_behavior": "show_evidence"
    },
    {
        "id": "Q22",
        "category": "unanswerable",
        "question": "What is the guaranteed uptime percentage?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain"
    },
    {
        "id": "Q23",
        "category": "unanswerable",
        "question": "Are customer-managed encryption keys supported?",
        "expected_status": "INSUFFICIENT_EVIDENCE",
        "expected_document_ids": [],
        "expected_evidence": [],
        "expected_behavior": "abstain"
    },
    {
        "id": "Q24",
        "category": "ambiguous",
        "question": "How often is access reviewed for customers and employees?",
        "expected_status": "AMBIGUOUS",
        "expected_document_ids": ["SEC-001"],
        "expected_evidence": ["Access permissions are reviewed quarterly."],
        "expected_behavior": "ask_clarification"
    },
    {
        "id": "Q25",
        "category": "conflicting",
        "question": "What single backup retention period should I put in questionnaire?",
        "expected_status": "CONFLICTING",
        "expected_document_ids": ["SEC-003", "SEC-005"],
        "expected_evidence": ["Backup retention is 30 days.", "Database backup retention is 90 days."],
        "expected_behavior": "flag_conflict"
    }
]

def get_evidence_for_method(question, method, workspace_id="ws_acme_corp"):
    if method == "bm25":
        return retrieve_by_bm25(question, workspace_id=workspace_id, top_k=3)
    elif method == "embedding":
        return retrieve_by_embedding(question, workspace_id=workspace_id, top_k=3)
    elif method == "hybrid_rrf":
        bm25_results = retrieve_by_bm25(question, workspace_id=workspace_id, top_k=20)
        emb_results = retrieve_by_embedding(question, workspace_id=workspace_id, top_k=20)
        return rrf_fuse(bm25_results, emb_results, k=60, top_k=10)
    elif method == "hybrid_cross_encoder":
        bm25_results = retrieve_by_bm25(question, workspace_id=workspace_id, top_k=20)
        emb_results = retrieve_by_embedding(question, workspace_id=workspace_id, top_k=20)
        rrf_candidates = rrf_fuse(bm25_results, emb_results, k=60, top_k=10)
        return rerank_evidence(question, rrf_candidates, top_k=5)

def score_retrieval(evidence_list, case):
    expected_docs = case.get("expected_document_ids", [])
    expected_evidence = case.get("expected_evidence", [])
    expected_behavior = case.get("expected_behavior", "")
    
    retrieved_docs = [e["document_id"] for e in evidence_list]

    if expected_behavior == "abstain":
        return "PASS", "No expected docs to retrieve for unanswerable question"
        
    if expected_behavior == "flag_conflict":
        if all(doc in retrieved_docs for doc in expected_docs):
            return "PASS", f"Found all conflicting docs: {expected_docs}"
        elif any(doc in retrieved_docs for doc in expected_docs):
            return "PARTIAL", f"Found some conflicting docs: {retrieved_docs}"
        else:
            return "FAIL", f"Found no conflicting docs: {retrieved_docs}"
            
    if expected_behavior == "ask_clarification":
        if any(doc in retrieved_docs for doc in expected_docs):
            return "PASS", f"Surfaced ambiguity docs: {expected_docs}"
        else:
            return "FAIL", f"Missed ambiguity docs: {retrieved_docs}"
            
    if not expected_docs:
        return "FAIL", "No expected docs specified"

    found_doc = False
    found_exact_passage = False
    
    for e in evidence_list:
        if e["document_id"] in expected_docs:
            found_doc = True
            if e["excerpt"] in expected_evidence:
                found_exact_passage = True
                break

    if found_exact_passage:
        return "PASS", "Found exact expected passage."
    elif found_doc:
        return "PARTIAL", "Found correct document but wrong passage."
    else:
        return "FAIL", f"Failed to retrieve expected doc."

def run_workspace_tests():
    print("\n" + "="*50)
    print("WORKSPACE ISOLATION TESTS")
    print("="*50)
    
    # Acme test
    acme_res = review("Where is production infrastructure hosted?", workspace_id="ws_acme_corp", method="hybrid")
    acme_docs = [e["document_id"] for e in acme_res.get("evidence", [])]
    acme_pass = ("SEC-G001" not in acme_docs) and (acme_res["status"] == "INSUFFICIENT_EVIDENCE")
    print(f"Acme Corp Test: {'PASS' if acme_pass else 'FAIL'} | Status: {acme_res['status']} | Docs: {acme_docs}")

    # Globex test
    globex_res = review("Where is production infrastructure hosted?", workspace_id='ws_acme_corp', method="hybrid", workspace_id="ws_globex_corp")
    globex_docs = [e["document_id"] for e in globex_res.get("evidence", [])]
    globex_pass = ("SEC-G001" in globex_docs) and (globex_res["status"] == "ANSWERABLE")
    print(f"Globex Corp Test: {'PASS' if globex_pass else 'FAIL'} | Status: {globex_res['status']} | Docs: {globex_docs}")
    
    return {
        "acme": {"pass": acme_pass, "status": acme_res["status"], "docs": acme_docs},
        "globex": {"pass": globex_pass, "status": globex_res["status"], "docs": globex_docs}
    }

def main():
    print("Starting Day 14 Held-Out Evaluation...")
    results = []
    
    methods = ["bm25", "embedding", "hybrid_rrf", "hybrid_cross_encoder"]

    for case in HELD_OUT_CASES:
        print(f"\nEvaluating {case['id']}: {case['question']}...")
        
        # 1. Primary pipeline execution (Hybrid RRF + Cross-Encoder + SLM)
        prod_res = review(case["question"], workspace_id="ws_acme_corp", method="hybrid")
        slm_status = prod_res.get("status")
        slm_reason = prod_res.get("reason", "")
        retrieved_evidence = prod_res.get("evidence", [])
        
        # Diagnostics for RRF vs Cross-Encoder
        bm25_top20 = retrieve_by_bm25(case["question"], workspace_id="ws_acme_corp", top_k=20)
        emb_top20 = retrieve_by_embedding(case["question"], workspace_id="ws_acme_corp", top_k=20)
        rrf_top10 = rrf_fuse(bm25_top20, emb_top20, k=60, top_k=10)
        ce_top5 = retrieved_evidence # rerank_evidence top-5
        
        # RRF & CE ranks for expected docs
        expected_docs = case.get("expected_document_ids", [])
        rrf_rank = None
        for rank, p in enumerate(rrf_top10, start=1):
            if p["document_id"] in expected_docs:
                rrf_rank = rank
                break
                
        ce_rank = None
        ce_score = None
        ce_chunk_id = None
        ce_doc_id = None
        for rank, p in enumerate(ce_top5, start=1):
            if p["document_id"] in expected_docs:
                ce_rank = rank
                ce_score = p.get("reranker_score")
                ce_chunk_id = p.get("chunk_id")
                ce_doc_id = p.get("document_id")
                break
                
        # Method comparison scoring (Layer 1)
        method_scores = {}
        for method in methods:
            ev = get_evidence_for_method(case["question"], method)
            sc, note = score_retrieval(ev, case)
            method_scores[method] = sc
            
        retrieval_result = method_scores["hybrid_cross_encoder"]
        slm_result_pass = (slm_status == case["expected_status"])
        
        # Final Status
        if retrieval_result == "PASS" and slm_result_pass:
            final_status = "PASS"
        elif retrieval_result == "PASS" or slm_result_pass:
            final_status = "PARTIAL"
        else:
            final_status = "FAIL"
            
        case_data = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_status": case["expected_status"],
            "expected_docs": expected_docs,
            "expected_evidence": case.get("expected_evidence", []),
            "retrieval_result": retrieval_result,
            "slm_status": slm_status,
            "slm_result_pass": slm_result_pass,
            "slm_reason": slm_reason,
            "final_status": final_status,
            "method_retrieval_scores": method_scores,
            "diagnostics": {
                "rrf_rank": rrf_rank,
                "ce_rank": ce_rank,
                "reranker_score": ce_score,
                "top_chunk_id": ce_chunk_id,
                "top_doc_id": ce_doc_id,
                "retrieved_evidence": retrieved_evidence
            }
        }
        results.append(case_data)

    ws_results = run_workspace_tests()
    
    output = {
        "held_out_results": results,
        "workspace_isolation": ws_results
    }

    os.makedirs("docs/evaluation", exist_ok=True)
    with open("docs/evaluation/day-14-raw-results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved raw evaluation results to docs/evaluation/day-14-raw-results.json")

    # Print Summary Tables
    print("\n" + "="*70)
    print("REQUIRED RESULTS TABLE")
    print("="*70)
    print("| Question | Category | Expected | Retrieval Result | SLM Result | Final Status |")
    print("|----------|----------|----------|------------------|------------|--------------|")
    for r in results:
        slm_display = r["slm_status"]
        print(f"| {r['id']} | {r['category'].capitalize()} | {r['expected_status']} | {r['retrieval_result']} | {slm_display} | {r['final_status']} |")

    # Metrics
    total = len(results)
    retrieval_passes = sum(1 for r in results if r["retrieval_result"] == "PASS")
    slm_passes = sum(1 for r in results if r["slm_result_pass"])
    final_passes = sum(1 for r in results if r["final_status"] == "PASS")

    print("\n" + "="*50)
    print("OVERALL METRICS")
    print("="*50)
    print(f"1. Retrieval Pass Rate: {retrieval_passes}/{total} ({retrieval_passes/total*100:.1f}%)")
    print(f"2. SLM Classification Accuracy: {slm_passes}/{total} ({slm_passes/total*100:.1f}%)")
    print(f"3. End-to-End Pass Rate: {final_passes}/{total} ({final_passes/total*100:.1f}%)")

    # Category Breakdown
    print("\n" + "="*50)
    print("CATEGORY BREAKDOWN")
    print("="*50)
    categories = sorted(list(set(r["category"] for r in results)))
    print("| Category | Count | Retrieval Pass | SLM Accuracy | End-to-End Pass |")
    print("|----------|-------|----------------|--------------|-----------------|")
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        c_tot = len(cat_items)
        c_ret = sum(1 for r in cat_items if r["retrieval_result"] == "PASS")
        c_slm = sum(1 for r in cat_items if r["slm_result_pass"])
        c_e2e = sum(1 for r in cat_items if r["final_status"] == "PASS")
        print(f"| {cat.capitalize()} | {c_tot} | {c_ret}/{c_tot} | {c_slm}/{c_tot} | {c_e2e}/{c_tot} |")

if __name__ == "__main__":
    main()
