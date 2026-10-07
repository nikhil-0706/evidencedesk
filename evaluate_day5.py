import json
import sys
import os
sys.path.append(os.getcwd())

from evidencedesk import review

def score_retrieval(result, case):
    expected_docs = case.get("expected_document_ids", [])
    expected_evidence = case.get("expected_evidence", [])
    expected_behavior = case.get("expected_behavior", "")
    
    if expected_behavior == "abstain":
        if result["status"] == "insufficient_evidence":
            return "PASS", "Correctly abstained"
        else:
            return "FAIL", f"Falsely returned: {result['candidate_excerpt']}"
            
    if result["status"] == "insufficient_evidence" or not result["candidate_excerpt"]:
        return "FAIL", "Failed to retrieve sufficient evidence"
        
    top_doc = result["evidence"][0]["document_id"]
    top_excerpt = result["evidence"][0]["excerpt"]
    
    if top_doc in expected_docs:
        if top_excerpt in expected_evidence:
            return "PASS", top_excerpt
        else:
            return "PARTIAL", f"Got right doc but wrong passage: {top_excerpt}"
    else:
        return "FAIL", f"Got wrong doc ({top_doc}): {top_excerpt}"

def main():
    with open("data/eval_questions.json", "r") as f:
        cases = json.load(f)
        
    dev_cases = [c for c in cases if c["split"] == "development"]
    
    results = []
    
    for case in dev_cases:
        lex_res = review(case["question"], method="lexical")
        emb_res = review(case["question"], method="embedding")
        
        lex_score, lex_note = score_retrieval(lex_res, case)
        emb_score, emb_note = score_retrieval(emb_res, case)
        
        winner = "Tie"
        if lex_score == emb_score:
            winner = "Tie"
        elif lex_score == "PASS":
            winner = "Lexical"
        elif emb_score == "PASS":
            winner = "Embedding"
        elif lex_score == "PARTIAL":
            winner = "Lexical"
        elif emb_score == "PARTIAL":
            winner = "Embedding"
            
        results.append({
            "id": case["id"],
            "category": case["category"],
            "lexical": lex_score,
            "lexical_note": lex_note,
            "embedding": emb_score,
            "embedding_note": emb_note,
            "winner": winner
        })

    # Print summary
    print("## Summary")
    lex_pass = sum(1 for r in results if r["lexical"] == "PASS")
    lex_partial = sum(1 for r in results if r["lexical"] == "PARTIAL")
    lex_fail = sum(1 for r in results if r["lexical"] == "FAIL")
    
    emb_pass = sum(1 for r in results if r["embedding"] == "PASS")
    emb_partial = sum(1 for r in results if r["embedding"] == "PARTIAL")
    emb_fail = sum(1 for r in results if r["embedding"] == "FAIL")
    
    print(f"Lexical: PASS: {lex_pass}, PARTIAL: {lex_partial}, FAIL: {lex_fail} (Pass Rate: {lex_pass/len(dev_cases)*100:.1f}%)")
    print(f"Embedding: PASS: {emb_pass}, PARTIAL: {emb_partial}, FAIL: {emb_fail} (Pass Rate: {emb_pass/len(dev_cases)*100:.1f}%)")
    
    # Breakdown by category
    categories = set(c["category"] for c in dev_cases)
    print("\n## By Category (PASS count)")
    for cat in categories:
        cat_cases = [r for r in results if r["category"] == cat]
        cat_lex_pass = sum(1 for r in cat_cases if r["lexical"] == "PASS")
        cat_emb_pass = sum(1 for r in cat_cases if r["embedding"] == "PASS")
        print(f"- {cat.capitalize()}: Lexical {cat_lex_pass}/{len(cat_cases)}, Embedding {cat_emb_pass}/{len(cat_cases)}")
        
    print("\n## Detailed Table")
    print("| ID | Category | Expected behavior | Lexical result | Embedding result | Winner | Notes |")
    print("|---|---|---|---|---|---|---|")
    
    for case in dev_cases:
        r = next(res for res in results if res["id"] == case["id"])
        # Format a quick note summarizing the difference
        if r["winner"] == "Lexical":
            note = f"Embedding {r['embedding_note']}"
        elif r["winner"] == "Embedding":
            note = f"Lexical {r['lexical_note']}"
        else:
            note = f"Lex: {r['lexical']}, Emb: {r['embedding']}"
            
        print(f"| {case['id']} | {case['category']} | {case.get('expected_behavior', 'show_evidence')} | {r['lexical']} | {r['embedding']} | {r['winner']} | {note} |")

if __name__ == "__main__":
    main()
