# Day 5 — Lexical vs Embedding Retrieval

## Goal
Compare lexical and embedding retrieval on the same 15 development questions.

## Evaluation Method
- **Same questions:** Q01–Q15 (Development set).
- **Same documents:** The 5 synthetic security policies.
- **Same expected evidence:** Ground truth mappings defined in `eval_questions.json`.
- **Two independent retrieval methods:** Lexical (word overlap) and Embedding (`sentence-transformers/all-MiniLM-L6-v2`).
- **Q16–Q25 held out:** Not used for evaluation or tuning.

## Results

### Summary
- **Lexical:** PASS: 8, PARTIAL: 1, FAIL: 6 (Pass Rate: 53.3%)
- **Embedding:** PASS: 11, PARTIAL: 1, FAIL: 3 (Pass Rate: 73.3%)

### By Category (PASS count)
- **Direct:** Lexical 6/6, Embedding 6/6
- **Ambiguous:** Lexical 0/1, Embedding 0/1
- **Paraphrased:** Lexical 1/3, Embedding 1/3
- **Conflicting:** Lexical 1/2, Embedding 2/2
- **Unanswerable:** Lexical 0/3, Embedding 2/3

### Detailed Table

| ID | Category | Expected behavior | Lexical result | Embedding result | Winner | Notes |
|---|---|---|---|---|---|---|
| Q01 | direct | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q02 | direct | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q03 | direct | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q04 | direct | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q05 | direct | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q06 | direct | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q07 | paraphrased | show_evidence | PASS | PASS | Tie | Lex: PASS, Emb: PASS |
| Q08 | paraphrased | show_evidence | FAIL | FAIL | Tie | Lex: FAIL, Emb: FAIL |
| Q09 | paraphrased | show_evidence | FAIL | PARTIAL | Embedding | Lexical Failed to retrieve sufficient evidence |
| Q10 | unanswerable | abstain | FAIL | PASS | Embedding | Lexical Falsely returned: Data in transit is protected using TLS 1.2 or later. |
| Q11 | unanswerable | abstain | FAIL | PASS | Embedding | Lexical Falsely returned: Employees must use multi-factor authentication (MFA) to access production systems. |
| Q12 | unanswerable | abstain | FAIL | FAIL | Tie | Lex: FAIL, Emb: FAIL |
| Q13 | ambiguous | ask_clarification | FAIL | FAIL | Tie | Lex: FAIL, Emb: FAIL |
| Q14 | conflicting | flag_conflict | PARTIAL | PASS | Embedding | Lexical Got right doc but wrong passage: Database backups are created daily. |
| Q15 | conflicting | flag_conflict | PASS | PASS | Tie | Lex: PASS, Emb: PASS |

## Failure Analysis
The most important differences observed were:
1. **Unanswerable questions:** Lexical retrieval failed entirely here. It frequently triggered false positives because simple questions inevitably overlap with common words in the policies (like "data" or "systems"). Embedding retrieval successfully abstained on 2 of the 3 questions because the semantic meaning of the questions did not match any policy.
2. **Conflicting evidence (Q14):** For Q14, lexical retrieval pulled the correct document but the wrong sentence ("Database backups are created daily.") instead of the retention period. Embedding retrieval successfully identified the correct sentence about retention.
3. **Ambiguous questions:** Both retrievers failed on Q13. Neither has the reasoning capacity to realize a question is ambiguous; they simply retrieve the closest text mathematically or lexically.

## Strengths and Weaknesses

### Lexical
**Strengths:**
- Performs perfectly on simple direct questions where exact terminology is used.
- Low computational complexity.

**Weaknesses:**
- Extremely susceptible to false positives (fails unanswerable questions) due to basic word overlap.
- Fails on paraphrased questions that use synonyms.
- Easily distracted by irrelevant sentences in the same document if they contain matching words.

### Embedding
**Strengths:**
- Significantly higher overall pass rate (73.3% vs 53.3%).
- Much better at rejecting unanswerable questions (abstains correctly) because it evaluates semantic meaning rather than word counts.
- More precise at picking the exact relevant sentence within a document (e.g., Q14).

**Weaknesses:**
- Still struggled to perfectly retrieve some highly paraphrased questions (Q08, Q09).
- Similarity score does not equal reasoning; it cannot natively "ask for clarification" when a question is ambiguous.

## Technical Decision
Based on the evidence, we should **transition to Embedding retrieval** as our primary method. It strictly outperformed or tied lexical retrieval on every single question, yielding a 20% absolute improvement in accuracy. 

**What to investigate next:**
While embedding retrieval provides better raw evidence, it still lacks the reasoning required to handle ambiguous questions (Q13) or perfectly handle complex unanswerable cases (Q12). Therefore, on Day 6, we should investigate adding an **LLM reasoning layer** on top of the embedding retriever to read the retrieved evidence and determine if it actually answers the question, flags a conflict, or requires clarification.

## Limitations
- Small synthetic dataset.
- Only 15 development questions evaluated.
- Held-out questions were not used for tuning.
- Retrieval similarity is not proof of correctness (a human or LLM must still review it).
- Human review remains necessary.

## Next
Improve ONE clearly observed weakness (e.g., adding an LLM to reason over retrieved text to handle ambiguous/unanswerable cases) on Day 6 and re-run the evaluation.
