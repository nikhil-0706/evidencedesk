# Day 09 — Retrieval Comparison

## Objective
Compare the retrieval quality of the following methods on the development question set (Q01–Q15):
- Original Lexical
- BM25
- Embedding
- Hybrid RRF

## Evaluation Protocol
- **Questions**: Exactly the same Q01–Q15 questions were queried across all four methods.
- **Corpus**: Exactly the same policy documents (`PASSAGES`) were used.
- **Expected Evidence**: The same labeled expected documents and expected behaviors were used.
- **Criteria**:
  - `PASS`: The retriever surfaced the expected evidence in the Top-K. For ambiguous/conflicting, both or relevant sources were surfaced. For unanswerable, it didn't inappropriately retrieve a misleading fact (bypassed LLM fragility by checking if no exact matching fake expected docs were flagged).
  - `PARTIAL`: The correct document was retrieved, but the specific expected excerpt was missing.
  - `FAIL`: The retriever failed to surface the expected documents.
- **No LLM Interference**: The underlying retrieval output (Top-K) was checked directly rather than routing through the 3B LLM, preventing LLM reasoning flaws from misrepresenting retrieval quality.
- **Held-out Data**: Q16–Q25 were absolutely untouched and not evaluated. No tuning was performed.

## Overall Results
| Method     | PASS | PARTIAL | FAIL | Pass Rate |
|------------|------|---------|------|-----------|
| Lexical    | 13   | 1       | 1    | 86.7%     |
| BM25       | 13   | 1       | 1    | 86.7%     |
| Embedding  | 15   | 0       | 0    | 100.0%    |
| Hybrid RRF | 15   | 0       | 0    | 100.0%    |

## Results by Question Type
| Question Type     | Lexical | BM25 | Embedding | Hybrid |
|-------------------|---------|------|-----------|--------|
| Direct (6)        | 6       | 6    | 6         | 6      |
| Paraphrased (3)   | 1       | 1    | 3         | 3      |
| Unanswerable (3)  | 3       | 3    | 3         | 3      |
| Ambiguous (1)     | 1       | 1    | 1         | 1      |
| Conflicting (2)   | 2       | 2    | 2         | 2      |

## Question-Level Results
| QID | Type | Lexical | BM25 | Embedding | Hybrid |
|-----|------|---------|------|-----------|--------|
| Q01 | direct | PASS | PASS | PASS | PASS |
| Q02 | direct | PASS | PASS | PASS | PASS |
| Q03 | direct | PASS | PASS | PASS | PASS |
| Q04 | direct | PASS | PASS | PASS | PASS |
| Q05 | direct | PASS | PASS | PASS | PASS |
| Q06 | direct | PASS | PASS | PASS | PASS |
| Q07 | paraphrased | PASS | PASS | PASS | PASS |
| Q08 | paraphrased | FAIL | FAIL | PASS | PASS |
| Q09 | paraphrased | PARTIAL | PARTIAL | PASS | PASS |
| Q10 | unanswerable | PASS | PASS | PASS | PASS |
| Q11 | unanswerable | PASS | PASS | PASS | PASS |
| Q12 | unanswerable | PASS | PASS | PASS | PASS |
| Q13 | ambiguous | PASS | PASS | PASS | PASS |
| Q14 | conflicting | PASS | PASS | PASS | PASS |
| Q15 | conflicting | PASS | PASS | PASS | PASS |

## Error Analysis
1. **Q08 (paraphrased)**: "What safeguards information while it travels between systems?"
   - **Lexical/BM25**: `FAIL`. They overmatched the word "systems" and returned SEC-001 ("Employees must use multi-factor authentication to access production systems.") but completely missed SEC-002 ("Data in transit is protected...").
   - **Reason**: Exact terminology mismatch. The query used "travels between systems" instead of "in transit".
2. **Q09 (paraphrased)**: "How regularly do you check that backups can be restored?"
   - **Lexical/BM25**: `PARTIAL`. They retrieved the correct document (SEC-003) but surfaced the excerpt "Database backups are created daily" instead of "Restore procedures are tested quarterly."
   - **Reason**: Lexical matching failed to align "check that backups can be restored" with "Restore procedures are tested".

## Hybrid Analysis
- **Where RRF helped vs Lexical/BM25**: Hybrid RRF correctly absorbed the semantic understanding of Embedding on Q08 and Q09. BM25 completely failed on Q08, but Hybrid retrieved the exact expected excerpt because the Embedding method had a high rank for it, keeping its RRF score high enough to make the Top-10.
- **Where RRF matched its components**: Embedding alone already scored 100% on this small development set. Hybrid RRF identically matched Embedding's perfect success rate, proving it safely merged the lexical and semantic lists.
- **Where RRF did not help**: RRF did not retrieve anything uniquely that its individual components missed completely (since Embedding alone was flawless on this specific subset of data).
- **Regression**: **No regression occurred.** The Hybrid system maintained the 100% Pass Rate of the Embedding system without losing any relevant evidence due to blending.

## Key Findings
- **Embedding heavily outperforms Lexical on Paraphrased Questions.** Lexical and BM25 fail immediately when exact keywords ("in transit" vs "travels") aren't present.
- **BM25 performs identically to the naïve word-overlap baseline** on this tiny synthetic corpus, largely because the corpus lacks heavy frequency/TF-IDF distribution complexity.
- **Hybrid RRF is extremely safe.** The fusion technique successfully combines exact match signals (BM25) and semantic signals (Embedding) without diluting the quality or dropping expected passages out of the Top-K.

## Limitations
- **Small Corpus & Question Set**: Embedding scoring 100% is a byproduct of testing only 15 development questions on a tiny 5-document synthetic policy corpus. 
- **Top-K Difference**: Lexical/BM25/Embedding evaluated using `Top-3`, whereas Hybrid RRF pulls `Top-20` candidates and returns `Top-10` fused results. The larger `Top-10` boundary for Hybrid increases its chance of scoring a PASS (since checking expected evidence is easier with 10 slots than 3).
- **No LLM Interaction**: The evaluation script strictly measured underlying Top-K retrieval, bypassing the LLM. In reality, retrieving 10 passages sometimes distracts the 3B LLM (as seen in Day 08 fixes).

## Next Step
The next planned engineering step is metadata/versioning and workspace-aware retrieval. Cross-encoder reranking is planned later.
