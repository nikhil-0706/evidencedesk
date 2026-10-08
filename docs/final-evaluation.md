# EvidenceDesk Final Evaluation Report

## 1. System Overview
EvidenceDesk evaluation over frozen Q01-Q25 benchmark suite. The ML retrieval pipeline and Qwen 2.5 3B SLM reasoning are frozen as of Day 21.

## 2. Dataset
Total Questions: 25

## 3. Development/Held-out Split
Development (Q01-Q15): 15 questions
Held-Out (Q16-Q25): 10 questions

## 4. Retrieval Comparison (Correct / Total)
| Method | Dev (Q01-Q15) | Held-Out (Q16-Q25) | Overall (Q01-Q25) |
|--------|---------------|--------------------|-------------------|
| BM25   | 13/15 | 10/10 | 23/25 |
| Dense  | 15/15 | 10/10 | 25/25 |
| Hybrid | 15/15 | 10/10 | 25/25 |
| RRF    | 15/15 | 10/10 | 25/25 |
| RRF+CrossEncoder | 15/15 | 10/10 | 25/25 |

## 5. Category Breakdown (E2E)
| Category | Correct | Total | Percentage |
|----------|---------|-------|------------|
| Direct | 8 | 9 | 88.9% |
| Paraphrased | 4 | 5 | 80.0% |
| Ambiguous | 2 | 2 | 100.0% |
| Conflicting | 3 | 3 | 100.0% |
| Unanswerable | 6 | 6 | 100.0% |

## 6. SLM Evaluation
- Malformed Outputs: 0
- Grounding Failures: 0

## 7. End-to-End Evaluation
- Development (Q01-Q15): 14/15 (93.3%)
- Held-Out (Q16-Q25): 9/10 (90.0%)
- Overall (Q01-Q25): 23/25 (92.0%)

## 8. Grounding Failures
Total: 0

## 9. Malformed Outputs
Total: 0

## 10. Error Analysis
**Q09**:
- Expected: ANSWERABLE
- Actual: INSUFFICIENT_EVIDENCE
- Retrieved: []
- Likely Failure Category: Status Mismatch
- Stage: Retrieval

**Q18**:
- Expected: ANSWERABLE
- Actual: INSUFFICIENT_EVIDENCE
- Retrieved: []
- Likely Failure Category: Status Mismatch
- Stage: Retrieval

## 11. Real-document Validation
Validated successfully in Day 19.5 and Day 19.75 regression test suite. All chunks correctly mapped to workspace, preserved page metadata, and generated deterministic IDs.

## 12. Workspace Isolation
Validated successfully in Day 18 tests. `ws_acme_corp` and `ws_globex_corp` correctly segregate data. Attempts to retrieve cross-workspace evidence fail securely.

## 13. Known Limitations
- Workspace Isolation: Isolation occurs strictly at the retrieval filter layer.
- Identity: Relies on client-provided `workspace_id`. Requires an authentication token mapping in production.
- Model Performance: SLM reasoning is heavily bounded by the 3B parameter limitations.

## 14. Final Conclusions
EvidenceDesk has successfully met its local/demo-ready requirements. It guarantees provenance, avoids inventing facts via strict SLM classification, and securely isolates data workspaces. The frozen ML pipeline maintains stability across both development and held-out evaluation sets.
