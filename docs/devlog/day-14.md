# Day 14 Devlog — Held-Out Retrieval and Reasoning Evaluation

## Summary of Accomplishments

Today we performed the formal **HELD-OUT evaluation** of the EvidenceDesk hybrid retrieval and reasoning pipeline on unseen test questions Q16–Q25. 

Per the evaluation protocol:
- **No tuning** was conducted (no prompt changes, no reranker tweaking, no threshold changes).
- Retrieval (Layer 1) and Reasoning/SLM classification (Layer 2) were measured independently.
- Workspace isolation regression was verified.

## Key Metrics

- **Retrieval Pass Rate (Layer 1)**: **100.0%** (10/10)
- **SLM Classification Accuracy (Layer 2)**: **80.0%** (8/10)
- **End-to-End Pass Rate**: **80.0%** (8/10)
- **Workspace Isolation**: **100.0% PASS** (Acme vs Globex isolated correctly without data leakage)

## Question-Level Breakdown

| QID | Category | Expected | Retrieval (L1) | SLM (L2) | Final Status |
|-----|----------|----------|----------------|----------|--------------|
| Q16 | Direct | ANSWERABLE | PASS | INSUFFICIENT_EVIDENCE | PARTIAL |
| Q17 | Direct | ANSWERABLE | PASS | ANSWERABLE | PASS |
| Q18 | Direct | ANSWERABLE | PASS | ANSWERABLE | PASS |
| Q19 | Direct | ANSWERABLE | PASS | ANSWERABLE | PASS |
| Q20 | Paraphrased | ANSWERABLE | PASS | ANSWERABLE | PASS |
| Q21 | Paraphrased | ANSWERABLE | PASS | AMBIGUOUS | PARTIAL |
| Q22 | Unanswerable | INSUFFICIENT_EVIDENCE | PASS | INSUFFICIENT_EVIDENCE | PASS |
| Q23 | Unanswerable | INSUFFICIENT_EVIDENCE | PASS | INSUFFICIENT_EVIDENCE | PASS |
| Q24 | Ambiguous | AMBIGUOUS | PASS | AMBIGUOUS | PASS |
| Q25 | Conflicting | CONFLICTING | PASS | CONFLICTING | PASS |

## Failure Analysis

Both failures were classified as **Stage C: Reasoning/classification failures**:

1. **Q16**: SLM returned `INSUFFICIENT_EVIDENCE` despite Top-1 retrieved evidence explicitly stating *"Encryption keys are managed in a dedicated key management service."*
2. **Q21**: SLM returned `AMBIGUOUS` instead of `ANSWERABLE`, over-analyzing the distinction between triage ("on-call engineer") and escalation ("security lead").

No retrieval or reranking failures occurred.

## Verification & Regressions

All previous unit and integration regression suites passed cleanly:
- `test_day08.py` (PASS)
- `test_day08_fix.py` (PASS)
- `test_day11.py` (PASS)
- `test_day12.py` (PASS)
- `test_day12_bugfix.py` (PASS)
- `test_metadata.py` (PASS)
- `test_rrf.py` (PASS)

## Next Steps

- Log Q16 and Q21 SLM classification failures as findings for future prompt refinement or larger model evaluation.
- Document system performance in project milestone notes.
