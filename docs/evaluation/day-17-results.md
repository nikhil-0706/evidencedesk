# Day 17 Evaluation Benchmark — Final Evidence Reasoning Stabilization & Freeze

## 1. Executive Summary

Day 17 concludes Week 3 of the **EvidenceDesk** project with the final verification, stabilization, and freeze of the SLM Evidence Reasoning layer.

The retrieval pipeline remains **100% frozen** (BM25 Top-20 + Qdrant Dense Top-20 $\rightarrow$ RRF $k=60$ Top-10 $\rightarrow$ Cross-Encoder Top-5).

### Key Performance Highlights
- **Phase 1 Development Set (Q01–Q15)**: **100.0% (15/15)** SLM Reasoning Accuracy (**+6.7%** over Day 15 baseline).
- **Phase 2 Held-Out Benchmark (Q16–Q25)**: **80.0% (8/10)** SLM Reasoning Accuracy (**+10.0%** over Day 15 baseline).
- **Retrieval Pass Rate**: **100.0%** across all 25 test cases.
- **Regressions**: **0** (Zero regressions across both test suites).
- **Malformed Outputs / Grounding Failures**: **0** (Pydantic schema validation and verbatim quote grounding achieved 100% compliance).

---

## 2. Benchmark Comparison (Day 15 vs Day 16 vs Day 17)

| Metric | Day 15 Baseline | Day 16 Baseline | Day 17 Frozen | Net Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Dev Retrieval Pass Rate (Q01–Q15)** | 100.0% (15/15) | 100.0% (15/15) | **100.0% (15/15)** | 0.0% |
| **Dev SLM Classification Acc (Q01–Q15)** | 93.3% (14/15) | 100.0% (15/15) | **100.0% (15/15)** | **+6.7%** |
| **Dev End-to-End Pass Rate (Q01–Q15)** | 93.3% (14/15) | 100.0% (15/15) | **100.0% (15/15)** | **+6.7%** |
| | | | | |
| **Held-Out Retrieval Pass Rate (Q16–Q25)** | 100.0% (10/10) | 100.0% (10/10) | **100.0% (10/10)** | 0.0% |
| **Held-Out SLM Classification Acc (Q16–Q25)** | 70.0% (7/10) | 80.0% (8/10) | **80.0% (8/10)** | **+10.0%** |
| **Held-Out End-to-End Pass Rate (Q16–Q25)** | 70.0% (7/10) | 80.0% (8/10) | **80.0% (8/10)** | **+10.0%** |
| | | | | |
| **Regression Count** | - | 0 | **0** | - |
| **Malformed SLM Outputs** | 0 | 0 | **0** | 0 |
| **Grounding Failures** | 0 | 0 | **0** | 0 |

---

## 3. Category Breakdown (Day 17 Final)

### Phase 1 — Development Set (Q01–Q15)

| Category | Count | Retrieval Pass | SLM Accuracy | E2E Pass | Pass Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| Direct Lookups (Q01–Q06) | 6 | 6/6 | 6/6 | 6/6 | 100.0% |
| Paraphrased (Q07–Q09) | 3 | 3/3 | 3/3 | 3/3 | 100.0% |
| Unanswerable (Q10–Q12) | 3 | 3/3 | 3/3 | 3/3 | 100.0% |
| Ambiguous (Q13) | 1 | 1/1 | 1/1 | 1/1 | 100.0% |
| Conflicting (Q14–Q15) | 2 | 2/2 | 2/2 | 2/2 | 100.0% |
| **Total Development** | **15** | **15/15** | **15/15** | **15/15** | **100.0%** |

### Phase 2 — Held-Out Benchmark (Q16–Q25)

| Category | Count | Retrieval Pass | SLM Accuracy | E2E Pass | Pass Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| Direct Lookups (Q16–Q19) | 4 | 4/4 | 4/4 | 4/4 | 100.0% |
| Paraphrased (Q20–Q21) | 2 | 2/2 | 2/2 | 2/2 | 100.0% |
| Unanswerable (Q22–Q23) | 2 | 2/2 | 2/2 | 2/2 | 100.0% |
| Ambiguous (Q24) | 1 | 1/1 | 0/1 | 0/1 | 0.0% |
| Conflicting (Q25) | 1 | 1/1 | 0/1 | 0/1 | 0.0% |
| **Total Held-Out** | **10** | **10/10** | **8/10** | **8/10** | **80.0%** |

---

## 4. Failure Analysis of Remaining Held-Out Cases

The retrieval layer successfully surfaced all relevant passages for 100% of cases. The 2 remaining held-out failures are isolated to complex scope & edge-case classification in the local 3B SLM:

1. **Q24 — *"How often is access reviewed for customers and employees?"***
   - **Expected**: `AMBIGUOUS`
   - **Retrieved Evidence**: `[SEC-001]` *"Access permissions are reviewed quarterly."*
   - **SLM Classification**: `ANSWERABLE`
   - **Root Cause**: The question asks about access reviews for *both* customers and employees. The policy passage only covers employee access permissions. The 3B SLM matched the phrase "access permissions are reviewed quarterly" and missed the missing customer scope.

2. **Q25 — *"What single backup retention period should I put in questionnaire?"***
   - **Expected**: `CONFLICTING`
   - **Retrieved Evidence**: `[SEC-003]` *"Backup retention is 30 days."* and `[SEC-005]` *"Database backup retention is 90 days."*
   - **SLM Classification**: `INSUFFICIENT_EVIDENCE`
   - **Root Cause**: The question asks for a "single backup retention period". Seeing two different durations (30 days vs 90 days), the 3B SLM concluded that "the evidence does not provide a single retention period" and returned `INSUFFICIENT_EVIDENCE` rather than `CONFLICTING`.

---

## 5. Honest Assessment of SLM Limitations

While local 3B parameter models (`qwen2.5:3b`) demonstrate high precision on direct lookups, semantic paraphrasing, and explicit unanswerable queries when guided by structured Pydantic schemas, their primary limitations remain:
- **Subtle Scope Discrepancies**: High sensitivity to surface matches when questions combine multiple subjects (e.g. customers + employees) but evidence covers only one.
- **Syntactic Cue Over-indexing**: When questions explicitly request a single unified value while candidate evidence contains conflicting policy exception passages, 3B models occasionally classify the situation as "no single value available" (`INSUFFICIENT_EVIDENCE`) rather than flagging the policy disagreement (`CONFLICTING`).

---

## 6. Freeze Status Confirmation

The SLM Evidence Reasoning Layer (`day16_reasoning.py`), prompt strategy, Pydantic validation schema (`EvidenceDecision`), and evidence grounding rules are officially **STABILIZED and FROZEN**.
