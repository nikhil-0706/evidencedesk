# Day 14 — Held-Out Evaluation

## 1. Objective

Questions Q16–Q25 were intentionally held out and untouched throughout the development process (Days 1–13). The objective of Day 14 is to perform a formal, un-tuned evaluation of the EvidenceDesk pipeline on these unseen test questions to measure system generalization across retrieval, reranking, and SLM reasoning capabilities.

## 2. Evaluation protocol

This is a strict evaluation day. **Zero post-hoc tuning** was performed:
- SLM prompt was not altered.
- No model weights or fine-tuning were modified.
- BM25, Qdrant embeddings, RRF parameter ($k=60$), Cross-Encoder settings, chunking strategy, and retrieval thresholds were unchanged.
- Test question labels were preserved exactly as defined.
- Failures discovered during evaluation are recorded as findings for future improvement cycles and were not fixed today.

## 3. System evaluated

The primary production pipeline evaluated is:
$$\text{Question} \rightarrow \text{Workspace Context} \rightarrow \text{BM25 Top-20 + Qdrant Embedding Top-20} \rightarrow \text{RRF (k=60)} \rightarrow \text{Top-10} \rightarrow \text{Cross-Encoder Rerank} \rightarrow \text{Top-5} \rightarrow \text{Few-shot SLM Reasoning (qwen2.5:3b)} \rightarrow \text{Human Review}$$

## 4. Question-level results

| Question ID | Category | Expected Outcome | Retrieval Result (Layer 1) | SLM Result (Layer 2) | Final Status |
|-------------|----------|------------------|----------------------------|----------------------|--------------|
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

## 5. Retrieval results

Retrieval quality is evaluated independently of LLM reasoning (Layer 1). The target evidence passage was retrieved within the Top-5 Cross-Encoder reranked results for **10 out of 10** questions.

- **Overall Retrieval Pass Rate**: **100.0%** (10/10)
- **Direct Questions (4)**: 4/4 (100.0%)
- **Paraphrased Questions (2)**: 2/2 (100.0%)
- **Unanswerable Questions (2)**: 2/2 (100.0%)
- **Ambiguous Questions (1)**: 1/1 (100.0%)
- **Conflicting Questions (1)**: 1/1 (100.0%)

### Diagnostic Retrieval Method Comparison

| Question ID | Category | BM25 | Embedding | Hybrid RRF | Hybrid RRF + Cross-Encoder |
|-------------|----------|------|-----------|------------|----------------------------|
| Q16 | Direct | PASS | PASS | PASS | PASS |
| Q17 | Direct | PASS | PASS | PASS | PASS |
| Q18 | Direct | PASS | PASS | PASS | PASS |
| Q19 | Direct | PASS | PASS | PASS | PASS |
| Q20 | Paraphrased | FAIL | PASS | PASS | PASS |
| Q21 | Paraphrased | PASS | PASS | PASS | PASS |
| Q22 | Unanswerable | PASS | PASS | PASS | PASS |
| Q23 | Unanswerable | PASS | PASS | PASS | PASS |
| Q24 | Ambiguous | PASS | PASS | PASS | PASS |
| Q25 | Conflicting | PASS | PASS | PASS | PASS |

*Note: For Q20 ("Who gives permission for an employee to receive admin privileges?"), BM25 alone failed due to lexical gap ("admin privileges" vs "administrative access"), whereas Embedding and Hybrid pipelines correctly retrieved `SEC-001`.*

## 6. SLM classification results

Layer 2 evaluates whether the few-shot SLM (`qwen2.5:3b`) correctly classifies the question based on the retrieved Top-5 evidence into `ANSWERABLE`, `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS`, or `CONFLICTING`.

- **Overall SLM Accuracy**: **80.0%** (8/10)
- **Direct Questions (4)**: 3/4 (75.0%)
- **Paraphrased Questions (2)**: 1/2 (50.0%)
- **Unanswerable Questions (2)**: 2/2 (100.0%)
- **Ambiguous Questions (1)**: 1/1 (100.0%)
- **Conflicting Questions (1)**: 1/1 (100.0%)

## 7. End-to-end results

An end-to-end PASS requires both **PASS** in Layer 1 (Retrieval) and an **EXACT MATCH** in Layer 2 (SLM Reasoning classification).

- **End-to-End Pass Rate**: **80.0%** (8/10)
- **Passed Completely (8 questions)**: Q17, Q18, Q19, Q20, Q22, Q23, Q24, Q25.
- **Partial Pass (2 questions)**: Q16, Q21 (Retrieval succeeded, SLM classification failed).

## 8. Failure analysis

Both system failures occurred exclusively in **Layer 2 (SLM Reasoning)**. The retrieval pipeline successfully provided the exact ground-truth evidence in the Top-1 reranked position for both failing questions.

### Failure 1: Q16 ("Where are encryption keys managed?")
- **Expected**: `ANSWERABLE`
- **Retrieved Evidence**: `[SEC-002] Encryption keys are managed in a dedicated key management service.` (RRF Rank #1, CE Rank #1, score: 0.7986)
- **SLM Classification**: `INSUFFICIENT_EVIDENCE`
- **SLM Reason**: *"The evidence does not establish where encryption keys are managed."*
- **Failure Classification**: **C. Reasoning/classification failure**
- **Analysis**: The SLM hallucinated that "dedicated key management service" was not an explicit location or management system, failing to map the literal policy statement to the question.

### Failure 2: Q21 ("Who takes the first look at a reported security incident?")
- **Expected**: `ANSWERABLE`
- **Retrieved Evidence**: `[SEC-004] Security incidents are triaged by the on-call engineer.` (RRF Rank #1, CE Rank #1, score: 0.1066)
- **SLM Classification**: `AMBIGUOUS`
- **SLM Reason**: *"The evidence does not specify whether the first look at a security incident is taken by the on-call engineer or the security lead."*
- **Failure Classification**: **C. Reasoning/classification failure**
- **Analysis**: The SLM over-analysed the evidence by comparing the triage step ("on-call engineer") with the escalation step ("security lead"), misclassifying a straightforward paraphrased question ("takes the first look" $\approx$ "triages") as ambiguous.

## 9. Workspace isolation

Workspace scoping and isolation regression tests were executed:

1. **Acme Corp (`ws_acme_corp`)**:
   - Question: *"Where is production infrastructure hosted?"*
   - Outcome: **PASS** (`INSUFFICIENT_EVIDENCE`). `SEC-G001` (Globex policy) was **strictly excluded** from candidate retrieval and reranking.

2. **Globex Corp (`ws_globex_corp`)**:
   - Question: *"Where is production infrastructure hosted?"*
   - Outcome: **PASS** (`ANSWERABLE`). Evidence retrieved: `[SEC-G001] Production infrastructure is hosted on ExampleCloud.`

3. **Data Leakage Verification**:
   - Cross-Encoder reranking operates exclusively on workspace-filtered candidates emitted by the RRF stage. Zero cross-workspace evidence leakage occurred.

## 10. Limitations

- **Small Evaluation Set**: The held-out benchmark consists of only 10 questions (Q16–Q25).
- **Synthetic Policy Corpus**: The evidence corpus consists of synthetic policy documents (`SEC-001` to `SEC-005`, `SEC-G001`).
- **Scale**: The results do not represent production-scale enterprise document retrieval (10,000+ chunks).
- **Model Size**: The SLM used is a compact 3B model (`qwen2.5:3b`), which exhibits occasional reasoning rigidity on subtle semantic phrasing.
- **Further Evaluation Required**: Larger, real-world policy corpora and questionnaires are required for production validation.

## 11. Conclusion

The current pipeline **generalized well to the held-out set**, achieving a **100% Retrieval Pass Rate** and an **80% End-to-End Accuracy**. 

Retrieval components (BM25 + Qdrant Embeddings + RRF + Cross-Encoder reranking) proved robust against unseen questions, including paraphrased and unanswerable cases. The remaining errors were isolated to SLM reasoning boundaries on small local models.

*Note: These results demonstrate baseline generalization on synthetic benchmarks, but do not imply that the system is production-ready for arbitrary enterprise workloads.*
