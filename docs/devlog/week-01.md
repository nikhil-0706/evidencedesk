# Week 1 — EvidenceDesk Retrieval and Review Baseline

## Goal
Build a measurable evidence retrieval baseline and test an initial improvement.

## What Was Built
- FastAPI application serving a simple HTML/JS frontend.
- Lexical retrieval (word-overlap baseline).
- Embedding retrieval (`sentence-transformers/all-MiniLM-L6-v2`) serving as the primary retriever.
- 25-question labeled evaluation set (15 dev, 10 held-out).
- Human review interface ensuring the user always reviews the retrieved evidence.
- Constrained evidence reasoning layer using a local LLM (`ollama`, `qwen2.5:3b`).
- Ambiguity detection, insufficient-evidence handling, and conflict classification based strictly on retrieved text.

## Development Evaluation
On Day 5, we evaluated the baseline retrieval on the 15-question development set:
- **Lexical:** 53.3% pass rate
- **Embedding:** 73.3% pass rate

*Technical Decision:* Embedding retrieval significantly outperformed lexical on Unanswerable and Conflicting questions. It was chosen as the primary retrieval method, with lexical kept as an available baseline. On Day 6, an LLM reasoning layer was added to improve Ambiguous question handling without changing the retriever itself.

## Held-Out Evaluation
On Day 7, we ran the first evaluation on the 10 unseen held-out questions (Q16–Q25) against the Day 6 Embedding + LLM Reasoning system.

**Summary:** PASS: 3, PARTIAL: 7, FAIL: 0 (Pass Rate: 30.0%)

**By Category (PASS count):**
- Unanswerable: 3/3
- Ambiguous: 0/1
- Direct: 0/3
- Conflicting: 0/1
- Paraphrased: 0/2

## What Worked
- **Unanswerable Questions (Safety):** The LLM reasoning layer handled all 3 held-out unanswerable questions perfectly (Q21, Q22, Q24) by safely returning `INSUFFICIENT_EVIDENCE` rather than hallucinating an answer.
- **Retrieval Engine Reliability:** Embedding retrieval continued to successfully retrieve the correct top-k documents (e.g., retrieving SEC-004 for Incident Response questions), even when the downstream reasoning failed.

## What Failed
- **Reasoning Layer Fragility (Generalization Drop):** The system pass rate dropped from 73% on dev to 30% on held-out.
- **Direct & Paraphrased Questions:** Q16-Q20 failed to reach a full `PASS` because the small LLM (3B parameters) incorrectly classified them as `INSUFFICIENT_EVIDENCE` or `AMBIGUOUS` despite the retriever fetching the correct sentence (e.g., "Security incidents are triaged by the on-call engineer").
- **Conflicting & Ambiguous Examples:** Q23 (Ambiguous) and Q25 (Conflicting) were marked as `INSUFFICIENT_EVIDENCE` by the LLM instead of correctly recognizing the ambiguity or conflict, meaning the prompt structure or model size did not generalize well beyond the few-shot examples we tuned on the dev set.

## What We Learned
- **Retrieval quality matters:** Semantic embedding retrieval was objectively superior to basic lexical matching at fetching relevant context.
- **Retrieval alone cannot resolve ambiguity:** We need a reasoning layer to detect underspecified questions or conflicting sources.
- **LLM reasoning must remain grounded:** The `INSUFFICIENT_EVIDENCE` fallback successfully prevented hallucinations, proving the system is safe, even if overly conservative.
- **Small models struggle to generalize:** The drastic drop in performance on unseen questions (over-classifying direct questions as insufficient evidence) shows that prompt-tuning a small 3B model on 15 questions does not guarantee robust zero-shot instruction following.
- **Human review remains important:** Evidence similarity is not proof, and AI classification is not infallible. Keeping the human in the loop is essential.

## Current Limitations
- Small synthetic dataset.
- Small held-out evaluation set (only 10 questions).
- No production deployment.
- No authentication.
- LLM reasoning can still make classification mistakes (especially false negatives/over-conservatism on small models).
- No automated long-term monitoring.
- No large-scale performance evaluation.

## Next Week
- Upgrade the reasoning layer (explore better prompting techniques or a larger model API to reduce false `INSUFFICIENT_EVIDENCE` classifications).
- Expand the evaluation dataset to capture more diverse edge cases.
- Analyze individual held-out failures to refine the criteria without overfitting.

---
### Week 1 Final Checklist
- [x] Lexical baseline exists
- [x] Embedding retriever exists
- [x] 25-question evaluation exists
- [x] 15 development questions evaluated
- [x] 10 held-out questions evaluated
- [x] LLM reasoning layer exists
- [x] Ambiguity handling tested
- [x] Insufficient evidence handling tested
- [x] Conflict handling tested
- [x] Human review preserved
- [x] README works
- [x] Setup instructions verified
- [x] Repository cleaned
- [x] Results documented
