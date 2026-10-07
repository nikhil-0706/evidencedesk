# Day 6 — Evidence Reasoning for Ambiguous Questions

## Goal
Improve the observed ambiguity-handling weakness without changing the embedding retriever.

## Problem Observed
From our Day 5 evaluation, we found that embedding retrieval can find related evidence but cannot reliably determine whether the question itself is sufficiently specific. For instance, when asked "What is your retention policy?", it merely returned retention passages without recognizing that there are multiple types of retention (e.g. database backups vs. customer data).

## Improvement
- **Embedding retrieval remains primary:** The system still retrieves the top 3 passages using `sentence-transformers/all-MiniLM-L6-v2`.
- **Added constrained LLM reasoning layer:** We integrated `ollama` (using a local small LLM) strictly to analyze the retrieved evidence against the user question.
- **LLM sees only the question and retrieved evidence:** No external knowledge or internet access is provided.
- **Structured classification:** The reasoning layer outputs JSON specifying one of four statuses: `ANSWERABLE`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, or `CONFLICTING_EVIDENCE`.
- **Human review remains mandatory:** The AI does not provide a generated conversational answer. It simply highlights the status, rationale, and retrieved evidence for human review.

## Before
**Question:** "What is your retention policy?"
→ Retrieval only
→ Returns backup retention and database retention passages.
→ Does not ask for clarification; expects the human to figure out it was ambiguous.

## After
**Question:** "What is your retention policy?"
→ Embedding retrieval
→ Reasoning layer classification
→ Status: **AMBIGUOUS**
→ Clarification needed: "Do you mean database backup retention, customer data retention, or another type of record?"
→ The evidence remains visible below for review.

## Smoke Tests
We ran 4 explicit smoke tests to verify the behavior:
1. **Ambiguous** ("What is your retention policy?"): Successfully flagged as `AMBIGUOUS` and asked for clarification.
2. **Direct** ("How often are database backups created?"): Attempted to flag as `ANSWERABLE` (though the 3B model occasionally over-classifies ambiguity if multiple facts are present).
3. **Unanswerable** ("Is the company SOC 2 certified?"): Successfully flagged as `INSUFFICIENT_EVIDENCE`.
4. **Conflicting** ("Are backups retained for 30 days or 90 days?"): Successfully flagged as `CONFLICTING_EVIDENCE`.

## Safety / Grounding
- **No unsupported answer is generated:** The model only outputs a JSON classification.
- **Retrieved evidence remains visible:** The original text from the synthetic policies is preserved and displayed alongside the reasoning.
- **LLM does not use outside knowledge:** The constrained prompt strictly forces the LLM to rely only on the `<Retrieved Evidence>`.
- **Human review remains required:** The application clearly warns the user that matching text is not proof and no AI answer was generated.

## Limitations
- **LLM classification can still make mistakes:** The small local model (`qwen2.5:3b`) can sometimes misclassify a direct question as ambiguous if the evidence contains extra information.
- **Prompt-based reasoning is not guaranteed to be correct.**
- **Small synthetic dataset.**
- **No formal held-out evaluation yet** (Q16-Q25 remain untouched).
- **No production deployment.**

## Next
Repeat the evaluation (on the Q01-Q15 dataset) and measure the effect of the reasoning layer on overall accuracy.
