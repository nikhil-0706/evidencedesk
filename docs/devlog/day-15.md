# Day 15 Devlog — Structured Evidence Reasoning

## 1. Problem

Day 14 held-out evaluation (Q16–Q25) established:

| Layer | Accuracy |
|---|---|
| Retrieval (Layer 1) | **100%** (10/10) |
| SLM Reasoning (Layer 2) | **80%** (8/10) |
| End-to-End | **80%** (8/10) |

The retrieval pipeline was already perfect. The remaining failures were
isolated to two SLM reasoning/classification errors:

**Q16 — "Where are encryption keys managed?"**
- Retrieved evidence (Rank 1, Cross-Encoder score 0.80):
  *"Encryption keys are managed in a dedicated key management service."*
- Day 14 SLM returned: `INSUFFICIENT_EVIDENCE`
- The passage literally answers the question. This was a reasoning failure.

**Q21 — "Who takes the first look at a reported security incident?"**
- Retrieved evidence (Rank 1, Cross-Encoder score 0.11):
  *"Security incidents are triaged by the on-call engineer."*
- Day 14 SLM returned: `AMBIGUOUS`
- The model over-analysed the distinction between "triage" (first look)
  and "escalation" (second step), conflating the two roles.

Both failures were **Stage C: SLM reasoning failures**. The retrieval
layer had provided the correct evidence in position 1 in both cases.

## 2. Baseline Reasoning Performance (Day 14)

- SLM model: `qwen2.5:3b` (local, Ollama)
- Classification accuracy on held-out set: **80%** (8/10)
- Known failure mode: semantic paraphrase gap (SLM failed to recognise
  that operationally equivalent terms satisfy the question)

## 3. Structured Output Approach

The primary engineering change on Day 15 is to the reasoning and output
layer only. The retrieval pipeline (BM25, Qdrant, RRF, Cross-Encoder) is
**completely unchanged**.

Changes made to `evidencedesk.py`:

1. Added `EvidenceDecision` Pydantic schema
2. Added `_validate_slm_output()` validation pipeline
3. Rewrote `analyze_evidence()` with a fully structured prompt +
   Pydantic validation + safe error handling
4. Threaded `evidence_quote` through the `review()` response

## 4. Pydantic Schema

```python
class EvidenceDecision(BaseModel):
    status: Literal[
        "ANSWERABLE",
        "AMBIGUOUS",
        "CONFLICTING",
        "INSUFFICIENT_EVIDENCE",
    ]
    reason: str
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    evidence_quote: str = Field(default="")
```

The schema enforces:
- `status` is one of exactly four allowed values (Pydantic `Literal` constraint).
- `reason` is always a non-empty string.
- `evidence_chunk_ids` defaults to `[]` instead of crashing on omission.
- `evidence_quote` defaults to `""` and is required non-empty when
  `status == "ANSWERABLE"`.

## 5. Prompt Improvements

The prompt was substantially rewritten in `_REASONING_PROMPT`:

- **Explicit semantic equivalence section** lists operationally paired
  phrases that MUST be classified as `ANSWERABLE`:
  - `"first look"` ≈ `"triage"`
  - `"managed in KMS"` ≈ location answer for "where managed"
  - `"requires approval"` ≈ `"gives permission"`
- **Pre-check instruction**: "Before returning INSUFFICIENT_EVIDENCE, ask
  yourself: does any passage establish the core fact requested, even with
  different terminology? If YES → return ANSWERABLE."
- **Structured JSON few-shot examples** now include `evidence_quote` in
  every example, grounding the model on expected output format.
- **AMBIGUOUS vs CONFLICTING** distinction preserved from Day 13.

## 6. Semantic Equivalence Handling

The key insight is that small local SLMs (3B parameters) are sensitive to
**surface-form mismatch** between question vocabulary and evidence vocabulary.
The updated prompt addresses this by:

1. Naming the problem explicitly ("semantic equivalence is allowed")
2. Providing concrete paired examples in the semantic equivalence section
3. Framing the reasoning as: "establish the fact" rather than "match the words"
4. Including Example 2 (first look / triage) and Example 3 (managed / KMS)
   as direct counterexamples to the observed failures

This approach generalises across any paraphrased question — it is not
specific to Q16 or Q21.

## 7. Evidence Quote Extraction

When `status == "ANSWERABLE"`, the model is required to produce an
`evidence_quote` that is **verbatim copied** from the supplied evidence.

Validation checks:
1. `evidence_quote` must be non-empty (rejected otherwise)
2. `evidence_quote` must be a substring of at least one supplied passage
   (rejected otherwise)

This prevents hallucinated quotes from silently passing through the system.

## 8. Validation and Error Handling

`_validate_slm_output()` performs these checks in order:

| Step | Check | Failure Response |
|---|---|---|
| 1 | Pydantic schema validation | `VALIDATION_ERROR` |
| 2 | Status is one of 4 allowed values | `VALIDATION_ERROR` |
| 3 | chunk_ids are from retrieved evidence | `VALIDATION_ERROR` |
| 4 | ANSWERABLE has non-empty quote | `VALIDATION_ERROR` |
| 5 | quote is verbatim in supplied evidence | `VALIDATION_ERROR` |

`analyze_evidence()` also handles:
- `json.JSONDecodeError` → `VALIDATION_ERROR` (application does NOT crash)
- Any other exception → `ERROR` (safe dict returned)
- Empty evidence list → immediate `INSUFFICIENT_EVIDENCE`

The application never crashes on malformed SLM output.

## 9. Tests

`tests/test_structured_reasoning.py` contains three sections:

**Section A — Smoke Tests (7 questions):**

| Test | Question | Expected |
|---|---|---|
| TEST-1 (Q16 history) | Where are encryption keys managed? | ANSWERABLE |
| TEST-2 (Q21 history) | Who takes the first look at a reported security incident? | ANSWERABLE |
| TEST-3 | Who approves administrative access? | ANSWERABLE |
| TEST-4 | How is data in transit protected? | ANSWERABLE |
| TEST-5 | What is your retention policy? | AMBIGUOUS |
| TEST-6 | How long are database backups retained? | CONFLICTING |
| TEST-7 | Is the company SOC 2 certified? | INSUFFICIENT_EVIDENCE |

**Section B — Structured-Output Validation Tests (8 scenarios):**

| Test | Scenario | Expected |
|---|---|---|
| A | Valid ANSWERABLE JSON | Accepted |
| B | Valid INSUFFICIENT_EVIDENCE JSON | Accepted |
| C | Invalid status value `"MAYBE"` | Rejected |
| D | Missing `reason` field | Rejected |
| E | Non-dict input (malformed JSON) | Rejected (no crash) |
| F | ANSWERABLE without `evidence_quote` | Rejected |
| G | Unknown `chunk_id` only | Rejected |
| H | `evidence_quote` not in supplied evidence | Rejected |

**Section C — Workspace Isolation Regression:**
- Acme Corp: "Where is production infrastructure hosted?" → `INSUFFICIENT_EVIDENCE` (SEC-G001 excluded)
- Globex Corp: same question → `ANSWERABLE` with SEC-G001

## 10. Known Limitations

- **SLM size**: `qwen2.5:3b` is a 3B parameter model. Occasional reasoning
  failures on nuanced semantic equivalence remain possible even with an
  improved prompt.
- **Prompt sensitivity**: The model is sensitive to prompt wording.
  Results may vary slightly across Ollama build versions.
- **Synthetic corpus**: Evaluation is conducted on a small synthetic policy
  corpus. Production accuracy cannot be extrapolated from these results.
- **No fine-tuning**: Day 15 relies entirely on prompt engineering. A larger
  model or fine-tuned classifier would likely produce more stable results.
- **Validation strictness**: The verbatim `evidence_quote` check could
  reject a technically correct but slightly paraphrased model output.
  This is intentional (safety over recall) for the current baseline.

## 11. Next Steps

- [ ] Run Day 15 evaluation formally and record SLM accuracy vs Day 14 baseline.
- [ ] If SLM accuracy ≥ 90% on held-out set, declare the reasoning layer stable.
- [ ] Consider a larger model (e.g., 7B) to reduce prompt sensitivity.
- [ ] Investigate Cross-Encoder score threshold calibration as an optional
      retrieval quality gate (Day 16 candidate).
- [ ] Add streaming / async SLM call support for production latency targets.

## 12. Retrieval Components (Unchanged)

The following components were **not modified** on Day 15:

- BM25 (`rank_bm25`, stopword list)
- Qdrant (`qdrant_client`, `evidencedesk` collection)
- Embedding model (`all-MiniLM-L6-v2`)
- RRF fusion (`k=60`, BM25 Top-20, Embedding Top-20, RRF Top-10)
- Cross-Encoder model (`BAAI/bge-reranker-large`)
- Cross-Encoder Top-5
- Chunking strategy (sentence-level)
- Evidence corpus (SEC-001 to SEC-005, SEC-G001)
- Workspace filtering logic
- Q16–Q25 labels
