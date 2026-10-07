# Day 08 — Hybrid Retrieval with RRF

## Goal
Build a hybrid retriever combining BM25 and semantic embeddings using Reciprocal Rank Fusion.

## Why Hybrid Retrieval?
BM25 is strong for exact terms and identifiers.
Embeddings are stronger for semantic similarity and paraphrased questions.
RRF combines their rankings without requiring incompatible raw scores to be calibrated.

## Architecture
Question
→ BM25 Top-20
→ Embedding Top-20
→ RRF
→ Top-10
→ Existing reasoning
→ Human review

## Implementation
- **BM25 implementation:** Used the `rank_bm25` python library (`BM25Okapi`).
- **BM25 dependency:** Added `rank_bm25==0.2.2` to `requirements.txt`.
- **RRF implementation:** Custom `rrf_fuse` function iterates through BM25 and Embedding lists, merging them using `RRF(d) = Σ 1 / (k + rank_m(d))`.
- **k = 60:** Standard constant used for Rank Fusion.
- **Top-20 candidate retrieval:** Both BM25 and Embedding methods pull 20 results.
- **Top-10 fused results:** The RRF function cuts the merged sorted list to 10.
- **Metadata preservation:** Chunk/document uniqueness is matched via `document_id` and `excerpt`, ensuring `version`, `title`, and `rrf_score` are correctly retained.
- **`/review` integration:** Added `method="hybrid"`, triggering BM25 -> Embedding -> RRF -> LLM pipeline. UI was updated to include a Hybrid radio button.

## Smoke Tests
1. **Direct question:** "What encryption protects customer data at rest?"
   - Expected: SEC-002
   - Observed: SEC-002 was ranked #1 by Hybrid RRF.
   - Pass/Fail: PASS

2. **Paraphrased question:** "What safeguards information while it travels between systems?"
   - Expected: SEC-002
   - Observed: SEC-002 (Data in transit) was ranked #3 by Hybrid RRF, successfully leveraging Embedding's semantic search.
   - Pass/Fail: PASS

3. **Conflicting question:** "Are backups retained for 30 days or 90 days?"
   - Expected: SEC-003 and SEC-005 remain available and trigger CONFLICTING_EVIDENCE.
   - Observed: Both passages returned at Rank 1 and 2, triggering CONFLICTING_EVIDENCE classification.
   - Pass/Fail: PASS

4. **Ambiguous question:** "What is your retention policy?"
   - Expected: AMBIGUOUS
   - Observed: Ranked retention passages #1, #2, #3, triggering AMBIGUOUS classification.
   - Pass/Fail: PASS

## Important Design Decision
RRF was used instead of raw score addition because BM25 scores (often > 1.0, unbounded) and Cosine Similarity scores (0 to 1) operate on completely incompatible scales. Raw addition would let BM25 overpower semantic embeddings, or vice versa, without intense calibration. RRF purely relies on rank positions to fuse the results.

## Limitations
- RRF improves candidate fusion but does not deeply understand query/chunk relevance.
- RRF does not replace reranking.
- Cross-encoder reranking is planned for a later day.
- Workspace isolation and version-aware filtering are planned for later.

## Next
Next step will be evaluation of:
Original lexical vs BM25 vs Embedding vs Hybrid RRF
