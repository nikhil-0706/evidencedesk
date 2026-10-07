# Day 10 — Evidence Metadata and Versioning

## Goal
Add canonical evidence metadata and document versioning to EvidenceDesk. Ensure that the source document, document version, workspace, chunk identity, and exact supporting text are preserved throughout the retrieval pipeline without rebuilding the retrieval system itself.

## Why This Matters
A production evidence system must provide traceability and audibility. When an AI or human reviewer evaluates compliance answers, they must know exactly:
- **Which workspace** does this evidence belong to?
- **Which document** does it belong to?
- **What is the document name/title**?
- **What version** is this?
- **Which chunk** produced this result?
- **What exact text** was retrieved?

Without these, it is impossible to trace an AI-generated answer back to the canonical truth source, rendering the system unsafe for security compliance tasks.

## Metadata Schema
The canonical evidence schema was updated to include:
```json
{
  "chunk_id": "chk_sec003_001",
  "workspace_id": "ws_acme_corp",
  "document_id": "SEC-003",
  "title": "Backup policy",
  "version": "2026-01",
  "excerpt": "Database backups are created daily."
}
```

## Implementation
1. **Creation**: The `PASSAGES` generation in `evidencedesk.py` now deterministically generates a `chunk_id` (e.g. `chk_sec003_001`) and assigns a default `workspace_id` (`ws_acme_corp`) during chunking.
2. **Lexical Retrieval**: Refactored to iterate over the pre-built `PASSAGES` list instead of dynamically re-splitting `DOCUMENTS` on the fly. This ensures the exact same chunk identity is used.
3. **BM25 & Embedding**: These already used `PASSAGES` and simply copy the objects, naturally preserving all metadata fields.
4. **RRF**: Updated the fusion `make_key()` logic to strictly fuse candidates based on `chunk_id` instead of dynamically concatenating `document_id` and text.
5. **Review UI**: Updated the frontend payload parsing to display the new Workspace, Document, Version, and Chunk fields in a clean, readable provenance indicator.

## Tests
Extensive testing (`test_metadata.py`) was introduced to enforce metadata preservation:
- `test_lexical_preserves_metadata`
- `test_bm25_preserves_metadata`
- `test_embedding_preserves_metadata`
- `test_hybrid_preserves_metadata`
- `test_rrf_no_duplicates`: Ensured chunk identity works for RRF fusion.
- `test_rrf_version_survives`, `test_rrf_workspace_survives`
- `test_chunk_identity_stable`: Confirmed all 3 underlying retrievers yield exactly matching `chunk_id` for the same text.

## Regression Tests
The Day 08 smoke tests were run against the updated codebase:
1. **Direct (Encryption)**: Retrieved SEC-002; ANSWERABLE.
2. **Paraphrased (Safeguards in transit)**: Retrieved SEC-002 at rank 3.
3. **Conflict (30 vs 90 days)**: Retrieved both SEC-003 and SEC-005; CONFLICTING_EVIDENCE.
4. **Ambiguous (Retention policy)**: Retrieved SEC-003 and SEC-005; AMBIGUOUS.
*Behavior is completely unchanged.*

## Design Decision
**Stable Chunk Identity in RRF:** By assigning a deterministic `chunk_id` at the time the corpus is chunked, RRF fusion becomes extremely robust. It eliminates the risk of slight text whitespace differences or concatenation collisions causing the same semantic passage to appear twice in the fused list. 

## Limitations
Workspace isolation/filtering is not yet implemented. All documents currently use a hardcoded default workspace (`ws_acme_corp`). 

## Next
Day 11: Workspace isolation and security tests.
