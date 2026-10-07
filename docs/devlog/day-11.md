# Day 11 — Qdrant Vector Store and Workspace Isolation

## Goal
Introduce Qdrant as the vector storage layer and enforce workspace-scoped semantic retrieval. 

## Why Qdrant?
We've replaced the in-memory cosine similarity array with Qdrant for the following reasons:
- **Vector similarity search**: Efficient, scalable approximate nearest neighbors instead of exhaustive tensor matching.
- **Metadata payloads**: We can embed our Day 10 canonical provenance chunks natively into each point.
- **Filtering**: We can push metadata filters directly down into the database engine.
- **Workspace isolation**: Crucial for security, preventing cross-tenant data leakage by evaluating filters alongside vector distances.
- **Production-oriented storage**: Gives a deterministic foundation for scaling.

## Architecture
The retrieval pipeline is now logically isolated per workspace:
```
Question
→ workspace filter
→ Qdrant embedding retrieval 
→ BM25 (workspace filtered)
→ RRF (Top-10)
→ reasoning
→ human review
```

## Metadata
The vector database strictly stores each chunk's canonical properties in its point payload:
- `chunk_id`: A deterministic UUID identifier derived from the chunk ID text.
- `workspace_id`: Tenant ID scope (e.g. `ws_acme_corp`).
- `document_id`: Parent document ID.
- `title`: Document name.
- `version`: The document's version string.
- `excerpt`: The exact passage string.

## Workspace Isolation
Filtering occurs at the vector database query layer (using Qdrant's `models.Filter` structure) *before* vector similarities are returned. If we were to retrieve all workspaces first and filter them afterward in Python, we risk returning a Top-20 list heavily skewed by another tenant's documents, effectively masking legitimate results for the queried tenant (and risking an incomplete slice of their own data). Database-level filtering ensures we extract the absolute best matches *from within* the requested workspace boundary.

## Tests
Testing focused specifically on boundary controls via a new test suite (`test_day11.py`):
- **Positive workspace retrieval**: Assured that querying the `ws_globex_corp` space actively yielded Globex's synthetic test policy.
- **Cross-workspace isolation**: Queried the primary `ws_acme_corp` space for a concept exclusively known by Globex. Qdrant correctly retrieved zero relevant Globex points. The API correctly reported INSUFFICIENT EVIDENCE without leaking data.
- **No-result case**: Verified semantic similarity does not bypass workspace isolation logic.
- **Regression smoke tests**: Re-ran the Day 08 test suite on Acme. 100% of the hybrid retrieval rankings exactly matched expectations from previous days.

## Limitations
- **Authentication is not implemented**: There are no JWTs or user sessions. 
- **Authorization is not implemented**: The API trusts any `workspace_id` passed by the client.
- **Retrieval-Level Only**: This represents retrieval-level isolation, not a complete identity or role-based access control system.

## Next
Day 12: Cross-Encoder reranking of RRF Top-10 candidates.
