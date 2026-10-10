# EvidenceDesk Architectural Report — Chunking & Retrieval Engineering

## 1. Baseline System Assessment & Failure Analysis

### 1.1 End-to-End Ingestion and Retrieval Flow (Pre-Fix)
Prior to this architectural upgrade, EvidenceDesk had fragmented document processing paths:
1. **Synthetic Corpus**: Synthetic documents in `evidencedesk.py` were split by raw regex sentence matching `(?<=[.!?])\s+`. Chunks were assigned `chk_{doc_id}_{idx:03d}` IDs without sequence ordering metadata (`chunk_index`).
2. **PDF Ingestion**: `pdf_ingestion.py` extracted text page-by-page and chunked line-by-line using sentence splitting, adding `chk_pdf_...` IDs.
3. **Multi-Document Upload**: `document_ingestion.py` handled PDF, DOCX, and TXT using distinct extraction functions, but applied separate single-sentence splitting into `evidencedesk_documents` Qdrant collection.

### 1.2 Chunking Limitations & Evidence Separation Risks
- **Single-Sentence Fragmentation**: Splitting strictly on single sentence boundaries fragmented questions from answers, table headers from table rows, and policy statements from their qualifications.
- **Lost Context**: When a questionnaire or document had a question on line 1 and answer on line 2, sentence chunking placed them into separate chunks. The answer chunk lacked question context/control IDs, while the question chunk lacked the answer.
- **Omitted Neighboring Chunks**: Standard top-k retrieval (BM25 or dense embeddings) fetched individual isolated chunks. If an answer's qualifying clause (e.g. "...unless authorized by CISO") sat in chunk `i+1`, top-k search often missed chunk `i+1`.
- **Stale Vector & Data Integrity Risks**: Ingestion lacked atomic document deletion/re-indexing mechanisms. Updating a file created a new SHA256 document ID while leaving old chunks in Qdrant and in-memory lists, creating stale vector pollution.

---

## 2. Architectural Upgrades Implemented

### 2.1 Shared, Structure-Aware Chunking Engine (`chunking.py`)
Created a unified, format-agnostic chunking engine used by all PDF, DOCX, TXT, and synthetic ingestion paths:
- **Format Adapters**:
  - `PDF`: Page-by-page PyMuPDF extraction preserving reading order and page numbers.
  - `DOCX`: Paragraph, heading, and structured table cell extraction using `python-docx`.
  - `TXT`: Structural paragraph, heading, list, and key-value record parsing.
- **Splitting Preference Hierarchy**:
  1. Complete Question & Answer records / pairs.
  2. Key-Value pairs and table rows.
  3. Sections and associated paragraphs.
  4. Sentence boundaries.
  5. Recursive window fallback (token/character limits with configurable overlap).
- **Conservative Text Normalization**:
  Preserves exact textual fidelity. Crucially, meaningful qualifiers such as `NOT`, `no`, `never`, `unless`, control IDs (`SEC-256`), and dates are preserved without destructive stop-word filtering.
- **Sequence Provenance**:
  Every chunk receives an explicit, zero-indexed `chunk_index` integer, enabling deterministic neighbor resolution.

### 2.2 Carefully Bounded Adjacent-Chunk Expansion (`expansion.py`)
Integrated adjacent-chunk expansion directly into the retrieval pipeline:
- **Safe Expansion Policy**:
  After initial BM25 + Dense + RRF + Cross-Encoder reranking identifies top candidate seeds, expansion fetches adjacent preceding (`chunk_index - 1`) and following (`chunk_index + 1`) neighbor chunks.
- **Strict Boundary Enforcements**:
  - `neighbor.workspace_id == seed.workspace_id`
  - `neighbor.document_id == seed.document_id`
  - `neighbor.version == seed.version`
  - **Zero Cross-Boundary Expansion**: Never crosses workspace, document, or version boundaries.
- **Ranking Semantics & Traceability**:
  Neighbor chunks are explicitly flagged (`is_expanded = True`, `expansion_type = "neighbor"`). Retrieval scores (`reranker_score`, `bm25_score`) are NOT faked or copied from seeds, preserving true score integrity.
- **Budgeting & Deduplication**:
  Enforces `max_seeds` (3), `max_neighbors_per_seed` (1), `max_total_expanded` (5), and text deduplication.

### 2.3 Safe Re-Indexing & Data Integrity (`document_ingestion.py`)
Added `delete_document(document_id, workspace_id)` and `overwrite=True` re-indexing:
- Deletes obsolete Qdrant points using workspace and document filter selectors.
- Purges in-memory chunk lists and content hash caches.
- Rebuilds BM25 indices atomically to prevent stale vector leakage.

---

## 3. Automated Test Suite & Verification Results

### 3.1 Unit & Integration Test Suite (`test_day21_chunking_expansion.py`)
Verified 20 automated test cases covering all prompt requirements:
1. **Question & Direct Answer**: Q&A pairs retained in unified chunk.
2. **Q&A Across Page Boundaries**: Metadata preserved across pages without invalid cross-page block merging.
3. **Headings & Paragraphs**: Section titles attached to child paragraphs.
4. **Multi-Row Tables**: DOCX table rows extracted cleanly.
5. **Recursive Splitting**: Long paragraphs split safely under character limits with overlap.
6. **TXT Key-Value Records**: `Key: Value` records preserved.
7. **DOCX Headings & Tables**: Heading levels and table contents extracted.
8. **PDF Reading Order**: PyMuPDF block ordering preserved across pages.
9. **Empty File Handling**: Graceful rejection of empty pages/files.
10. **Negation & Qualification Preservation**: Words like `NOT` and `unless` retained.
11. **Full Provenance**: `document_id`, `version`, `workspace_id`, `page`, `section`, `chunk_index` verified.
12. **No Silent Truncation**: Text length verified before and after chunking.
13. **Adjacent Neighbor Expansion**: Preceding and following neighbors successfully recovered.
14. **Strict Boundary Rules**: Verified zero expansion across workspaces, documents, or versions.
15. **Deduplication & Budgeting**: Overlapping text deduplicated, context limits enforced.
16. **Ranking Semantics**: Neighbor chunks clearly distinguished without score fabrication.

Result: **20/20 PASSED**

### 3.2 Regression Test Suite Execution
Ran full test suite (`run_all_tests.py`):
- `test_day08.py` - PASSED
- `test_day08_fix.py` - PASSED
- `test_day11.py` - PASSED
- `test_day12.py` - PASSED
- `test_day12_bugfix.py` - PASSED
- `test_metadata.py` - PASSED
- `test_rrf.py` - PASSED
- `test_day16.py` - PASSED
- `test_day17.py` - PASSED
- `test_day18.py` - PASSED
- `test_day19.py` - PASSED
- `test_day19_5.py` - PASSED
- `test_day19_75.py` - PASSED
- `test_day20.py` - PASSED
- `test_day21_chunking_expansion.py` - PASSED

Result: **ALL REGRESSION TESTS PASSED SUCCESSFULLY!**
