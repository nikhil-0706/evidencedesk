# Day 19.5 Evaluation: Real PDF Ingestion & Realistic Corpus Validation

## Executive Summary

Day 19.5 implemented a realistic PDF document ingestion pipeline (`pdf_ingestion.py`), extending EvidenceDesk beyond synthetic evidence into real multi-page document processing with PyMuPDF text extraction, structure-aware chunking, rich provenance metadata (`page`, `section`, `source_type`), and deterministic chunk identity.

The downstream frozen pipeline (BM25, Qdrant, RRF k=60, Cross-Encoder reranking, SLM reasoning, Human Review Workspace, and workspace isolation) was preserved 100% intact. All **18 mandatory Day 19.5 tests passed successfully**.

---

## Controlled Benchmark vs. Realistic PDF Validation

> [!NOTE]
> **Corpus Separation & Evaluation Design**
> 
> - **Benchmark A (Synthetic Corpus)**: The original Q01–Q25 benchmark remains completely untouched as the controlled evaluation baseline.
> - **Benchmark B (Realistic PDF Corpus)**: `data/acme_security_policy.pdf` (`SEC-REAL-001`, Version `2026-01`) serves as the realistic multi-page document validation layer.

---

## Test Execution Matrix (`test_day19_5.py`)

| Test ID | PDF Ingestion & Pipeline Requirement | Result | Verified Target |
| :---: | :--- | :---: | :--- |
| **TEST 1** | PDF loaded successfully | **PASS** | `data/acme_security_policy.pdf` exists and loads |
| **TEST 2** | Text extraction succeeds | **PASS** | PyMuPDF page-by-page text extraction succeeded |
| **TEST 3** | Multiple chunks generated | **PASS** | Structure-aware chunking generated 12 policy chunks |
| **TEST 4** | Chunks preserve page info | **PASS** | `page` metadata integer (1-based) attached |
| **TEST 5** | Chunks preserve section info | **PASS** | Section headings preserved per chunk |
| **TEST 6** | Chunk IDs are deterministic | **PASS** | SHA-256 stable chunk ID generation verified |
| **TEST 7** | Every chunk has `workspace_id` | **PASS** | `ws_acme_corp` tagged on all chunks |
| **TEST 8** | Every chunk has `document_id` | **PASS** | `SEC-REAL-001` tagged on all chunks |
| **TEST 9** | Every chunk has `version` | **PASS** | `2026-01` tagged on all chunks |
| **TEST 10** | Every chunk has `excerpt` | **PASS** | Clean, non-empty excerpt text |
| **TEST 11** | BM25 retrieves PDF evidence | **PASS** | BM25Okapi successfully indexes & retrieves PDF chunks |
| **TEST 12** | Dense retrieval retrieves PDF evidence | **PASS** | Qdrant `all-MiniLM-L6-v2` vectors match PDF queries |
| **TEST 13** | RRF receives BM25 + dense candidates | **PASS** | RRF k=60 merges PDF BM25 and dense sets |
| **TEST 14** | Cross-Encoder reranking works on PDF | **PASS** | `BAAI/bge-reranker-large` scores PDF chunks |
| **TEST 15** | SLM receives final PDF evidence | **PASS** | Local 3B SLM reasons on Top-5 PDF chunks |
| **TEST 16** | Review payload displays PDF provenance | **PASS** | `page`, `section`, `source_type` present in review payload |
| **TEST 17** | Page/section metadata survives | **PASS** | Complete metadata survives end-to-end to UI |
| **TEST 18** | Workspace isolation works on PDF | **PASS** | Acme PDF queries exclude Globex evidence |

---

## Provenance Trace

```
PDF Document (data/acme_security_policy.pdf, Page 2, Section 4)
             ↓
Deterministic Chunk (chk_pdf_secreal001_p02_e4f1a2b9)
             ↓
Dense Embedding (all-MiniLM-L6-v2, 384d) + BM25 Index
             ↓
Qdrant Payload (workspace_id: ws_acme_corp, page: 2, section: Section 4)
             ↓
RRF Fusion (k=60) -> Cross-Encoder Reranking (BAAI/bge-reranker-large)
             ↓
Local SLM Reasoning (Qwen 2.5 3B)
             ↓
Human Review Workspace UI (Displays Page 2, Section 4, Document ID, Version)
```
