# Day 19.75 Devlog: Multi-Document Upload & Automatic Document Ingestion Workflow

## 1. Objective

Day 19.75 converts EvidenceDesk's fixed-document ingestion into a **User-Provided Multi-Document Ingestion Workflow**.

Users and reviewers can upload multiple policy documents in **PDF**, **DOCX**, or **TXT** formats through the EvidenceDesk UI or REST API (`/upload`). Uploaded documents are automatically extracted, chunked using structure-aware rules, embedded with 384-dimensional `all-MiniLM-L6-v2` vectors, indexed into Qdrant and BM25, and made immediately searchable by the frozen downstream RRF, Cross-Encoder reranking, SLM reasoning, and Human Review pipeline.

---

## 2. Supported Formats & Extraction Methods

| Format | Extension | Library | Extracted Structural Metadata | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **PDF** | `.pdf` | PyMuPDF (`fitz`) | `page` (int), `section` (heading string), `lines` | Page-aware structure preservation |
| **DOCX** | `.docx` | `python-docx` | `section` (heading string), paragraph blocks | `page`: `None` (pages not fixed in DOCX) |
| **TXT** | `.txt` | Native UTF-8 | `section` ("General" / section titles), paragraph blocks | `page`: `None` |

*Unsupported formats (e.g. `.xlsx`, `.pptx`, `.exe`, `.zip`, `.doc`) are cleanly rejected with a per-file status report without interrupting valid files in the same batch.*

---

## 3. Ingestion Architecture & Common Document Model

### Workflow:
$$\text{USER UPLOADS FILES} \rightarrow \text{VALIDATION} \rightarrow \text{TEXT EXTRACTION} \rightarrow \text{STRUCTURE-AWARE CHUNKING} \rightarrow \text{EMBEDDINGS} \rightarrow \text{QDRANT + BM25 INDEX} \rightarrow \text{RRF} \rightarrow \text{CROSS-ENCODER} \rightarrow \text{SLM} \rightarrow \text{HUMAN REVIEW}$$

### Common Document Model:
```json
{
  "document_id": "DOC-UPL-A1B2C3D4",
  "workspace_id": "ws_acme_corp",
  "title": "Incident Response Policy",
  "version": "2026-01",
  "source_type": "docx",
  "filename": "incident_response.docx",
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

### Chunk Model with Rich Provenance:
```json
{
  "chunk_id": "chk_docx_docupla1b2c3d4_p0_7f8a9b0c",
  "workspace_id": "ws_acme_corp",
  "document_id": "DOC-UPL-A1B2C3D4",
  "title": "Incident Response Policy",
  "version": "2026-01",
  "page": null,
  "section": "Section 1: Incident Triage Policy",
  "source_type": "docx",
  "filename": "incident_response.docx",
  "excerpt": "Security incidents are triaged by the senior security response team within 15 minutes."
}
```

---

## 4. Key Capabilities & Safety Features

1. **Deterministic Document & Chunk IDs**:
   - `document_id`: Generated via SHA-256 hash of `(workspace_id, filename, content_hash)`.
   - `chunk_id`: Generated via SHA-256 hash of `(document_id, version, page, section, text, index)`.
2. **Duplicate Detection**:
   - Duplicate uploads of the exact same content within the same workspace are detected via content hash and skipped without creating duplicate chunks.
3. **Multi-File Batch Upload**:
   - Single HTTP multipart request (`POST /upload`) supports uploading multiple files simultaneously. Processing is per-file so failure of one corrupted file does not abort valid files.
4. **Strict Workspace Isolation**:
   - Every chunk inherits `workspace_id`. Qdrant filters and BM25 retrievers enforce strict `workspace_id` scoping in backend queries. Documents uploaded to `ws_acme_corp` can never leak into `ws_globex_corp`.
5. **File Size & Safety Limits**:
   - `MAX_UPLOAD_MB = 10` MB per file. Uploaded files are parsed as static data and never executed.

---

## 5. Downstream ML Preservation Guarantee

The ML and reasoning architecture remains **100% FROZEN and UNCHANGED**:

- **BM25 Retrieval Algorithm**: UNCHANGED
- **Embedding Model**: UNCHANGED (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions)
- **Qdrant Vector DB & Filter Semantics**: UNCHANGED
- **RRF Fusion ($k=60$)**: UNCHANGED
- **Cross-Encoder Reranker**: UNCHANGED (`BAAI/bge-reranker-large`)
- **SLM Model & Prompt**: UNCHANGED (`Qwen 2.5 3B`, JSON mode)
- **Pydantic Decision Schema**: UNCHANGED (`EvidenceDecision`)
- **Human Review Governance**: UNCHANGED (Approve / Edit Response / Reject)
- **Synthetic Q01–Q25 Benchmark**: UNCHANGED (Benchmark A)

---

## 6. Verification & Test Results

### 1. Day 19.75 Test Suite (`test_day19_75.py`)
27/27 tests PASSED:
- **TEST 1**: Upload one PDF $\rightarrow$ SUCCESS
- **TEST 2**: Upload one DOCX $\rightarrow$ SUCCESS
- **TEST 3**: Upload one TXT $\rightarrow$ SUCCESS
- **TEST 4**: Upload multiple files in one request $\rightarrow$ SUCCESS
- **TEST 5**: Distinct document IDs generated $\rightarrow$ SUCCESS
- **TEST 6–8**: Workspace ID, Document ID, Version inherited by all chunks $\rightarrow$ SUCCESS
- **TEST 9–11**: PDF page metadata, DOCX heading section metadata, TXT blocks preserved $\rightarrow$ SUCCESS
- **TEST 12**: Chunk IDs are deterministic $\rightarrow$ SUCCESS
- **TEST 13**: Duplicate document detection works $\rightarrow$ SUCCESS
- **TEST 14**: Unsupported file type rejected $\rightarrow$ SUCCESS
- **TEST 15**: Empty file handled safely $\rightarrow$ SUCCESS
- **TEST 16–17**: 384-dim embeddings generated and indexed into Qdrant $\rightarrow$ SUCCESS
- **TEST 18–21**: BM25, Dense retrieval, RRF k=60, and Cross-Encoder reranking operational $\rightarrow$ SUCCESS
- **TEST 22–23**: SLM reasoning and Human Review provenance payload operational $\rightarrow$ SUCCESS
- **TEST 24–25**: Strict workspace isolation verified (Acme documents never leak to Globex) $\rightarrow$ SUCCESS
- **TEST 26**: Multi-document evidence retrieval working $\rightarrow$ SUCCESS
- **TEST 27**: Conflicting evidence from two uploaded documents preserved $\rightarrow$ SUCCESS

### 2. Full Regression Suite (`run_all_tests.py`)
All regression tests passed:
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

---

## 7. Known Limitations

1. **Scanned PDFs**: Scanned image-only PDFs require OCR (e.g. Tesseract) before text extraction.
2. **DOCX Page Numbers**: Page numbers are omitted (`null`) for DOCX files because DOCX format relies on dynamic renderer pagination.
3. **Legacy `.doc` Files**: Binary `.doc` (Word 97-2003) files are not supported; convert to `.docx` before uploading.
4. **Local Qwen 2.5 3B SLM**: Requires local Ollama instance running `qwen2.5:3b` for SLM evidence classification.

---

## 8. Statement of Readiness

"EvidenceDesk supports user-provided PDF, DOCX, and TXT documents. Uploaded documents are converted into provenance-preserving chunks and indexed using the same retrieval and reasoning pipeline used by the controlled corpus."
