"""
document_ingestion.py — Multi-document upload and automatic ingestion for EvidenceDesk.

Supports:
- PDF (.pdf via PyMuPDF)
- DOCX (.docx via python-docx)
- TXT (.txt via UTF-8 text reader)

Preserves:
- Document identity & versioning
- Structure-aware chunking
- Deterministic chunk IDs
- Provenance metadata (workspace_id, document_id, title, version, page, section, source_type, filename)
- Qdrant dense vector search (384-dim all-MiniLM-L6-v2)
- BM25 retrieval
- Downstream RRF (k=60), Cross-Encoder reranking, SLM reasoning, and Human Review.
"""

import os
import io
import re
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import docx  # python-docx
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models

MAX_UPLOAD_MB = 10
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

STOP_WORDS = set("a an the is are do does what which how who your you we our for to of in at and or with required".split())

def _tokenize(text: str) -> List[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]



def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file byte content."""
    return hashlib.sha256(content).hexdigest()


def generate_document_id(workspace_id: str, filename: str, content_hash: str) -> str:
    """Generate a stable, deterministic document_id based on workspace, filename, and content hash."""
    raw = f"{workspace_id}_{filename.lower()}_{content_hash}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8].upper()
    return f"DOC-UPL-{digest}"


def extract_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text and line structure from a PDF byte buffer using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid PDF file: {exc}")

    if doc.page_count == 0:
        raise ValueError(f"PDF is empty: {filename}")

    pages_data = []
    total_text_len = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        total_text_len += len(text.strip())
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pages_data.append({
            "page": page_num,
            "text": text,
            "lines": lines,
        })

    if total_text_len == 0:
        raise ValueError(f"PDF contains no extractable text: {filename}")

    return pages_data


def extract_docx(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text and headings from a DOCX byte buffer using python-docx."""
    try:
        file_stream = io.BytesIO(file_bytes)
        doc = docx.Document(file_stream)
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid DOCX file: {exc}")

    paragraphs_data = []
    total_text_len = 0
    current_section = "General"

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        total_text_len += len(text)
        style_name = para.style.name if para.style else ""

        # Heading detection heuristic
        if "Heading" in style_name or text.startswith("Section ") or (text.isupper() and len(text) < 60):
            current_section = text
            continue

        paragraphs_data.append({
            "page": None,
            "section": current_section,
            "text": text,
        })

    if total_text_len == 0:
        raise ValueError(f"DOCX contains no extractable text: {filename}")

    return paragraphs_data


def extract_txt(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text from a TXT byte buffer with UTF-8 encoding."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except Exception as exc:
            raise ValueError(f"Invalid text encoding for TXT file: {exc}")

    if not text.strip():
        raise ValueError(f"TXT file is empty: {filename}")

    # Split into logical paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [p.strip() for p in text.splitlines() if p.strip()]

    txt_data = []
    current_section = "General"

    for p in raw_paragraphs:
        if p.startswith("Section ") or p.startswith("Title:") or (p.isupper() and len(p) < 60):
            current_section = p
            continue

        txt_data.append({
            "page": None,
            "section": current_section,
            "text": p,
        })

    if not txt_data:
        raise ValueError(f"TXT file contains no valid content: {filename}")

    return txt_data


def structure_aware_chunking_multi(
    extracted_blocks: List[Dict[str, Any]],
    document_id: str,
    title: str,
    version: str,
    workspace_id: str,
    source_type: str,
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Common structure-aware chunker for PDF, DOCX, and TXT documents.
    Generates deterministic chunk IDs.
    """
    chunks = []
    doc_id_clean = document_id.lower().replace("-", "")

    for idx, block in enumerate(extracted_blocks, start=1):
        page = block.get("page")
        section = block.get("section", "General")
        text = block.get("text", "")

        # Split text into sentences for sentence/paragraph-level chunks
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text.strip()]

        for s_idx, sentence in enumerate(sentences, start=1):
            clean_excerpt = " ".join(sentence.split())
            if not clean_excerpt:
                continue

            # Skip standalone short title/header repetitions
            if len(clean_excerpt) < 15 and ("Section" in clean_excerpt or "Policy" in clean_excerpt):
                continue

            page_str = f"p{page:02d}" if page is not None else "p0"
            hash_input = f"{document_id}_{version}_{page_str}_{section}_{clean_excerpt}_{idx}_{s_idx}"
            digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
            chunk_id = f"chk_{source_type}_{doc_id_clean}_{page_str}_{digest}"

            chunks.append({
                "chunk_id": chunk_id,
                "workspace_id": workspace_id,
                "document_id": document_id,
                "title": title,
                "version": version,
                "page": page,
                "section": section,
                "source_type": source_type,
                "filename": filename,
                "excerpt": clean_excerpt,
            })

    return chunks


class MultiDocumentCorpus:
    """
    Manages multi-document ingestion (PDF, DOCX, TXT),
    BM25 indexing, Qdrant vector storage, and workspace-isolated retrieval.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embed_model,
        collection_name: str = "evidencedesk_documents"
    ):
        self.qdrant_client = qdrant_client
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.chunks: List[Dict[str, Any]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> doc metadata
        self.content_hashes: Dict[Tuple[str, str], str] = {}  # (workspace_id, hash) -> doc_id
        self.bm25_model: Optional[BM25Okapi] = None

    def _ensure_collection(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )

    def validate_file(self, filename: str, file_bytes: bytes) -> Optional[str]:
        """Validate file size and extension."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return f"Unsupported file type '{ext}'. Supported formats: PDF, DOCX, TXT."

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            return f"File size ({size_mb:.2f} MB) exceeds maximum limit of {MAX_UPLOAD_MB} MB."

        if len(file_bytes) == 0:
            return "File is empty (0 bytes)."

        return None

    def ingest_single_document(
        self,
        filename: str,
        file_bytes: bytes,
        workspace_id: str = "ws_acme_corp",
        version: str = "2026-01",
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single uploaded file (PDF, DOCX, TXT).
        Handles validation, duplicate detection, text extraction, chunking,
        embedding generation, and Qdrant + BM25 indexing.
        """

        validation_err = self.validate_file(filename, file_bytes)
        if validation_err:
            return {
                "filename": filename,
                "status": "FAILED",
                "reason": validation_err,
                "chunks_created": 0,
            }

        content_hash = compute_content_hash(file_bytes)
        dup_key = (workspace_id, content_hash)

        if dup_key in self.content_hashes:
            existing_doc_id = self.content_hashes[dup_key]
            return {
                "filename": filename,
                "status": "DUPLICATE",
                "reason": f"Document already indexed as {existing_doc_id}.",
                "document_id": existing_doc_id,
                "chunks_created": 0,
            }

        ext = os.path.splitext(filename)[1].lower()
        source_type = ext.lstrip(".")
        doc_title = title or os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        document_id = generate_document_id(workspace_id, filename, content_hash)

        try:
            if source_type == "pdf":
                blocks = extract_pdf(file_bytes, filename)
            elif source_type == "docx":
                blocks = extract_docx(file_bytes, filename)
            elif source_type == "txt":
                blocks = extract_txt(file_bytes, filename)
            else:
                return {
                    "filename": filename,
                    "status": "FAILED",
                    "reason": f"Unsupported extension: {ext}",
                    "chunks_created": 0,
                }
        except Exception as exc:
            return {
                "filename": filename,
                "status": "FAILED",
                "reason": f"Text extraction failed: {exc}",
                "chunks_created": 0,
            }

        new_chunks = structure_aware_chunking_multi(
            extracted_blocks=blocks,
            document_id=document_id,
            title=doc_title,
            version=version,
            workspace_id=workspace_id,
            source_type=source_type,
            filename=filename,
        )

        if not new_chunks:
            return {
                "filename": filename,
                "status": "FAILED",
                "reason": "No valid chunk excerpts generated from document.",
                "chunks_created": 0,
            }

        # Store metadata
        self.documents[document_id] = {
            "document_id": document_id,
            "workspace_id": workspace_id,
            "title": doc_title,
            "version": version,
            "source_type": source_type,
            "filename": filename,
            "content_hash": content_hash,
            "chunks_count": len(new_chunks),
        }
        self.content_hashes[dup_key] = document_id
        self.chunks.extend(new_chunks)

        # Rebuild BM25 index
        tokenized_corpus = [_tokenize(c["excerpt"]) for c in self.chunks]
        self.bm25_model = BM25Okapi(tokenized_corpus)

        # Generate embeddings and index into Qdrant
        self._ensure_collection()
        points = []
        for chunk in new_chunks:
            vector = self.embed_model.encode(chunk["excerpt"]).tolist()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk,
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return {
            "filename": filename,
            "status": "SUCCESS",
            "document_id": document_id,
            "title": doc_title,
            "version": version,
            "workspace_id": workspace_id,
            "source_type": source_type,
            "chunks_created": len(new_chunks),
            "message": f"Successfully indexed {len(new_chunks)} chunks.",
        }

    def ingest_multiple_documents(
        self,
        files: List[Tuple[str, bytes]],
        workspace_id: str = "ws_acme_corp",
        version: str = "2026-01",
    ) -> Dict[str, Any]:
        """Ingest multiple files in a single request."""
        results = []
        successful_docs = 0
        total_chunks = 0

        for filename, file_bytes in files:
            res = self.ingest_single_document(
                filename=filename,
                file_bytes=file_bytes,
                workspace_id=workspace_id,
                version=version,
            )
            results.append(res)
            if res["status"] == "SUCCESS":
                successful_docs += 1
                total_chunks += res.get("chunks_created", 0)

        return {
            "workspace_id": workspace_id,
            "version": version,
            "total_files": len(files),
            "successful_documents": successful_docs,
            "total_chunks_added": total_chunks,
            "results": results,
        }

    def retrieve_bm25(self, question: str, workspace_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks using BM25 with strict workspace filtering."""
        if not self.bm25_model or not self.chunks:
            return []
        query_toks = _tokenize(question)
        scores = self.bm25_model.get_scores(query_toks)
        ranked = []
        for i, score in enumerate(scores):
            if score > 0:
                chunk = self.chunks[i]
                if chunk["workspace_id"] == workspace_id:
                    c = chunk.copy()
                    c["bm25_score"] = round(float(score), 4)
                    ranked.append(c)
        return sorted(ranked, key=lambda x: x["bm25_score"], reverse=True)[:top_k]

    def retrieve_embedding(self, question: str, workspace_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks using Qdrant dense vector search with workspace filter."""
        self._ensure_collection()
        query_vector = self.embed_model.encode(question).tolist()
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id",
                        match=models.MatchValue(value=workspace_id),
                    )
                ]
            ),
            limit=top_k,
        )
        ranked = []
        for hit in search_result.points:
            chunk = hit.payload.copy()
            chunk["similarity_score"] = round(float(hit.score), 4)
            ranked.append(chunk)
        return ranked

    def review_documents(self, question: str, workspace_id: str = "ws_acme_corp") -> Dict[str, Any]:
        """Execute RRF -> Cross-Encoder -> SLM -> Human Review pipeline on uploaded document corpus."""
        from evidencedesk import rrf_fuse, rerank_evidence, analyze_evidence
        bm25_res = self.retrieve_bm25(question, workspace_id, top_k=20)
        emb_res = self.retrieve_embedding(question, workspace_id, top_k=20)
        rrf_cand = rrf_fuse(bm25_res, emb_res, k=60, top_k=10)
        evidence = rerank_evidence(question, rrf_cand, top_k=5)
        analysis = analyze_evidence(question, evidence)

        return {
            "question": question,
            "workspace_id": workspace_id,
            "status": analysis.get("status", "INSUFFICIENT_EVIDENCE"),
            "candidate_excerpt": evidence[0]["excerpt"] if evidence else None,
            "evidence": evidence,
            "mode": "multi_document_hybrid_rrf_reranked_reasoning",
            "reason": analysis.get("reason", ""),
            "evidence_chunk_ids": analysis.get("evidence_chunk_ids", []),
            "evidence_quote": analysis.get("evidence_quote", ""),
            "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
        }

    def list_documents(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List indexed documents."""
        docs = list(self.documents.values())
        if workspace_id:
            docs = [d for d in docs if d["workspace_id"] == workspace_id]
        return docs

