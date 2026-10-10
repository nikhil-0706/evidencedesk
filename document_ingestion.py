"""
document_ingestion.py — Multi-document upload and automatic ingestion for EvidenceDesk.

Supports:
- PDF (.pdf via PyMuPDF)
- DOCX (.docx via python-docx)
- TXT (.txt via UTF-8 text reader)

Preserves:
- Document identity & versioning
- Shared structure-aware chunking (chunking.py)
- Deterministic chunk IDs and ordered chunk_index
- Provenance metadata (workspace_id, document_id, title, version, page, section, source_type, filename)
- Qdrant dense vector search (384-dim all-MiniLM-L6-v2)
- BM25 retrieval
- Safe document deletion & re-indexing
- Downstream RRF (k=60), Cross-Encoder reranking, Bounded Adjacent-Chunk Expansion, SLM reasoning, and Human Review.
"""

import os
import io
import re
import hashlib
from collections import Counter
import uuid
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import docx  # python-docx
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models

from chunking import (
    extract_pdf_blocks,
    extract_docx_blocks,
    extract_txt_blocks,
    structure_aware_chunking_shared,
    conservative_normalize_text,
)
from expansion import expand_adjacent_chunks

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
    return extract_pdf_blocks(file_bytes, filename)


def extract_docx(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text and headings from a DOCX byte buffer using python-docx."""
    return extract_docx_blocks(file_bytes, filename)


def extract_txt(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text from a TXT byte buffer with UTF-8 encoding."""
    return extract_txt_blocks(file_bytes, filename)


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
    Generates deterministic chunk IDs and sequence ordering (chunk_index).
    """
    return structure_aware_chunking_shared(
        extracted_blocks=extracted_blocks,
        document_id=document_id,
        title=title,
        version=version,
        workspace_id=workspace_id,
        source_type=source_type,
        filename=filename,
    )


class MultiDocumentCorpus:
    """
    Manages multi-document ingestion (PDF, DOCX, TXT),
    BM25 indexing, Qdrant vector storage, safe re-indexing/deletion,
    and workspace-isolated retrieval.
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
        self._load_from_storage()

    def _load_from_storage(self):
        """Restore in-memory chunks, document metadata, and BM25 index from persistent Qdrant collection."""
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                return
            scroll_res = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            points = scroll_res[0] if scroll_res else []
            if not points:
                return

            restored_chunks = []
            for p in points:
                if p.payload:
                    restored_chunks.append(p.payload)

            restored_chunks.sort(key=lambda c: (
                c.get("workspace_id", ""),
                c.get("document_id", ""),
                c.get("version", ""),
                c.get("chunk_index", 0)
            ))
            self.chunks = restored_chunks

            # Restore document metadata
            doc_counts = Counter(c.get("document_id") for c in self.chunks if c.get("document_id"))
            for c in self.chunks:
                doc_id = c.get("document_id")
                ws_id = c.get("workspace_id")
                if doc_id and doc_id not in self.documents:
                    self.documents[doc_id] = {
                        "document_id": doc_id,
                        "workspace_id": ws_id,
                        "title": c.get("title", doc_id),
                        "version": c.get("version", "2026-01"),
                        "source_type": c.get("source_type", "pdf"),
                        "filename": c.get("filename", f"{doc_id}.pdf"),
                        "chunks_count": doc_counts.get(doc_id, 1),
                    }

            if self.chunks:
                tokenized_corpus = [_tokenize(c["excerpt"]) for c in self.chunks]
                self.bm25_model = BM25Okapi(tokenized_corpus)
        except Exception as exc:
            print(f"Warning: Could not restore corpus from Qdrant: {exc}")

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

    def delete_document(self, document_id: str, workspace_id: str) -> bool:
        """
        Safely remove obsolete or updated document chunks from Qdrant, BM25, and memory.
        Prevents stale vectors remaining searchable.
        """
        self._ensure_collection()
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="workspace_id",
                                match=models.MatchValue(value=workspace_id),
                            ),
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            ),
                        ]
                    )
                ),
            )
        except Exception as exc:
            print(f"Warning: Failed to delete Qdrant points for {document_id}: {exc}")

        # Clean up in-memory chunks
        self.chunks = [c for c in self.chunks if not (c.get("workspace_id") == workspace_id and c.get("document_id") == document_id)]
        
        # Remove from document dict and content hashes
        if document_id in self.documents:
            doc_info = self.documents.pop(document_id)
            c_hash = doc_info.get("content_hash")
            if c_hash and (workspace_id, c_hash) in self.content_hashes:
                self.content_hashes.pop((workspace_id, c_hash))

        # Rebuild BM25 index
        if self.chunks:
            tokenized_corpus = [_tokenize(c["excerpt"]) for c in self.chunks]
            self.bm25_model = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_model = None

        return True

    def ingest_single_document(
        self,
        filename: str,
        file_bytes: bytes,
        workspace_id: str = "ws_acme_corp",
        version: str = "2026-01",
        title: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a single uploaded file (PDF, DOCX, TXT).
        Handles validation, duplicate detection, text extraction, shared structure-aware chunking,
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

        if dup_key in self.content_hashes and not overwrite:
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

        # If overwriting existing document, remove stale vectors first
        if overwrite or document_id in self.documents:
            self.delete_document(document_id, workspace_id)

        try:
            if source_type == "pdf":
                blocks = extract_pdf_blocks(file_bytes, filename)
            elif source_type == "docx":
                blocks = extract_docx_blocks(file_bytes, filename)
            elif source_type == "txt":
                blocks = extract_txt_blocks(file_bytes, filename)
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

        new_chunks = structure_aware_chunking_shared(
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

    def review_documents(
        self,
        question: str,
        workspace_id: str = "ws_acme_corp",
        enable_expansion: bool = True,
    ) -> Dict[str, Any]:
        """Execute RRF -> Cross-Encoder -> Bounded Expansion -> SLM -> Human Review pipeline on uploaded document corpus."""
        from evidencedesk import rrf_fuse, rerank_evidence, analyze_evidence
        bm25_res = self.retrieve_bm25(question, workspace_id, top_k=20)
        emb_res = self.retrieve_embedding(question, workspace_id, top_k=20)
        rrf_cand = rrf_fuse(bm25_res, emb_res, k=60, top_k=10)
        evidence = rerank_evidence(question, rrf_cand, top_k=5)

        if enable_expansion and self.chunks:
            workspace_chunks = [c for c in self.chunks if c.get("workspace_id") == workspace_id]
            evidence = expand_adjacent_chunks(
                candidate_chunks=evidence,
                all_corpus_chunks=workspace_chunks,
                max_seeds=3,
                max_neighbors_per_seed=1,
                max_total_expanded=5,
            )

        analysis = analyze_evidence(question, evidence)

        return {
            "question": question,
            "workspace_id": workspace_id,
            "status": analysis.get("status", "INSUFFICIENT_EVIDENCE"),
            "candidate_excerpt": evidence[0]["excerpt"] if evidence else None,
            "evidence": evidence,
            "mode": "multi_document_hybrid_rrf_reranked_expanded_reasoning",
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
