"""
pdf_ingestion.py — Structure-aware PDF ingestion, chunking, and indexing for EvidenceDesk.

Day 19.5 / Day 21 Architecture:
REAL PDF -> TEXT EXTRACTION (PyMuPDF) -> SHARED STRUCTURE-AWARE CHUNKING -> RICH PROVENANCE
         -> BM25 INDEX + QDRANT VECTORS -> RRF k=60 -> CROSS-ENCODER -> BOUNDED EXPANSION -> SLM -> HUMAN REVIEW

Preserves:
- Downstream retrieval, RRF, Cross-Encoder, SLM reasoning, and Human Review.
- Synthetic Q01-Q25 benchmark (Benchmark A) is kept untouched.
- PDF corpus operates as a realistic document validation layer (Benchmark B).
"""

import os
import re
import fitz  # PyMuPDF
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from chunking import (
    extract_pdf_blocks,
    structure_aware_chunking_shared,
    conservative_normalize_text,
)
from expansion import expand_adjacent_chunks

from evidencedesk import (
    embed_model,
    qdrant,
    rrf_fuse,
    rerank_evidence,
    analyze_evidence,
    tokens,
    STOP,
)


def extract_pdf_structure(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract page text and line structure from a PDF using PyMuPDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    return extract_pdf_blocks(file_bytes, os.path.basename(pdf_path))


def structure_aware_chunking(
    pages_data: List[Dict[str, Any]],
    document_id: str = "SEC-REAL-001",
    title: str = "Acme Enterprise Security & Compliance Policy",
    version: str = "2026-01",
    workspace_id: str = "ws_acme_corp",
) -> List[Dict[str, Any]]:
    """
    Structure-aware chunking priority:
    1. Section boundaries
    2. Paragraph & policy statement boundaries
    3. Controlled window size limit
    Generates deterministic chunk_id values and ordered chunk_index.
    """
    return structure_aware_chunking_shared(
        extracted_blocks=pages_data,
        document_id=document_id,
        title=title,
        version=version,
        workspace_id=workspace_id,
        source_type="pdf",
        filename=f"{document_id}.pdf",
    )


class PDFEvidenceCorpus:
    """Manages PDF chunk indexing, BM25 retrieval, Qdrant dense vector search, and adjacent expansion."""

    def __init__(self, collection_name: str = "evidencedesk_pdf"):
        self.collection_name = collection_name
        self.chunks: List[Dict[str, Any]] = []
        self.bm25_model: Optional[BM25Okapi] = None

        if not qdrant.collection_exists(self.collection_name):
            qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )

    def ingest_pdf(
        self,
        pdf_path: str,
        document_id: str = "SEC-REAL-001",
        title: str = "Acme Enterprise Security & Compliance Policy",
        version: str = "2026-01",
        workspace_id: str = "ws_acme_corp",
    ) -> List[Dict[str, Any]]:
        """Ingest PDF, chunk, generate embeddings, and populate Qdrant & BM25."""
        pages_data = extract_pdf_structure(pdf_path)
        new_chunks = structure_aware_chunking(
            pages_data,
            document_id=document_id,
            title=title,
            version=version,
            workspace_id=workspace_id,
        )

        self.chunks.extend(new_chunks)

        # Build BM25 index for PDF chunks
        tokenized_chunks = [tokens(c["excerpt"]) for c in self.chunks]
        self.bm25_model = BM25Okapi(tokenized_chunks)

        # Index in Qdrant
        points = []
        for chunk in new_chunks:
            vector = embed_model.encode(chunk["excerpt"]).tolist()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk,
                )
            )

        qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return new_chunks

    def retrieve_bm25(self, question: str, workspace_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.bm25_model or not self.chunks:
            return []
        query_toks = tokens(question)
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
        query_vector = embed_model.encode(question).tolist()
        search_result = qdrant.query_points(
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

    def review_pdf(
        self,
        question: str,
        workspace_id: str = "ws_acme_corp",
        enable_expansion: bool = True,
    ) -> Dict[str, Any]:
        """Execute complete pipeline on PDF corpus with optional adjacent chunk expansion."""
        bm25_res = self.retrieve_bm25(question, workspace_id, top_k=20)
        emb_res = self.retrieve_embedding(question, workspace_id, top_k=20)
        rrf_cand = rrf_fuse(bm25_res, emb_res, k=60, top_k=10)
        evidence = rerank_evidence(question, rrf_cand, top_k=5)

        if enable_expansion and self.chunks:
            evidence = expand_adjacent_chunks(
                candidate_chunks=evidence,
                all_corpus_chunks=self.chunks,
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
            "mode": "pdf_hybrid_rrf_reranked_expanded_reasoning",
            "reason": analysis.get("reason", ""),
            "evidence_chunk_ids": analysis.get("evidence_chunk_ids", []),
            "evidence_quote": analysis.get("evidence_quote", ""),
            "warning": "Matching text is not proof that a question is answered. A human must review it. No AI answer was generated.",
        }
