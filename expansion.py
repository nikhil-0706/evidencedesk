"""
expansion.py — Bounded Adjacent-Chunk Expansion for EvidenceDesk Retrieval.

Enforces strict boundary isolation:
1. Never cross workspace_id boundaries.
2. Never cross document_id boundaries.
3. Never cross document version boundaries.
4. Enforces configurable seed limits, max neighbors, total expansion budget, and text deduplication.
5. Preserves ranking semantics (does NOT fake seed scores) and retains full evidence traceability.
"""

from typing import List, Dict, Any, Optional, Set, Tuple


def expand_adjacent_chunks(
    candidate_chunks: List[Dict[str, Any]],
    all_corpus_chunks: List[Dict[str, Any]],
    max_seeds: int = 3,
    max_neighbors_per_seed: int = 1,
    max_total_expanded: int = 6,
    max_context_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Perform carefully bounded adjacent-chunk expansion on initial retrieval candidates.

    Args:
        candidate_chunks: Top-k initial retrieval or reranked chunks.
        all_corpus_chunks: Full corpus chunk list (or lookup dict) with `chunk_index` metadata.
        max_seeds: Maximum number of top initial candidate chunks to use as expansion seeds.
        max_neighbors_per_seed: Number of preceding and following neighbors to fetch (e.g. 1 -> -1 and +1).
        max_total_expanded: Maximum total expanded neighbor chunks allowed.
        max_context_chars: Maximum total character count for final expanded context set.

    Returns:
        Deduplicated, boundary-enforced list of chunks containing seed chunks and their neighbors.
    """
    if not candidate_chunks:
        return []

    # Map corpus by (workspace_id, document_id, version, chunk_index) for O(1) boundary-safe lookup
    corpus_lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for c in all_corpus_chunks:
        ws = c.get("workspace_id")
        doc_id = c.get("document_id")
        ver = c.get("version")
        idx = c.get("chunk_index")
        if ws and doc_id and ver is not None and idx is not None:
            corpus_lookup[(ws, doc_id, ver, idx)] = c

    # Select seed chunks up to max_seeds
    seed_chunks = candidate_chunks[:max_seeds]
    seen_chunk_ids: Set[str] = {c["chunk_id"] for c in candidate_chunks}
    seen_texts: Set[str] = {c.get("excerpt", "").strip().lower() for c in candidate_chunks}

    expanded_results: List[Dict[str, Any]] = list(candidate_chunks)
    added_neighbors_count = 0
    current_char_count = sum(len(c.get("excerpt", "")) for c in candidate_chunks)

    for seed in seed_chunks:
        ws_id = seed.get("workspace_id")
        doc_id = seed.get("document_id")
        ver = seed.get("version")
        seed_idx = seed.get("chunk_index")

        if ws_id is None or doc_id is None or ver is None or seed_idx is None:
            continue

        # Look for preceding and following neighbors
        neighbor_indices = []
        for step in range(1, max_neighbors_per_seed + 1):
            neighbor_indices.append(seed_idx - step)  # preceding
            neighbor_indices.append(seed_idx + step)  # following

        for n_idx in neighbor_indices:
            if added_neighbors_count >= max_total_expanded:
                break
            if current_char_count >= max_context_chars:
                break

            lookup_key = (ws_id, doc_id, ver, n_idx)
            neighbor = corpus_lookup.get(lookup_key)

            if not neighbor:
                continue

            # Strict Boundary Checks
            if neighbor.get("workspace_id") != ws_id:
                continue  # Workspace boundary breach prevented
            if neighbor.get("document_id") != doc_id:
                continue  # Document boundary breach prevented
            if neighbor.get("version") != ver:
                continue  # Version boundary breach prevented

            n_id = neighbor["chunk_id"]
            n_text = neighbor.get("excerpt", "").strip().lower()

            if n_id in seen_chunk_ids or n_text in seen_texts:
                continue  # Deduplicate text or repeated chunk ID

            # Prepare expanded chunk metadata
            expanded_chunk = neighbor.copy()
            expanded_chunk["is_expanded"] = True
            expanded_chunk["expansion_type"] = "neighbor"
            expanded_chunk["seed_chunk_id"] = seed["chunk_id"]
            
            # Do NOT fake seed chunk retrieval scores! Set scores to None/0 to preserve ranking semantics.
            expanded_chunk["bm25_score"] = None
            expanded_chunk["similarity_score"] = None
            expanded_chunk["reranker_score"] = None
            expanded_chunk["rrf_score"] = None

            expanded_results.append(expanded_chunk)
            seen_chunk_ids.add(n_id)
            seen_texts.add(n_text)
            added_neighbors_count += 1
            current_char_count += len(neighbor.get("excerpt", ""))

    return expanded_results
