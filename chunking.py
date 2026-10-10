"""
chunking.py — Shared, format-agnostic, structure-aware document chunker for EvidenceDesk.

Implements structure-aware document extraction and chunking:
1. Format adapters for PDF, DOCX, and TXT.
2. Structure-aware splitting hierarchy (Q&A, Key-Value/Tables, Sections, Paragraphs, Sentences, Window fallback).
3. Conservative text normalization preserving meaningful words, negations, control IDs, dates.
4. Rich, stable provenance metadata including deterministic sequence order (chunk_index).
"""

import os
import io
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import docx  # python-docx

DEFAULT_MAX_CHUNK_CHARS = 250
DEFAULT_MIN_CHUNK_CHARS = 30
DEFAULT_OVERLAP_CHARS = 50


def conservative_normalize_text(text: str) -> str:
    """
    Conservatively normalize text without removing stop words, negations ('not', 'no', 'never', 'unless'),
    dates, numbers, or control identifiers.
    """
    if not text:
        return ""
    # Standardize whitespace while preserving case and all characters
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_pdf_blocks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract structured text blocks and page metadata from PDF byte buffer."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid PDF file: {exc}")

    if doc.page_count == 0:
        raise ValueError(f"PDF is empty: {filename}")

    blocks_data = []
    total_text_len = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        cleaned_text = conservative_normalize_text(text)
        if not cleaned_text:
            continue

        total_text_len += len(cleaned_text)
        
        # PyMuPDF blocks preserve reading order and page structure
        page_blocks = page.get_text("blocks")
        if page_blocks:
            for b in page_blocks:
                b_text = conservative_normalize_text(b[4]) if len(b) > 4 else ""
                if b_text:
                    blocks_data.append({
                        "page": page_num,
                        "section": None,
                        "text": b_text,
                        "block_type": "text",
                    })
        else:
            # Fallback to paragraph splitting
            paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
            for p in paragraphs:
                blocks_data.append({
                    "page": page_num,
                    "section": None,
                    "text": p,
                    "block_type": "text",
                })

    if total_text_len == 0:
        raise ValueError(f"PDF contains no extractable text: {filename}")

    return blocks_data


def extract_docx_blocks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract structured text paragraphs, headings, and tables from DOCX byte buffer."""
    try:
        file_stream = io.BytesIO(file_bytes)
        doc = docx.Document(file_stream)
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid DOCX file: {exc}")

    blocks_data = []
    total_text_len = 0
    current_section = "General"

    # Extract body elements (paragraphs and tables)
    for element in doc.element.body:
        tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag_name == "p":
            para = docx.text.paragraph.Paragraph(element, doc)
            text = conservative_normalize_text(para.text)
            if not text:
                continue

            total_text_len += len(text)
            style_name = para.style.name if para.style else ""

            # Detect heading vs body
            is_heading = (
                "Heading" in style_name
                or text.startswith("Section ")
                or text.startswith("Chapter ")
                or (text.isupper() and len(text) < 80)
            )

            if is_heading:
                current_section = text
                blocks_data.append({
                    "page": None,
                    "section": current_section,
                    "text": text,
                    "block_type": "heading",
                })
            else:
                blocks_data.append({
                    "page": None,
                    "section": current_section,
                    "text": text,
                    "block_type": "paragraph",
                })

        elif tag_name == "tbl":
            table = docx.table.Table(element, doc)
            table_rows_text = []
            for row in table.rows:
                row_cells = [conservative_normalize_text(cell.text) for cell in row.cells]
                row_str = " | ".join([c for c in row_cells if c])
                if row_str:
                    table_rows_text.append(row_str)

            if table_rows_text:
                full_table_text = "\n".join(table_rows_text)
                total_text_len += len(full_table_text)
                blocks_data.append({
                    "page": None,
                    "section": current_section,
                    "text": full_table_text,
                    "block_type": "table",
                })

    if total_text_len == 0:
        raise ValueError(f"DOCX contains no extractable text: {filename}")

    return blocks_data


def extract_txt_blocks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text blocks, headings, lists, and key-values from TXT byte buffer."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except Exception as exc:
            raise ValueError(f"Invalid text encoding for TXT file: {exc}")

    normalized = conservative_normalize_text(text)
    if not normalized:
        raise ValueError(f"TXT file is empty: {filename}")

    # Split into logical blocks by double newline
    raw_blocks = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    if not raw_blocks:
        raw_blocks = [p.strip() for p in normalized.splitlines() if p.strip()]

    blocks_data = []
    current_section = "General"

    for b in raw_blocks:
        # Check if header
        is_heading = (
            b.startswith("Section ")
            or b.startswith("Title:")
            or b.startswith("Chapter ")
            or (b.isupper() and len(b) < 80 and "\n" not in b)
        )

        if is_heading:
            current_section = b
            blocks_data.append({
                "page": None,
                "section": current_section,
                "text": b,
                "block_type": "heading",
            })
        else:
            blocks_data.append({
                "page": None,
                "section": current_section,
                "text": b,
                "block_type": "text",
            })

    return blocks_data


def split_text_recursively(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[str]:
    """
    Recursively split text that exceeds max_chars using a preference hierarchy:
    Paragraphs -> Lines -> Sentences -> Words -> Character boundaries.
    Applies overlap between sub-chunks.
    """
    text = conservative_normalize_text(text)
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    
    # Priority 1: Split by paragraphs (\n\n)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        current_chunk = ""
        for p in paragraphs:
            if not current_chunk:
                current_chunk = p
            elif len(current_chunk) + len(p) + 2 <= max_chars:
                current_chunk += "\n\n" + p
            else:
                chunks.extend(split_text_recursively(current_chunk, max_chars, overlap_chars))
                current_chunk = p
        if current_chunk:
            chunks.extend(split_text_recursively(current_chunk, max_chars, overlap_chars))
        return chunks

    # Priority 2: Split by lines (\n)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1:
        current_chunk = ""
        for line in lines:
            if not current_chunk:
                current_chunk = line
            elif len(current_chunk) + len(line) + 1 <= max_chars:
                current_chunk += "\n" + line
            else:
                chunks.extend(split_text_recursively(current_chunk, max_chars, overlap_chars))
                current_chunk = line
        if current_chunk:
            chunks.extend(split_text_recursively(current_chunk, max_chars, overlap_chars))
        return chunks

    # Priority 3: Split by sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) > 1:
        current_chunk = ""
        for s in sentences:
            if not current_chunk:
                current_chunk = s
            elif len(current_chunk) + len(s) + 1 <= max_chars:
                current_chunk += " " + s
            else:
                chunks.append(current_chunk)
                # Apply overlap by taking trailing portion of previous chunk
                overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else ""
                if overlap_text and " " in overlap_text:
                    overlap_text = overlap_text[overlap_text.index(" ") + 1:]
                current_chunk = (overlap_text + " " + s).strip() if overlap_text else s
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # Priority 4: Split by words as last resort
    words = text.split()
    current_chunk = ""
    for w in words:
        if not current_chunk:
            current_chunk = w
        elif len(current_chunk) + len(w) + 1 <= max_chars:
            current_chunk += " " + w
        else:
            chunks.append(current_chunk)
            current_chunk = w
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def is_qa_pair(text: str) -> bool:
    """Detect if a text block contains an explicit Question and Answer structure."""
    if not text:
        return False
    pattern_labeled = (
        r"(?:^|\n)\s*(?:Q(?:uestion)?|Req(?:uirement)?|Control(?:\s+ID)?|CAIQ(?:\s+ID)?|[A-Z]{2,5}-\d{1,3}(?:\.\d+)?)\s*[:\.\-?]"
        r".*(?:\n|\s+)+(?:A(?:nswer)?|Response|Status|Implementation)\s*[:\.\-]"
    )
    if re.search(pattern_labeled, text, re.IGNORECASE | re.DOTALL):
        return True

    pattern_qa = (
        r"(?:^|\n)\s*(?:[\"']?(?:Do|Does|Is|Are|Can|Could|Will|Would|How|What|Where|Who|Which|Why)\b[^?\n]+\?[\"']?|"
        r"(?:Q(?:uestion)?|Req(?:uirement)?|Control(?:\s+ID)?|CAIQ(?:\s+ID)?)\s*[:\.\-].*?\?[\"']?)"
        r"\s*(?:\n|\s{2,})\s*[\"']?(?:Yes|No|True|False|Implemented|Partially|Not Applicable|N/A|NA|Compliant|This feature|Customer data|All|We|Our)\b"
    )
    if re.search(pattern_qa, text, re.IGNORECASE | re.DOTALL):
        return True

    return False


def structure_aware_chunking_shared(
    extracted_blocks: List[Dict[str, Any]],
    document_id: str,
    title: str,
    version: str,
    workspace_id: str,
    source_type: str,
    filename: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """
    Shared, structure-aware chunker for PDF, DOCX, TXT, and synthetic documents.
    
    Order of preference for boundaries:
    1. Complete Q&A records
    2. Key-Value records & table rows
    3. Section / Paragraph units
    4. Sentence boundaries
    5. Token / character boundaries as last resort
    
    Generates ordered `chunk_index` and stable, deterministic `chunk_id`.
    """
    chunks = []
    doc_id_clean = document_id.lower().replace("-", "")
    current_section = "General"
    chunk_counter = 0

    # Group adjacent blocks by section where section header appears
    processed_units = []
    i = 0
    while i < len(extracted_blocks):
        block = extracted_blocks[i]
        page = block.get("page")
        if block.get("section"):
            current_section = block["section"]
        section = current_section
        text = block.get("text", "").strip()
        block_type = block.get("block_type", "text")

        if not text:
            i += 1
            continue

        # Update current section if block is a heading or section title
        if block_type == "heading" or text.startswith("Section ") or text.startswith("Chapter ") or (text.isupper() and len(text) < 60):
            current_section = text
            # Check if next block is paragraph belonging to this heading on the SAME page
            if i + 1 < len(extracted_blocks):
                next_block = extracted_blocks[i + 1]
                next_page = next_block.get("page")
                next_text = next_block.get("text", "").strip()
                same_page = (page == next_page) or (page is None and next_page is None)
                if same_page and len(text) + len(next_text) + 2 <= max_chunk_chars:
                    combined_unit = f"{text}\n{next_text}"
                    processed_units.append({
                        "page": page,
                        "section": current_section,
                        "text": combined_unit,
                    })
                    i += 2
                    continue

            # Standalone section title chunk if long enough, else store section state
            if len(text) >= DEFAULT_MIN_CHUNK_CHARS:
                processed_units.append({
                    "page": page,
                    "section": current_section,
                    "text": text,
                })
            i += 1
            continue

        # Check for 3-block pairing: [Control ID/Header] + [Question] + [Answer] on the SAME page
        if i + 2 < len(extracted_blocks):
            next_b = extracted_blocks[i + 1]
            third_b = extracted_blocks[i + 2]
            p_next = next_b.get("page")
            p_third = third_b.get("page")
            same_page_3 = (page == p_next == p_third) or (page is None and p_next is None and p_third is None)

            clean_t = text.strip().rstrip("\"' )")
            is_id_header = bool(re.match(r"^[\"']?(?:Control(?:\s+ID)?|CAIQ(?:\s+ID)?|Requirement|[A-Z]{2,5}-\d{1,3}(?:\.\d+)?)\s*[:\.\-]?\s*.*$", text, re.IGNORECASE)) and not clean_t.endswith("?")
            clean_next = next_b.get("text", "").strip().rstrip("\"' )")
            is_next_q = clean_next.endswith("?") or bool(re.search(r"^(?:Q|Question|\d+[\.\)]\s+(?:Who|What|How|Is|Are|Where|Which|Do|Does|Can|Could|Will|Would|Has|Have|Should))\b", next_b.get("text", ""), re.IGNORECASE))
            clean_third = third_b.get("text", "").strip()
            is_third_a = (
                bool(re.search(r"^[\"']?(?:A|Answer|Response|Status|Implementation|Yes|No|True|False|Implemented|Partially|Not Applicable|N/A|NA|Compliant)\b", clean_third, re.IGNORECASE))
                or bool(re.search(r"^[\"']?(?:This feature|All|We|Our)\b", clean_third, re.IGNORECASE))
            )

            if same_page_3 and is_id_header and is_next_q and is_third_a:
                combined_unit = f"{text}\n{next_b.get('text', '').strip()}\n{clean_third}"
                if len(combined_unit) <= max(max_chunk_chars * 3, 750):
                    processed_units.append({
                        "page": page,
                        "section": current_section,
                        "text": combined_unit,
                        "is_qa": True,
                    })
                    i += 3
                    continue

        # Check for 2-block Question + Answer pairing across adjacent blocks on the SAME page
        if i + 1 < len(extracted_blocks):
            next_block = extracted_blocks[i + 1]
            next_page = next_block.get("page")
            next_text = next_block.get("text", "").strip()
            same_page = (page == next_page) or (page is None and next_page is None)
            
            clean_t = text.strip().rstrip("\"' )")
            is_q = (
                clean_t.endswith("?")
                or bool(re.search(r"^(?:Q|Question|\d+[\.\)]\s+(?:Who|What|How|Is|Are|Where|Which|Do|Does|Can|Could|Will|Would|Has|Have|Should))\b", text, re.IGNORECASE))
                or bool(re.search(r"\b(?:Who|What|How|Is|Are|Where|Which|Do|Does|Can|Could|Will|Would|Has|Have|Should)\b[^?\n]+\?", text, re.IGNORECASE))
            )
            is_a = (
                bool(re.search(r"^[\"']?(?:A|Answer|Response|Status|Implementation)\b", next_text, re.IGNORECASE))
                or bool(re.search(r"^[\"']?(?:Yes|No|True|False|Implemented|Partially|Not Applicable|N/A|NA|Compliant|Non-compliant)\b", next_text, re.IGNORECASE))
                or bool(re.search(r"^[\"']?(?:This feature|All customer|Customer data|All)\b", next_text, re.IGNORECASE))
            )
            
            if same_page and (is_q or is_a or is_qa_pair(f"{text}\n{next_text}")):
                combined_qa = f"{text}\n{next_text}"
                if len(combined_qa) <= max(max_chunk_chars * 3, 750):
                    processed_units.append({
                        "page": page,
                        "section": current_section,
                        "text": combined_qa,
                        "is_qa": True,
                    })
                    i += 2
                    continue

        processed_units.append({
            "page": page,
            "section": current_section,
            "text": text,
        })
        i += 1

    # Final chunk generation with recursive splitting and deterministic chunk_index
    for unit in processed_units:
        page = unit["page"]
        section = unit["section"]
        unit_text = unit["text"]
        is_qa = unit.get("is_qa") or is_qa_pair(unit_text)

        if is_qa and len(unit_text) <= max(max_chunk_chars * 3, 750):
            sub_excerpts = [unit_text]
        else:
            sub_excerpts = split_text_recursively(unit_text, max_chars=max_chunk_chars, overlap_chars=overlap_chars)
        
        for excerpt in sub_excerpts:
            clean_excerpt = conservative_normalize_text(excerpt)
            if len(clean_excerpt) < DEFAULT_MIN_CHUNK_CHARS and len(sub_excerpts) > 1:
                continue

            page_str = f"p{page:02d}" if page is not None else "p0"
            hash_input = f"{document_id}_{version}_{page_str}_{section}_{clean_excerpt}_{chunk_counter}"
            digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
            
            chunk_id = f"chk_{source_type}_{doc_id_clean}_{page_str}_idx{chunk_counter:04d}_{digest}"

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_counter,
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
            chunk_counter += 1

    return chunks
