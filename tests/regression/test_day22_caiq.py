import os
import sys
import fitz  # PyMuPDF
from document_ingestion import MultiDocumentCorpus

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

from evidencedesk import qdrant, embed_model, review

def create_octopus_caiq_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Octopus Cloud Security Assessment (CAIQ)\n\n"
        "Control ID: IAM-01.1\n"
        "Question: Do you use Role-Based Access Control (RBAC) to manage access to the platform?\n"
        "Answer: Yes, Octopus Cloud enforces strict RBAC across all tenant environments and administrative portals.\n\n"
        "Control ID: DCS-02.1\n"
        "Question: Is customer data encrypted at rest?\n"
        "Answer: Customer data is encrypted at rest using AES-256 managed by Octopus KMS.\n",
        fontsize=11
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_octopus_caiq():
    print("--- Running Day 22 Octopus CAIQ Diagnostic Test ---")
    corpus = MultiDocumentCorpus(
        qdrant_client=qdrant,
        embed_model=embed_model
    )
    
    pdf_bytes = create_octopus_caiq_pdf()
    
    # Ingest into an isolated workspace
    workspace = "ws_octopus"
    corpus.ingest_single_document(
        filename="octopus_caiq_v3.pdf",
        file_bytes=pdf_bytes,
        workspace_id=workspace,
        title="Octopus CAIQ",
        version="v3"
    )
    
    # Query 1
    q1 = "Do you use Role-Based Access Control (RBAC) to manage access to the platform?"
    res1 = review(q1, method='hybrid', workspace_id=workspace, enable_expansion=True, debug=True)
    
    if res1["status"] != "ANSWERABLE":
        print(f"FAILED: Expected ANSWERABLE for RBAC question, got {res1['status']}")
        sys.exit(1)
        
    print("  [TEST 1] RBAC question successfully reached LLM context and answered: PASSED")
    
    # Query 2
    q2 = "Is customer data encrypted at rest?"
    res2 = review(q2, method='hybrid', workspace_id=workspace, enable_expansion=True)
    
    if res2["status"] != "ANSWERABLE":
        print(f"FAILED: Expected ANSWERABLE for Encryption question, got {res2['status']}")
        sys.exit(1)
        
    print("  [TEST 2] Encryption question successfully reached LLM context and answered: PASSED")
    print("\nALL DAY 22 OCTOPUS CAIQ DIAGNOSTIC TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_octopus_caiq()
