import unittest

from tests.fixtures.synthetic_data import inject_legacy_fixtures
from evidencedesk import multi_doc_corpus, qdrant, embed_model
inject_legacy_fixtures(multi_doc_corpus, qdrant, embed_model, multi_doc_corpus.collection_name)

from evidencedesk import rrf_fuse

class TestRRF(unittest.TestCase):
    def setUp(self):
        self.doc_a = {"chunk_id": "chk_A", "document_id": "A", "excerpt": "text A", "version": "1", "title": "A title"}
        self.doc_b = {"chunk_id": "chk_B", "document_id": "B", "excerpt": "text B", "version": "1", "title": "B title"}
        self.doc_c = {"chunk_id": "chk_C", "document_id": "C", "excerpt": "text C", "version": "1", "title": "C title"}

    def test_rrf_merges_highly_ranked_documents(self):
        bm25 = [self.doc_a, self.doc_b, self.doc_c]
        embed = [self.doc_b, self.doc_c, self.doc_a]
        
        result = rrf_fuse(bm25, embed, k=60, top_k=10)
        
        # doc_b is rank 2 in bm25 and rank 1 in embed -> 1/62 + 1/61 = 0.0325
        # doc_a is rank 1 in bm25 and rank 3 in embed -> 1/61 + 1/63 = 0.0322
        # doc_c is rank 3 in bm25 and rank 2 in embed -> 1/63 + 1/62 = 0.0320
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["document_id"], "B")
        self.assertEqual(result[1]["document_id"], "A")
        self.assertEqual(result[2]["document_id"], "C")
        self.assertIn("rrf_score", result[0])

    def test_rrf_handles_unique_documents(self):
        bm25 = [self.doc_a]
        embed = [self.doc_b]
        
        result = rrf_fuse(bm25, embed, k=60, top_k=10)
        
        self.assertEqual(len(result), 2)
        # both rank 1, so tie. We just check they both appear.
        doc_ids = [r["document_id"] for r in result]
        self.assertIn("A", doc_ids)
        self.assertIn("B", doc_ids)

    def test_rrf_returns_top_k(self):
        bm25 = [self.doc_a, self.doc_b, self.doc_c]
        embed = [self.doc_b, self.doc_c, self.doc_a]
        
        result = rrf_fuse(bm25, embed, k=60, top_k=2)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["document_id"], "B")
        self.assertEqual(result[1]["document_id"], "A")

    def test_rrf_preserves_metadata(self):
        bm25 = [self.doc_a]
        embed = []
        
        result = rrf_fuse(bm25, embed, k=60, top_k=10)
        
        self.assertEqual(result[0]["title"], "A title")
        self.assertEqual(result[0]["version"], "1")
        self.assertEqual(result[0]["excerpt"], "text A")

if __name__ == '__main__':
    unittest.main()
