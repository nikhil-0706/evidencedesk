# EvidenceDesk

An evidence-backed security questionnaire assistant with retrieval evaluation and human review.

**Problem:** Help reviewers find reliable evidence for security questionnaires without treating matching text as proof.

**Current implementation:** FastAPI, synthetic security policies, lexical retrieval, evidence ranking, and human review interface.

**Evaluation:** 25 labeled questions covering direct, paraphrased, unanswerable, ambiguous, and conflicting cases, with expected evidence and behavior recorded for each.

**How to run:** Installation and launch commands.

**Limitations:** No LLM, embeddings, reranking, authentication, or public deployment yet.

**Next milestone:** Compare improved retrieval methods against the current lexical baseline using development and held-out questions.
