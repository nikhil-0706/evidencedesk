# EvidenceDesk

An evidence-backed security questionnaire assistant with retrieval evaluation and human review.

**Problem:** Help reviewers find reliable evidence for security questionnaires without treating matching text as proof.

**Current implementation:** FastAPI, synthetic security policies, lexical retrieval baseline, embedding retrieval (primary), constrained LLM reasoning layer (Ollama) to classify evidence, retrieval evaluation, and human review interface.

**Evaluation:** 25 labeled questions covering direct, paraphrased, unanswerable, ambiguous, and conflicting cases, with expected evidence and behavior recorded for each.

**How to run:** Installation and launch commands.

**Limitations:** Human review is still completely required. The AI does not generate conversational final answers. Not for production use.

**Next milestone:** Measure the impact of the new LLM reasoning layer on the development evaluation set.
