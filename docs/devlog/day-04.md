# Day 4 — Embedding Retriever

## Goal
Add an embedding-based retriever while preserving the lexical baseline.

## What was built
- Embedding-based passage retrieval
- Sentence-transformer model used
- Cosine similarity
- Top-k retrieval
- Separate lexical and embedding retrieval paths
- Same review interface
- No AI-generated answers

## Architecture

Question
↓
Retriever selection
↓
Lexical OR Embedding
↓
Top-k evidence
↓
Human review

## Smoke tests
1. **Direct**: "What encryption protects customer data at rest?"
   - Embedding: "Customer data is encrypted at rest using AES-256." (Score: 0.7301)
2. **Paraphrased**: "What safeguards information while it travels between systems?"
   - Embedding: "Security incidents are triaged by the on-call engineer." (Score: 0.3809)
3. **Conflict**: "Are backups retained for 30 days or 90 days?"
   - Embedding: "Backup retention is 30 days." (Score: 0.8531)

## Important design decision
The lexical retriever was preserved unchanged so that both methods can be compared fairly on Day 5.

## Limitations
- Small synthetic dataset
- In-memory embedding index
- No vector database
- No reranker
- No LLM
- No formal comparison yet

## Next
Compare lexical and embedding retrieval using the development questions.
