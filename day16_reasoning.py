"""
day16_reasoning.py — Day 16 & Day 17 Semantic Reasoning Implementation
========================================================================

Implements generalized semantic reasoning prompt strategy for SLM evidence
classification without altering retrieval components or Day 15 baseline.
"""

import json
from typing import List, Literal, Optional
import ollama
from pydantic import BaseModel, Field, ValidationError

# Re-use schema and validation from evidencedesk for exact compliance
from evidencedesk import EvidenceDecision, _validate_slm_output


_REASONING_PROMPT_DAY16 = """\
You are an evidence classifier for a security-policy questionnaire system.

Your job is to classify whether the supplied evidence passages answer the user's question.

===========================================================
REASONING RULES
===========================================================

1. ANSWERABLE:
   Classify as ANSWERABLE if AT LEAST ONE evidence passage directly establishes the fact requested by the question.
   - Semantic equivalence is ALLOWED: The question and evidence do NOT need to use identical words.
   - Focus on whether the evidence establishes the requested fact using domain/operational equivalence:
     * "takes the first look" / "initial handling" ≈ "triages"
     * "who gives permission" / "authorizes" ≈ "requires approval"
     * "travels between systems" / "moves between systems" ≈ "in transit"
     * "where are encryption keys managed" is answered by "managed in a dedicated key management service"
     * "how regularly do you check that backups can be restored" is answered by "restore procedures are tested quarterly"
   - Do NOT mark a specific question as AMBIGUOUS merely because multiple evidence passages discuss different policies. If any single passage answers the specific question, return ANSWERABLE.

2. AMBIGUOUS:
   Classify as AMBIGUOUS ONLY when the QUESTION ITSELF is vague, underspecified, or lacks essential scope.
   - Example: "What is your retention policy?" (The question is ambiguous because it does not state whether it refers to database backups, logs, or customer data).
   - If the question is specific (e.g. "Who approves administrative access?" or "How frequently are database backups created?"), it is NOT ambiguous.

3. CONFLICTING:
   Classify as CONFLICTING ONLY when the question is specific, but two or more evidence passages give materially different or contradictory values for that exact same fact.
   - Example: "How long are database backups retained?" when one passage states "Backup retention is 30 days" and another states "Database backup retention is 90 days".

4. INSUFFICIENT_EVIDENCE:
   Classify as INSUFFICIENT_EVIDENCE when NO supplied passage establishes or contains the requested fact (such as resolution SLAs, tabletop drill frequency, SOC 2 certification, cloud host, or uptime guarantee).
   - Do NOT guess, infer, or assume facts not explicitly stated in the evidence.

===========================================================
GENERALIZED FEW-SHOT EXAMPLES
===========================================================

Example 1 — ANSWERABLE (Semantic Equivalence: 'first look' ≈ 'triages')

Question: Who takes the first look at a reported security incident?
Evidence: [chk_sec004_001] Security incidents are triaged by the on-call engineer.

{{
  "status": "ANSWERABLE",
  "reason": "The question asks who initially handles/triages a security incident, and the evidence identifies the on-call engineer.",
  "evidence_chunk_ids": ["chk_sec004_001"],
  "evidence_quote": "Security incidents are triaged by the on-call engineer."
}}


Example 2 — ANSWERABLE (Semantic Equivalence: 'gives permission' ≈ 'requires approval')

Question: Who gives permission for administrative access?
Evidence: [chk_sec001_003] Administrative access requires manager approval.

{{
  "status": "ANSWERABLE",
  "reason": "The evidence directly establishes that manager approval is required for administrative access.",
  "evidence_chunk_ids": ["chk_sec001_003"],
  "evidence_quote": "Administrative access requires manager approval."
}}


Example 3 — ANSWERABLE (Semantic Equivalence: 'moves between systems' ≈ 'in transit')

Question: What safeguards customer information while it moves between systems?
Evidence: [chk_sec002_002] Data in transit is protected using TLS 1.2 or later.

{{
  "status": "ANSWERABLE",
  "reason": "Information moving between systems refers to data in transit. The evidence states it is protected using TLS 1.2 or later.",
  "evidence_chunk_ids": ["chk_sec002_002"],
  "evidence_quote": "Data in transit is protected using TLS 1.2 or later."
}}


Example 4 — AMBIGUOUS (Underspecified Question Scope)

Question: What is your retention policy?
Evidence: [chk_sec003_002] Backup retention is 30 days.
         [chk_sec005_001] Database backup retention is 90 days.

{{
  "status": "AMBIGUOUS",
  "reason": "The question does not specify what type of retention policy (e.g. database backups, logs, or customer data) is being requested.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}


Example 5 — CONFLICTING (Specific Question, Contradictory Evidence)

Question: How long are database backups retained?
Evidence: [chk_sec003_002] Backup retention is 30 days.
         [chk_sec005_001] Database backup retention is 90 days.

{{
  "status": "CONFLICTING",
  "reason": "The question specifically asks about database backup retention, and the evidence gives two materially different retention periods (30 days vs 90 days).",
  "evidence_chunk_ids": ["chk_sec003_002", "chk_sec005_001"],
  "evidence_quote": ""
}}


Example 6 — INSUFFICIENT_EVIDENCE (Fact Not Established in Corpus)

Question: What is your SLA for resolving high-severity security incidents?
Evidence: [chk_sec004_001] Security incidents are triaged by the on-call engineer.
         [chk_sec004_002] Confirmed incidents are escalated to the security lead.

{{
  "status": "INSUFFICIENT_EVIDENCE",
  "reason": "The supplied evidence identifies triage and escalation roles but does not establish a resolution SLA.",
  "evidence_chunk_ids": [],
  "evidence_quote": ""
}}


===========================================================
NOW CLASSIFY THE CURRENT REQUEST
===========================================================

Question:
{question}

Evidence:
{evidence_text}

Return ONLY valid JSON matching this schema exactly:
{{
  "status": "ANSWERABLE | AMBIGUOUS | INSUFFICIENT_EVIDENCE | CONFLICTING",
  "reason": "short explanation based on the semantic meaning of the evidence",
  "evidence_chunk_ids": ["chunk_id_1"],
  "evidence_quote": "exact verbatim quote from supplied evidence if ANSWERABLE, else empty string"
}}
"""


def analyze_evidence_day16(question: str, retrieved_evidence: list) -> dict:
    """
    Day 16 / Day 17 SLM evidence reasoning function.
    
    Uses generalized semantic reasoning prompt strategy, calls local SLM (qwen2.5:3b),
    and validates output against EvidenceDecision Pydantic schema.
    """
    if not retrieved_evidence:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "No evidence was retrieved.",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
        }

    evidence_text = "\n".join(
        [f"[{doc['chunk_id']}] {doc['excerpt']}" for doc in retrieved_evidence]
    )
    prompt = _REASONING_PROMPT_DAY16.format(
        question=question, evidence_text=evidence_text
    )

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        raw = json.loads(response["message"]["content"])
    except json.JSONDecodeError as exc:
        return {
            "status": "VALIDATION_ERROR",
            "reason": f"SLM returned malformed JSON: {exc}",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "_validation_error": True,
        }
    except Exception as exc:
        return {
            "status": "VALIDATION_ERROR",
            "reason": f"SLM execution error: {exc}",
            "evidence_chunk_ids": [],
            "evidence_quote": "",
            "_validation_error": True,
        }

    # Validate and sanitize using standard schema enforcement
    return _validate_slm_output(raw, retrieved_evidence)
