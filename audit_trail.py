import sqlite3
import json
import hashlib
import threading
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

class AuditTrail:
    """
    Cryptographically verifiable append-only audit log for human review decisions.
    
    Security Limitations & Authorization Note:
    In a production environment, `workspace_id` and `reviewer_identity` MUST NOT be 
    trusted from client-supplied HTTP request bodies. They should be securely extracted 
    from a verified session token (e.g., JWT) or authentication context by the API gateway.
    This implementation currently accepts them from the client for demonstration purposes.
    
    Integrity Note:
    Hashing alone (SHA-256 chain) does not guarantee absolute immutability if an attacker
    can truncate or overwrite the entire database. It provides tamper-evidence within the chain.
    """
    def __init__(self, db_path="audit_trail.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    reviewer_identity TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    question TEXT NOT NULL,
                    ai_status TEXT NOT NULL,
                    original_ai_response TEXT NOT NULL,
                    final_reviewed_response TEXT NOT NULL,
                    reviewer_notes TEXT,
                    document_ids TEXT NOT NULL,
                    document_versions TEXT NOT NULL,
                    evidence_chunk_ids TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
            ''')
            
    def _canonicalize(self, record: dict) -> str:
        """Create a canonical JSON string representation for stable hashing."""
        return json.dumps(record, separators=(',', ':'), sort_keys=True)
        
    def _compute_hash(self, prev_hash: str, canonical_record: str) -> str:
        h = hashlib.sha256()
        h.update(prev_hash.encode('utf-8'))
        h.update(canonical_record.encode('utf-8'))
        return h.hexdigest()
        
    def record_decision(
        self,
        workspace_id: str,
        reviewer_identity: str,
        decision: str,
        question: str,
        ai_status: str,
        original_ai_response: str,
        final_reviewed_response: str,
        reviewer_notes: str,
        document_ids: List[str],
        document_versions: List[str],
        evidence_chunk_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Records a decision in a tamper-evident append-only log.
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                prev_hash = row[0] if row else "GENESIS"
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                record_dict = {
                    "workspace_id": workspace_id,
                    "reviewer_identity": reviewer_identity,
                    "decision": decision,
                    "question": question,
                    "ai_status": ai_status,
                    "original_ai_response": original_ai_response,
                    "final_reviewed_response": final_reviewed_response,
                    "reviewer_notes": reviewer_notes or "",
                    "document_ids": sorted(list(set(document_ids))),
                    "document_versions": sorted(list(set(document_versions))),
                    "evidence_chunk_ids": sorted(list(set(evidence_chunk_ids))),
                    "timestamp": timestamp
                }
                
                canonical_record = self._canonicalize(record_dict)
                record_hash = self._compute_hash(prev_hash, canonical_record)
                
                cursor.execute('''
                    INSERT INTO audit_log (
                        workspace_id, reviewer_identity, decision, question, ai_status,
                        original_ai_response, final_reviewed_response, reviewer_notes,
                        document_ids, document_versions, evidence_chunk_ids,
                        timestamp, previous_hash, hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    workspace_id, reviewer_identity, decision, question, ai_status,
                    original_ai_response, final_reviewed_response, reviewer_notes or "",
                    json.dumps(record_dict["document_ids"]),
                    json.dumps(record_dict["document_versions"]),
                    json.dumps(record_dict["evidence_chunk_ids"]),
                    timestamp, prev_hash, record_hash
                ))
                
                decision_id = cursor.lastrowid
                
                return {
                    "id": decision_id,
                    **record_dict,
                    "previous_hash": prev_hash,
                    "hash": record_hash
                }
                
    def verify_chain(self) -> Tuple[bool, str]:
        """
        Verifies the cryptographic integrity of the audit log.
        Returns (is_valid, message).
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, workspace_id, reviewer_identity, decision, question, ai_status,
                           original_ai_response, final_reviewed_response, reviewer_notes,
                           document_ids, document_versions, evidence_chunk_ids,
                           timestamp, previous_hash, hash
                    FROM audit_log ORDER BY id ASC
                ''')
                rows = cursor.fetchall()
                
        prev_hash = "GENESIS"
        for row in rows:
            record_dict = {
                "workspace_id": row[1],
                "reviewer_identity": row[2],
                "decision": row[3],
                "question": row[4],
                "ai_status": row[5],
                "original_ai_response": row[6],
                "final_reviewed_response": row[7],
                "reviewer_notes": row[8],
                "document_ids": json.loads(row[9]),
                "document_versions": json.loads(row[10]),
                "evidence_chunk_ids": json.loads(row[11]),
                "timestamp": row[12]
            }
            stored_prev = row[13]
            stored_hash = row[14]
            
            if prev_hash != stored_prev:
                return False, f"Chain broken at record ID {row[0]}: previous_hash mismatch."
                
            canonical_record = self._canonicalize(record_dict)
            computed_hash = self._compute_hash(prev_hash, canonical_record)
            
            if computed_hash != stored_hash:
                return False, f"Chain broken at record ID {row[0]}: hash mismatch. Record was tampered with."
                
            prev_hash = stored_hash
            
        return True, "Audit log integrity verified."

    def clear(self):
        """Used strictly for test isolation."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DROP TABLE IF EXISTS audit_log")
        self._init_db()

# Global singleton
audit_trail = AuditTrail()
