from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from smriti.models import Memory, MemoryVersion, AuditLog, Project, Task
from smriti.schemas import ProvenanceExplanation
from smriti.graph import graph_service

class MemoryManager:
    """Handles memory lifecycle, conflict tracking, diff identification, and provenance explainability."""

    def record_diff(self, db: Session, old_memories: List[Memory], new_candidates: List[Any]) -> List[Dict[str, Any]]:
        """Identifies diffs (new decisions, superseded, conflicts) before insertion."""
        diffs = []
        for cand in new_candidates:
            matched = False
            for old in old_memories:
                if old.memory_type == cand.memory_type and cand.memory_type == "decision":
                    # Check for direct conflict or supersession
                    if old.statement.lower() != cand.statement.lower():
                        # Mark diff
                        diffs.append({
                            "type": "SUPERSEDED_DECISION" if "instead of" in cand.statement.lower() else "CONFLICTING_DECISION",
                            "existing_id": old.id,
                            "existing_statement": old.statement,
                            "new_statement": cand.statement,
                            "recommendation": "supersede" if "instead of" in cand.statement.lower() else "review"
                        })
                        matched = True
                        break
            if not matched:
                diffs.append({
                    "type": f"NEW_{cand.memory_type.upper()}",
                    "statement": cand.statement
                })
        return diffs

    def resolve_conflict(
        self,
        db: Session,
        memory_id: str,
        resolution_action: str,  # "keep_active", "supersede", "deprecate", "forget"
        superseded_by_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Memory:
        mem = db.query(Memory).filter(Memory.id == memory_id).first()
        if not mem:
            raise ValueError(f"Memory with ID {memory_id} not found")

        old_status = mem.status
        now = datetime.now(timezone.utc)

        if resolution_action == "keep_active":
            mem.status = "active"
            mem.last_confirmed_at = now
        elif resolution_action == "supersede":
            mem.status = "superseded"
            mem.superseded_by_id = superseded_by_id
            if superseded_by_id:
                graph_service.add_edge(
                    db=db,
                    source_type="memory",
                    source_id=superseded_by_id,
                    relation="SUPERSEDES",
                    target_type="memory",
                    target_id=mem.id
                )
        elif resolution_action == "deprecate":
            mem.status = "deprecated"
        elif resolution_action == "forget":
            mem.status = "forgotten"

        # Increment version and record version history
        new_version_num = round(mem.version + 1.0, 1)
        mem.version = new_version_num
        mem.updated_at = now

        version_record = MemoryVersion(
            memory_id=mem.id,
            version_number=new_version_num,
            statement=mem.statement,
            rationale=mem.rationale,
            details=mem.details,
            status=mem.status,
            change_reason=reason or f"Conflict resolved: {resolution_action}"
        )
        db.add(version_record)

        audit = AuditLog(
            entity_type="memory",
            entity_id=mem.id,
            action="resolve_conflict",
            details={"old_status": old_status, "new_status": mem.status, "action": resolution_action, "reason": reason}
        )
        db.add(audit)
        db.commit()
        db.refresh(mem)
        return mem

    def explain_memory(self, db: Session, memory_id: str) -> ProvenanceExplanation:
        """Explains provenance: source conversation, message, extraction method, versions."""
        mem = db.query(Memory).filter(Memory.id == memory_id).first()
        if not mem:
            raise ValueError(f"Memory with ID {memory_id} not found")

        source_msg_dict = None
        source_conv_dict = None
        relevant_content = None

        if mem.source_message:
            source_msg_dict = {
                "id": mem.source_message.id,
                "role": mem.source_message.role,
                "created_at": mem.source_message.created_at,
                "external_id": mem.source_message.external_id
            }
            relevant_content = mem.source_message.content
            if mem.source_message.conversation:
                source_conv_dict = {
                    "id": mem.source_message.conversation.id,
                    "title": mem.source_message.conversation.title,
                    "created_at": mem.source_message.conversation.created_at,
                    "provider": mem.source_message.conversation.provider_account.provider if mem.source_message.conversation.provider_account else "unknown"
                }

        versions = [
            {
                "version": v.version_number,
                "statement": v.statement,
                "status": v.status,
                "change_reason": v.change_reason,
                "created_at": v.created_at
            }
            for v in mem.versions
        ]

        return ProvenanceExplanation(
            memory_id=mem.id,
            statement=mem.statement,
            memory_type=mem.memory_type,
            status=mem.status,
            confidence=mem.confidence,
            extraction_method=mem.extraction_method,
            created_at=mem.created_at,
            source_conversation=source_conv_dict,
            source_message=source_msg_dict,
            relevant_source_content=relevant_content,
            history_versions=versions
        )

memory_manager = MemoryManager()
