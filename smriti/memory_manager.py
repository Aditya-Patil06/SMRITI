from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from smriti.models import Memory, MemoryVersion, AuditLog, Project, Task, Message, Conversation, ProviderAccount
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
        """Explains full provenance: source message, conversation, provider account, extraction method, versions."""
        mem = db.query(Memory).filter(Memory.id == memory_id).first()
        if not mem:
            raise ValueError(f"Memory with ID {memory_id} not found")

        is_manual = (mem.source_message_id is None and mem.source_conversation_id is None) or (mem.extraction_method == "user_explicit")
        source_msg_dict = None
        source_conv_dict = None
        provider_account_dict = None
        provider_name = None
        relevant_content = None
        source_unavailable = False

        if mem.source_message_id:
            msg = db.query(Message).filter(Message.id == mem.source_message_id).first()
            if msg:
                source_msg_dict = {
                    "id": msg.id,
                    "role": msg.role,
                    "created_at": msg.created_at,
                    "external_id": msg.external_id
                }
                relevant_content = msg.content
            else:
                source_unavailable = True

        conv_id = mem.source_conversation_id or (mem.source_message.conversation_id if mem.source_message else None)
        if conv_id:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if conv:
                if conv.is_deleted_source:
                    source_unavailable = True
                source_conv_dict = {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at,
                    "is_deleted_source": conv.is_deleted_source
                }
                if conv.provider_account:
                    pa = conv.provider_account
                    provider_name = pa.provider
                    provider_account_dict = {
                        "id": pa.id,
                        "account_label": pa.account_label,
                        "provider": pa.provider,
                        "user_id": pa.user_id
                    }
                    source_conv_dict["provider"] = pa.provider
                    source_conv_dict["account_label"] = pa.account_label
            else:
                source_unavailable = True

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
            workspace_id=mem.workspace_id,
            project_id=mem.project_id,
            is_manual=is_manual,
            source_unavailable=source_unavailable,
            superseded_by_id=mem.superseded_by_id,
            provider=provider_name,
            provider_account=provider_account_dict,
            source_conversation=source_conv_dict,
            source_message=source_msg_dict,
            relevant_source_content=relevant_content,
            history_versions=versions
        )

memory_manager = MemoryManager()
