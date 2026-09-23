from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from smriti.models import Memory, MemoryVersion, AuditLog, Project, Task, Message, Conversation, ProviderAccount, RelationshipEdge
from smriti.schemas import ProvenanceExplanation
from smriti.graph import graph_service
from smriti.memory_diff import memory_diff_engine, DiffResult
from smriti.extraction import ExtractedCandidate, ClaimNormalizer

class MemoryManager:
    """Handles memory lifecycle, conflict tracking, diff identification, and provenance explainability."""

    def record_diff(
        self,
        db: Session,
        existing_memories: List[Memory],
        new_candidates: List[Any],
        workspace_id: str,
        project_id: Optional[str] = None
    ) -> List[DiffResult]:
        """Identifies diffs (NEW, UNCHANGED, SUPERSEDED, CONFLICTING, COEXISTING) using MemoryDiffEngine."""
        diffs = []
        for cand in new_candidates:
            diff = memory_diff_engine.classify_diff(cand, existing_memories, workspace_id, project_id)
            diffs.append(diff)
        return diffs

    def _are_scopes_distinct(self, scope_a: Dict[str, Any], scope_b: Dict[str, Any]) -> bool:
        """Determines whether two scopes are explicitly distinct."""
        for k in set(scope_a.keys()).union(set(scope_b.keys())):
            v1 = scope_a.get(k)
            v2 = scope_b.get(k)
            if v1 and v2 and v1 != v2:
                return True
        return False

    def get_conflicts(
        self,
        db: Session,
        workspace_id: str,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves pending conflict and review items across scoped memories.
        Identifies pairs of conflicting or coexisting items needing review.
        Respects persisted coexistence edges, multi-valued predicates, and distinct scopes.
        """
        q = db.query(Memory).filter(
            Memory.workspace_id == workspace_id,
            Memory.status.in_(["review_required", "active"])
        )
        if project_id:
            q = q.filter(Memory.project_id == project_id)
        memories = q.all()

        # Load persisted coexistence edges across memories in workspace
        coexisting_edges = db.query(RelationshipEdge).filter(
            RelationshipEdge.source_type == "memory",
            RelationshipEdge.target_type == "memory",
            RelationshipEdge.relation == "COEXISTS_WITH"
        ).all()
        confirmed_coexisting_pairs = set()
        for e in coexisting_edges:
            confirmed_coexisting_pairs.add(tuple(sorted([e.source_id, e.target_id])))

        conflicts = []
        seen_pairs = set()
        paired_memory_ids = set()

        # 1. Identify pairs of competing claims
        for i, m1 in enumerate(memories):
            if not m1.structured_claim:
                continue

            claim1 = ClaimNormalizer.normalize_claim(m1.structured_claim)
            for m2 in memories[i+1:]:
                if not m2.structured_claim:
                    continue
                claim2 = ClaimNormalizer.normalize_claim(m2.structured_claim)

                # If same subject and predicate but different object
                if (
                    claim1.get("subject") == claim2.get("subject")
                    and claim1.get("predicate") == claim2.get("predicate")
                    and claim1.get("object") != claim2.get("object")
                ):
                    predicate = claim1.get("predicate")
                    # Check if predicate naturally supports multiple coexisting values
                    if predicate in memory_diff_engine.MULTI_VALUED_PREDICATES:
                        continue

                    pair_key = tuple(sorted([m1.id, m2.id]))

                    # If pair already has a confirmed COEXISTS_WITH relationship edge in DB, skip
                    if pair_key in confirmed_coexisting_pairs:
                        continue

                    # If both are active with distinct non-overlapping scopes, they coexist cleanly
                    scope1 = claim1.get("scope") or {}
                    scope2 = claim2.get("scope") or {}
                    if m1.status == "active" and m2.status == "active" and self._are_scopes_distinct(scope1, scope2):
                        continue

                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        paired_memory_ids.add(m1.id)
                        paired_memory_ids.add(m2.id)
                        conflicts.append({
                            "id": f"pair_{m1.id}_{m2.id}",
                            "type": "CLAIM_CONFLICT",
                            "subject": claim1.get("subject"),
                            "predicate": claim1.get("predicate"),
                            "memory_a": {
                                "id": m1.id,
                                "statement": m1.statement,
                                "object": claim1.get("object"),
                                "scope": claim1.get("scope"),
                                "status": m1.status,
                                "confidence": m1.confidence,
                                "created_at": m1.created_at.isoformat() if m1.created_at else None
                            },
                            "memory_b": {
                                "id": m2.id,
                                "statement": m2.statement,
                                "object": claim2.get("object"),
                                "scope": claim2.get("scope"),
                                "status": m2.status,
                                "confidence": m2.confidence,
                                "created_at": m2.created_at.isoformat() if m2.created_at else None
                            },
                            "reason": f"Competing values for '{claim1.get('subject')}.{claim1.get('predicate')}': '{claim1.get('object')}' vs '{claim2.get('object')}'"
                        })

        # 2. Add unpaired review_required memories as SINGLE_REVIEW
        for m in memories:
            if m.status == "review_required" and m.id not in paired_memory_ids:
                conflicts.append({
                    "id": f"single_{m.id}",
                    "type": "SINGLE_REVIEW",
                    "memory_a": {
                        "id": m.id,
                        "statement": m.statement,
                        "status": m.status,
                        "confidence": m.confidence,
                        "structured_claim": m.structured_claim,
                        "created_at": m.created_at.isoformat() if m.created_at else None
                    },
                    "memory_b": None,
                    "reason": "Low confidence or unverified extraction requiring user confirmation"
                })

        return conflicts

    def resolve_conflict(
        self,
        db: Session,
        memory_id: str,
        resolution_action: str,  # "keep_active", "keep_both", "supersede", "deprecate", "forget"
        superseded_by_id: Optional[str] = None,
        paired_memory_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Memory:
        """
        Resolves conflicts transactionally with version history, graph updates, and audit logging.
        Supports:
        - keep_active: Marks target active and last_confirmed_at=now
        - keep_both: Both memory_id and paired_memory_id are kept active with distinct scopes,
                     persisting a COEXISTS_WITH relationship edge in graph and database.
        - supersede: Target is superseded by superseded_by_id with SUPERSEDES edge
        - deprecate: Target is marked deprecated
        - forget: Target is soft-deleted as forgotten
        """
        mem = db.query(Memory).filter(Memory.id == memory_id).first()
        if not mem:
            raise ValueError(f"Memory with ID {memory_id} not found")

        old_status = mem.status
        now = datetime.now(timezone.utc)

        if resolution_action == "keep_active":
            mem.status = "active"
            mem.last_confirmed_at = now
            if paired_memory_id:
                p_mem = db.query(Memory).filter(Memory.id == paired_memory_id).first()
                if p_mem:
                    p_mem.status = "deprecated"
                    p_mem.updated_at = now
                    p_version = MemoryVersion(
                        memory_id=p_mem.id,
                        version_number=round(p_mem.version + 1.0, 1),
                        statement=p_mem.statement,
                        rationale=p_mem.rationale,
                        structured_claim=p_mem.structured_claim,
                        details=p_mem.details,
                        status=p_mem.status,
                        change_reason=f"Deprecated in favor of keeping {mem.id} active"
                    )
                    p_mem.version = p_version.version_number
                    db.add(p_version)

        elif resolution_action == "keep_both":
            mem.status = "active"
            mem.last_confirmed_at = now
            if paired_memory_id:
                p_mem = db.query(Memory).filter(Memory.id == paired_memory_id).first()
                if p_mem:
                    p_mem.status = "active"
                    p_mem.last_confirmed_at = now
                    p_version = MemoryVersion(
                        memory_id=p_mem.id,
                        version_number=round(p_mem.version + 1.0, 1),
                        statement=p_mem.statement,
                        rationale=p_mem.rationale,
                        structured_claim=p_mem.structured_claim,
                        details=p_mem.details,
                        status=p_mem.status,
                        change_reason=f"Coexistence confirmed alongside {mem.id}"
                    )
                    p_mem.version = p_version.version_number
                    db.add(p_version)

                    # Persist bidirectional COEXISTS_WITH relationship edges
                    existing_edge = db.query(RelationshipEdge).filter(
                        RelationshipEdge.source_type == "memory",
                        RelationshipEdge.target_type == "memory",
                        RelationshipEdge.relation == "COEXISTS_WITH",
                        ((RelationshipEdge.source_id == mem.id) & (RelationshipEdge.target_id == p_mem.id)) |
                        ((RelationshipEdge.source_id == p_mem.id) & (RelationshipEdge.target_id == mem.id))
                    ).first()
                    if not existing_edge:
                        graph_service.add_edge(
                            db=db,
                            source_type="memory",
                            source_id=mem.id,
                            relation="COEXISTS_WITH",
                            target_type="memory",
                            target_id=p_mem.id,
                            properties={"reason": reason or "Confirmed coexistence"}
                        )
                        graph_service.add_edge(
                            db=db,
                            source_type="memory",
                            source_id=p_mem.id,
                            relation="COEXISTS_WITH",
                            target_type="memory",
                            target_id=mem.id,
                            properties={"reason": reason or "Confirmed coexistence"}
                        )

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
        else:
            raise ValueError(f"Invalid resolution_action '{resolution_action}'")

        # Increment version and record version history
        new_version_num = round(mem.version + 1.0, 1)
        mem.version = new_version_num
        mem.updated_at = now

        version_record = MemoryVersion(
            memory_id=mem.id,
            version_number=new_version_num,
            statement=mem.statement,
            rationale=mem.rationale,
            structured_claim=mem.structured_claim,
            details=mem.details,
            status=mem.status,
            change_reason=reason or f"Conflict resolved: {resolution_action}"
        )
        db.add(version_record)

        audit = AuditLog(
            entity_type="memory",
            entity_id=mem.id,
            action="resolve_conflict",
            details={
                "old_status": old_status,
                "new_status": mem.status,
                "action": resolution_action,
                "reason": reason,
                "paired_memory_id": paired_memory_id,
                "superseded_by_id": superseded_by_id
            }
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
                "structured_claim": v.structured_claim,
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
