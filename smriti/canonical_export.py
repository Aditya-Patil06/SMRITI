import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from smriti.models import (
    Workspace, Project, Task, Memory, MemoryVersion, Conversation, Message, RelationshipEdge, User, ProviderAccount, AuditLog
)

class CanonicalExportEngine:
    """Handles complete export and import of SMRITI memory state without semantic loss."""

    VERSION = "1.1.0"
    DEFAULT_USER_ID = "default-user"

    def export_all(self, db: Session, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        ws_q = db.query(Workspace)
        if workspace_id:
            ws_q = ws_q.filter(Workspace.id == workspace_id)
        workspaces = ws_q.all()
        active_ws_ids = set(w.id for w in workspaces)

        if workspace_id:
            # Workspace-scoped export: restrict all entities to those owned by or related to this workspace
            target_user_ids = set(w.user_id for w in workspaces)
            users = db.query(User).filter(User.id.in_(target_user_ids)).all()
            provider_accounts = db.query(ProviderAccount).filter(ProviderAccount.user_id.in_(target_user_ids)).all()
            pa_ids = set(pa.id for pa in provider_accounts)

            proj_q = db.query(Project).filter(Project.workspace_id.in_(active_ws_ids))
            projects = proj_q.all()
            active_proj_ids = set(p.id for p in projects)

            tasks = db.query(Task).filter(Task.project_id.in_(active_proj_ids)).all()
            memories = db.query(Memory).filter(Memory.workspace_id.in_(active_ws_ids)).all()
            mem_ids = set(m.id for m in memories)

            memory_versions = db.query(MemoryVersion).filter(MemoryVersion.memory_id.in_(mem_ids)).all()

            # Conversations referenced by memories or provider accounts
            mem_conv_ids = set(m.source_conversation_id for m in memories if m.source_conversation_id)
            conv_q = db.query(Conversation).filter(
                (Conversation.id.in_(mem_conv_ids)) | (Conversation.provider_account_id.in_(pa_ids))
            )
            conversations = conv_q.all()
            conv_ids = set(c.id for c in conversations)

            messages = db.query(Message).filter(Message.conversation_id.in_(conv_ids)).all()

            # Edges where either source or target matches any scoped project, memory, or task
            scoped_node_ids = (
                set(f"project:{p.id}" for p in projects) |
                set(f"memory:{m.id}" for m in memories) |
                set(f"task:{t.id}" for t in tasks) |
                active_ws_ids | active_proj_ids | mem_ids
            )
            all_edges = db.query(RelationshipEdge).all()
            edges = [
                e for e in all_edges
                if e.source_id in scoped_node_ids or e.target_id in scoped_node_ids
            ]

            # Audit logs referencing scoped entities
            all_audit = db.query(AuditLog).all()
            audit_logs = [
                a for a in all_audit
                if a.entity_id in scoped_node_ids
            ]
        else:
            # Full system export
            users = db.query(User).all()
            provider_accounts = db.query(ProviderAccount).all()
            proj_q = db.query(Project).filter(Project.workspace_id.in_(active_ws_ids))
            projects = proj_q.all()
            active_proj_ids = set(p.id for p in projects)

            tasks = db.query(Task).filter(Task.project_id.in_(active_proj_ids)).all()
            memories = db.query(Memory).filter(Memory.workspace_id.in_(active_ws_ids)).all()
            mem_ids = set(m.id for m in memories)

            memory_versions = db.query(MemoryVersion).filter(MemoryVersion.memory_id.in_(mem_ids)).all()

            conv_q = db.query(Conversation)
            conversations = conv_q.all()
            conv_ids = set(c.id for c in conversations)

            messages = db.query(Message).filter(Message.conversation_id.in_(conv_ids)).all()
            edges = db.query(RelationshipEdge).all()
            audit_logs = db.query(AuditLog).all()

        manifest = {
            "version": self.VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": workspace_id,
            "entity_counts": {
                "users": len(users),
                "workspaces": len(workspaces),
                "provider_accounts": len(provider_accounts),
                "projects": len(projects),
                "tasks": len(tasks),
                "memories": len(memories),
                "memory_versions": len(memory_versions),
                "conversations": len(conversations),
                "messages": len(messages),
                "edges": len(edges),
                "audit_logs": len(audit_logs)
            }
        }

        def serialize_dt(dt):
            return dt.isoformat() if dt else None

        bundle = {
            "manifest": manifest,
            "users": [
                {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "created_at": serialize_dt(u.created_at)
                }
                for u in users
            ],
            "workspaces": [
                {
                    "id": w.id,
                    "user_id": w.user_id,
                    "name": w.name,
                    "description": w.description,
                    "is_default": w.is_default,
                    "created_at": serialize_dt(w.created_at)
                }
                for w in workspaces
            ],
            "provider_accounts": [
                {
                    "id": pa.id,
                    "user_id": pa.user_id,
                    "provider": pa.provider,
                    "account_label": pa.account_label,
                    "auth_metadata": pa.auth_metadata,
                    "is_active": pa.is_active,
                    "created_at": serialize_dt(pa.created_at)
                }
                for pa in provider_accounts
            ],
            "projects": [
                {
                    "id": p.id,
                    "workspace_id": p.workspace_id,
                    "name": p.name,
                    "description": p.description,
                    "goal": p.goal,
                    "status": p.status,
                    "architecture_overview": p.architecture_overview,
                    "tech_stack": p.tech_stack,
                    "constraints": p.constraints,
                    "created_at": serialize_dt(p.created_at),
                    "updated_at": serialize_dt(p.updated_at),
                    "last_confirmed_at": serialize_dt(p.last_confirmed_at)
                }
                for p in projects
            ],
            "tasks": [
                {
                    "id": t.id,
                    "project_id": t.project_id,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status,
                    "priority": t.priority,
                    "due_date": serialize_dt(t.due_date),
                    "created_at": serialize_dt(t.created_at),
                    "updated_at": serialize_dt(t.updated_at)
                }
                for t in tasks
            ],
            "memories": [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "project_id": m.project_id,
                    "source_message_id": m.source_message_id,
                    "source_conversation_id": m.source_conversation_id,
                    "memory_type": m.memory_type,
                    "statement": m.statement,
                    "rationale": m.rationale,
                    "details": m.details,
                    "status": m.status,
                    "confidence": m.confidence,
                    "extraction_method": m.extraction_method,
                    "version": m.version,
                    "superseded_by_id": m.superseded_by_id,
                    "created_at": serialize_dt(m.created_at),
                    "updated_at": serialize_dt(m.updated_at),
                    "last_confirmed_at": serialize_dt(m.last_confirmed_at)
                }
                for m in memories
            ],
            "memory_versions": [
                {
                    "id": mv.id,
                    "memory_id": mv.memory_id,
                    "version_number": mv.version_number,
                    "statement": mv.statement,
                    "rationale": mv.rationale,
                    "details": mv.details,
                    "status": mv.status,
                    "change_reason": mv.change_reason,
                    "created_at": serialize_dt(mv.created_at)
                }
                for mv in memory_versions
            ],
            "conversations": [
                {
                    "id": c.id,
                    "provider_account_id": c.provider_account_id,
                    "external_id": c.external_id,
                    "title": c.title,
                    "is_deleted_source": c.is_deleted_source,
                    "raw_metadata": c.raw_metadata,
                    "created_at": serialize_dt(c.created_at),
                    "updated_at": serialize_dt(c.updated_at)
                }
                for c in conversations
            ],
            "messages": [
                {
                    "id": msg.id,
                    "conversation_id": msg.conversation_id,
                    "external_id": msg.external_id,
                    "role": msg.role,
                    "content": msg.content,
                    "sequence_index": msg.sequence_index,
                    "raw_metadata": msg.raw_metadata,
                    "created_at": serialize_dt(msg.created_at)
                }
                for msg in messages
            ],
            "relationships": [
                {
                    "id": e.id,
                    "source_type": e.source_type,
                    "source_id": e.source_id,
                    "relation": e.relation,
                    "target_type": e.target_type,
                    "target_id": e.target_id,
                    "properties": e.properties,
                    "created_at": serialize_dt(e.created_at)
                }
                for e in edges
            ],
            "audit_logs": [
                {
                    "id": a.id,
                    "entity_type": a.entity_type,
                    "entity_id": a.entity_id,
                    "action": a.action,
                    "details": a.details,
                    "created_at": serialize_dt(a.created_at)
                }
                for a in audit_logs
            ]
        }
        return bundle

    def validate_bundle(self, bundle: Dict[str, Any]):
        """Validate format and essential structures prior to writing to database."""
        if not isinstance(bundle, dict):
            raise ValueError("Export bundle must be a JSON object")
        manifest = bundle.get("manifest")
        if not manifest or not isinstance(manifest, dict):
            raise ValueError("Export bundle is missing a valid manifest")
        version = manifest.get("version")
        if not version:
            raise ValueError("Manifest missing export schema version")

    def preflight_check(self, db: Session, bundle: Dict[str, Any]):
        """Perform read-only validation for ownership collisions or invalid hierarchies before writing anything."""
        # 1. Workspace ownership collisions
        for w_data in bundle.get("workspaces", []):
            existing_ws = db.query(Workspace).filter(Workspace.id == w_data["id"]).first()
            if existing_ws:
                incoming_user = w_data.get("user_id") or self.DEFAULT_USER_ID
                if existing_ws.user_id != incoming_user:
                    raise ValueError(
                        f"Workspace collision: workspace '{w_data['id']}' already exists and belongs to user '{existing_ws.user_id}', but import claims user '{incoming_user}'"
                    )

        # 2. Project workspace collisions
        for p_data in bundle.get("projects", []):
            existing_p = db.query(Project).filter(Project.id == p_data["id"]).first()
            if existing_p:
                if existing_p.workspace_id != p_data.get("workspace_id"):
                    raise ValueError(
                        f"Project collision: project '{p_data['id']}' already belongs to workspace '{existing_p.workspace_id}', not '{p_data.get('workspace_id')}'"
                    )

        # 3. Task project collisions
        for t_data in bundle.get("tasks", []):
            existing_t = db.query(Task).filter(Task.id == t_data["id"]).first()
            if existing_t:
                if existing_t.project_id != t_data.get("project_id"):
                    raise ValueError(
                        f"Task collision: task '{t_data['id']}' belongs to project '{existing_t.project_id}', not '{t_data.get('project_id')}'"
                    )

        # 4. Memory workspace collisions
        for m_data in bundle.get("memories", []):
            existing_m = db.query(Memory).filter(Memory.id == m_data["id"]).first()
            if existing_m:
                if existing_m.workspace_id != m_data.get("workspace_id"):
                    raise ValueError(
                        f"Memory collision: memory '{m_data['id']}' belongs to workspace '{existing_m.workspace_id}', not '{m_data.get('workspace_id')}'"
                    )

    def import_all(self, db: Session, bundle: Dict[str, Any]) -> Dict[str, int]:
        from dateutil import parser
        self.validate_bundle(bundle)
        self.preflight_check(db, bundle)

        counts = {
            "users": 0, "workspaces": 0, "provider_accounts": 0, "projects": 0,
            "tasks": 0, "conversations": 0, "messages": 0, "memories": 0,
            "memory_versions": 0, "relationships": 0, "audit_logs": 0
        }

        def parse_dt(val):
            return parser.parse(val) if val else datetime.now(timezone.utc)

        # Atomic transaction
        try:
            # 0. Ensure DEFAULT_USER_ID exists if any incoming workspace or provider account relies on it
            needs_default_user = any(
                not w.get("user_id") for w in bundle.get("workspaces", [])
            ) or any(
                not pa.get("user_id") for pa in bundle.get("provider_accounts", [])
            )
            if needs_default_user:
                if not db.query(User).filter(User.id == self.DEFAULT_USER_ID).first():
                    default_user = User(
                        id=self.DEFAULT_USER_ID,
                        username=self.DEFAULT_USER_ID,
                        email=f"{self.DEFAULT_USER_ID}@smriti.local"
                    )
                    db.add(default_user)
                    db.flush()
                    counts["users"] += 1

            # 1. Users
            for u_data in bundle.get("users", []):
                if not db.query(User).filter(User.id == u_data["id"]).first():
                    u = User(
                        id=u_data["id"],
                        username=u_data["username"],
                        email=u_data.get("email"),
                        created_at=parse_dt(u_data.get("created_at"))
                    )
                    db.add(u)
                    counts["users"] += 1

            # 2. Workspaces
            for w_data in bundle.get("workspaces", []):
                if not db.query(Workspace).filter(Workspace.id == w_data["id"]).first():
                    w = Workspace(
                        id=w_data["id"],
                        user_id=w_data.get("user_id") or self.DEFAULT_USER_ID,
                        name=w_data["name"],
                        description=w_data.get("description"),
                        is_default=w_data.get("is_default", False),
                        created_at=parse_dt(w_data.get("created_at"))
                    )
                    db.add(w)
                    counts["workspaces"] += 1

            # 3. Provider Accounts
            for pa_data in bundle.get("provider_accounts", []):
                if not db.query(ProviderAccount).filter(ProviderAccount.id == pa_data["id"]).first():
                    pa = ProviderAccount(
                        id=pa_data["id"],
                        user_id=pa_data.get("user_id") or self.DEFAULT_USER_ID,
                        provider=pa_data["provider"],
                        account_label=pa_data["account_label"],
                        auth_metadata=pa_data.get("auth_metadata", {}),
                        is_active=pa_data.get("is_active", True),
                        created_at=parse_dt(pa_data.get("created_at"))
                    )
                    db.add(pa)
                    counts["provider_accounts"] += 1

            # 4. Projects
            for p_data in bundle.get("projects", []):
                if not db.query(Project).filter(Project.id == p_data["id"]).first():
                    p = Project(
                        id=p_data["id"],
                        workspace_id=p_data["workspace_id"],
                        name=p_data["name"],
                        description=p_data.get("description"),
                        goal=p_data.get("goal"),
                        status=p_data.get("status", "active"),
                        architecture_overview=p_data.get("architecture_overview"),
                        tech_stack=p_data.get("tech_stack", []),
                        constraints=p_data.get("constraints", []),
                        created_at=parse_dt(p_data.get("created_at")),
                        updated_at=parse_dt(p_data.get("updated_at")),
                        last_confirmed_at=parse_dt(p_data.get("last_confirmed_at"))
                    )
                    db.add(p)
                    counts["projects"] += 1

            # 5. Tasks
            for t_data in bundle.get("tasks", []):
                if not db.query(Task).filter(Task.id == t_data["id"]).first():
                    t = Task(
                        id=t_data["id"],
                        project_id=t_data["project_id"],
                        title=t_data["title"],
                        description=t_data.get("description"),
                        status=t_data.get("status", "todo"),
                        priority=t_data.get("priority", "medium"),
                        due_date=parse_dt(t_data.get("due_date")) if t_data.get("due_date") else None,
                        created_at=parse_dt(t_data.get("created_at")),
                        updated_at=parse_dt(t_data.get("updated_at"))
                    )
                    db.add(t)
                    counts["tasks"] += 1

            # 6. Conversations
            for c_data in bundle.get("conversations", []):
                if not db.query(Conversation).filter(Conversation.id == c_data["id"]).first():
                    c = Conversation(
                        id=c_data["id"],
                        provider_account_id=c_data.get("provider_account_id"),
                        external_id=c_data.get("external_id"),
                        title=c_data["title"],
                        is_deleted_source=c_data.get("is_deleted_source", False),
                        raw_metadata=c_data.get("raw_metadata", {}),
                        created_at=parse_dt(c_data.get("created_at")),
                        updated_at=parse_dt(c_data.get("updated_at"))
                    )
                    db.add(c)
                    counts["conversations"] += 1

            # 7. Messages
            for m_data in bundle.get("messages", []):
                if not db.query(Message).filter(Message.id == m_data["id"]).first():
                    msg = Message(
                        id=m_data["id"],
                        conversation_id=m_data["conversation_id"],
                        external_id=m_data.get("external_id"),
                        role=m_data["role"],
                        content=m_data["content"],
                        sequence_index=m_data.get("sequence_index", 0.0),
                        raw_metadata=m_data.get("raw_metadata", {}),
                        created_at=parse_dt(m_data.get("created_at"))
                    )
                    db.add(msg)
                    counts["messages"] += 1

            # 8. Memories
            for mem_data in bundle.get("memories", []):
                if not db.query(Memory).filter(Memory.id == mem_data["id"]).first():
                    mem = Memory(
                        id=mem_data["id"],
                        workspace_id=mem_data["workspace_id"],
                        project_id=mem_data.get("project_id"),
                        source_message_id=mem_data.get("source_message_id"),
                        source_conversation_id=mem_data.get("source_conversation_id"),
                        memory_type=mem_data["memory_type"],
                        statement=mem_data["statement"],
                        rationale=mem_data.get("rationale"),
                        details=mem_data.get("details", {}),
                        status=mem_data.get("status", "active"),
                        confidence=mem_data.get("confidence", 1.0),
                        extraction_method=mem_data.get("extraction_method", "rule_heuristic"),
                        version=mem_data.get("version", 1.0),
                        superseded_by_id=mem_data.get("superseded_by_id"),
                        created_at=parse_dt(mem_data.get("created_at")),
                        updated_at=parse_dt(mem_data.get("updated_at")),
                        last_confirmed_at=parse_dt(mem_data.get("last_confirmed_at"))
                    )
                    db.add(mem)
                    counts["memories"] += 1

            # 9. Memory Versions
            for mv_data in bundle.get("memory_versions", []):
                if not db.query(MemoryVersion).filter(MemoryVersion.id == mv_data["id"]).first():
                    mv = MemoryVersion(
                        id=mv_data["id"],
                        memory_id=mv_data["memory_id"],
                        version_number=mv_data["version_number"],
                        statement=mv_data["statement"],
                        rationale=mv_data.get("rationale"),
                        details=mv_data.get("details", {}),
                        status=mv_data["status"],
                        change_reason=mv_data.get("change_reason"),
                        created_at=parse_dt(mv_data.get("created_at"))
                    )
                    db.add(mv)
                    counts["memory_versions"] += 1

            # 10. Relationships
            for e_data in bundle.get("relationships", []):
                if not db.query(RelationshipEdge).filter(RelationshipEdge.id == e_data["id"]).first():
                    edge = RelationshipEdge(
                        id=e_data["id"],
                        source_type=e_data["source_type"],
                        source_id=e_data["source_id"],
                        relation=e_data["relation"],
                        target_type=e_data["target_type"],
                        target_id=e_data["target_id"],
                        properties=e_data.get("properties", {}),
                        created_at=parse_dt(e_data.get("created_at"))
                    )
                    db.add(edge)
                    counts["relationships"] += 1

            # 11. Audit Logs
            for a_data in bundle.get("audit_logs", []):
                if not db.query(AuditLog).filter(AuditLog.id == a_data["id"]).first():
                    al = AuditLog(
                        id=a_data["id"],
                        entity_type=a_data["entity_type"],
                        entity_id=a_data["entity_id"],
                        action=a_data["action"],
                        details=a_data.get("details", {}),
                        created_at=parse_dt(a_data.get("created_at"))
                    )
                    db.add(al)
                    counts["audit_logs"] += 1

            db.commit()
        except Exception:
            db.rollback()
            raise

        return counts

canonical_export_engine = CanonicalExportEngine()
