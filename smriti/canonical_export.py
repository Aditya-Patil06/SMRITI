import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from smriti.models import (
    Workspace, Project, Task, Memory, Conversation, Message, RelationshipEdge, User, ProviderAccount
)
from smriti.schemas import MemoryRead, ProjectRead

class CanonicalExportEngine:
    """Handles full export and import of SMRITI memory state without semantic loss."""

    VERSION = "1.0.0"

    def export_all(self, db: Session, workspace_id: str = None) -> Dict[str, Any]:
        projects = db.query(Project).all()
        tasks = db.query(Task).all()
        memories = db.query(Memory).all()
        conversations = db.query(Conversation).all()
        messages = db.query(Message).all()
        edges = db.query(RelationshipEdge).all()
        workspaces = db.query(Workspace).all()

        manifest = {
            "version": self.VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entity_counts": {
                "workspaces": len(workspaces),
                "projects": len(projects),
                "tasks": len(tasks),
                "memories": len(memories),
                "conversations": len(conversations),
                "messages": len(messages),
                "edges": len(edges)
            }
        }

        def serialize_dt(dt):
            return dt.isoformat() if dt else None

        bundle = {
            "manifest": manifest,
            "workspaces": [
                {
                    "id": w.id,
                    "name": w.name,
                    "description": w.description,
                    "is_default": w.is_default,
                    "created_at": serialize_dt(w.created_at)
                }
                for w in workspaces
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
            ]
        }
        return bundle

    def import_all(self, db: Session, bundle: Dict[str, Any]) -> Dict[str, int]:
        from dateutil import parser
        counts = {"workspaces": 0, "projects": 0, "tasks": 0, "memories": 0, "conversations": 0, "messages": 0, "relationships": 0}

        def parse_dt(val):
            return parser.parse(val) if val else datetime.now(timezone.utc)

        # Workspaces
        for w_data in bundle.get("workspaces", []):
            if not db.query(Workspace).filter(Workspace.id == w_data["id"]).first():
                w = Workspace(
                    id=w_data["id"],
                    user_id="default-user",
                    name=w_data["name"],
                    description=w_data.get("description"),
                    is_default=w_data.get("is_default", False),
                    created_at=parse_dt(w_data.get("created_at"))
                )
                db.add(w)
                counts["workspaces"] += 1

        # Projects
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

        # Tasks
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

        # Conversations
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

        # Messages
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

        # Memories
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

        # Relationships
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

        db.commit()
        return counts

canonical_export_engine = CanonicalExportEngine()
