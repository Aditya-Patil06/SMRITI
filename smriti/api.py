from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import json

from smriti.config import settings
from smriti.models import (
    init_db, get_db, User, Workspace, ProviderAccount,
    Conversation, Message, Project, Task, Memory, RelationshipEdge, AuditLog
)
from smriti.schemas import (
    UserRead, WorkspaceRead, WorkspaceCreate,
    ProviderAccountRead, ProviderAccountCreate,
    ConversationRead, ConversationCreate, MessageRead,
    ProjectRead, ProjectCreate, TaskRead, TaskCreate,
    MemoryRead, MemoryCreate, MemoryUpdate,
    ProvenanceExplanation, PortableContextPackage,
    GraphData, SearchResultItem
)
from smriti.importers import registry as importer_registry
from smriti.extraction import MemoryExtractor
from smriti.vector_store import vector_store
from smriti.graph import graph_service
from smriti.retrieval import retrieval_engine
from smriti.context_engine import context_engine
from smriti.memory_manager import memory_manager
from smriti.providers.adapters import provider_registry
from smriti.canonical_export import canonical_export_engine

app = FastAPI(
    title="Project SMRITI Memory Core API",
    description="Persistent, portable personal AI memory layer for connecting knowledge across AI platforms.",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = MemoryExtractor()

@app.on_event("startup")
def on_startup():
    init_db()
    # Initialize graph from db on startup
    from smriti.models import SessionLocal
    db = SessionLocal()
    try:
        graph_service.sync_from_db(db)
        # Seed vector store with existing active memories
        for m in db.query(Memory).filter(Memory.status != "forgotten").all():
            vector_store.upsert(m.id, m.statement, {"project_id": m.project_id, "status": m.status})
    finally:
        db.close()

# --- Health Check ---
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Project SMRITI Memory Core",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# --- Workspaces ---
@app.get("/api/v1/workspaces", response_model=List[WorkspaceRead])
def list_workspaces(db: Session = Depends(get_db)):
    return db.query(Workspace).all()

@app.post("/api/v1/workspaces", response_model=WorkspaceRead)
def create_workspace(data: WorkspaceCreate, db: Session = Depends(get_db)):
    ws = Workspace(
        user_id="default-user",
        name=data.name,
        description=data.description,
        is_default=data.is_default
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws

# --- Provider Accounts ---
@app.get("/api/v1/accounts", response_model=List[ProviderAccountRead])
def list_provider_accounts(db: Session = Depends(get_db)):
    return db.query(ProviderAccount).all()

@app.post("/api/v1/accounts", response_model=ProviderAccountRead)
def create_provider_account(data: ProviderAccountCreate, db: Session = Depends(get_db)):
    account = ProviderAccount(
        user_id=data.user_id or "default-user",
        provider=data.provider,
        account_label=data.account_label,
        auth_metadata=data.auth_metadata or {},
        is_active=data.is_active
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account

# --- Projects ---
@app.get("/api/v1/projects", response_model=List[ProjectRead])
def list_projects(workspace_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Project)
    if workspace_id:
        q = q.filter(Project.workspace_id == workspace_id)
    return q.all()

@app.post("/api/v1/projects", response_model=ProjectRead)
def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    ws_id = data.workspace_id or "default-workspace"
    proj = Project(
        workspace_id=ws_id,
        name=data.name,
        description=data.description,
        goal=data.goal,
        status=data.status,
        architecture_overview=data.architecture_overview,
        tech_stack=data.tech_stack,
        constraints=data.constraints
    )
    db.add(proj)
    db.commit()
    db.refresh(proj)
    graph_service.sync_from_db(db)
    return proj

@app.get("/api/v1/projects/{project_id}", response_model=ProjectRead)
def get_project(project_id: str, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj

# --- Tasks ---
@app.post("/api/v1/tasks", response_model=TaskRead)
def create_task(data: TaskCreate, db: Session = Depends(get_db)):
    task = Task(
        project_id=data.project_id,
        title=data.title,
        description=data.description,
        status=data.status,
        priority=data.priority,
        due_date=data.due_date
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    graph_service.sync_from_db(db)
    return task

@app.patch("/api/v1/tasks/{task_id}", response_model=TaskRead)
def update_task(task_id: str, updates: Dict[str, Any], db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    for k, v in updates.items():
        if hasattr(task, k):
            setattr(task, k, v)
    task.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(task)
    return task

# --- Conversations & Messages ---
@app.get("/api/v1/conversations", response_model=List[ConversationRead])
def list_conversations(db: Session = Depends(get_db)):
    return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()

@app.get("/api/v1/conversations/{conv_id}", response_model=ConversationRead)
def get_conversation(conv_id: str, db: Session = Depends(get_db)):
    c = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return c

@app.post("/api/v1/conversations", response_model=ConversationRead)
def create_conversation(data: ConversationCreate, db: Session = Depends(get_db)):
    conv = Conversation(
        title=data.title,
        provider_account_id=data.provider_account_id,
        external_id=data.external_id,
        raw_metadata=data.raw_metadata or {}
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    # Add messages
    for m in data.messages:
        msg = Message(
            conversation_id=conv.id,
            role=m.role,
            content=m.content,
            sequence_index=m.sequence_index,
            external_id=m.external_id,
            created_at=m.created_at or datetime.now(timezone.utc),
            raw_metadata=m.raw_metadata or {}
        )
        db.add(msg)
    db.commit()
    db.refresh(conv)
    return conv

# --- Memories ---
@app.get("/api/v1/memories", response_model=List[MemoryRead])
def list_memories(
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    memory_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Memory).filter(Memory.status != "forgotten")
    if project_id:
        q = q.filter(Memory.project_id == project_id)
    if status:
        q = q.filter(Memory.status == status)
    if memory_type:
        q = q.filter(Memory.memory_type == memory_type)
    return q.order_by(Memory.updated_at.desc()).all()

@app.post("/api/v1/memories", response_model=MemoryRead)
def create_memory(data: MemoryCreate, db: Session = Depends(get_db)):
    ws_id = data.workspace_id or "default-workspace"
    mem = Memory(
        workspace_id=ws_id,
        project_id=data.project_id,
        source_message_id=data.source_message_id,
        source_conversation_id=data.source_conversation_id,
        memory_type=data.memory_type,
        statement=data.statement,
        rationale=data.rationale,
        details=data.details or {},
        status=data.status,
        confidence=data.confidence,
        extraction_method=data.extraction_method
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)

    # Update Vector Store & Graph
    vector_store.upsert(mem.id, mem.statement, {"project_id": mem.project_id, "status": mem.status})
    graph_service.sync_from_db(db)
    return mem

@app.get("/api/v1/memories/{memory_id}/explain", response_model=ProvenanceExplanation)
def explain_memory(memory_id: str, db: Session = Depends(get_db)):
    try:
        return memory_manager.explain_memory(db, memory_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/v1/memories/{memory_id}/resolve", response_model=MemoryRead)
def resolve_memory_conflict(
    memory_id: str,
    resolution_action: str = Query(..., description="keep_active, supersede, deprecate, or forget"),
    superseded_by_id: Optional[str] = None,
    reason: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        mem = memory_manager.resolve_conflict(
            db=db,
            memory_id=memory_id,
            resolution_action=resolution_action,
            superseded_by_id=superseded_by_id,
            reason=reason
        )
        if mem.status == "forgotten":
            vector_store.delete(mem.id)
        else:
            vector_store.upsert(mem.id, mem.statement, {"project_id": mem.project_id, "status": mem.status})
        graph_service.sync_from_db(db)
        return mem
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- Ingestion & Extraction ---
@app.post("/api/v1/imports/conversations")
def import_conversations(
    provider: str,
    payload: List[Dict[str, Any]],
    project_id: Optional[str] = None,
    auto_extract: bool = True,
    db: Session = Depends(get_db)
):
    normalized_convs = importer_registry.import_conversations(provider, payload)
    imported_ids = []
    total_extracted_memories = 0
    all_diffs = []

    for nc in normalized_convs:
        conv = Conversation(
            title=nc.title,
            external_id=nc.external_id,
            created_at=nc.created_at or datetime.now(timezone.utc),
            updated_at=nc.updated_at or datetime.now(timezone.utc),
            raw_metadata=nc.raw_metadata
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        imported_ids.append(conv.id)

        # Insert messages
        db_messages = []
        for nm in nc.messages:
            msg = Message(
                conversation_id=conv.id,
                role=nm.role,
                content=nm.content,
                sequence_index=nm.sequence_index,
                external_id=nm.external_id,
                created_at=nm.created_at or datetime.now(timezone.utc),
                raw_metadata=nm.raw_metadata
            )
            db.add(msg)
            db_messages.append(msg)
        db.commit()

        if auto_extract:
            for msg in db_messages:
                candidates = extractor.extract_from_message(msg.role, msg.content)
                if candidates:
                    existing_mems = db.query(Memory).filter(Memory.project_id == project_id).all() if project_id else []
                    diffs = memory_manager.record_diff(db, existing_mems, candidates)
                    all_diffs.extend(diffs)

                    for cand in candidates:
                        mem = Memory(
                            workspace_id="default-workspace",
                            project_id=project_id,
                            source_message_id=msg.id,
                            source_conversation_id=conv.id,
                            memory_type=cand.memory_type,
                            statement=cand.statement,
                            rationale=cand.rationale,
                            details=cand.details,
                            status=cand.status,
                            confidence=cand.confidence,
                            extraction_method="rule_heuristic"
                        )
                        db.add(mem)
                        db.commit()
                        db.refresh(mem)
                        vector_store.upsert(mem.id, mem.statement, {"project_id": mem.project_id, "status": mem.status})
                        total_extracted_memories += 1

    graph_service.sync_from_db(db)

    return {
        "status": "success",
        "provider": provider,
        "imported_conversations": len(imported_ids),
        "extracted_memories": total_extracted_memories,
        "diffs_identified": len(all_diffs),
        "diffs": all_diffs[:10]
    }

# --- Hybrid Search ---
@app.get("/api/v1/search", response_model=List[SearchResultItem])
def search_memory(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 15,
    include_superseded: bool = False,
    db: Session = Depends(get_db)
):
    return retrieval_engine.search(
        db=db,
        query=query,
        project_id=project_id,
        limit=limit,
        include_superseded=include_superseded
    )

# --- Context Engine ---
@app.get("/api/v1/context", response_model=PortableContextPackage)
def get_context_package(
    query: str,
    project_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return context_engine.build_context(db=db, query=query, project_id=project_id)

# --- Knowledge Graph ---
@app.get("/api/v1/graph", response_model=GraphData)
def get_knowledge_graph(center_node: Optional[str] = None, depth: int = 2, db: Session = Depends(get_db)):
    graph_service.sync_from_db(db)
    return graph_service.get_subgraph(center_node=center_node, depth=depth)

# --- Provider Capabilities ---
@app.get("/api/v1/providers/capabilities")
def get_provider_capabilities():
    return provider_registry.list_all_capabilities()

# --- Export & Import ---
@app.get("/api/v1/exports/canonical")
def export_canonical_memory(db: Session = Depends(get_db)):
    return canonical_export_engine.export_all(db)

@app.post("/api/v1/imports/canonical")
def import_canonical_memory(payload: Dict[str, Any], db: Session = Depends(get_db)):
    counts = canonical_export_engine.import_all(db, payload)
    graph_service.sync_from_db(db)
    # Refresh vector store
    for m in db.query(Memory).filter(Memory.status != "forgotten").all():
        vector_store.upsert(m.id, m.statement, {"project_id": m.project_id, "status": m.status})
    return {"status": "success", "imported_counts": counts}

# --- Dashboard Stats ---
@app.get("/api/v1/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_memories = db.query(Memory).filter(Memory.status != "forgotten").count()
    active_projects = db.query(Project).filter(Project.status == "active").count()
    total_conversations = db.query(Conversation).count()
    pending_reviews = db.query(Memory).filter(Memory.status == "review_required").count()
    stale_memories = db.query(Memory).filter(Memory.status == "review_required").count()
    accounts = db.query(ProviderAccount).count()

    recent_mems = db.query(Memory).order_by(Memory.updated_at.desc()).limit(5).all()

    return {
        "total_memories": total_memories,
        "active_projects": active_projects,
        "total_conversations": total_conversations,
        "pending_reviews": pending_reviews,
        "stale_memories": stale_memories,
        "provider_accounts": accounts,
        "recent_memories": [
            {
                "id": m.id,
                "statement": m.statement,
                "type": m.memory_type,
                "status": m.status,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat()
            }
            for m in recent_mems
        ]
    }
