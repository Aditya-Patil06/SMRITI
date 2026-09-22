from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    from smriti.models import SessionLocal
    db = SessionLocal()
    try:
        graph_service.sync_from_db(db)
        vector_store.rebuild_from_db(db)
    finally:
        db.close()
    yield

app = FastAPI(
    title="Project SMRITI Memory Core API",
    description="Persistent, portable personal AI memory layer for connecting knowledge across AI platforms.",
    version="0.1.0",
    lifespan=lifespan
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

# --- Global Exception Handling ---
@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "type": "ValueError"}
    )

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
def list_workspaces(user_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Workspace)
    if user_id:
        q = q.filter(Workspace.user_id == user_id)
    return q.all()

@app.post("/api/v1/workspaces", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
def create_workspace(data: WorkspaceCreate, db: Session = Depends(get_db)):
    target_user_id = data.user_id or "default-user"
    user = db.query(User).filter(User.id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{target_user_id}' does not exist")

    ws = Workspace(
        user_id=target_user_id,
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
def list_provider_accounts(user_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(ProviderAccount)
    if user_id:
        q = q.filter(ProviderAccount.user_id == user_id)
    return q.all()

@app.post("/api/v1/accounts", response_model=ProviderAccountRead, status_code=status.HTTP_201_CREATED)
def create_provider_account(data: ProviderAccountCreate, db: Session = Depends(get_db)):
    target_user_id = data.user_id or "default-user"
    user = db.query(User).filter(User.id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{target_user_id}' does not exist")

    account = ProviderAccount(
        user_id=target_user_id,
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

@app.post("/api/v1/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    ws_id = data.workspace_id or "default-workspace"
    ws = db.query(Workspace).filter(Workspace.id == ws_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{ws_id}' not found")

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
def get_project(project_id: str, workspace_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Project).filter(Project.id == project_id)
    if workspace_id:
        q = q.filter(Project.workspace_id == workspace_id)
    proj = q.first()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj

# --- Tasks ---
@app.post("/api/v1/tasks", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def create_task(data: TaskCreate, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == data.project_id).first()
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{data.project_id}' not found")

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
def list_conversations(
    provider_account_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Conversation)
    if provider_account_id:
        q = q.filter(Conversation.provider_account_id == provider_account_id)
    return q.order_by(Conversation.updated_at.desc()).all()

@app.get("/api/v1/conversations/{conv_id}", response_model=ConversationRead)
def get_conversation(conv_id: str, db: Session = Depends(get_db)):
    c = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return c

@app.post("/api/v1/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
def create_conversation(data: ConversationCreate, db: Session = Depends(get_db)):
    if data.provider_account_id:
        pa = db.query(ProviderAccount).filter(ProviderAccount.id == data.provider_account_id).first()
        if not pa:
            raise HTTPException(status_code=404, detail=f"ProviderAccount '{data.provider_account_id}' not found")

    conv = Conversation(
        title=data.title,
        provider_account_id=data.provider_account_id,
        external_id=data.external_id,
        raw_metadata=data.raw_metadata or {}
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

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
    workspace_id: Optional[str] = None,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    memory_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Memory).filter(Memory.status != "forgotten")
    if workspace_id:
        q = q.filter(Memory.workspace_id == workspace_id)
    if project_id:
        q = q.filter(Memory.project_id == project_id)
    if status:
        q = q.filter(Memory.status == status)
    if memory_type:
        q = q.filter(Memory.memory_type == memory_type)
    return q.order_by(Memory.updated_at.desc()).all()

@app.post("/api/v1/memories", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def create_memory(data: MemoryCreate, db: Session = Depends(get_db)):
    ws_id = data.workspace_id or "default-workspace"
    ws = db.query(Workspace).filter(Workspace.id == ws_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{ws_id}' not found")

    if data.project_id:
        proj = db.query(Project).filter(Project.id == data.project_id).first()
        if not proj:
            raise HTTPException(status_code=404, detail=f"Project '{data.project_id}' not found")
        if proj.workspace_id != ws_id:
            raise HTTPException(status_code=400, detail="Project does not belong to specified workspace")

    extraction_method = data.extraction_method
    if not data.source_message_id and not data.source_conversation_id:
        extraction_method = "user_explicit"

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
        extraction_method=extraction_method
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)

    # Update Vector Store & Graph
    vector_store.upsert(
        mem.id,
        f"{mem.statement} {mem.rationale or ''}",
        {"workspace_id": mem.workspace_id, "project_id": mem.project_id, "status": mem.status}
    )
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
            vector_store.upsert(
                mem.id,
                f"{mem.statement} {mem.rationale or ''}",
                {"workspace_id": mem.workspace_id, "project_id": mem.project_id, "status": mem.status}
            )
        graph_service.sync_from_db(db)
        return mem
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- Ingestion & Extraction ---
@app.post("/api/v1/imports/conversations")
def import_conversations(
    provider: str,
    payload: List[Dict[str, Any]],
    workspace_id: Optional[str] = None,
    provider_account_id: Optional[str] = None,
    project_id: Optional[str] = None,
    auto_extract: bool = True,
    db: Session = Depends(get_db)
):
    ws_id = workspace_id or "default-workspace"
    ws = db.query(Workspace).filter(Workspace.id == ws_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{ws_id}' not found")

    if project_id:
        proj = db.query(Project).filter(Project.id == project_id).first()
        if not proj:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
        if proj.workspace_id != ws_id:
            raise HTTPException(status_code=400, detail="Project does not belong to specified workspace")

    if provider_account_id:
        pa = db.query(ProviderAccount).filter(ProviderAccount.id == provider_account_id).first()
        if not pa:
            raise HTTPException(status_code=404, detail=f"ProviderAccount '{provider_account_id}' not found")

    normalized_convs = importer_registry.import_conversations(provider, payload)
    imported_ids = []
    total_extracted_memories = 0
    all_diffs = []

    for nc in normalized_convs:
        conv = Conversation(
            title=nc.title,
            provider_account_id=provider_account_id,
            external_id=nc.external_id,
            created_at=nc.created_at or datetime.now(timezone.utc),
            updated_at=nc.updated_at or datetime.now(timezone.utc),
            raw_metadata=nc.raw_metadata
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        imported_ids.append(conv.id)

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
                            workspace_id=ws_id,
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
                        vector_store.upsert(
                            mem.id,
                            f"{mem.statement} {mem.rationale or ''}",
                            {"workspace_id": mem.workspace_id, "project_id": mem.project_id, "status": mem.status}
                        )
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
    workspace_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 15,
    include_superseded: bool = False,
    db: Session = Depends(get_db)
):
    return retrieval_engine.search(
        db=db,
        query=query,
        workspace_id=workspace_id,
        project_id=project_id,
        limit=limit,
        include_superseded=include_superseded
    )

# --- Context Engine ---
@app.get("/api/v1/context", response_model=PortableContextPackage)
def get_context_package(
    query: str,
    workspace_id: Optional[str] = None,
    project_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return context_engine.build_context(db=db, query=query, workspace_id=workspace_id, project_id=project_id)

# --- Knowledge Graph ---
@app.get("/api/v1/graph", response_model=GraphData)
def get_knowledge_graph(
    workspace_id: Optional[str] = None,
    center_node: Optional[str] = None,
    depth: int = 2,
    db: Session = Depends(get_db)
):
    graph_service.sync_from_db(db, workspace_id=workspace_id)
    return graph_service.get_subgraph(center_node=center_node, depth=depth)

# --- Provider Capabilities ---
@app.get("/api/v1/providers/capabilities")
def get_provider_capabilities():
    return provider_registry.list_all_capabilities()

# --- Export & Import ---
@app.get("/api/v1/exports/canonical")
def export_canonical_memory(
    workspace_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return canonical_export_engine.export_all(db, workspace_id=workspace_id)

@app.post("/api/v1/imports/canonical")
def import_canonical_memory(payload: Dict[str, Any], db: Session = Depends(get_db)):
    counts = canonical_export_engine.import_all(db, payload)
    graph_service.sync_from_db(db)
    vector_store.rebuild_from_db(db)
    return {"status": "success", "imported_counts": counts}

# --- Dashboard Stats ---
@app.get("/api/v1/dashboard/stats")
def get_dashboard_stats(workspace_id: Optional[str] = None, db: Session = Depends(get_db)):
    mem_q = db.query(Memory).filter(Memory.status != "forgotten")
    proj_q = db.query(Project).filter(Project.status == "active")
    conv_q = db.query(Conversation)
    rev_q = db.query(Memory).filter(Memory.status == "review_required")

    if workspace_id:
        mem_q = mem_q.filter(Memory.workspace_id == workspace_id)
        proj_q = proj_q.filter(Project.workspace_id == workspace_id)
        rev_q = rev_q.filter(Memory.workspace_id == workspace_id)

    total_memories = mem_q.count()
    active_projects = proj_q.count()
    total_conversations = conv_q.count()
    pending_reviews = rev_q.count()
    stale_memories = rev_q.count()
    accounts = db.query(ProviderAccount).count()

    recent_mems = mem_q.order_by(Memory.updated_at.desc()).limit(5).all()

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
