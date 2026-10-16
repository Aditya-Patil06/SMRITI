"""SMRITI Phase 4.2 — Dedicated Read-Only ChatGPT Custom GPT Actions Application.

Provides a dedicated, strictly read-only ASGI application with an isolated OpenAPI
schema for ChatGPT Custom GPT Actions.
"""

from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from smriti.config import settings
from smriti.context_engine import context_engine
from smriti.graph import graph_service
from smriti.models import (
    Project,
    SessionLocal,
    User,
    Workspace,
    engine,
    get_db,
)
from smriti.retrieval import retrieval_engine
from smriti.schemas import (
    PortableContextPackage,
    ProjectRead,
    SearchResultItem,
    WorkspaceRead,
)
from smriti.vector_store import (
    InMemoryVectorStore,
    get_vector_store,
    vector_store,
)

logger = logging.getLogger("smriti.actions_app")

# Process-local lock protecting cache synchronization and search/context execution.
# This prevents race conditions where concurrent requests interleave graph clearing,
# graph rebuilding, or vector cache reloads.
_actions_lock = threading.Lock()

# Vector cache signature tracking: (st_mtime_ns, st_size)
_last_vector_signature: Optional[Tuple[int, int]] = None

# HTTP Bearer scheme for OpenAPI and credential extraction.
# auto_error=False allows uniform, explicit 401 handling with WWW-Authenticate: Bearer.
bearer_scheme = HTTPBearer(scheme_name="BearerAuth", auto_error=False)


def check_database_initialized(db: Session) -> None:
    """Verify that required database schema tables exist without executing DDL."""
    target_engine = db.get_bind() if hasattr(db, "get_bind") else engine
    insp = inspect(target_engine)
    required_tables = ("users", "workspaces", "projects", "memories")
    missing_tables = [t for t in required_tables if not insp.has_table(t)]
    if missing_tables:
        raise RuntimeError(
            f"SMRITI database is not initialized (missing tables: {missing_tables}). "
            "Initialize the database using standard migrations or smriti.mcp_server --init-db before starting Actions."
        )


def validate_startup_configuration() -> str:
    """Validate SMRITI_ACTIONS_API_KEY and SMRITI_ACTIONS_USER_ID at startup.

    Fails closed immediately if either variable is unset or blank, or if the user
    does not exist in the database. Never creates tables or seeds records.
    """
    api_key = os.getenv("SMRITI_ACTIONS_API_KEY") or settings.actions_api_key
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "SMRITI_ACTIONS_API_KEY is not configured or is empty. "
            "You must configure a secret bearer token to start the ChatGPT Actions application."
        )

    user_id = os.getenv("SMRITI_ACTIONS_USER_ID") or settings.actions_user_id
    if not user_id or not user_id.strip():
        raise RuntimeError(
            "SMRITI_ACTIONS_USER_ID is not configured or is empty. "
            "You must explicitly specify the bound user ID for the ChatGPT Actions application."
        )

    clean_user_id = user_id.strip()

    # Verify database schema and user presence without modifying state
    db: Session = SessionLocal()
    try:
        check_database_initialized(db)
        user = db.query(User).filter(User.id == clean_user_id).first()
        if not user:
            raise RuntimeError(
                f"Configured SMRITI_ACTIONS_USER_ID '{clean_user_id}' does not exist in the database. "
                "Ensure the user exists before launching the Actions application."
            )
        return clean_user_id
    finally:
        db.close()


def get_configured_api_key() -> str:
    """Retrieve the configured API key or raise an error."""
    api_key = os.getenv("SMRITI_ACTIONS_API_KEY") or settings.actions_api_key
    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChatGPT Actions API is not configured on this server.",
        )
    return api_key.strip()


def get_configured_user_id() -> str:
    """Retrieve the configured user ID or raise an error."""
    user_id = os.getenv("SMRITI_ACTIONS_USER_ID") or settings.actions_user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChatGPT Actions User ID is not configured on this server.",
        )
    return user_id.strip()


def verify_actions_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Verify bearer authentication and return the bound User.

    Uses constant-time comparison against SMRITI_ACTIONS_API_KEY.
    Returns HTTP 401 with WWW-Authenticate: Bearer on failure.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expected_key = get_configured_api_key()
    provided_key = credentials.credentials.strip()

    if not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    target_user_id = get_configured_user_id()
    user = db.query(User).filter(User.id == target_user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configured Actions user could not be found.",
        )
    return user


def refresh_caches_if_needed(db: Session) -> None:
    """Synchronize process-local vector cache and knowledge graph read-only.

    Must be called inside _actions_lock.
    - Vector index: checks storage/vectors.json signature (st_mtime_ns, st_size) and
      re-reads the file read-only if it changed. Never saves or rebuilds.
    - Knowledge graph: synchronizes the graph from DB without a workspace filter
      so that all workspaces remain queryable and one workspace request cannot
      purge another workspace's nodes via graph.clear().
    """
    global _last_vector_signature

    # 1. Read-only vector file signature reload (only for JSON-backed in-memory store)
    storage_path = getattr(vector_store, "storage_path", None)
    load_fn = getattr(vector_store, "load", None)
    if (
        isinstance(get_vector_store(), InMemoryVectorStore)
        and callable(load_fn)
        and storage_path
        and os.path.exists(storage_path)
    ):
        try:
            stat_res = os.stat(storage_path)
            sig = (stat_res.st_mtime_ns, stat_res.st_size)
            if _last_vector_signature is None or sig != _last_vector_signature:
                # Pre-validate file read and JSON parsing because InMemoryVectorStore.load()
                # suppresses exceptions and silently clears vectors to {} on corruption.
                try:
                    with open(storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("Vector cache payload must be a JSON object mapping doc IDs.")
                except Exception as parse_err:
                    logger.exception("Corrupted or unreadable vector store cache file detected")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Memory search cache is currently unavailable.",
                    ) from parse_err

                load_fn()
                _last_vector_signature = sig
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error refreshing vector store cache from disk")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory search cache is currently unavailable.",
            ) from exc

    # 2. Knowledge graph synchronization (unfiltered)
    # Rebuilding on every search/context request ensures cross-process freshness
    # against mutations from the primary API. This is an explicit MVP correctness trade-off
    # with a performance cost as graph size increases. Note that a brief eventual-consistency
    # window remains if database transactions and vector file writes occur concurrently in the primary API.
    try:
        graph_service.sync_from_db(db, workspace_id=None)
    except Exception as exc:
        logger.exception("Error syncing knowledge graph from database")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory context graph is currently unavailable.",
        ) from exc


def get_verified_workspace(db: Session, workspace_id: str, user_id: str) -> Workspace:
    """Look up a workspace by id and verify ownership by the bound Actions user."""
    ws = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id, Workspace.user_id == user_id)
        .first()
    )
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )
    return ws


def get_verified_project(db: Session, project_id: str, workspace_id: str) -> Project:
    """Look up a project by id and verify it belongs to workspace_id."""
    proj = (
        db.query(Project)
        .filter(Project.id == project_id, Project.workspace_id == workspace_id)
        .first()
    )
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found in workspace '{workspace_id}'.",
        )
    return proj


@asynccontextmanager
async def actions_lifespan(app_instance: FastAPI):
    """Validate environment configuration and database readiness on startup."""
    validate_startup_configuration()
    # Initial read-only vector load for JSON-backed in-memory store
    load_fn = getattr(vector_store, "load", None)
    if isinstance(get_vector_store(), InMemoryVectorStore) and callable(load_fn):
        load_fn()
    yield


app = FastAPI(
    title="Project SMRITI ChatGPT Actions API",
    description="Dedicated read-only API surface for ChatGPT Custom GPT Actions to retrieve SMRITI workspaces, projects, memories, and context packages.",
    version="0.1.0",
    docs_url=None,       # Disable interactive Swagger UI
    redoc_url=None,      # Disable ReDoc UI
    openapi_url=None,    # Disabled unauthenticated default route
    lifespan=actions_lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    # Filter routes to exclude the openapi.json route itself from the exported schema
    schema_routes = [
        route for route in app.routes
        if getattr(route, "path", None) != "/actions/openapi.json"
    ]
    openapi_schema = get_openapi(
        title="Project SMRITI ChatGPT Actions API",
        version="0.1.0",
        description="Dedicated read-only API surface for ChatGPT Custom GPT Actions.",
        routes=schema_routes,
    )
    # Add Bearer authentication security scheme to OpenAPI
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "APIKey",
            "description": "Provide your SMRITI Actions API Key as a Bearer token.",
        }
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    # Ensure every operation's security requirement explicitly references BearerAuth
    for path_item in openapi_schema.get("paths", {}).values():
        for op in path_item.values():
            op["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# --- Authenticated OpenAPI Schema Endpoint ---
@app.get("/actions/openapi.json", include_in_schema=False)
def get_actions_openapi_schema(current_user: User = Depends(verify_actions_auth)):
    """Return the dedicated OpenAPI schema for ChatGPT Actions, protected by bearer authentication."""
    return custom_openapi()


# --- 1. GET /actions/workspaces ---
@app.get("/actions/workspaces", response_model=List[WorkspaceRead])
def list_workspaces(
    current_user: User = Depends(verify_actions_auth),
    db: Session = Depends(get_db),
):
    """List workspaces owned by the authenticated Actions user."""
    return (
        db.query(Workspace)
        .filter(Workspace.user_id == current_user.id)
        .order_by(Workspace.created_at.asc())
        .all()
    )


# --- 2. GET /actions/projects ---
@app.get("/actions/projects", response_model=List[ProjectRead])
def list_projects(
    workspace_id: str = Query(..., description="Target workspace ID (must be owned by the user)"),
    current_user: User = Depends(verify_actions_auth),
    db: Session = Depends(get_db),
):
    """List projects in an owned workspace."""
    get_verified_workspace(db, workspace_id, current_user.id)
    return (
        db.query(Project)
        .filter(Project.workspace_id == workspace_id)
        .order_by(Project.created_at.asc())
        .all()
    )


# --- 3. GET /actions/search ---
@app.get("/actions/search", response_model=List[SearchResultItem])
def search_memories(
    query: str = Query(..., min_length=1, description="Search query string"),
    workspace_id: str = Query(..., min_length=1, description="Target workspace ID (must be owned by the user)"),
    project_id: Optional[str] = Query(None, description="Optional project ID to scope search within workspace"),
    limit: int = Query(15, ge=1, le=20, description="Maximum results to return (1-20)"),
    current_user: User = Depends(verify_actions_auth),
    db: Session = Depends(get_db),
):
    """Search memories within an owned workspace using hybrid lexical and vector retrieval."""
    with _actions_lock:
        get_verified_workspace(db, workspace_id, current_user.id)
        if project_id:
            get_verified_project(db, project_id, workspace_id)
        refresh_caches_if_needed(db)
        return retrieval_engine.search(
            db=db,
            query=query,
            workspace_id=workspace_id,
            project_id=project_id,
            limit=limit,
            include_superseded=False,
        )


# --- 4. GET /actions/context ---
@app.get("/actions/context", response_model=PortableContextPackage)
def get_context_package(
    query: str = Query(..., min_length=1, description="Query string for relevant context retrieval"),
    workspace_id: str = Query(..., min_length=1, description="Target workspace ID (must be owned by the user)"),
    project_id: Optional[str] = Query(None, description="Optional project ID to scope context package"),
    current_user: User = Depends(verify_actions_auth),
    db: Session = Depends(get_db),
):
    """Synthesize a portable context package for a query scoped to an owned workspace."""
    with _actions_lock:
        get_verified_workspace(db, workspace_id, current_user.id)
        if project_id:
            get_verified_project(db, project_id, workspace_id)
        refresh_caches_if_needed(db)
        return context_engine.build_context(
            db=db,
            query=query,
            workspace_id=workspace_id,
            project_id=project_id,
        )
