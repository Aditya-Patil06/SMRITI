"""SMRITI Phase 4.1 — Local Model Context Protocol (MCP) Server (Read-Only MVP).

Provides a local stdio MCP server for Claude Desktop and compatible MCP clients
to access SMRITI workspaces, projects, hybrid memory search, and synthesized context packages.
"""

import argparse
from contextlib import contextmanager
import logging
import os
import sys
from typing import Any, Dict, Generator, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import StrictBool, StrictInt, StrictStr
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from smriti.config import settings
from smriti.context_engine import context_engine
from smriti.graph import graph_service
from smriti.models import Project, SessionLocal, User, Workspace, engine, init_db
from smriti.retrieval import retrieval_engine
from smriti.vector_store import vector_store

# Configure standard error logging to keep stdout strictly for JSON-RPC MCP framing
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smriti.mcp_server")

# Track server bootstrap state to avoid duplicate DB/index syncs
_bootstrap_completed = False


def check_database_initialized(db: Session) -> None:
    """Verify that required database schema tables exist before running read-only operations."""
    target_engine = db.get_bind() if hasattr(db, "get_bind") else engine
    insp = inspect(target_engine)
    required_tables = ("users", "workspaces", "projects", "memories")
    missing_tables = [t for t in required_tables if not insp.has_table(t)]
    if missing_tables:
        raise RuntimeError(
            f"SMRITI database is not initialized (missing tables: {missing_tables}). "
            "Run 'python -m smriti.mcp_server --init-db' or the SMRITI API/migrations to prepare the database."
        )



def get_active_mcp_user_id(db: Session) -> Optional[str]:
    """Retrieve and validate the configured or default MCP user ID."""
    configured_user_id = os.getenv("SMRITI_MCP_USER_ID") or settings.mcp_user_id
    if configured_user_id:
        clean_user_id = configured_user_id.strip()
        if not clean_user_id:
            raise RuntimeError(
                "SMRITI_MCP_USER_ID is set but empty. Provide a valid user ID or leave unset."
            )
        user = db.query(User).filter(User.id == clean_user_id).first()
        if not user:
            raise RuntimeError(
                f"Configured SMRITI_MCP_USER_ID '{clean_user_id}' does not exist in the database."
            )
        return clean_user_id

    # If no explicit SMRITI_MCP_USER_ID is provided, inspect existing users
    all_users = db.query(User).all()
    if not all_users:
        raise RuntimeError(
            "SMRITI database has no registered users. "
            "Run administrative initialization ('python -m smriti.mcp_server --init-db') before starting the server."
        )
    if len(all_users) == 1:
        return all_users[0].id

    # Multiple users found and no SMRITI_MCP_USER_ID specified
    raise RuntimeError(
        f"Multiple users ({len(all_users)}) exist in the database. "
        "You must configure SMRITI_MCP_USER_ID to isolate the MCP session to a specific user."
    )


def bootstrap_server(allow_write_init: bool = False, rebuild_indices: bool = False) -> None:
    """Validate database readiness and sync in-memory graph.

    By default, ordinary MCP server startup is strictly read-only and will never execute DDL,
    seed users/workspaces, or rebuild persistent vector files.
    """
    global _bootstrap_completed
    if _bootstrap_completed:
        return

    if allow_write_init:
        logger.info("Executing administrative database initialization (--init-db)...")
        init_db()

    db: Session = SessionLocal()
    try:
        check_database_initialized(db)

        if rebuild_indices:
            logger.info("Rebuilding vector store indices from database (--rebuild-indices)...")
            vector_store.rebuild_from_db(db)

        # In-memory graph sync is a read-only projection into memory
        graph_service.sync_from_db(db)
        _bootstrap_completed = True
        logger.info("SMRITI read-only startup verification completed successfully.")
    except Exception:
        logger.exception("Failed during SMRITI bootstrap")
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional DB session with guaranteed cleanup."""
    bootstrap_server()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def validate_string_identifier(
    value: Any,
    field_name: str,
    max_length: int = 255,
) -> str:
    """Validate and sanitize a required string identifier field."""
    if not isinstance(value, str):
        raise ToolError(f"Invalid input: '{field_name}' must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ToolError(f"Invalid input: '{field_name}' must be a non-empty string.")
    if len(cleaned) > max_length:
        raise ToolError(
            f"Invalid input: '{field_name}' exceeds maximum length of {max_length} characters."
        )
    return cleaned


def validate_optional_string(
    value: Optional[Any],
    field_name: str,
    max_length: int,
) -> Optional[str]:
    """Validate an optional string parameter, returning cleaned value or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError(f"Invalid input: '{field_name}' must be a string or null.")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise ToolError(
            f"Invalid input: '{field_name}' exceeds maximum length of {max_length} characters."
        )
    return cleaned


def validate_limit(limit: Any) -> int:
    """Validate search memory limit parameter."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 20:
        raise ToolError("Invalid input: 'limit' must be an integer between 1 and 20.")
    return limit


def get_verified_workspace(db: Session, workspace_id: str) -> Workspace:
    """Look up a workspace by id and verify ownership by the active MCP user."""
    active_user_id = get_active_mcp_user_id(db)
    query = db.query(Workspace).filter(Workspace.id == workspace_id)
    if active_user_id:
        query = query.filter(Workspace.user_id == active_user_id)

    workspace = query.first()
    if not workspace:
        # Uniform safe error message for missing or unauthorized workspaces
        raise ToolError(f"Workspace not found: '{workspace_id}'")
    return workspace


def get_verified_project(db: Session, project_id: str, workspace_id: str) -> Project:
    """Look up a project by id and verify it belongs to workspace_id."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project or project.workspace_id != workspace_id:
        raise ToolError(f"Project '{project_id}' not found in workspace '{workspace_id}'.")
    return project


def execute_list_workspaces() -> Dict[str, Any]:
    """Execute list_workspaces scoped strictly to the authenticated/configured user."""
    try:
        with get_db_session() as db:
            active_user_id = get_active_mcp_user_id(db)
            query = db.query(Workspace)
            if active_user_id:
                query = query.filter(Workspace.user_id == active_user_id)
            workspaces = query.order_by(Workspace.created_at.asc()).all()
            result = [
                {
                    "id": w.id,
                    "name": w.name,
                    "description": w.description or "",
                    "is_default": bool(w.is_default),
                }
                for w in workspaces
            ]
            return {"workspaces": result}
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("Error in list_workspaces")
        raise ToolError("Failed to list workspaces due to an internal error.") from exc


def execute_list_projects(workspace_id: StrictStr) -> Dict[str, Any]:
    """Execute list_projects business logic after validating workspace ownership."""
    w_id = validate_string_identifier(workspace_id, "workspace_id")
    try:
        with get_db_session() as db:
            get_verified_workspace(db, w_id)
            projects = (
                db.query(Project)
                .filter(Project.workspace_id == w_id)
                .order_by(Project.created_at.asc())
                .all()
            )
            result = [
                {
                    "id": p.id,
                    "workspace_id": p.workspace_id,
                    "name": p.name,
                    "description": p.description or "",
                    "goal": p.goal or "",
                    "status": p.status,
                }
                for p in projects
            ]
            return {"projects": result}
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("Error in list_projects")
        raise ToolError("Failed to list projects due to an internal error.") from exc


def execute_search_memories(
    query: StrictStr,
    workspace_id: StrictStr,
    project_id: Optional[StrictStr] = None,
    limit: StrictInt = 15,
    include_superseded: StrictBool = False,
) -> Dict[str, Any]:
    """Execute search_memories business logic after validating workspace ownership."""
    q = validate_string_identifier(query, "query", max_length=2000)
    w_id = validate_string_identifier(workspace_id, "workspace_id")
    p_id = validate_optional_string(project_id, "project_id", max_length=255)
    checked_limit = validate_limit(limit)

    if not isinstance(include_superseded, bool):
        raise ToolError("Invalid input: 'include_superseded' must be a boolean.")

    try:
        with get_db_session() as db:
            get_verified_workspace(db, w_id)
            if p_id:
                get_verified_project(db, p_id, w_id)

            hits = retrieval_engine.search(
                db=db,
                query=q,
                project_id=p_id,
                workspace_id=w_id,
                limit=checked_limit,
                include_superseded=bool(include_superseded),
            )

            items = [
                {
                    "id": hit.id,
                    "type": hit.type,
                    "title": hit.title,
                    "content": hit.content,
                    "score": round(float(hit.score), 4),
                    "metadata": hit.metadata,
                }
                for hit in hits
            ]
            return {"results": items}
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("Error in search_memories")
        raise ToolError("Failed to search memories due to an internal error.") from exc


def execute_build_context_package(
    workspace_id: StrictStr,
    project_id: StrictStr,
    query: Optional[StrictStr] = None,
    task_description: Optional[StrictStr] = None,
) -> Dict[str, Any]:
    """Execute build_context_package business logic after validating workspace ownership."""
    w_id = validate_string_identifier(workspace_id, "workspace_id")
    p_id = validate_string_identifier(project_id, "project_id")
    q_cleaned = validate_optional_string(query, "query", max_length=2000) or ""
    t_desc_cleaned = validate_optional_string(task_description, "task_description", max_length=4000)

    # Combine query and task_description if both are provided for memory retrieval and prompt synthesis
    effective_query = q_cleaned
    if t_desc_cleaned:
        if effective_query:
            effective_query = f"{effective_query} - {t_desc_cleaned}"
        else:
            effective_query = t_desc_cleaned

    try:
        with get_db_session() as db:
            get_verified_workspace(db, w_id)
            get_verified_project(db, p_id, w_id)

            package = context_engine.build_context(
                db=db,
                query=effective_query,
                project_id=p_id,
                workspace_id=w_id,
            )

            return package.model_dump()
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("Error in build_context_package")
        raise ToolError("Failed to build context package due to an internal error.") from exc


def create_mcp_server() -> MCPServer:
    """Factory creating and configuring the SMRITI MCP server with 4 read-only tools."""
    server = MCPServer(
        name="smriti",
        version="0.1.0",
        instructions=(
            "SMRITI Model Context Protocol server. Provides read-only access to "
            "workspaces, projects, hybrid memory search, and synthesized context packages."
        ),
    )

    @server.tool(
        name="list_workspaces",
        description="List all available SMRITI workspaces with their IDs, names, and descriptions.",
    )
    def list_workspaces() -> Dict[str, Any]:
        """List available workspaces without exposing user credentials or private auth metadata."""
        return execute_list_workspaces()

    @server.tool(
        name="list_projects",
        description="List projects belonging to a specified workspace.",
    )
    def list_projects(workspace_id: StrictStr) -> Dict[str, Any]:
        """List projects belonging to workspace_id.

        Validates workspace existence and isolates results to the given workspace.
        """
        return execute_list_projects(workspace_id)

    @server.tool(
        name="search_memories",
        description=(
            "Search SMRITI memories using hybrid retrieval (lexical, vector, and knowledge graph). "
            "Requires query and workspace_id. Optional project_id, limit (1-20), and include_superseded."
        ),
    )
    def search_memories(
        query: StrictStr,
        workspace_id: StrictStr,
        project_id: Optional[StrictStr] = None,
        limit: StrictInt = 15,
        include_superseded: StrictBool = False,
    ) -> Dict[str, Any]:
        """Perform hybrid memory search scoped strictly to the given workspace.

        Validates workspace existence and ensures any supplied project belongs to that workspace.
        """
        return execute_search_memories(
            query=query,
            workspace_id=workspace_id,
            project_id=project_id,
            limit=limit,
            include_superseded=include_superseded,
        )

    @server.tool(
        name="build_context_package",
        description=(
            "Synthesize a portable context package containing relevant memories, active tasks, "
            "architecture choices, constraints, and a formatted prompt for target AI models. "
            "Requires workspace_id and project_id. Optional query and task_description."
        ),
    )
    def build_context_package(
        workspace_id: StrictStr,
        project_id: StrictStr,
        query: Optional[StrictStr] = None,
        task_description: Optional[StrictStr] = None,
    ) -> Dict[str, Any]:
        """Synthesize a portable context package scoped strictly to a workspace.

        Validates workspace existence and ensures the supplied project belongs to that workspace.
        """
        return execute_build_context_package(
            workspace_id=workspace_id,
            project_id=project_id,
            query=query,
            task_description=task_description,
        )

    return server


# Global module-level server instance
mcp_server = create_mcp_server()


def main() -> None:
    """Run the SMRITI MCP server using stdio transport or execute administrative initialization."""
    parser = argparse.ArgumentParser(description="SMRITI Model Context Protocol (MCP) server")
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Explicitly initialize database tables and seed default entities",
    )
    parser.add_argument(
        "--rebuild-indices",
        action="store_true",
        help="Explicitly rebuild vector store and graph indices from the database",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Execute administrative initialization and index rebuild, then exit without starting stdio server",
    )
    args, _ = parser.parse_known_args()

    allow_write = args.init_db or args.init_only
    rebuild = args.rebuild_indices or args.init_only

    bootstrap_server(allow_write_init=allow_write, rebuild_indices=rebuild)

    if args.init_only:
        logger.info("Administrative initialization completed. Exiting.")
        return

    logger.info("Starting SMRITI MCP server on stdio transport...")
    mcp_server.run(transport="stdio")



if __name__ == "__main__":
    main()
