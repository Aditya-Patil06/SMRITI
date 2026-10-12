"""SMRITI Phase 4.1 — Local Model Context Protocol (MCP) Server (Read-Only MVP).

Provides a local stdio MCP server for Claude Desktop and compatible MCP clients
to access SMRITI workspaces, projects, hybrid memory search, and synthesized context packages.
"""

from contextlib import contextmanager
import logging
import os
import sys
from typing import Any, Dict, Generator, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import StrictBool, StrictInt, StrictStr
from sqlalchemy.orm import Session

from smriti.context_engine import context_engine
from smriti.graph import graph_service
from smriti.models import Project, SessionLocal, Workspace, init_db
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


def bootstrap_server() -> None:
    """Initialize database tables and sync graph/vector stores if needed."""
    global _bootstrap_completed
    if _bootstrap_completed:
        return

    logger.info("Bootstrapping SMRITI database and memory indices...")
    init_db()

    db: Session = SessionLocal()
    try:
        graph_service.sync_from_db(db)
        vector_store.rebuild_from_db(db)
        _bootstrap_completed = True
        logger.info("SMRITI bootstrap completed successfully.")
    except Exception as exc:
        logger.error("Failed during SMRITI bootstrap: %s", exc)
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
        try:
            with get_db_session() as db:
                workspaces = db.query(Workspace).order_by(Workspace.created_at.asc()).all()
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
            logger.error("Error in list_workspaces: %s", exc)
            raise ToolError("Failed to list workspaces due to an internal error.") from exc

    @server.tool(
        name="list_projects",
        description="List projects belonging to a specified workspace.",
    )
    def list_projects(workspace_id: StrictStr) -> Dict[str, Any]:
        """List projects belonging to workspace_id.

        Validates workspace existence and isolates results to the given workspace.
        """
        if not isinstance(workspace_id, str):
            raise ToolError("Invalid input: 'workspace_id' must be a string.")

        w_id = workspace_id.strip()
        if not w_id:
            raise ToolError("Invalid input: 'workspace_id' must be a non-empty string.")
        if len(w_id) > 255:
            raise ToolError("Invalid input: 'workspace_id' exceeds maximum length of 255 characters.")

        try:
            with get_db_session() as db:
                workspace = db.query(Workspace).filter(Workspace.id == w_id).first()
                if not workspace:
                    raise ToolError(f"Workspace not found: '{w_id}'")

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
            logger.error("Error in list_projects: %s", exc)
            raise ToolError("Failed to list projects due to an internal error.") from exc

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
        if not isinstance(query, str):
            raise ToolError("Invalid input: 'query' must be a string.")
        q = query.strip()
        if not q:
            raise ToolError("Invalid input: 'query' must be a non-empty string.")
        if len(q) > 2000:
            raise ToolError("Invalid input: 'query' exceeds maximum length of 2000 characters.")

        if not isinstance(workspace_id, str):
            raise ToolError("Invalid input: 'workspace_id' must be a string.")
        w_id = workspace_id.strip()
        if not w_id:
            raise ToolError("Invalid input: 'workspace_id' must be a non-empty string.")
        if len(w_id) > 255:
            raise ToolError("Invalid input: 'workspace_id' exceeds maximum length of 255 characters.")

        p_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str):
                raise ToolError("Invalid input: 'project_id' must be a string or null.")
            p_id_cleaned = project_id.strip()
            if p_id_cleaned:
                if len(p_id_cleaned) > 255:
                    raise ToolError("Invalid input: 'project_id' exceeds maximum length of 255 characters.")
                p_id = p_id_cleaned

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ToolError("Invalid input: 'limit' must be an integer between 1 and 20.")
        if limit < 1 or limit > 20:
            raise ToolError("Invalid input: 'limit' must be an integer between 1 and 20.")

        if not isinstance(include_superseded, bool):
            raise ToolError("Invalid input: 'include_superseded' must be a boolean.")

        try:
            with get_db_session() as db:
                workspace = db.query(Workspace).filter(Workspace.id == w_id).first()
                if not workspace:
                    raise ToolError(f"Workspace not found: '{w_id}'")

                if p_id:
                    project = db.query(Project).filter(Project.id == p_id).first()
                    if not project or project.workspace_id != w_id:
                        raise ToolError(
                            f"Project '{p_id}' not found in workspace '{w_id}'."
                        )

                hits = retrieval_engine.search(
                    db=db,
                    query=q,
                    project_id=p_id,
                    workspace_id=w_id,
                    limit=limit,
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
            logger.error("Error in search_memories: %s", exc)
            raise ToolError("Failed to search memories due to an internal error.") from exc

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
        if not isinstance(workspace_id, str):
            raise ToolError("Invalid input: 'workspace_id' must be a string.")
        w_id = workspace_id.strip()
        if not w_id:
            raise ToolError("Invalid input: 'workspace_id' must be a non-empty string.")
        if len(w_id) > 255:
            raise ToolError("Invalid input: 'workspace_id' exceeds maximum length of 255 characters.")

        if not isinstance(project_id, str):
            raise ToolError("Invalid input: 'project_id' must be a string.")
        p_id = project_id.strip()
        if not p_id:
            raise ToolError("Invalid input: 'project_id' must be a non-empty string.")
        if len(p_id) > 255:
            raise ToolError("Invalid input: 'project_id' exceeds maximum length of 255 characters.")

        q_cleaned = ""
        if query is not None:
            if not isinstance(query, str):
                raise ToolError("Invalid input: 'query' must be a string or null.")
            q_candidate = query.strip()
            if len(q_candidate) > 2000:
                raise ToolError("Invalid input: 'query' exceeds maximum length of 2000 characters.")
            q_cleaned = q_candidate

        t_desc_cleaned: Optional[str] = None
        if task_description is not None:
            if not isinstance(task_description, str):
                raise ToolError("Invalid input: 'task_description' must be a string or null.")
            t_candidate = task_description.strip()
            if len(t_candidate) > 4000:
                raise ToolError("Invalid input: 'task_description' exceeds maximum length of 4000 characters.")
            t_desc_cleaned = t_candidate

        # Combine query and task_description if both are provided for memory retrieval and prompt synthesis
        effective_query = q_cleaned
        if t_desc_cleaned:
            if effective_query:
                effective_query = f"{effective_query} - {t_desc_cleaned}"
            else:
                effective_query = t_desc_cleaned

        try:
            with get_db_session() as db:
                workspace = db.query(Workspace).filter(Workspace.id == w_id).first()
                if not workspace:
                    raise ToolError(f"Workspace not found: '{w_id}'")

                project = db.query(Project).filter(Project.id == p_id).first()
                if not project or project.workspace_id != w_id:
                    raise ToolError(
                        f"Project '{p_id}' not found in workspace '{w_id}'."
                    )

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
            logger.error("Error in build_context_package: %s", exc)
            raise ToolError("Failed to build context package due to an internal error.") from exc

    return server


# Global module-level server instance
mcp_server = create_mcp_server()


def main() -> None:
    """Run the SMRITI MCP server using stdio transport."""
    bootstrap_server()
    logger.info("Starting SMRITI MCP server on stdio transport...")
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
