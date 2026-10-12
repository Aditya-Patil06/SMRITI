"""Tests for SMRITI Phase 4.1 Local MCP Server (Read-Only MVP).

Validates:
1. Tool registration: exactly 4 tools with valid metadata and descriptions.
2. Workspace and project listing with strict scoping and credential exclusion.
3. Hybrid memory search with workspace/project scoping, bounds checking, and superseded filtering.
4. Context package generation with valid formatted prompt.
5. Missing workspace and cross-workspace project access rejection.
6. Input validation (empty queries, query length, limit out of bounds).
7. Database session lifecycle and cleanup on success and error.
8. Official MCP SDK in-memory client session tool invocation.
"""

import asyncio
from datetime import datetime, timezone
import json
import pytest
import anyio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from smriti.models import Base, User, Workspace, Project, Task, Memory, Message, Conversation
from smriti.vector_store import vector_store
from smriti.graph import graph_service
import smriti.mcp_server as mcp_server_module
from smriti.mcp_server import create_mcp_server
from mcp.server.mcpserver.exceptions import ToolError
from mcp.client.session import ClientSession


@pytest.fixture
def mcp_test_db(tmp_path, monkeypatch):
    """Create an isolated SQLite database and patch SessionLocal and bootstrap state."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Seed test entities
    session = TestingSessionLocal()
    user = User(id="user-1", username="testuser", email="test@smriti.local")
    session.add(user)

    ws_personal = Workspace(id="ws-personal", user_id="user-1", name="Personal Workspace", description="Personal items", is_default=True)
    ws_work = Workspace(id="ws-work", user_id="user-1", name="Work Workspace", description="Work projects", is_default=False)
    session.add_all([ws_personal, ws_work])

    # Projects
    proj_personal = Project(
        id="proj-p1",
        workspace_id="ws-personal",
        name="Home Automation",
        description="Smart home setup",
        goal="Automate living room lights",
        status="active",
        architecture_overview="MQTT + HomeAssistant",
        tech_stack=["python", "mqtt"],
        constraints=["Local network only"],
    )
    proj_work = Project(
        id="proj-w1",
        workspace_id="ws-work",
        name="Enterprise Cloud",
        description="Cloud deployment",
        goal="Migrate workloads to Kubernetes",
        status="active",
        architecture_overview="Microservices on EKS",
        tech_stack=["kubernetes", "go"],
        constraints=["SOC2 compliance"],
    )
    session.add_all([proj_personal, proj_work])

    # Tasks
    task_p = Task(id="task-p1", project_id="proj-p1", title="Setup zigbee broker", status="todo", priority="high")
    task_w = Task(id="task-w1", project_id="proj-w1", title="Configure RBAC", status="in_progress", priority="critical")
    session.add_all([task_p, task_w])

    # Memories
    mem_active = Memory(
        id="mem-1",
        workspace_id="ws-personal",
        project_id="proj-p1",
        statement="Home Assistant requires Zigbee2MQTT for Sonoff dongle",
        rationale="Official documentation recommends Zigbee2MQTT",
        memory_type="decision",
        status="active",
        confidence=0.95,
    )
    mem_superseded = Memory(
        id="mem-2",
        workspace_id="ws-personal",
        project_id="proj-p1",
        statement="Home Assistant uses ZHA integration",
        rationale="Initial prototype test",
        memory_type="decision",
        status="superseded",
        confidence=0.7,
    )
    mem_work = Memory(
        id="mem-3",
        workspace_id="ws-work",
        project_id="proj-w1",
        statement="Kubernetes cluster requires Calico CNI plugin",
        rationale="Needed for network policy enforcement",
        memory_type="decision",
        status="active",
        confidence=0.9,
    )
    session.add_all([mem_active, mem_superseded, mem_work])
    session.commit()

    # Isolate vector_store storage path and index, capturing original state for teardown
    orig_storage_path = vector_store.storage_path
    orig_vectors = dict(vector_store.vectors)
    orig_metadata = dict(vector_store.metadata)
    orig_graph = graph_service.graph.copy()

    test_storage = str(tmp_path / "test_mcp_vectors.json")
    vector_store.storage_path = test_storage
    vector_store.vectors.clear()
    vector_store.metadata.clear()
    vector_store.upsert(mem_active.id, mem_active.statement, {"workspace_id": "ws-personal", "project_id": "proj-p1", "memory_type": "decision", "status": "active"})
    vector_store.upsert(mem_superseded.id, mem_superseded.statement, {"workspace_id": "ws-personal", "project_id": "proj-p1", "memory_type": "decision", "status": "superseded"})
    vector_store.upsert(mem_work.id, mem_work.statement, {"workspace_id": "ws-work", "project_id": "proj-w1", "memory_type": "decision", "status": "active"})

    # Sync graph
    graph_service.sync_from_db(session)

    session.close()

    # Patch SessionLocal and bypass re-initialization of Base.metadata.create_all on production engine
    monkeypatch.setattr(mcp_server_module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(mcp_server_module, "init_db", lambda: None)
    monkeypatch.setattr(mcp_server_module, "_bootstrap_completed", True)

    try:
        yield TestingSessionLocal
    finally:
        vector_store.storage_path = orig_storage_path
        vector_store.vectors = orig_vectors
        vector_store.metadata = orig_metadata
        graph_service.graph = orig_graph


@pytest.mark.asyncio
async def test_tool_registration(mcp_test_db):
    """Ensure exactly four expected tools are registered with proper descriptions."""
    server = create_mcp_server()
    tools = await server.list_tools()

    tool_names = [t.name for t in tools]
    assert len(tool_names) == 4
    assert sorted(tool_names) == sorted([
        "list_workspaces",
        "list_projects",
        "search_memories",
        "build_context_package"
    ])

    tool_map = {t.name: t for t in tools}
    assert "workspaces" in tool_map["list_workspaces"].description.lower()
    assert "projects" in tool_map["list_projects"].description.lower()
    assert "search" in tool_map["search_memories"].description.lower()
    assert "context" in tool_map["build_context_package"].description.lower()


@pytest.mark.asyncio
async def test_list_workspaces(mcp_test_db):
    """Verify list_workspaces returns sanitized, scoped workspace records."""
    server = create_mcp_server()
    res = await server.call_tool("list_workspaces", {})

    workspaces = json.loads(res.content[0].text)["workspaces"]
    assert len(workspaces) >= 2
    ws_ids = [w["id"] for w in workspaces]
    assert "ws-personal" in ws_ids
    assert "ws-work" in ws_ids

    # Verify no credential, user email, or password data leaks
    for w in workspaces:
        assert "user_id" not in w
        assert "auth_metadata" not in w
        assert "email" not in w


@pytest.mark.asyncio
async def test_list_projects_scoped_and_validation(mcp_test_db):
    """Verify list_projects strictly isolates projects to the requested workspace."""
    server = create_mcp_server()

    # Personal workspace projects
    res_p = await server.call_tool("list_projects", {"workspace_id": "ws-personal"})
    projects_p = json.loads(res_p.content[0].text)["projects"]
    assert len(projects_p) == 1
    assert projects_p[0]["id"] == "proj-p1"
    assert projects_p[0]["name"] == "Home Automation"

    # Work workspace projects
    res_w = await server.call_tool("list_projects", {"workspace_id": "ws-work"})
    projects_w = json.loads(res_w.content[0].text)["projects"]
    assert len(projects_w) == 1
    assert projects_w[0]["id"] == "proj-w1"
    assert projects_w[0]["name"] == "Enterprise Cloud"

    # Missing workspace error
    with pytest.raises(ToolError) as exc_info:
        await server.call_tool("list_projects", {"workspace_id": "nonexistent-ws"})
    assert "Workspace not found" in str(exc_info.value)

    # Empty workspace input error
    with pytest.raises(ToolError) as exc_info_empty:
        await server.call_tool("list_projects", {"workspace_id": "   "})
    assert "Invalid input" in str(exc_info_empty.value)


@pytest.mark.asyncio
async def test_search_memories_workspace_and_project_scoping(mcp_test_db):
    """Verify search_memories enforces workspace boundaries and project validation."""
    server = create_mcp_server()

    # Search in personal workspace
    res = await server.call_tool("search_memories", {
        "query": "Zigbee2MQTT",
        "workspace_id": "ws-personal",
    })
    results = json.loads(res.content[0].text)["results"]
    assert len(results) >= 1
    assert results[0]["id"] == "mem-1"
    assert "Zigbee2MQTT" in results[0]["content"]

    # Ensure search in personal workspace NEVER returns work memories
    res_cross = await server.call_tool("search_memories", {
        "query": "Kubernetes",
        "workspace_id": "ws-personal",
    })
    results_cross = json.loads(res_cross.content[0].text)["results"]
    assert all("Kubernetes" not in item["content"] for item in results_cross)

    # Superseded exclusion by default
    res_super_default = await server.call_tool("search_memories", {
        "query": "ZHA integration",
        "workspace_id": "ws-personal",
        "include_superseded": False,
    })
    res_super_items = json.loads(res_super_default.content[0].text)["results"]
    assert all(item["id"] != "mem-2" for item in res_super_items)

    # Superseded inclusion when requested
    res_super_inc = await server.call_tool("search_memories", {
        "query": "ZHA integration",
        "workspace_id": "ws-personal",
        "include_superseded": True,
    })
    res_inc_items = json.loads(res_super_inc.content[0].text)["results"]
    assert any(item["id"] == "mem-2" for item in res_inc_items)


@pytest.mark.asyncio
async def test_search_memories_cross_workspace_project_rejected(mcp_test_db):
    """Verify passing a project from another workspace raises ToolError without leaking data."""
    server = create_mcp_server()

    # proj-w1 belongs to ws-work, not ws-personal
    with pytest.raises(ToolError) as exc_info:
        await server.call_tool("search_memories", {
            "query": "test",
            "workspace_id": "ws-personal",
            "project_id": "proj-w1",
        })
    assert "not found in workspace" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_memories_input_validation(mcp_test_db):
    """Verify query length, empty query, and limit bounds checking."""
    server = create_mcp_server()

    # Empty query
    with pytest.raises(ToolError) as exc_empty:
        await server.call_tool("search_memories", {"query": "", "workspace_id": "ws-personal"})
    assert "query" in str(exc_empty.value).lower()

    # Query too long
    with pytest.raises(ToolError) as exc_long:
        await server.call_tool("search_memories", {"query": "x" * 2005, "workspace_id": "ws-personal"})
    assert "maximum length" in str(exc_long.value).lower()

    # Limit <= 0
    with pytest.raises(ToolError) as exc_limit_low:
        await server.call_tool("search_memories", {"query": "test", "workspace_id": "ws-personal", "limit": 0})
    assert "limit" in str(exc_limit_low.value).lower()

    # Limit = 20 boundary (valid)
    res_boundary_20 = await server.call_tool("search_memories", {"query": "Zigbee2MQTT", "workspace_id": "ws-personal", "limit": 20})
    assert not res_boundary_20.is_error

    # Limit = 1 boundary (valid)
    res_boundary_1 = await server.call_tool("search_memories", {"query": "Zigbee2MQTT", "workspace_id": "ws-personal", "limit": 1})
    assert not res_boundary_1.is_error

    # Limit > 20
    with pytest.raises(ToolError) as exc_limit_high:
        await server.call_tool("search_memories", {"query": "test", "workspace_id": "ws-personal", "limit": 21})
    assert "limit" in str(exc_limit_high.value).lower()

    # Over-length workspace_id
    with pytest.raises(ToolError) as exc_ws_long:
        await server.call_tool("search_memories", {"query": "test", "workspace_id": "w" * 256})
    assert "workspace_id" in str(exc_ws_long.value).lower()
    assert "256" not in str(exc_ws_long.value)
    assert "maximum length" in str(exc_ws_long.value).lower()

    # Over-length project_id
    with pytest.raises(ToolError) as exc_proj_long:
        await server.call_tool("search_memories", {"query": "test", "workspace_id": "ws-personal", "project_id": "p" * 256})
    assert "project_id" in str(exc_proj_long.value).lower()
    assert "maximum length" in str(exc_proj_long.value).lower()


@pytest.mark.asyncio
async def test_build_context_package_input_validation(mcp_test_db):
    """Verify input lengths and boundary checks for build_context_package."""
    server = create_mcp_server()

    # Empty workspace_id
    with pytest.raises(ToolError) as exc_ws_empty:
        await server.call_tool("build_context_package", {"workspace_id": "   ", "project_id": "proj-p1"})
    assert "workspace_id" in str(exc_ws_empty.value).lower()

    # Empty project_id
    with pytest.raises(ToolError) as exc_proj_empty:
        await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "   "})
    assert "project_id" in str(exc_proj_empty.value).lower()

    # Over-length workspace_id (> 255)
    with pytest.raises(ToolError) as exc_ws_long:
        await server.call_tool("build_context_package", {"workspace_id": "w" * 256, "project_id": "proj-p1"})
    assert "workspace_id" in str(exc_ws_long.value).lower()
    assert "maximum length of 255" in str(exc_ws_long.value).lower()

    # Over-length project_id (> 255)
    with pytest.raises(ToolError) as exc_proj_long:
        await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "p" * 256})
    assert "project_id" in str(exc_proj_long.value).lower()
    assert "maximum length of 255" in str(exc_proj_long.value).lower()

    # Optional query: empty/whitespace is permitted and treated as empty
    res_empty_q = await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "proj-p1", "query": ""})
    assert not res_empty_q.is_error

    # Optional query: null is permitted
    res_null_q = await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "proj-p1", "query": None})
    assert not res_null_q.is_error

    # Over-length query (> 2000)
    with pytest.raises(ToolError) as exc_long:
        await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "proj-p1", "query": "a" * 2001})
    assert "query" in str(exc_long.value).lower()
    assert "maximum length of 2000" in str(exc_long.value).lower()

    # Query boundary at 2000 chars (valid)
    res_q_2000 = await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "proj-p1", "query": "a" * 2000})
    assert not res_q_2000.is_error

    # Optional task_description: null is permitted
    res_null_task = await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "proj-p1", "task_description": None})
    assert not res_null_task.is_error

    # Over-length task_description (> 4000)
    with pytest.raises(ToolError) as exc_task_long:
        await server.call_tool(
            "build_context_package",
            {"workspace_id": "ws-personal", "project_id": "proj-p1", "task_description": "t" * 4001},
        )
    assert "task_description" in str(exc_task_long.value).lower()
    assert "maximum length of 4000" in str(exc_task_long.value).lower()

    # Boundary task_description at 4000 chars (valid)
    res_task_4000 = await server.call_tool(
        "build_context_package",
        {"workspace_id": "ws-personal", "project_id": "proj-p1", "task_description": "t" * 4000},
    )
    assert not res_task_4000.is_error


@pytest.mark.asyncio
async def test_mcp_input_type_validation_malformed_arguments(mcp_test_db):
    """Verify that malformed argument types produce controlled ToolError without internal exception leakage."""
    server = create_mcp_server()

    def assert_safe_error(err_text: str):
        assert "traceback" not in err_text.lower()
        assert "attributeerror" not in err_text.lower()
        assert "sqlite" not in err_text.lower()
        assert "smriti.db" not in err_text.lower()
        assert "c:/" not in err_text.lower()

    # list_projects: non-string workspace_id (int, dict, list, bool)
    for bad_ws in [123, True, False, ["ws"], {"ws": "id"}]:
        with pytest.raises(ToolError) as exc_lp:
            await server.call_tool("list_projects", {"workspace_id": bad_ws})
        err = str(exc_lp.value)
        assert any(phrase in err.lower() for phrase in ["string", "validation error", "input should be a valid string"])
        assert_safe_error(err)

    # search_memories: non-string query
    for bad_q in [123, True, False, ["query"], {"q": "search"}]:
        with pytest.raises(ToolError) as exc_sm_q:
            await server.call_tool("search_memories", {"query": bad_q, "workspace_id": "ws-personal"})
        err = str(exc_sm_q.value)
        assert "query" in err.lower()
        assert any(phrase in err.lower() for phrase in ["string", "validation error", "input should be a valid string"])
        assert_safe_error(err)

    # search_memories: non-string workspace_id
    with pytest.raises(ToolError) as exc_sm_ws:
        await server.call_tool("search_memories", {"query": "valid", "workspace_id": 999})
    assert "workspace_id" in str(exc_sm_ws.value).lower()
    assert_safe_error(str(exc_sm_ws.value))

    # search_memories: non-string project_id (when provided)
    for bad_pid in [123, True, False, ["proj"], {"p": "id"}]:
        with pytest.raises(ToolError) as exc_sm_pid:
            await server.call_tool(
                "search_memories",
                {"query": "valid", "workspace_id": "ws-personal", "project_id": bad_pid},
            )
        err = str(exc_sm_pid.value)
        assert "project_id" in err.lower()
        assert_safe_error(err)

    # search_memories: project_id null is permitted
    res_sm_null_pid = await server.call_tool(
        "search_memories",
        {"query": "Zigbee2MQTT", "workspace_id": "ws-personal", "project_id": None},
    )
    assert not res_sm_null_pid.is_error

    # search_memories: non-integer / boolean limit
    for bad_lim in [True, False, "10", 3.14, [10]]:
        with pytest.raises(ToolError) as exc_sm_lim:
            await server.call_tool(
                "search_memories",
                {"query": "valid", "workspace_id": "ws-personal", "limit": bad_lim},
            )
        err = str(exc_sm_lim.value)
        assert "limit" in err.lower()
        assert any(phrase in err.lower() for phrase in ["integer", "int", "validation error"])
        assert_safe_error(err)

    # search_memories: non-boolean include_superseded
    for bad_sup in ["true", 1, 0, [True], {"inc": True}]:
        with pytest.raises(ToolError) as exc_sm_sup:
            await server.call_tool(
                "search_memories",
                {
                    "query": "valid",
                    "workspace_id": "ws-personal",
                    "include_superseded": bad_sup,
                },
            )
        err = str(exc_sm_sup.value)
        assert "include_superseded" in err.lower()
        assert any(phrase in err.lower() for phrase in ["boolean", "bool", "validation error"])
        assert_safe_error(err)

    # build_context_package: non-string workspace_id
    for bad_ws in [123, True, False, ["ws"], {"id": 1}]:
        with pytest.raises(ToolError) as exc_bcp_ws:
            await server.call_tool("build_context_package", {"workspace_id": bad_ws, "project_id": "proj-p1"})
        assert "workspace_id" in str(exc_bcp_ws.value).lower()
        assert_safe_error(str(exc_bcp_ws.value))

    # build_context_package: non-string project_id
    for bad_pid in [123, True, False, ["p"], {"id": 1}]:
        with pytest.raises(ToolError) as exc_bcp_pid:
            await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": bad_pid})
        assert "project_id" in str(exc_bcp_pid.value).lower()
        assert_safe_error(str(exc_bcp_pid.value))

    # build_context_package: non-string query (when provided)
    for bad_q in [1234, True, False, ["q"], {"query": "text"}]:
        with pytest.raises(ToolError) as exc_bcp_q:
            await server.call_tool("build_context_package", {"workspace_id": "ws-personal", "project_id": "proj-p1", "query": bad_q})
        assert "query" in str(exc_bcp_q.value).lower()
        assert_safe_error(str(exc_bcp_q.value))

    # build_context_package: non-string task_description (when provided)
    for bad_tdesc in [42, True, False, ["t"], {"desc": "text"}]:
        with pytest.raises(ToolError) as exc_bcp_tdesc:
            await server.call_tool(
                "build_context_package",
                {"workspace_id": "ws-personal", "project_id": "proj-p1", "task_description": bad_tdesc},
            )
        assert "task_description" in str(exc_bcp_tdesc.value).lower()
        assert_safe_error(str(exc_bcp_tdesc.value))


@pytest.mark.asyncio
async def test_error_sanitization_no_sensitive_or_raw_exception_leak(mcp_test_db, monkeypatch):
    """Verify that raw internal exceptions, database paths, and SQL traces are never leaked to clients."""
    server = create_mcp_server()

    # Simulate an internal DB crash containing sensitive path and table name details
    sensitive_diagnostic = "OperationalError: table 'smriti_secrets' locked at C:/secrets/db.sqlite: syntax error at line 42"

    def faulty_search(*args, **kwargs):
        raise RuntimeError(sensitive_diagnostic)

    monkeypatch.setattr(mcp_server_module.retrieval_engine, "search", faulty_search)

    with pytest.raises(ToolError) as exc_info:
        await server.call_tool("search_memories", {"query": "test query", "workspace_id": "ws-personal"})

    err_msg = str(exc_info.value)
    # The client-facing message must be sanitized
    assert "Failed to search memories due to an internal error." in err_msg
    # Ensure sensitive diagnostics are absent from the message
    assert "OperationalError" not in err_msg
    assert "smriti_secrets" not in err_msg
    assert "C:/secrets" not in err_msg
    assert "syntax error" not in err_msg

    # Test build_context_package sanitization
    def faulty_build(*args, **kwargs):
        raise ValueError("Sensitive token leaked in exception: sk-proj-1234567890abcdef")

    monkeypatch.setattr(mcp_server_module.context_engine, "build_context", faulty_build)

    with pytest.raises(ToolError) as exc_ctx:
        await server.call_tool(
            "build_context_package",
            {"workspace_id": "ws-personal", "project_id": "proj-p1", "query": "test query"},
        )

    err_ctx_msg = str(exc_ctx.value)
    assert "Failed to build context package due to an internal error." in err_ctx_msg
    assert "sk-proj" not in err_ctx_msg


@pytest.mark.asyncio
async def test_build_context_package_success_and_scoping(mcp_test_db):
    """Verify build_context_package returns a structured package with formatted prompt."""
    server = create_mcp_server()

    res = await server.call_tool("build_context_package", {
        "query": "How do I configure lights?",
        "workspace_id": "ws-personal",
        "project_id": "proj-p1",
    })

    data = json.loads(res.content[0].text)
    assert data["project_id"] == "proj-p1"
    assert data["project_name"] == "Home Automation"
    assert "formatted_prompt" in data
    assert "# PROJECT SMRITI CONTEXT PACKAGE" in data["formatted_prompt"]
    assert "Home Automation" in data["formatted_prompt"]

    # Cross-workspace project mismatch rejection
    with pytest.raises(ToolError) as exc_mismatch:
        await server.call_tool("build_context_package", {
            "query": "How do I configure lights?",
            "workspace_id": "ws-personal",
            "project_id": "proj-w1",
        })
    assert "not found in workspace" in str(exc_mismatch.value)


@pytest.mark.asyncio
async def test_db_session_cleanup_on_error(mcp_test_db, monkeypatch):
    """Verify database sessions are closed on tool error or exception."""
    closed_sessions = []
    base_factory = mcp_test_db

    def instrumented_session_factory():
        sess = base_factory()
        orig_close = sess.close

        def tracking_close():
            closed_sessions.append(sess)
            orig_close()

        sess.close = tracking_close
        return sess

    monkeypatch.setattr(mcp_server_module, "SessionLocal", instrumented_session_factory)

    server = create_mcp_server()

    # Trigger ToolError via invalid workspace
    with pytest.raises(ToolError):
        await server.call_tool("list_projects", {"workspace_id": "nonexistent"})

    assert len(closed_sessions) >= 1

    # Trigger successful tool call
    closed_before = len(closed_sessions)
    await server.call_tool("list_workspaces", {})
    assert len(closed_sessions) > closed_before


@pytest.mark.asyncio
async def test_mcp_client_session_stdio_transport(mcp_test_db):
    """Test full MCP client-server interaction over in-memory streams using official ClientSession."""
    server = create_mcp_server()

    server_send, client_recv = anyio.create_memory_object_stream(10)
    client_send, server_recv = anyio.create_memory_object_stream(10)

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            server._lowlevel_server.run,
            server_recv,
            server_send,
            server._lowlevel_server.create_initialization_options(),
        )

        async with ClientSession(client_recv, client_send) as session:
            await session.initialize()

            # Discovered tools
            tool_list = await session.list_tools()
            tool_names = [t.name for t in tool_list.tools]
            assert "list_workspaces" in tool_names
            assert "search_memories" in tool_names

            # Tool call via client session
            ws_res = await session.call_tool("list_workspaces", {})
            assert not ws_res.is_error
            ws_data = json.loads(ws_res.content[0].text)
            assert "workspaces" in ws_data
            assert any(w["id"] == "ws-personal" for w in ws_data["workspaces"])

        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_mcp_subprocess_stdio_launch(tmp_path):
    """End-to-end integration test launching the actual SMRITI MCP server subprocess over stdio.

    Verifies:
    1. Subprocess starts cleanly with python -m smriti.mcp_server.
    2. Uses an isolated temporary SQLite database and mock embedding provider.
    3. External AI-provider credentials (OPENAI_*, ANTHROPIC_*, GEMINI_*, AWS_*) are stripped.
    4. Official MCP SDK stdio client initializes and discovers exactly 4 tools.
    5. Successful tool call executes (list_workspaces).
    6. Clean subprocess shutdown with valid protocol output on stdout and logs on stderr.
    """
    import sys
    import os
    from mcp.client.stdio import stdio_client, StdioServerParameters

    db_file = tmp_path / "subprocess_test.db"
    db_url = f"sqlite:///{db_file}"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Construct a minimal, isolated environment without leaking host AI provider credentials
    isolated_env = {}
    # Retain standard platform / system environment variables required for Windows and Python
    preserved_keys = [
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "PYTHONHOME",
    ]
    for key in preserved_keys:
        val = os.environ.get(key)
        if val is not None:
            isolated_env[key] = val

    # Configure SMRITI-specific variables
    isolated_env["SMRITI_DATABASE_URL"] = db_url
    isolated_env["SMRITI_EMBEDDING_PROVIDER"] = "mock"
    isolated_env["PYTHONPATH"] = repo_root

    # Ensure no API keys or credentials exist in the subprocess environment
    for k in list(isolated_env.keys()):
        if any(prefix in k.upper() for prefix in ["OPENAI", "ANTHROPIC", "GEMINI", "AWS", "API_KEY", "SECRET"]):
            if k != "SMRITI_DATABASE_URL":
                del isolated_env[k]

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "smriti.mcp_server"],
        env=isolated_env,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Verify tool discovery
            tool_list = await session.list_tools()
            tool_names = [t.name for t in tool_list.tools]
            assert len(tool_names) == 4
            assert sorted(tool_names) == sorted([
                "list_workspaces",
                "list_projects",
                "search_memories",
                "build_context_package",
            ])

            # Verify tool execution
            res = await session.call_tool("list_workspaces", {})
            assert not res.is_error
            data = json.loads(res.content[0].text)
            assert "workspaces" in data
            assert len(data["workspaces"]) >= 1
            assert data["workspaces"][0]["name"] == "Personal Workspace"
