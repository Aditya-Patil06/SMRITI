"""Tests for SMRITI Phase 4.2 — Dedicated Read-Only ChatGPT Custom GPT Actions Application."""

import hashlib
import json
import os
import threading
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from smriti.models import Base, User, Workspace, Project, Memory
import smriti.actions_app as actions_module
from smriti.actions_app import app as actions_app
from smriti.config import settings
import smriti.vector_store
from smriti.vector_store import vector_store


TEST_ACTIONS_API_KEY = "test-actions-secret-key-12345"
TEST_ACTIONS_USER_ID = "action-user-1"


@pytest.fixture
def actions_test_db(tmp_path) -> Generator[Session, None, None]:
    """Create an isolated SQLite database and vectors storage for actions tests."""
    db_file = tmp_path / "test_actions.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()

    # Seed User 1 and User 2
    u1 = User(id="action-user-1", email="u1@example.com", username="user1")
    u2 = User(id="action-user-2", email="u2@example.com", username="user2")
    w1 = Workspace(id="ws-u1", user_id="action-user-1", name="U1 Workspace 1")
    w1_b = Workspace(id="ws-u1-b", user_id="action-user-1", name="U1 Workspace 2")
    w2 = Workspace(id="ws-u2", user_id="action-user-2", name="U2 Workspace")
    p1 = Project(id="proj-u1", workspace_id="ws-u1", name="Project 1")
    p1_b = Project(id="proj-u1-b", workspace_id="ws-u1-b", name="Project 1B")
    p2 = Project(id="proj-u2", workspace_id="ws-u2", name="Project 2")

    m1 = Memory(
        id="mem-u1-1",
        workspace_id="ws-u1",
        project_id="proj-u1",
        memory_type="decision",
        statement="Use PostgreSQL for production database",
        rationale="ACID compliance and pgvector",
        status="active",
        confidence=1.0,
    )
    m1_b = Memory(
        id="mem-u1-b-1",
        workspace_id="ws-u1-b",
        project_id="proj-u1-b",
        memory_type="decision",
        statement="Use Redis for caching and session store",
        rationale="In-memory sub-millisecond latency",
        status="active",
        confidence=1.0,
    )
    m2 = Memory(
        id="mem-u2-1",
        workspace_id="ws-u2",
        project_id="proj-u2",
        memory_type="decision",
        statement="Use MongoDB for document storage",
        rationale="Schema flexibility",
        status="active",
        confidence=1.0,
    )

    db.add_all([u1, u2, w1, w1_b, w2, p1, p1_b, p2, m1, m1_b, m2])
    db.commit()

    # Configure vector store to point to tmp storage
    vectors_file = str(tmp_path / "vectors.json")
    orig_storage_path = getattr(vector_store, "storage_path", None)
    vector_store.storage_path = vectors_file
    vector_store.clear(auto_save=False)
    vector_store.upsert(m1.id, f"{m1.statement} {m1.rationale}", {"workspace_id": m1.workspace_id, "project_id": m1.project_id, "status": m1.status})
    vector_store.upsert(m1_b.id, f"{m1_b.statement} {m1_b.rationale}", {"workspace_id": m1_b.workspace_id, "project_id": m1_b.project_id, "status": m1_b.status})
    vector_store.upsert(m2.id, f"{m2.statement} {m2.rationale}", {"workspace_id": m2.workspace_id, "project_id": m2.project_id, "status": m2.status})

    # Override get_db in actions_app
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    actions_app.dependency_overrides[actions_module.get_db] = override_get_db
    actions_module.SessionLocal = TestingSessionLocal

    yield db

    db.close()
    actions_app.dependency_overrides.clear()
    if orig_storage_path:
        vector_store.storage_path = orig_storage_path


@pytest.fixture
def auth_client(actions_test_db, monkeypatch) -> TestClient:
    """Provide a TestClient with valid Actions configuration."""
    monkeypatch.setenv("SMRITI_ACTIONS_API_KEY", TEST_ACTIONS_API_KEY)
    monkeypatch.setenv("SMRITI_ACTIONS_USER_ID", TEST_ACTIONS_USER_ID)
    settings.actions_api_key = TEST_ACTIONS_API_KEY
    settings.actions_user_id = TEST_ACTIONS_USER_ID
    return TestClient(actions_app)


def test_auth_missing_or_invalid_bearer_token(auth_client):
    """Missing or invalid token returns 401 with WWW-Authenticate: Bearer across all endpoints including openapi.json."""
    # 1. No Authorization header
    res_none = auth_client.get("/actions/workspaces")
    assert res_none.status_code == 401
    assert "Bearer" in res_none.headers.get("WWW-Authenticate", "")

    res_none_openapi = auth_client.get("/actions/openapi.json")
    assert res_none_openapi.status_code == 401
    assert "Bearer" in res_none_openapi.headers.get("WWW-Authenticate", "")

    # 2. Invalid Bearer token
    res_invalid = auth_client.get(
        "/actions/workspaces",
        headers={"Authorization": "Bearer wrong-token-value"},
    )
    assert res_invalid.status_code == 401
    assert "Bearer" in res_invalid.headers.get("WWW-Authenticate", "")

    res_invalid_openapi = auth_client.get(
        "/actions/openapi.json",
        headers={"Authorization": "Bearer wrong-token-value"},
    )
    assert res_invalid_openapi.status_code == 401
    assert "Bearer" in res_invalid_openapi.headers.get("WWW-Authenticate", "")


def test_fail_closed_when_key_or_user_id_missing(actions_test_db, monkeypatch):
    """Missing or blank API key or user ID fails closed."""
    # Missing API key
    monkeypatch.setenv("SMRITI_ACTIONS_API_KEY", "")
    monkeypatch.setenv("SMRITI_ACTIONS_USER_ID", TEST_ACTIONS_USER_ID)
    monkeypatch.setattr(settings, "actions_api_key", "")
    monkeypatch.setattr(settings, "actions_user_id", TEST_ACTIONS_USER_ID)
    client = TestClient(actions_app, raise_server_exceptions=False)
    res = client.get("/actions/workspaces", headers={"Authorization": "Bearer any-token"})
    assert res.status_code == 503

    # Missing User ID
    monkeypatch.setenv("SMRITI_ACTIONS_API_KEY", TEST_ACTIONS_API_KEY)
    monkeypatch.setenv("SMRITI_ACTIONS_USER_ID", "")
    monkeypatch.setattr(settings, "actions_api_key", TEST_ACTIONS_API_KEY)
    monkeypatch.setattr(settings, "actions_user_id", "")
    res_user = client.get(
        "/actions/workspaces",
        headers={"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"},
    )
    assert res_user.status_code == 503


def test_unknown_configured_user_fails_startup(tmp_path, monkeypatch):
    """Unknown configured user fails startup without initializing or seeding the database."""
    db_file = tmp_path / "empty_db.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setenv("SMRITI_ACTIONS_API_KEY", TEST_ACTIONS_API_KEY)
    monkeypatch.setenv("SMRITI_ACTIONS_USER_ID", "non-existent-user-999")
    monkeypatch.setattr(settings, "actions_api_key", TEST_ACTIONS_API_KEY)
    monkeypatch.setattr(settings, "actions_user_id", "non-existent-user-999")
    monkeypatch.setattr(actions_module, "SessionLocal", TestingSessionLocal)

    with pytest.raises(RuntimeError, match="does not exist in the database"):
        actions_module.validate_startup_configuration()


def test_workspace_listing_enforces_owner_scoping(auth_client):
    """Workspace listing returns only workspaces owned by SMRITI_ACTIONS_USER_ID."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    res = auth_client.get("/actions/workspaces", headers=headers)
    assert res.status_code == 200
    workspaces = res.json()
    assert len(workspaces) == 2
    ws_ids = {w["id"] for w in workspaces}
    assert ws_ids == {"ws-u1", "ws-u1-b"}
    assert all(w["user_id"] == TEST_ACTIONS_USER_ID for w in workspaces)
    # Must NOT include ws-u2
    assert not any(w["id"] == "ws-u2" for w in workspaces)


def test_project_listing_enforces_workspace_scoping(auth_client):
    """Projects listing requires workspace_id and returns only projects in an owned workspace."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # Missing workspace_id parameter -> 422
    res_missing = auth_client.get("/actions/projects", headers=headers)
    assert res_missing.status_code == 422

    # Owned workspace -> 200
    res_owned = auth_client.get("/actions/projects?workspace_id=ws-u1", headers=headers)
    assert res_owned.status_code == 200
    projects = res_owned.json()
    assert len(projects) == 1
    assert projects[0]["id"] == "proj-u1"

    # Cross-user workspace -> 404
    res_cross = auth_client.get("/actions/projects?workspace_id=ws-u2", headers=headers)
    assert res_cross.status_code == 404
    assert "not found" in res_cross.json()["detail"].lower()


def test_search_and_context_parameter_and_scope_validation(auth_client):
    """Search & context require query and workspace_id; reject cross-scope access with 404."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # 1. Missing workspace_id on search -> 422
    res_no_ws = auth_client.get("/actions/search?query=database", headers=headers)
    assert res_no_ws.status_code == 422

    # 2. Missing query on search -> 422
    res_no_q = auth_client.get("/actions/search?workspace_id=ws-u1", headers=headers)
    assert res_no_q.status_code == 422

    # 3. Limit boundary check (0 or 25) -> 422
    res_bad_limit = auth_client.get("/actions/search?query=db&workspace_id=ws-u1&limit=25", headers=headers)
    assert res_bad_limit.status_code == 422
    res_bad_limit_zero = auth_client.get("/actions/search?query=db&workspace_id=ws-u1&limit=0", headers=headers)
    assert res_bad_limit_zero.status_code == 422

    # 4. Cross-user workspace on search -> 404
    res_cross_search = auth_client.get("/actions/search?query=database&workspace_id=ws-u2", headers=headers)
    assert res_cross_search.status_code == 404

    # 5. Project not in workspace -> 404
    res_proj_mismatch = auth_client.get("/actions/search?query=database&workspace_id=ws-u1&project_id=proj-u2", headers=headers)
    assert res_proj_mismatch.status_code == 404

    # 6. Valid search returns only ws-u1 memories
    res_valid_search = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res_valid_search.status_code == 200
    hits = res_valid_search.json()
    assert any(h["id"] == "mem-u1-1" for h in hits)
    assert not any(h["id"] == "mem-u2-1" for h in hits)

    # 7. Valid context package returns only ws-u1 content
    res_valid_ctx = auth_client.get("/actions/context?query=database&workspace_id=ws-u1", headers=headers)
    assert res_valid_ctx.status_code == 200
    ctx = res_valid_ctx.json()
    assert "PostgreSQL" in ctx["formatted_prompt"]
    assert "MongoDB" not in ctx["formatted_prompt"]

    # 8. Cross-user context -> 404
    res_cross_ctx = auth_client.get("/actions/context?query=database&workspace_id=ws-u2", headers=headers)
    assert res_cross_ctx.status_code == 404


def test_write_and_legacy_routes_forbidden_and_not_found(auth_client):
    """Write endpoints are not allowed and legacy/admin routes return 404."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # POST /actions/memories -> 405 Method Not Allowed (since route is not defined for POST)
    res_post_mem = auth_client.post("/actions/memories", json={"statement": "Test"}, headers=headers)
    assert res_post_mem.status_code in (404, 405)

    # Legacy routes do not exist on actions_app -> 404
    assert auth_client.get("/api/v1/health", headers=headers).status_code == 404
    assert auth_client.get("/api/v1/memories", headers=headers).status_code == 404
    assert auth_client.post("/api/v1/imports/canonical", json={}, headers=headers).status_code == 404
    assert auth_client.get("/api/v1/exports/canonical", headers=headers).status_code == 404
    assert auth_client.post("/api/v1/conversations", json={}, headers=headers).status_code == 404
    assert auth_client.get("/api/v1/dashboard/stats", headers=headers).status_code == 404
    assert auth_client.get("/docs", headers=headers).status_code == 404
    assert auth_client.get("/redoc", headers=headers).status_code == 404


def test_openapi_schema_contains_only_four_operations(auth_client):
    """OpenAPI schema contains only four approved operations with valid and resolved BearerAuth security requirements."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    res = auth_client.get("/actions/openapi.json", headers=headers)
    assert res.status_code == 200
    schema = res.json()
    paths = schema.get("paths", {})

    expected_paths = {
        "/actions/workspaces",
        "/actions/projects",
        "/actions/search",
        "/actions/context",
    }
    # Exactly the four approved paths exist
    assert set(paths.keys()) == expected_paths

    sec_schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in sec_schemes

    # Each path contains only a GET operation, has security requirements, and all schemes exist in components
    for path, methods in paths.items():
        assert set(methods.keys()) == {"get"}, f"Path {path} has non-GET methods: {methods.keys()}"
        for method_name, op in methods.items():
            security_reqs = op.get("security")
            assert security_reqs is not None, f"Operation {method_name.upper()} {path} missing security declaration"
            assert len(security_reqs) > 0, f"Operation {method_name.upper()} {path} has empty security requirements list"
            # Every scheme referenced by an operation's security requirement must exist in components.securitySchemes
            for req in security_reqs:
                assert isinstance(req, dict)
                for scheme_name in req.keys():
                    assert scheme_name in sec_schemes, f"Operation {method_name.upper()} {path} references undeclared scheme '{scheme_name}'"
                    # The expected bearer scheme is used consistently
                    assert scheme_name == "BearerAuth", f"Operation {method_name.upper()} {path} uses unexpected scheme '{scheme_name}'"


def test_actions_requests_are_strictly_read_only(auth_client, actions_test_db, monkeypatch):
    """Search/context requests do not modify files, DB, or call write methods."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # 1. Capture database record snapshot and table counts before requests
    counts_before = {
        "users": actions_test_db.query(User).count(),
        "workspaces": actions_test_db.query(Workspace).count(),
        "projects": actions_test_db.query(Project).count(),
        "memories": actions_test_db.query(Memory).count(),
    }
    mems_before = [
        (m.id, m.statement, m.status, m.version)
        for m in actions_test_db.query(Memory).order_by(Memory.id).all()
    ]

    # 2. Check vector file checksum before and after
    v_path = vector_store.storage_path
    with open(v_path, "rb") as f:
        before_hash = hashlib.sha256(f.read()).hexdigest()
    before_mtime = os.path.getmtime(v_path)

    # 3. Spy on mutation methods
    mutated = []
    def fail_mutation(*args, **kwargs):
        mutated.append("mutated")
        raise RuntimeError("Forbidden write call!")

    target_vs = smriti.vector_store.get_vector_store()
    monkeypatch.setattr(target_vs, "save", fail_mutation)
    monkeypatch.setattr(target_vs, "upsert", fail_mutation)
    monkeypatch.setattr(target_vs, "rebuild_from_db", fail_mutation)

    # 4. Run valid search & context requests
    res_s = auth_client.get("/actions/search?query=database&workspace_id=ws-u1", headers=headers)
    assert res_s.status_code == 200
    res_c = auth_client.get("/actions/context?query=database&workspace_id=ws-u1", headers=headers)
    assert res_c.status_code == 200

    assert len(mutated) == 0

    # 5. Verify vector file state is strictly identical
    with open(v_path, "rb") as f:
        after_hash = hashlib.sha256(f.read()).hexdigest()
    after_mtime = os.path.getmtime(v_path)

    assert before_hash == after_hash
    assert before_mtime == after_mtime

    # 6. Verify database records and counts are strictly identical
    counts_after = {
        "users": actions_test_db.query(User).count(),
        "workspaces": actions_test_db.query(Workspace).count(),
        "projects": actions_test_db.query(Project).count(),
        "memories": actions_test_db.query(Memory).count(),
    }
    mems_after = [
        (m.id, m.statement, m.status, m.version)
        for m in actions_test_db.query(Memory).order_by(Memory.id).all()
    ]

    assert counts_before == counts_after
    assert mems_before == mems_after


def test_vector_file_update_detected_and_reloaded(auth_client, actions_test_db):
    """Vector file changes made by the primary API are detected and reloaded on the next request."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # Search initially returns mem-u1-1
    res1 = auth_client.get("/actions/search?query=Kubernetes&workspace_id=ws-u1", headers=headers)
    assert res1.status_code == 200
    assert not any(h["id"] == "mem-u1-new" for h in res1.json())

    # Simulate primary API writing a new memory to DB and vectors.json
    m_new = Memory(
        id="mem-u1-new",
        workspace_id="ws-u1",
        project_id="proj-u1",
        memory_type="decision",
        statement="Use Kubernetes for orchestration",
        rationale="Autoscaling capability",
        status="active",
        confidence=1.0,
    )
    actions_test_db.add(m_new)
    actions_test_db.commit()

    # Emulate primary API vector store saving to disk
    vector_store.upsert(m_new.id, f"{m_new.statement} {m_new.rationale}", {"workspace_id": m_new.workspace_id, "project_id": m_new.project_id, "status": m_new.status})

    # Execute search via actions_app -> should automatically detect new signature and find new memory
    res2 = auth_client.get("/actions/search?query=Kubernetes&workspace_id=ws-u1", headers=headers)
    assert res2.status_code == 200
    hits = res2.json()
    assert any(h["id"] == "mem-u1-new" for h in hits)


def test_cache_refresh_failure_fails_closed_with_sanitized_503(auth_client, monkeypatch):
    """Failure during cache refresh fails closed with sanitized HTTP 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    def failing_graph_sync(*args, **kwargs):
        raise RuntimeError("Internal database connectivity error during graph sync")

    monkeypatch.setattr(actions_module.graph_service, "sync_from_db", failing_graph_sync)

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 503
    detail = res.json().get("detail", "")
    assert detail == "Memory context graph is currently unavailable."
    # Ensure internal exception details are NOT leaked
    assert "Internal database connectivity error" not in detail


def test_concurrent_requests_for_different_workspaces(auth_client):
    """Concurrent search and context requests for two authorized workspaces and one unauthorized workspace do not interfere."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    errors = []
    search_results_u1 = []
    search_results_u1_b = []
    ctx_results_u1 = []
    ctx_results_u1_b = []

    def task_ws1():
        try:
            # 1. Search ws-u1
            res_s = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
            if res_s.status_code != 200:
                errors.append(f"ws-1 search failed: {res_s.status_code}")
            else:
                search_results_u1.append(res_s.json())

            # 2. Context ws-u1
            res_c = auth_client.get("/actions/context?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
            if res_c.status_code != 200:
                errors.append(f"ws-1 context failed: {res_c.status_code}")
            else:
                ctx_results_u1.append(res_c.json())
        except Exception as e:
            errors.append(str(e))

    def task_ws1_b():
        try:
            # 1. Search ws-u1-b
            res_s = auth_client.get("/actions/search?query=Redis&workspace_id=ws-u1-b", headers=headers)
            if res_s.status_code != 200:
                errors.append(f"ws-1-b search failed: {res_s.status_code}")
            else:
                search_results_u1_b.append(res_s.json())

            # 2. Context ws-u1-b
            res_c = auth_client.get("/actions/context?query=Redis&workspace_id=ws-u1-b", headers=headers)
            if res_c.status_code != 200:
                errors.append(f"ws-1-b context failed: {res_c.status_code}")
            else:
                ctx_results_u1_b.append(res_c.json())
        except Exception as e:
            errors.append(str(e))

    def task_ws2_cross_user():
        try:
            # ws-u2 belongs to action-user-2 so it must consistently return 404 for action-user-1
            res_s = auth_client.get("/actions/search?query=MongoDB&workspace_id=ws-u2", headers=headers)
            if res_s.status_code != 404:
                errors.append(f"ws-2 search expected 404, got: {res_s.status_code}")
            res_c = auth_client.get("/actions/context?query=MongoDB&workspace_id=ws-u2", headers=headers)
            if res_c.status_code != 404:
                errors.append(f"ws-2 context expected 404, got: {res_c.status_code}")
        except Exception as e:
            errors.append(str(e))

    threads = []
    for _ in range(4):
        t1 = threading.Thread(target=task_ws1)
        t1_b = threading.Thread(target=task_ws1_b)
        t2 = threading.Thread(target=task_ws2_cross_user)
        threads.extend([t1, t1_b, t2])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent execution encountered errors: {errors}"
    assert len(search_results_u1) == 4
    assert len(search_results_u1_b) == 4
    assert len(ctx_results_u1) == 4
    assert len(ctx_results_u1_b) == 4

    # Workspace 1 search hits: must contain mem-u1-1 (PostgreSQL) and not mem-u1-b-1 (Redis) or mem-u2-1 (MongoDB)
    for hits in search_results_u1:
        assert any(h["id"] == "mem-u1-1" for h in hits)
        assert not any(h["id"] == "mem-u1-b-1" for h in hits)
        assert not any(h["id"] == "mem-u2-1" for h in hits)

    # Workspace 1B search hits: must contain mem-u1-b-1 (Redis) and not mem-u1-1 (PostgreSQL) or mem-u2-1 (MongoDB)
    for hits in search_results_u1_b:
        assert any(h["id"] == "mem-u1-b-1" for h in hits)
        assert not any(h["id"] == "mem-u1-1" for h in hits)
        assert not any(h["id"] == "mem-u2-1" for h in hits)

    # Workspace 1 context: formatted_prompt and memories must include PostgreSQL and never Redis or MongoDB
    for pkg in ctx_results_u1:
        prompt = pkg["formatted_prompt"]
        assert "PostgreSQL" in prompt
        assert "Redis" not in prompt
        assert "MongoDB" not in prompt
        mem_ids = {m["id"] for m in pkg["relevant_memories"]}
        assert "mem-u1-1" in mem_ids
        assert "mem-u1-b-1" not in mem_ids
        assert "mem-u2-1" not in mem_ids

    # Workspace 1B context: formatted_prompt and memories must include Redis and never PostgreSQL or MongoDB
    for pkg in ctx_results_u1_b:
        prompt = pkg["formatted_prompt"]
        assert "Redis" in prompt
        assert "PostgreSQL" not in prompt
        assert "MongoDB" not in prompt
        mem_ids = {m["id"] for m in pkg["relevant_memories"]}
        assert "mem-u1-b-1" in mem_ids
        assert "mem-u1-1" not in mem_ids
        assert "mem-u2-1" not in mem_ids


def test_database_memory_update_reflected_in_subsequent_context(auth_client, actions_test_db):
    """A new or updated memory in the same workspace is immediately reflected in subsequent context requests."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # Initial context query for ws-u1 does not have RabbitMQ in decisions or relevant_memories
    res_initial = auth_client.get("/actions/context?query=RabbitMQ&workspace_id=ws-u1", headers=headers)
    assert res_initial.status_code == 200
    init_pkg = res_initial.json()
    assert not any("RabbitMQ" in m.get("statement", "") for m in init_pkg["relevant_memories"])
    assert not any("RabbitMQ" in d.get("statement", "") for d in init_pkg["decisions"])

    # Add a new memory directly to the database in ws-u1 (simulating write in primary API)
    m_rabbitmq = Memory(
        id="mem-u1-rabbitmq",
        workspace_id="ws-u1",
        project_id="proj-u1",
        memory_type="decision",
        statement="Use RabbitMQ for asynchronous task broker",
        rationale="Reliable AMQP message delivery",
        status="active",
        confidence=1.0,
    )
    actions_test_db.add(m_rabbitmq)
    actions_test_db.commit()

    # Query context again: graph synchronization on each request ensures this new memory is reflected
    res_updated = auth_client.get("/actions/context?query=RabbitMQ&workspace_id=ws-u1", headers=headers)
    assert res_updated.status_code == 200
    updated_pkg = res_updated.json()
    prompt_updated = updated_pkg["formatted_prompt"]
    assert "RabbitMQ" in prompt_updated
    mem_ids = {m["id"] for m in updated_pkg["relevant_memories"]}
    assert "mem-u1-rabbitmq" in mem_ids


def test_pgvector_like_store_without_load_method_succeeds(auth_client, monkeypatch):
    """A PostgreSQL-like vector store without load() method skips JSON reload and succeeds without 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    class MockPgVectorStore(smriti.vector_store.BaseVectorStore):
        def __init__(self):
            # Has a storage_path attribute like PgVectorStore, but no load() method
            self.storage_path = vector_store.storage_path

        def search(self, *args, **kwargs):
            return [("mem-u1-1", 0.95, {"workspace_id": "ws-u1", "project_id": "proj-u1", "status": "active"})]

    mock_pg_store = MockPgVectorStore()
    monkeypatch.setattr(smriti.vector_store, "get_vector_store", lambda: mock_pg_store)
    monkeypatch.setattr(actions_module, "get_vector_store", lambda: mock_pg_store)

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 200
    hits = res.json()
    assert any(h["id"] == "mem-u1-1" for h in hits)


def test_corrupted_json_cache_fails_closed_with_503(auth_client):
    """A corrupted or unreadable JSON vector file fails closed with HTTP 503 without leaking details."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    v_path = vector_store.storage_path
    # Write corrupted non-JSON data to the cache file
    with open(v_path, "w", encoding="utf-8") as f:
        f.write("{this is not valid json! corrupted file content")

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 503
    detail = res.json().get("detail", "")
    assert detail == "Memory search cache is currently unavailable."
    # Assert no traceback or internal parsing exception text leaked
    assert "JSONDecodeError" not in detail
    assert "corrupted" not in detail


def test_missing_optional_json_cache_file_does_not_fail(auth_client, monkeypatch):
    """A missing optional JSON cache file is not treated as corruption and succeeds."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}

    # Point to a non-existent file path
    non_existent_path = os.path.join(os.path.dirname(vector_store.storage_path), "non_existent_vectors.json")
    if os.path.exists(non_existent_path):
        os.remove(non_existent_path)

    target_vs = smriti.vector_store.get_vector_store()
    monkeypatch.setattr(target_vs, "storage_path", non_existent_path)

    # Search should succeed (returning empty or lexical results) without raising 503
    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 200


def test_syntactically_valid_json_with_missing_vector_fails_closed_with_503(auth_client):
    """Syntactically valid JSON where an entry misses the 'vector' key fails closed with HTTP 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    v_path = vector_store.storage_path
    payload = {
        "mem-u1-1": {
            "metadata": {"workspace_id": "ws-u1", "status": "active"}
            # Missing "vector" key
        }
    }
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 503
    assert res.json().get("detail") == "Memory search cache is currently unavailable."


def test_syntactically_valid_json_with_malformed_metadata_fails_closed_with_503(auth_client):
    """Syntactically valid JSON where metadata is not a dictionary fails closed with HTTP 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    v_path = vector_store.storage_path
    target_vs = smriti.vector_store.get_vector_store()
    dim = target_vs.embedding_provider.dimension
    payload = {
        "mem-u1-1": {
            "vector": [0.1] * dim,
            "metadata": "not-a-dict-string"  # Malformed metadata type
        }
    }
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 503
    assert res.json().get("detail") == "Memory search cache is currently unavailable."


def test_syntactically_valid_json_with_incompatible_dimension_fails_closed_with_503(auth_client):
    """Syntactically valid JSON with incompatible vector embedding dimension fails closed with HTTP 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    v_path = vector_store.storage_path
    target_vs = smriti.vector_store.get_vector_store()
    dim = target_vs.embedding_provider.dimension
    payload = {
        "mem-u1-1": {
            "vector": [0.1] * (dim + 5),  # Incompatible dimension
            "metadata": {"workspace_id": "ws-u1", "status": "active"}
        }
    }
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 503
    assert res.json().get("detail") == "Memory search cache is currently unavailable."


def test_syntactically_valid_json_with_boolean_vector_element_fails_closed_with_503(auth_client):
    """Syntactically valid JSON with boolean vector elements fails closed with HTTP 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    v_path = vector_store.storage_path
    target_vs = smriti.vector_store.get_vector_store()
    dim = target_vs.embedding_provider.dimension
    vec = [0.1] * (dim - 1) + [True]  # Boolean element in vector
    payload = {
        "mem-u1-1": {
            "vector": vec,
            "metadata": {"workspace_id": "ws-u1", "status": "active"}
        }
    }
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    res = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res.status_code == 503
    assert res.json().get("detail") == "Memory search cache is currently unavailable."


def test_syntactically_valid_json_with_nan_or_infinite_vector_element_fails_closed_with_503(auth_client):
    """Syntactically valid JSON with NaN or Infinity vector elements fails closed with HTTP 503."""
    headers = {"Authorization": f"Bearer {TEST_ACTIONS_API_KEY}"}
    v_path = vector_store.storage_path
    target_vs = smriti.vector_store.get_vector_store()
    dim = target_vs.embedding_provider.dimension

    # 1. Test NaN
    vec_nan = [0.1] * (dim - 1) + [float("nan")]
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump({"mem-u1-1": {"vector": vec_nan, "metadata": {}}}, f)

    res_nan = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res_nan.status_code == 503
    assert res_nan.json().get("detail") == "Memory search cache is currently unavailable."

    # 2. Test +Infinity
    vec_inf = [0.1] * (dim - 1) + [float("inf")]
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump({"mem-u1-1": {"vector": vec_inf, "metadata": {}}}, f)

    res_inf = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res_inf.status_code == 503
    assert res_inf.json().get("detail") == "Memory search cache is currently unavailable."

    # 3. Test -Infinity
    vec_ninf = [0.1] * (dim - 1) + [float("-inf")]
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump({"mem-u1-1": {"vector": vec_ninf, "metadata": {}}}, f)

    res_ninf = auth_client.get("/actions/search?query=PostgreSQL&workspace_id=ws-u1", headers=headers)
    assert res_ninf.status_code == 503
    assert res_ninf.json().get("detail") == "Memory search cache is currently unavailable."
