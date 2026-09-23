import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import json

from smriti.models import Base, get_db, User, Workspace, Project, Task, Memory, Conversation, Message, ProviderAccount
from smriti.api import app
from smriti.vector_store import vector_store, EmbeddingService, VectorStore

from sqlalchemy.pool import StaticPool

# Isolated in-memory testing database fixture
@pytest.fixture
def test_db_session():
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()

    # Seed default user and workspace
    user = User(id="user-1", username="testuser", email="test@smriti.local")
    session.add(user)
    ws1 = Workspace(id="ws-1", user_id="user-1", name="Personal Workspace", is_default=True)
    ws2 = Workspace(id="ws-2", user_id="user-1", name="Work Workspace", is_default=False)
    session.add_all([ws1, ws2])
    session.commit()

    yield session
    session.close()

@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path):
    """Ensure vector_store is isolated per test with its own temporary storage file."""
    orig_path = vector_store.storage_path
    orig_vectors = dict(vector_store.vectors)
    orig_metadata = dict(vector_store.metadata)

    test_storage = str(tmp_path / "test_vectors.json")
    vector_store.storage_path = test_storage
    vector_store.vectors.clear()
    vector_store.metadata.clear()

    yield vector_store

    vector_store.storage_path = orig_path
    vector_store.vectors = orig_vectors
    vector_store.metadata = orig_metadata

@pytest.fixture
def client(test_db_session):
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()

def test_health_check(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Project SMRITI Memory Core"

def test_projects_crud_and_validation(client):
    # 1. Create project in valid workspace
    res = client.post("/api/v1/projects", json={
        "workspace_id": "ws-1",
        "name": "Project Alpha",
        "goal": "Build decentralized memory layer",
        "tech_stack": ["FastAPI", "React"],
        "constraints": ["Zero data leakage"]
    })
    assert res.status_code == 201
    proj_id = res.json()["id"]

    # 2. Get project
    res = client.get(f"/api/v1/projects/{proj_id}")
    assert res.status_code == 200
    assert res.json()["name"] == "Project Alpha"

    # 3. Create project with invalid workspace -> 404
    res_bad = client.post("/api/v1/projects", json={
        "workspace_id": "nonexistent-ws",
        "name": "Project Fail"
    })
    assert res_bad.status_code == 404

def test_memories_crud_and_workspace_isolation(client):
    # Create memory in ws-1
    res1 = client.post("/api/v1/memories", json={
        "workspace_id": "ws-1",
        "memory_type": "decision",
        "statement": "Use MediaPipe for vision tracking",
        "rationale": "High mobile performance"
    })
    assert res1.status_code == 201
    mem1_id = res1.json()["id"]

    # Create memory in ws-2
    res2 = client.post("/api/v1/memories", json={
        "workspace_id": "ws-2",
        "memory_type": "decision",
        "statement": "Use PyTorch for model training",
        "rationale": "Research flexibility"
    })
    assert res2.status_code == 201

    # Query ws-1 only
    list_ws1 = client.get("/api/v1/memories?workspace_id=ws-1").json()
    assert len(list_ws1) == 1
    assert list_ws1[0]["id"] == mem1_id
    assert "MediaPipe" in list_ws1[0]["statement"]

    # Query ws-2 only
    list_ws2 = client.get("/api/v1/memories?workspace_id=ws-2").json()
    assert len(list_ws2) == 1
    assert "PyTorch" in list_ws2[0]["statement"]

def test_explain_provenance_manual_vs_imported(client, test_db_session):
    # 1. Manual memory -> should be marked is_manual=True and not invent sources
    res_manual = client.post("/api/v1/memories", json={
        "workspace_id": "ws-1",
        "memory_type": "decision",
        "statement": "Manual user architectural choice",
        "extraction_method": "user_explicit"
    })
    mem_manual_id = res_manual.json()["id"]

    exp_res = client.get(f"/api/v1/memories/{mem_manual_id}/explain")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["is_manual"] is True
    assert exp_data["source_conversation"] is None
    assert exp_data["source_message"] is None

    # 2. Imported memory with message and conversation
    pa = ProviderAccount(id="pa-1", user_id="user-1", provider="chatgpt", account_label="Personal")
    conv = Conversation(id="c-1", provider_account_id="pa-1", title="Design Session")
    msg = Message(id="m-1", conversation_id="c-1", role="assistant", content="We decided to use SQLite for portability")
    mem_imported = Memory(
        id="mem-imp-1",
        workspace_id="ws-1",
        source_conversation_id="c-1",
        source_message_id="m-1",
        memory_type="decision",
        statement="Decided to use SQLite",
        extraction_method="rule_heuristic"
    )
    test_db_session.add_all([pa, conv, msg, mem_imported])
    test_db_session.commit()

    exp_imp = client.get("/api/v1/memories/mem-imp-1/explain").json()
    assert exp_imp["is_manual"] is False
    assert exp_imp["source_conversation"]["title"] == "Design Session"
    assert exp_imp["source_conversation"]["provider"] == "chatgpt"
    assert exp_imp["source_message"]["id"] == "m-1"
    assert "SQLite" in exp_imp["relevant_source_content"]

def test_conflict_resolution_api(client, test_db_session):
    mem = Memory(
        id="mem-conflict",
        workspace_id="ws-1",
        memory_type="decision",
        statement="Use REST endpoints",
        status="active"
    )
    test_db_session.add(mem)
    test_db_session.commit()

    res = client.post("/api/v1/memories/mem-conflict/resolve?resolution_action=supersede&reason=Migrated%20to%20gRPC")
    assert res.status_code == 200
    assert res.json()["status"] == "superseded"
    assert res.json()["version"] == 2.0

def test_conversation_import_api(client):
    payload = [
        {
            "id": "conv-import-1",
            "title": "Robotics Vision Discussion",
            "create_time": 1710000000,
            "mapping": {
                "n1": {
                    "message": {
                        "id": "msg-imp-1",
                        "author": {"role": "user"},
                        "content": {"parts": ["We decided to use OpenCV and Python because of cross-platform portability. Next step: write test suite."]},
                        "create_time": 1710000001
                    }
                }
            }
        }
    ]
    res = client.post("/api/v1/imports/conversations?provider=chatgpt&workspace_id=ws-1", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["imported_conversations"] == 1
    assert data["extracted_memories"] >= 2

def test_canonical_export_import_validation_and_atomicity(client, test_db_session):
    # Add record to export
    client.post("/api/v1/projects", json={"workspace_id": "ws-1", "name": "Preserved Project"})
    export_res = client.get("/api/v1/exports/canonical")
    assert export_res.status_code == 200
    bundle = export_res.json()
    assert bundle["manifest"]["version"] == "1.2.0"
    assert len(bundle["projects"]) >= 1

    # Test malformed import -> 400
    bad_res = client.post("/api/v1/imports/canonical", json={"invalid": "payload"})
    assert bad_res.status_code == 400

    # Test valid import into fresh test session
    import_res = client.post("/api/v1/imports/canonical", json=bundle)
    assert import_res.status_code == 200
    assert import_res.json()["status"] == "success"

def test_vector_store_persistence_and_rebuild(tmp_path, test_db_session):
    storage_file = str(tmp_path / "test_vectors.json")
    emb = EmbeddingService(dimension=64)
    vstore = VectorStore(emb, storage_path=storage_file)

    # 1. Upsert and verify disk persistence
    vstore.upsert("doc-1", "FastAPI web services", {"workspace_id": "ws-1"})
    assert len(vstore.vectors) == 1

    # 2. Restart / reinitialize new instance pointing to same file
    vstore2 = VectorStore(emb, storage_path=storage_file)
    assert len(vstore2.vectors) == 1
    hits = vstore2.search("FastAPI", top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == "doc-1"

    # 3. Add memories to DB and trigger rebuild_from_db
    mem = Memory(
        id="db-mem-1",
        workspace_id="ws-1",
        memory_type="decision",
        statement="Deterministic hash embedding architecture",
        rationale="Guarantees vector stability across application restarts",
        status="active"
    )
    test_db_session.add(mem)
    test_db_session.commit()

    count = vstore2.rebuild_from_db(test_db_session)
    assert count >= 1
    rebuild_hits = vstore2.search("Deterministic hash embedding", top_k=1)
    assert len(rebuild_hits) == 1
    assert rebuild_hits[0][0] == "db-mem-1"

def test_workspace_isolation_in_search_and_context(client, test_db_session):
    # Add memory in ws-1 and ws-2
    m1 = Memory(id="m-ws1", workspace_id="ws-1", project_id="p-alpha", memory_type="decision", statement="Use Kafka in Workspace 1", status="active")
    m2 = Memory(id="m-ws2", workspace_id="ws-2", project_id="p-beta", memory_type="decision", statement="Use RabbitMQ in Workspace 2", status="active")
    test_db_session.add_all([m1, m2])
    test_db_session.commit()

    vector_store.upsert(m1.id, m1.statement, {"workspace_id": "ws-1", "project_id": "p-alpha", "status": "active"})
    vector_store.upsert(m2.id, m2.statement, {"workspace_id": "ws-2", "project_id": "p-beta", "status": "active"})

    # 1. Search scoped to ws-1 returns ws-1 memory and excludes ws-2 memory
    res1 = client.get("/api/v1/search?query=queue&workspace_id=ws-1").json()
    assert any(r["id"] == "m-ws1" for r in res1)
    assert not any(r["id"] == "m-ws2" for r in res1)

    # 2. Stale vector metadata simulation:
    # Suppose vector store has stale metadata claiming m2 belongs to ws-1 (or filter_fn returns it)
    vector_store.metadata["m-ws2"] = {"workspace_id": "ws-1", "project_id": "p-alpha", "status": "active"}
    res_stale = client.get("/api/v1/search?query=RabbitMQ&workspace_id=ws-1").json()
    # Database canonical scope check MUST prevent m-ws2 from being returned despite stale vector hit!
    assert not any(r["id"] == "m-ws2" for r in res_stale)

    # 3. Project scope enforcement at database lookup
    res_proj = client.get("/api/v1/search?query=Kafka&workspace_id=ws-1&project_id=p-other").json()
    assert not any(r["id"] == "m-ws1" for r in res_proj)

    # 4. Unscoped search returns both
    res_all = client.get("/api/v1/search?query=queue").json()
    all_ids = [r["id"] for r in res_all]
    assert "m-ws1" in all_ids
    assert "m-ws2" in all_ids

    # Context scoped to ws-1
    ctx1 = client.get("/api/v1/context?query=messaging&workspace_id=ws-1").json()
    assert "Kafka" in ctx1["formatted_prompt"]
    assert "RabbitMQ" not in ctx1["formatted_prompt"]


def test_provider_account_user_id_mismatch_validation(client, test_db_session):
    # Create another user and their provider account
    user2 = User(id="user-2", username="otheruser", email="other@smriti.local")
    test_db_session.add(user2)
    pa2 = ProviderAccount(id="pa-user2", user_id="user-2", provider="chatgpt", account_label="ChatGPT Account")
    test_db_session.add(pa2)
    test_db_session.commit()

    # Attempt to import conversations into ws-1 (owned by user-1) using pa2 (owned by user-2)
    payload = [{
        "id": "conv-test-1",
        "title": "Cross User Test",
        "mapping": {}
    }]
    res = client.post("/api/v1/imports/conversations", params={
        "provider": "chatgpt",
        "workspace_id": "ws-1",
        "provider_account_id": "pa-user2"
    }, json=payload)
    assert res.status_code == 400
    assert "does not match Workspace user_id" in res.json()["detail"]

def test_canonical_export_workspace_scoping(client, test_db_session):
    # Seed project and memory in ws-1 and ws-2
    p1 = Project(id="p-ws1", workspace_id="ws-1", name="Project 1")
    p2 = Project(id="p-ws2", workspace_id="ws-2", name="Project 2")
    m1 = Memory(id="m-ws1-scoped", workspace_id="ws-1", memory_type="decision", statement="WS1 memory", status="active")
    m2 = Memory(id="m-ws2-scoped", workspace_id="ws-2", memory_type="decision", statement="WS2 memory", status="active")
    test_db_session.add_all([p1, p2, m1, m2])
    test_db_session.commit()

    # Export ws-1 only
    res = client.get("/api/v1/exports/canonical?workspace_id=ws-1")
    assert res.status_code == 200
    bundle = res.json()
    assert bundle["manifest"]["workspace_id"] == "ws-1"

    ws_ids = [w["id"] for w in bundle["workspaces"]]
    assert ws_ids == ["ws-1"]

    proj_ids = [p["id"] for p in bundle["projects"]]
    assert "p-ws1" in proj_ids
    assert "p-ws2" not in proj_ids

    mem_ids = [m["id"] for m in bundle["memories"]]
    assert "m-ws1-scoped" in mem_ids
    assert "m-ws2-scoped" not in mem_ids

def test_canonical_import_preflight_collision_rejection(client, test_db_session):
    # Prepare a bundle where ws-1 is claimed by a different user
    bundle = {
        "manifest": {
            "version": "1.1.0",
            "entity_counts": {}
        },
        "workspaces": [
            {
                "id": "ws-1",
                "user_id": "impostor-user",
                "name": "Hijacked Workspace"
            }
        ]
    }
    res = client.post("/api/v1/imports/canonical", json=bundle)
    assert res.status_code == 400
    assert "Workspace collision" in res.json()["detail"]

    # Verify db was untouched
    ws1 = test_db_session.query(Workspace).filter(Workspace.id == "ws-1").first()
    assert ws1.user_id == "user-1"

def test_phase2_api_synthesize_and_milestones(client, test_db_session):
    # Create project and memory
    p_res = client.post("/api/v1/projects", json={"workspace_id": "ws-1", "name": "Synthesis Project"})
    p_id = p_res.json()["id"]

    client.post("/api/v1/memories", json={
        "workspace_id": "ws-1",
        "project_id": p_id,
        "memory_type": "decision",
        "statement": "Use FastAPI for REST API backend",
        "structured_claim": {
            "subject": "backend",
            "predicate": "uses_backend_framework",
            "object": "fastapi"
        }
    })

    # Call synthesize endpoint
    synth_res = client.post(f"/api/v1/projects/{p_id}/synthesize")
    assert synth_res.status_code == 200
    data = synth_res.json()
    assert data["status"] == "success"
    assert "fastapi" in data["synthesis"]["tech_stack"]

    # Call milestones endpoint
    ml_res = client.get(f"/api/v1/projects/{p_id}/milestones")
    assert ml_res.status_code == 200
    milestones = ml_res.json()
    assert len(milestones) >= 1
    types = [ml["milestone_type"] for ml in milestones]
    assert "project_created" in types

def test_phase2_api_conflicts_and_resolution_flow(client, test_db_session):
    # Create project and two competing memories
    p_res = client.post("/api/v1/projects", json={"workspace_id": "ws-1", "name": "Conflict Project"})
    p_id = p_res.json()["id"]

    m1_res = client.post("/api/v1/memories", json={
        "workspace_id": "ws-1",
        "project_id": p_id,
        "memory_type": "decision",
        "statement": "Backend uses PostgreSQL",
        "structured_claim": {
            "subject": "backend",
            "predicate": "uses_database",
            "object": "postgresql"
        }
    })
    m1_id = m1_res.json()["id"]

    m2_res = client.post("/api/v1/memories", json={
        "workspace_id": "ws-1",
        "project_id": p_id,
        "memory_type": "decision",
        "statement": "Backend uses MongoDB",
        "structured_claim": {
            "subject": "backend",
            "predicate": "uses_database",
            "object": "mongodb"
        }
    })
    m2_id = m2_res.json()["id"]

    # List conflicts
    confs_res = client.get(f"/api/v1/conflicts?workspace_id=ws-1&project_id={p_id}")
    assert confs_res.status_code == 200
    conflicts = confs_res.json()
    assert len(conflicts) >= 1

    # Resolve via conflict flow: keep_both
    res_flow = client.post(
        f"/api/v1/conflicts/resolve?memory_id={m1_id}&resolution_action=keep_both&paired_memory_id={m2_id}"
    )
    assert res_flow.status_code == 200
    assert res_flow.json()["status"] == "success"

def test_phase2_canonical_export_import_roundtrip(client, test_db_session):
    p_res = client.post("/api/v1/projects", json={"workspace_id": "ws-1", "name": "Roundtrip Project"})
    p_id = p_res.json()["id"]

    # Add memory with structured claim
    client.post("/api/v1/memories", json={
        "workspace_id": "ws-1",
        "project_id": p_id,
        "memory_type": "decision",
        "statement": "Use Alembic migrations",
        "structured_claim": {
            "subject": "database",
            "predicate": "uses_orm",
            "object": "alembic"
        }
    })

    # Trigger synthesis to create milestone
    client.post(f"/api/v1/projects/{p_id}/synthesize")

    # Export canonical bundle
    exp_res = client.get("/api/v1/exports/canonical?workspace_id=ws-1")
    assert exp_res.status_code == 200
    bundle = exp_res.json()
    assert bundle["manifest"]["version"] == "1.2.0"
    assert len(bundle["milestones"]) >= 1
    assert any(m.get("structured_claim") is not None for m in bundle["memories"])

