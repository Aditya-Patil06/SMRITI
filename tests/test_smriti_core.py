import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from smriti.models import Base, User, Workspace, Project, Task, Memory, Conversation, Message
from smriti.importers.chatgpt import ChatGPTImporter
from smriti.importers.claude import ClaudeImporter
from smriti.importers.gemini import GeminiImporter
from smriti.extraction import MemoryExtractor
from smriti.vector_store import EmbeddingService, VectorStore
from smriti.graph import KnowledgeGraphService
from smriti.retrieval import HybridRetrievalEngine
from smriti.context_engine import ContextEngine
from smriti.memory_manager import MemoryManager
from smriti.canonical_export import CanonicalExportEngine
from smriti.providers.adapters import ProviderAdapterRegistry

@pytest.fixture
def db_session():
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    
    # Seed default user and workspace
    user = User(id="test-user", username="testuser", email="test@smriti.local")
    session.add(user)
    ws = Workspace(id="test-ws", user_id="test-user", name="Test Workspace", is_default=True)
    session.add(ws)
    session.commit()

    yield session
    session.close()

def test_chatgpt_importer():
    importer = ChatGPTImporter()
    raw = [
        {
            "id": "conv-1",
            "title": "Build a Pose App",
            "create_time": 1700000000,
            "mapping": {
                "node-1": {
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"parts": ["We decided to use MediaPipe because of superior mobile performance"]},
                        "create_time": 1700000001
                    }
                }
            }
        }
    ]
    convs = importer.parse(raw)
    assert len(convs) == 1
    assert convs[0].title == "Build a Pose App"
    assert len(convs[0].messages) == 1
    assert "MediaPipe" in convs[0].messages[0].content

def test_claude_importer():
    importer = ClaudeImporter()
    raw = [
        {
            "uuid": "claude-conv-1",
            "name": "Backend Architecture",
            "chat_messages": [
                {"sender": "human", "text": "What database should we use?"},
                {"sender": "assistant", "text": "We decided to use SQLite for local-first storage."}
            ]
        }
    ]
    convs = importer.parse(raw)
    assert len(convs) == 1
    assert len(convs[0].messages) == 2
    assert convs[0].messages[1].role == "assistant"

def test_gemini_importer():
    importer = GeminiImporter()
    raw = [
        {
            "id": "gemini-1",
            "title": "API Design",
            "messages": [
                {"author": "user", "content": "Let's build with FastAPI."},
                {"author": "model", "content": "We should implement authentication next."}
            ]
        }
    ]
    convs = importer.parse(raw)
    assert len(convs) == 1
    assert len(convs[0].messages) == 2

def test_memory_extraction():
    extractor = MemoryExtractor()
    text = "We decided to use FastAPI because it provides automatic OpenAPI documentation. Next step: implement user authentication. Issue is: CORS failing."
    candidates = extractor.extract_from_message(role="assistant", content=text)
    types = [c.memory_type for c in candidates]
    assert "decision" in types
    assert "task" in types
    assert "problem" in types
    assert "technology" in types

def test_vector_store_similarity():
    emb = EmbeddingService(dimension=64)
    vstore = VectorStore(emb)
    vstore.upsert("doc1", "FastAPI web framework with Python", {"project": "smriti"})
    vstore.upsert("doc2", "React frontend user interface", {"project": "smriti"})

    hits = vstore.search("FastAPI backend", top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == "doc1"

def test_provenance_and_supersession(db_session):
    mgr = MemoryManager()
    mem1 = Memory(
        id="mem-1",
        workspace_id="test-ws",
        memory_type="decision",
        statement="Use OpenPose for tracking",
        status="active",
        confidence=0.9
    )
    db_session.add(mem1)
    db_session.commit()

    # Resolve conflict / supersede
    updated = mgr.resolve_conflict(
        db=db_session,
        memory_id="mem-1",
        resolution_action="supersede",
        reason="Switched to MediaPipe for mobile performance"
    )
    assert updated.status == "superseded"
    assert updated.version == 2.0
    assert len(updated.versions) == 1

def test_context_engine_generation(db_session):
    proj = Project(
        id="proj-1",
        workspace_id="test-ws",
        name="MotionCare",
        goal="AI physiotherapy mobile app",
        tech_stack=["React", "FastAPI"],
        constraints=["Local offline execution only"]
    )
    db_session.add(proj)
    mem = Memory(
        id="mem-pose",
        workspace_id="test-ws",
        project_id="proj-1",
        memory_type="decision",
        statement="Use MediaPipe instead of OpenPose",
        rationale="Better mobile performance",
        status="active"
    )
    db_session.add(mem)
    db_session.commit()

    ce = ContextEngine()
    ctx = ce.build_context(db=db_session, query="pose estimation system", project_id="proj-1")
    assert ctx.project_name == "MotionCare"
    assert len(ctx.decisions) >= 1
    assert "MediaPipe" in ctx.formatted_prompt

def test_canonical_export_roundtrip(db_session):
    proj = Project(
        id="proj-roundtrip",
        workspace_id="test-ws",
        name="Export Test",
        goal="Verify zero semantic loss"
    )
    db_session.add(proj)
    mem = Memory(
        id="mem-rt",
        workspace_id="test-ws",
        project_id="proj-roundtrip",
        memory_type="decision",
        statement="Deterministic testing",
        status="active"
    )
    db_session.add(mem)
    db_session.commit()

    engine = CanonicalExportEngine()
    bundle = engine.export_all(db_session)
    assert bundle["manifest"]["entity_counts"]["projects"] >= 1
    assert bundle["manifest"]["entity_counts"]["memories"] >= 1

    # Fresh session
    fresh_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=fresh_engine)
    FreshSession = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    fresh_session = FreshSession()

    counts = engine.import_all(fresh_session, bundle)
    assert counts["projects"] >= 1
    assert counts["memories"] >= 1

    imported_mem = fresh_session.query(Memory).filter(Memory.id == "mem-rt").first()
    assert imported_mem is not None
    assert imported_mem.statement == "Deterministic testing"

def test_provider_capability_discovery():
    reg = ProviderAdapterRegistry()
    caps = reg.list_all_capabilities()
    assert len(caps) == 3
    chatgpt_caps = next(c for c in caps if c.provider == "chatgpt")
    assert chatgpt_caps.capabilities["READ_CONTEXT"] is True
    assert chatgpt_caps.capabilities["REAL_TIME_EVENTS"] is False  # Unsupported without official webhook/MCP
