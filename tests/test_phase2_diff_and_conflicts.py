import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from smriti.models import Base, User, Workspace, Project, Memory
from smriti.memory_diff import MemoryDiffEngine, DiffResult
from smriti.memory_manager import MemoryManager
from smriti.extraction import ExtractedCandidate

@pytest.fixture
def db_session():
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()

    user = User(id="u1", username="testuser", email="test@smriti.local")
    session.add(user)
    ws = Workspace(id="ws-1", user_id="u1", name="WS 1", is_default=True)
    session.add(ws)
    proj = Project(id="p1", workspace_id="ws-1", name="Project 1", status="active")
    session.add(proj)
    session.commit()

    yield session
    session.close()

def test_five_way_diff_engine(db_session):
    diff_engine = MemoryDiffEngine()

    existing = [
        Memory(
            id="mem-1",
            workspace_id="ws-1",
            project_id="p1",
            memory_type="decision",
            statement="Backend uses PostgreSQL for data persistence",
            structured_claim={
                "subject": "backend",
                "predicate": "uses_database",
                "object": "postgresql",
                "scope": {"env": "prod"}
            },
            status="active",
            confidence=0.95
        )
    ]

    # 1. NEW
    new_cand = ExtractedCandidate(
        memory_type="technology",
        statement="Frontend built with React",
        structured_claim={
            "subject": "frontend",
            "predicate": "uses_frontend_framework",
            "object": "react"
        },
        confidence=0.92
    )
    diff_new = diff_engine.classify_diff(new_cand, existing, "ws-1", "p1")
    assert diff_new.change_type == "NEW"
    assert not diff_new.review_required

    # 2. UNCHANGED (Duplicate)
    dup_cand = ExtractedCandidate(
        memory_type="decision",
        statement="Backend uses Postgres for production data persistence",
        structured_claim={
            "subject": "backend",
            "predicate": "uses_database",
            "object": "postgres",
            "scope": {"env": "prod"}
        },
        confidence=0.95
    )
    diff_dup = diff_engine.classify_diff(dup_cand, existing, "ws-1", "p1")
    assert diff_dup.change_type == "UNCHANGED"
    assert diff_dup.existing_memory_id == "mem-1"

    # 3. SUPERSEDED (Explicit replacement)
    super_cand = ExtractedCandidate(
        memory_type="decision",
        statement="Switched to SQLite instead of PostgreSQL for simpler testing",
        structured_claim={
            "subject": "backend",
            "predicate": "uses_database",
            "object": "sqlite"
        },
        details={"replaces": "postgresql"},
        confidence=0.94
    )
    diff_super = diff_engine.classify_diff(super_cand, existing, "ws-1", "p1")
    assert diff_super.change_type == "SUPERSEDED"
    assert diff_super.existing_memory_id == "mem-1"

    # 4. COEXISTING (Clear scope: dev vs prod, high confidence)
    coexist_clear = ExtractedCandidate(
        memory_type="decision",
        statement="Use SQLite for local dev environment",
        structured_claim={
            "subject": "backend",
            "predicate": "uses_database",
            "object": "sqlite",
            "scope": {"env": "dev"}
        },
        confidence=0.94
    )
    diff_coexist_clear = diff_engine.classify_diff(coexist_clear, existing, "ws-1", "p1")
    assert diff_coexist_clear.change_type == "COEXISTING"
    assert not diff_coexist_clear.review_required

    # 4b. COEXISTING (Ambiguous / low confidence scope)
    coexist_ambig = ExtractedCandidate(
        memory_type="decision",
        statement="We also use SQLite sometimes",
        structured_claim={
            "subject": "backend",
            "predicate": "uses_database",
            "object": "sqlite",
            "scope": {"env": "local_experimental"}
        },
        confidence=0.79
    )
    diff_coexist_ambig = diff_engine.classify_diff(coexist_ambig, existing, "ws-1", "p1")
    assert diff_coexist_ambig.change_type == "COEXISTING"
    assert diff_coexist_ambig.review_required

    # 5. CONFLICTING (Same subject/predicate, different object, identical/unspecified scope, no superseding clause)
    conflict_cand = ExtractedCandidate(
        memory_type="decision",
        statement="Backend uses MongoDB for production storage",
        structured_claim={
            "subject": "backend",
            "predicate": "uses_database",
            "object": "mongodb",
            "scope": {"env": "prod"}
        },
        confidence=0.91
    )
    diff_conflict = diff_engine.classify_diff(conflict_cand, existing, "ws-1", "p1")
    assert diff_conflict.change_type == "CONFLICTING"
    assert diff_conflict.review_required

def test_interactive_conflict_resolution_actions(db_session):
    manager = MemoryManager()

    m1 = Memory(
        id="mem-a",
        workspace_id="ws-1",
        project_id="p1",
        memory_type="decision",
        statement="Backend uses PostgreSQL",
        structured_claim={"subject": "backend", "predicate": "uses_database", "object": "postgresql"},
        status="active"
    )
    m2 = Memory(
        id="mem-b",
        workspace_id="ws-1",
        project_id="p1",
        memory_type="decision",
        statement="Backend uses MongoDB",
        structured_claim={"subject": "backend", "predicate": "uses_database", "object": "mongodb"},
        status="review_required"
    )
    db_session.add_all([m1, m2])
    db_session.commit()

    # 1. get_conflicts should detect competing claim pair
    conflicts = manager.get_conflicts(db_session, "ws-1", "p1")
    assert len(conflicts) >= 1
    assert any(c["type"] == "CLAIM_CONFLICT" for c in conflicts)

    # 2. Test keep_both resolution
    res_both = manager.resolve_conflict(
        db=db_session,
        memory_id="mem-a",
        resolution_action="keep_both",
        paired_memory_id="mem-b",
        reason="Confirmed coexistence across distinct components"
    )
    assert res_both.status == "active"
    db_session.refresh(m2)
    assert m2.status == "active"
    assert m1.version == 2.0
    assert m2.version == 2.0

    # 3. Test supersede resolution
    res_super = manager.resolve_conflict(
        db=db_session,
        memory_id="mem-b",
        resolution_action="supersede",
        superseded_by_id="mem-a",
        reason="Superseded by mem-a"
    )
    assert res_super.status == "superseded"
    assert res_super.superseded_by_id == "mem-a"

    # 4. Test deprecate resolution
    res_dep = manager.resolve_conflict(
        db=db_session,
        memory_id="mem-b",
        resolution_action="deprecate",
        reason="Deprecated"
    )
    assert res_dep.status == "deprecated"

    # 5. Test forget resolution
    res_forget = manager.resolve_conflict(
        db=db_session,
        memory_id="mem-b",
        resolution_action="forget",
        reason="Forgotten"
    )
    assert res_forget.status == "forgotten"
