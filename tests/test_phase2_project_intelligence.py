import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from smriti.models import Base, User, Workspace, Project, Task, Milestone, Memory
from smriti.project_intelligence import ProjectIntelligenceService

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
    session.commit()

    yield session
    session.close()

def test_project_state_synthesis(db_session):
    service = ProjectIntelligenceService()

    proj = Project(
        id="p1",
        workspace_id="ws-1",
        name="AI Knowledge Core",
        status="active"
    )
    db_session.add(proj)
    db_session.commit()

    # Add active memories: decisions and technologies
    m1 = Memory(
        id="m1",
        workspace_id="ws-1",
        project_id="p1",
        memory_type="decision",
        statement="Build unified memory layer for autonomous AI",
        status="active"
    )
    m2 = Memory(
        id="m2",
        workspace_id="ws-1",
        project_id="p1",
        memory_type="technology",
        statement="Uses technology: PostgreSQL",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "postgresql"},
        status="active"
    )
    m3 = Memory(
        id="m3",
        workspace_id="ws-1",
        project_id="p1",
        memory_type="constraint",
        statement="Must preserve local privacy and avoid vendor lock-in",
        structured_claim={"subject": "project", "predicate": "constrained_by", "object": "local_privacy"},
        status="active"
    )
    db_session.add_all([m1, m2, m3])
    db_session.commit()

    # Run synthesis
    result = service.synthesize_project_state(db_session, "p1")
    assert result["project_id"] == "p1"
    assert "postgresql" in result["tech_stack"]
    assert "local_privacy" in result["constraints"]
    assert result["goal"] == "Build unified memory layer for autonomous AI"

    # Verify db record updated
    db_session.refresh(proj)
    assert "postgresql" in proj.tech_stack
    assert "local_privacy" in proj.constraints

def test_idempotent_milestone_detection(db_session):
    service = ProjectIntelligenceService()

    proj = Project(
        id="p1",
        workspace_id="ws-1",
        name="Smriti Milestone Project",
        tech_stack=["fastapi", "postgresql"],
        architecture_overview="Layered hexagonal architecture",
        status="active"
    )
    db_session.add(proj)

    # Completed task
    task = Task(
        id="t1",
        project_id="p1",
        title="Implement Phase 2 Extraction",
        description="Completed hybrid LLM and heuristic extraction",
        status="completed"
    )
    db_session.add(task)
    db_session.commit()

    # First detection
    milestones_1 = service.detect_milestones(db_session, "p1")
    types_1 = [ml.milestone_type for ml in milestones_1]
    assert "project_created" in types_1
    assert "technology_selected" in types_1
    assert "architecture_decided" in types_1
    assert "feature_completed" in types_1
    count_1 = len(milestones_1)

    # Second detection must be idempotent (no duplicate milestones created)
    milestones_2 = service.detect_milestones(db_session, "p1")
    assert len(milestones_2) == count_1

    # Add another completed task
    task2 = Task(
        id="t2",
        project_id="p1",
        title="Implement Fine-Grained Diff",
        status="done"
    )
    db_session.add(task2)
    db_session.commit()

    milestones_3 = service.detect_milestones(db_session, "p1")
    assert len(milestones_3) == count_1 + 1

def test_synthesis_ignores_superseded_and_forgotten(db_session):
    service = ProjectIntelligenceService()

    proj = Project(
        id="p-sup",
        workspace_id="ws-1",
        name="Superseded Project",
        status="active"
    )
    db_session.add(proj)
    db_session.commit()

    # Active memory
    m_active = Memory(
        id="m-act",
        workspace_id="ws-1",
        project_id="p-sup",
        memory_type="technology",
        statement="Uses technology: PostgreSQL",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "postgresql"},
        status="active"
    )
    # Superseded memory
    m_sup = Memory(
        id="m-sup",
        workspace_id="ws-1",
        project_id="p-sup",
        memory_type="technology",
        statement="Uses technology: MongoDB",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "mongodb"},
        status="superseded"
    )
    # Forgotten memory
    m_forg = Memory(
        id="m-forg",
        workspace_id="ws-1",
        project_id="p-sup",
        memory_type="technology",
        statement="Uses technology: SQLite",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "sqlite"},
        status="forgotten"
    )
    db_session.add_all([m_active, m_sup, m_forg])
    db_session.commit()

    res = service.synthesize_project_state(db_session, "p-sup")
    assert "postgresql" in res["tech_stack"]
    assert "mongodb" not in res["tech_stack"]
    assert "sqlite" not in res["tech_stack"]

def test_synthesis_rebuilds_and_removes_stale_tech_from_project_model(db_session):
    service = ProjectIntelligenceService()

    # Project starts with old tech in tech_stack
    proj = Project(
        id="p-stale",
        workspace_id="ws-1",
        name="Stale Project",
        tech_stack=["mongodb", "flask"],
        status="active"
    )
    db_session.add(proj)
    db_session.commit()
    proj.last_confirmed_at = None
    db_session.commit()

    # Only new active memories exist: PostgreSQL and FastAPI
    m1 = Memory(
        id="m-fresh-1",
        workspace_id="ws-1",
        project_id="p-stale",
        memory_type="technology",
        statement="Uses technology: PostgreSQL",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "postgresql"},
        status="active"
    )
    m2 = Memory(
        id="m-fresh-2",
        workspace_id="ws-1",
        project_id="p-stale",
        memory_type="technology",
        statement="Uses technology: FastAPI",
        structured_claim={"subject": "project", "predicate": "uses_backend_framework", "object": "fastapi"},
        status="active"
    )
    db_session.add_all([m1, m2])
    db_session.commit()

    res = service.synthesize_project_state(db_session, "p-stale")
    db_session.refresh(proj)

    # Stale technologies (mongodb, flask) must NOT be retained
    assert "mongodb" not in proj.tech_stack
    assert "flask" not in proj.tech_stack
    assert "postgresql" in proj.tech_stack
    assert "fastapi" in proj.tech_stack
    assert proj.tech_stack == ["fastapi", "postgresql"]


def test_synthesis_preserves_confirmed_and_rebuilds_derived(db_session):
    service = ProjectIntelligenceService()
    now = datetime.now(timezone.utc)

    # 1. Confirmed project with explicit values (must be preserved)
    proj_conf_full = Project(
        id="p-conf-full",
        workspace_id="ws-1",
        name="Confirmed Full",
        tech_stack=["user-tech"],
        constraints=["user-constraint"],
        status="active"
    )
    # 2. Confirmed project with empty values (must be populated from memories)
    proj_conf_empty = Project(
        id="p-conf-empty",
        workspace_id="ws-1",
        name="Confirmed Empty",
        tech_stack=[],
        constraints=[],
        status="active"
    )
    # 3. Unconfirmed project (must rebuild from scratch, dropping stale)
    proj_unconf = Project(
        id="p-unconf",
        workspace_id="ws-1",
        name="Unconfirmed",
        tech_stack=["stale-tech"],
        status="active"
    )
    db_session.add_all([proj_conf_full, proj_conf_empty, proj_unconf])
    db_session.commit()

    # Force unconfirmed project to be unconfirmed
    proj_unconf.last_confirmed_at = None
    db_session.commit()

    # Memory for conf_full (should be ignored)
    m_full = Memory(
        id="m-full", workspace_id="ws-1", project_id="p-conf-full",
        memory_type="technology", statement="Uses unwanted-tech",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "unwanted-tech"},
        status="active"
    )
    # Memory for conf_empty (should populate it)
    m_empty = Memory(
        id="m-empty", workspace_id="ws-1", project_id="p-conf-empty",
        memory_type="technology", statement="Uses fresh-tech",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "fresh-tech"},
        status="active"
    )
    # Memory for unconf (should rebuild, dropping stale-tech)
    m_unconf = Memory(
        id="m-unconf", workspace_id="ws-1", project_id="p-unconf",
        memory_type="technology", statement="Uses fresh-tech",
        structured_claim={"subject": "project", "predicate": "uses_database", "object": "fresh-tech"},
        status="active"
    )
    db_session.add_all([m_full, m_empty, m_unconf])
    db_session.commit()

    service.synthesize_project_state(db_session, "p-conf-full")
    service.synthesize_project_state(db_session, "p-conf-empty")
    service.synthesize_project_state(db_session, "p-unconf")
    db_session.refresh(proj_conf_full)
    db_session.refresh(proj_conf_empty)
    db_session.refresh(proj_unconf)

    # 1. Preserved
    assert proj_conf_full.tech_stack == ["user-tech"]
    assert "unwanted-tech" not in proj_conf_full.tech_stack

    # 2. Populated
    assert proj_conf_empty.tech_stack == ["fresh-tech"]

    # 3. Rebuilt and stale dropped
    assert "fresh-tech" in proj_unconf.tech_stack
    assert "stale-tech" not in proj_unconf.tech_stack

