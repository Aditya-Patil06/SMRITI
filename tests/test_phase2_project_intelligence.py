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
