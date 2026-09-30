"""
Targeted Phase 2 regression tests for commit 28d842b.

Commit under test:
    28d842b - fix(phase-2): close remaining Jules review findings

Production fixes covered:
1. get_conflicts() now requires exact project_id equality, including None.
2. Workspace-scoped canonical export now includes raw Task IDs when
   determining which relationship/audit records belong to the export scope.

These tests are intentionally narrow and should be run in addition to the
existing full Phase 2 suite.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from smriti.models import (
    Base,
    User,
    Workspace,
    Project,
    Memory,
    Task,
    RelationshipEdge,
)
from smriti.memory_manager import MemoryManager
from smriti.canonical_export import CanonicalExportEngine


@pytest.fixture
def db_session():
    """Create an isolated in-memory database for each regression test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    user = User(
        id="test-user",
        username="testuser",
        email="test@smriti.local",
    )
    workspace = Workspace(
        id="ws-1",
        user_id="test-user",
        name="Test Workspace",
        is_default=True,
    )
    session.add_all([user, workspace])
    session.commit()

    yield session
    session.close()


def _memory(
    memory_id,
    project_id,
    statement,
    database,
):
    return Memory(
        id=memory_id,
        workspace_id="ws-1",
        project_id=project_id,
        memory_type="decision",
        statement=statement,
        status="active",
        structured_claim={
            "subject": "project",
            "predicate": "uses_database",
            "object": database,
        },
    )


def test_get_conflicts_rejects_cross_project_pairs(db_session):
    """
    Regression: conflicting claims in different projects must not be paired.

    This catches the old condition that only rejected mismatched project IDs
    when both IDs were truthy.
    """
    manager = MemoryManager()

    project_a = _memory(
        "mem-a",
        "project-a",
        "Use PostgreSQL",
        "postgresql",
    )
    project_b = _memory(
        "mem-b",
        "project-b",
        "Use MongoDB",
        "mongodb",
    )

    db_session.add_all([project_a, project_b])
    db_session.commit()

    conflicts = manager.get_conflicts(db_session, "ws-1")

    assert conflicts == []


def test_get_conflicts_keeps_same_project_conflicts(db_session):
    """
    Regression guard: the project-scope fix must not suppress legitimate
    conflicts inside the same project.
    """
    manager = MemoryManager()

    first = _memory(
        "mem-a1",
        "project-a",
        "Use PostgreSQL",
        "postgresql",
    )
    second = _memory(
        "mem-a2",
        "project-a",
        "Use SQLite",
        "sqlite",
    )

    db_session.add_all([first, second])
    db_session.commit()

    conflicts = manager.get_conflicts(db_session, "ws-1")

    assert len(conflicts) == 1
    pair = {
        conflicts[0]["memory_a"]["id"],
        conflicts[0]["memory_b"]["id"],
    }
    assert pair == {"mem-a1", "mem-a2"}


def test_get_conflicts_treats_none_project_ids_as_same_scope(db_session):
    """
    Regression for the exact comparison introduced in 28d842b.

    Two unassigned memories have project_id=None, so None == None and they
    remain eligible for conflict detection. This prevents the fix from
    accidentally treating all project-less memories as unrelated.
    """
    manager = MemoryManager()

    first = _memory(
        "mem-none-1",
        None,
        "Use Redis",
        "redis",
    )
    second = _memory(
        "mem-none-2",
        None,
        "Use Memcached",
        "memcached",
    )

    db_session.add_all([first, second])
    db_session.commit()

    conflicts = manager.get_conflicts(db_session, "ws-1")

    assert len(conflicts) == 1
    pair = {
        conflicts[0]["memory_a"]["id"],
        conflicts[0]["memory_b"]["id"],
    }
    assert pair == {"mem-none-1", "mem-none-2"}


def test_scoped_canonical_export_preserves_task_relationships(db_session):
    """
    Regression for the canonical-export fix in 28d842b.

    A workspace-scoped export must retain relationships whose raw source or
    target ID is a Task ID. Before the fix, task IDs were missing from the
    scoped-node ID set, so those relationships could be dropped.
    """
    workspace = db_session.query(Workspace).filter_by(id="ws-1").one()

    project = Project(
        id="project-1",
        workspace_id=workspace.id,
        name="Export Project",
    )
    task = Task(
        id="task-1",
        project_id=project.id,
        title="Important Task",
    )

    unrelated_workspace = Workspace(
        id="ws-2",
        user_id="test-user",
        name="Other Workspace",
        is_default=False,
    )
    other_project = Project(
        id="project-2",
        workspace_id=unrelated_workspace.id,
        name="Other Project",
    )
    other_task = Task(
        id="task-2",
        project_id=other_project.id,
        title="Other Task",
    )

    db_session.add_all(
        [project, task, unrelated_workspace, other_project, other_task]
    )
    db_session.commit()

    scoped_edge = RelationshipEdge(
        source_type="task",
        source_id=task.id,
        relation="LINKED_TO",
        target_type="project",
        target_id=project.id,
    )
    unrelated_edge = RelationshipEdge(
        source_type="task",
        source_id=other_task.id,
        relation="LINKED_TO",
        target_type="project",
        target_id=other_project.id,
    )
    db_session.add_all([scoped_edge, unrelated_edge])
    db_session.commit()

    bundle = CanonicalExportEngine().export_all(
        db_session,
        workspace_id=workspace.id,
    )

    exported_edges = bundle["relationships"]

    assert any(
        edge["source_id"] == "task-1"
        and edge["target_id"] == "project-1"
        for edge in exported_edges
    )

    assert not any(
        edge["source_id"] == "task-2"
        and edge["target_id"] == "project-2"
        for edge in exported_edges
    )


def test_scoped_canonical_export_contains_exported_task(db_session):
    """
    Regression guard: the task itself must remain part of a workspace-scoped
    canonical export, not just its relationship.
    """
    workspace = db_session.query(Workspace).filter_by(id="ws-1").one()

    project = Project(
        id="project-task-export",
        workspace_id=workspace.id,
        name="Task Export Project",
    )
    task = Task(
        id="task-export",
        project_id=project.id,
        title="Export Me",
    )

    db_session.add_all([project, task])
    db_session.commit()

    bundle = CanonicalExportEngine().export_all(
        db_session,
        workspace_id=workspace.id,
    )

    exported_task_ids = {item["id"] for item in bundle["tasks"]}

    assert "task-export" in exported_task_ids
    assert bundle["manifest"]["entity_counts"]["tasks"] >= 1
