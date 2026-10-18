"""Tests for the read-only ChatGPT Actions preflight diagnostic."""

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from smriti.actions_preflight import build_preflight_report, inspect_json_cache
from smriti.models import Base, Memory, User, Workspace


def test_inspect_json_cache_reports_incompatible_dimensions(tmp_path):
    cache_path = tmp_path / "vectors.json"
    cache_path.write_text(
        json.dumps({
            "mem-1": {
                "vector": [0.1, 0.2],
                "metadata": {"workspace_id": "ws-1"},
            }
        }),
        encoding="utf-8",
    )

    result = inspect_json_cache(cache_path, expected_dimension=384)

    assert result["status"] == "warning"
    assert result["entries"] == 1
    assert result["dimensions"] == {2: 1}
    assert result["dimension_mismatches"] == 1
    assert result["entries_by_workspace"] == {"ws-1": 1}


def test_inspect_json_cache_handles_corrupt_and_missing_files(tmp_path):
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not-json", encoding="utf-8")

    assert inspect_json_cache(corrupt_path, 3)["status"] == "invalid"
    assert inspect_json_cache(tmp_path / "missing.json", 3)["status"] == "missing"
    assert inspect_json_cache(None, 3)["status"] == "not_applicable"


def test_preflight_reports_workspace_counts_without_flagging_other_owned_data(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    try:
        db.add_all([
            User(id="user-1", username="user1", email="user1@example.com"),
            User(id="user-2", username="user2", email="user2@example.com"),
            Workspace(id="ws-user-1", user_id="user-1", name="Personal", is_default=True),
            Workspace(id="ws-user-2", user_id="user-2", name="Other", is_default=True),
            Memory(
                id="mem-user-1",
                workspace_id="ws-user-1",
                memory_type="decision",
                statement="Use FastAPI",
                status="active",
            ),
            Memory(
                id="mem-user-2",
                workspace_id="ws-user-2",
                memory_type="decision",
                statement="Use PostgreSQL",
                status="active",
            ),
        ])
        db.commit()

        cache_path = tmp_path / "vectors.json"
        cache_path.write_text(
            json.dumps({
                "mem-user-1": {
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {"workspace_id": "ws-user-1"},
                },
                "mem-user-2": {
                    "vector": [0.3, 0.2, 0.1],
                    "metadata": {"workspace_id": "ws-user-2"},
                },
            }),
            encoding="utf-8",
        )

        report = build_preflight_report(db, "user-1", str(cache_path), expected_dimension=3)

        assert report["status"] == "ok"
        assert report["eligible_database_memories"] == 1
        assert report["workspaces"] == [{
            "id": "ws-user-1",
            "name": "Personal",
            "is_default": True,
            "projects": 0,
            "memories": 1,
        }]
        assert report["database_memories_missing_from_cache"] == []
        assert report["extra_cache_memory_ids"] == []
        assert report["warnings"] == []
    finally:
        db.close()
        engine.dispose()


def test_preflight_flags_database_memories_missing_from_cache(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    try:
        db.add_all([
            User(id="user-1", username="user1", email="user1@example.com"),
            Workspace(id="ws-user-1", user_id="user-1", name="Personal", is_default=True),
            Memory(
                id="mem-not-indexed",
                workspace_id="ws-user-1",
                memory_type="decision",
                statement="Use FastAPI",
                status="active",
            ),
        ])
        db.commit()

        cache_path = tmp_path / "vectors.json"
        cache_path.write_text("{}", encoding="utf-8")

        report = build_preflight_report(db, "user-1", str(cache_path), expected_dimension=3)

        assert report["status"] == "warning"
        assert report["database_memories_missing_from_cache"] == ["mem-not-indexed"]
        assert any("absent from the JSON vector cache" in item for item in report["warnings"])
    finally:
        db.close()
        engine.dispose()
