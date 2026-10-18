"""Read-only diagnostics for the ChatGPT Actions runtime configuration.

Run with: python -m smriti.actions_preflight

This command never creates tables, seeds records, rebuilds indices, or writes cache files.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import func

from smriti.config import settings
from smriti.models import Memory, Project, SessionLocal, User, Workspace
from smriti.vector_store import get_vector_store


def inspect_json_cache(
    cache_path: str | os.PathLike[str] | None,
    expected_dimension: int | None,
) -> dict[str, Any]:
    """Inspect cache shape and dimensions without loading or modifying the cache."""
    if not cache_path:
        return {"kind": "non-json-store", "status": "not_applicable"}

    path = Path(cache_path)
    if not path.exists():
        return {"kind": "json", "status": "missing", "path": str(path)}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "kind": "json",
            "status": "invalid",
            "path": str(path),
            "error": type(exc).__name__,
        }

    if not isinstance(payload, dict):
        return {"kind": "json", "status": "invalid", "path": str(path), "error": "root_not_object"}

    dimensions: Counter[int] = Counter()
    workspace_counts: Counter[str] = Counter()
    malformed = 0
    dimension_mismatches = 0
    cache_ids: set[str] = set()

    for memory_id, entry in payload.items():
        cache_ids.add(memory_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("vector"), list):
            malformed += 1
            continue

        dimension = len(entry["vector"])
        dimensions[dimension] += 1
        if expected_dimension is not None and dimension != expected_dimension:
            dimension_mismatches += 1

        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            malformed += 1
            continue
        workspace_id = metadata.get("workspace_id")
        if isinstance(workspace_id, str) and workspace_id:
            workspace_counts[workspace_id] += 1

    status = "warning" if malformed or dimension_mismatches else "ok"
    return {
        "kind": "json",
        "status": status,
        "path": str(path),
        "entries": len(payload),
        "dimensions": dict(dimensions),
        "expected_dimension": expected_dimension,
        "dimension_mismatches": dimension_mismatches,
        "malformed_entries": malformed,
        "entries_by_workspace": dict(workspace_counts),
        "memory_ids": cache_ids,
    }


def build_preflight_report(
    db,
    user_id: str,
    cache_path: str | None,
    expected_dimension: int | None,
) -> dict[str, Any]:
    """Build a read-only report of user workspaces, memory counts, and cache alignment."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return {"status": "error", "error": "configured_user_not_found", "user_id": user_id}

    workspaces = (
        db.query(Workspace)
        .filter(Workspace.user_id == user_id)
        .order_by(Workspace.created_at.asc())
        .all()
    )
    workspace_ids = {workspace.id for workspace in workspaces}
    owned_workspace_filter = Memory.workspace_id.in_(workspace_ids) if workspace_ids else False

    memory_counts = dict(
        db.query(Memory.workspace_id, func.count(Memory.id))
        .filter(owned_workspace_filter)
        .group_by(Memory.workspace_id)
        .all()
    )
    project_counts = dict(
        db.query(Project.workspace_id, func.count(Project.id))
        .filter(Project.workspace_id.in_(workspace_ids) if workspace_ids else False)
        .group_by(Project.workspace_id)
        .all()
    )

    eligible_memory_ids = {
        row[0]
        for row in db.query(Memory.id)
        .filter(owned_workspace_filter, Memory.status.notin_(("superseded", "forgotten")))
        .all()
    }
    all_eligible_memory_ids = {
        row[0]
        for row in db.query(Memory.id)
        .filter(Memory.status.notin_(("superseded", "forgotten")))
        .all()
    }

    cache = inspect_json_cache(cache_path, expected_dimension)
    cache_ids = cache.pop("memory_ids", set())
    cache_is_comparable = cache["kind"] == "json" and cache["status"] not in ("missing", "invalid")
    missing_cache_ids = sorted(eligible_memory_ids - cache_ids) if cache_is_comparable else []
    extra_cache_ids = sorted(cache_ids - all_eligible_memory_ids) if cache_is_comparable else []

    workspace_report = []
    for workspace in workspaces:
        workspace_report.append({
            "id": workspace.id,
            "name": workspace.name,
            "is_default": bool(workspace.is_default),
            "projects": int(project_counts.get(workspace.id, 0)),
            "memories": int(memory_counts.get(workspace.id, 0)),
        })

    warnings = []
    if not workspace_report:
        warnings.append("The configured user has no workspaces.")
    if workspace_report and not any(item["memories"] for item in workspace_report):
        warnings.append("No memories belong to any workspace owned by the configured user.")
    if cache["status"] in ("missing", "invalid", "warning"):
        warnings.append("The vector cache is missing, malformed, or has dimensions inconsistent with the configured embedding model.")
    if missing_cache_ids:
        warnings.append(f"{len(missing_cache_ids)} eligible database memories are absent from the JSON vector cache.")
    if extra_cache_ids:
        warnings.append(f"{len(extra_cache_ids)} cached memory IDs do not match eligible database memories.")

    return {
        "status": "warning" if warnings else "ok",
        "user_id": user_id,
        "workspaces": workspace_report,
        "cache": cache,
        "eligible_database_memories": len(eligible_memory_ids),
        "database_memories_missing_from_cache": missing_cache_ids,
        "extra_cache_memory_ids": extra_cache_ids,
        "warnings": warnings,
    }


def main() -> int:
    """Print the preflight report and return nonzero when setup needs attention."""
    user_id = (os.getenv("SMRITI_ACTIONS_USER_ID") or settings.actions_user_id or "").strip()
    if not user_id:
        print("ERROR: Set SMRITI_ACTIONS_USER_ID before running this diagnostic.")
        return 2

    try:
        active_store = get_vector_store()
        expected_dimension = getattr(getattr(active_store, "embedding_provider", None), "dimension", None)
        cache_path = getattr(active_store, "storage_path", None)
    except Exception as exc:
        print(f"ERROR: Could not initialize the configured vector store ({type(exc).__name__}).")
        return 2

    db = SessionLocal()
    try:
        report = build_preflight_report(db, user_id, cache_path, expected_dimension)
    except Exception as exc:
        print(f"ERROR: Preflight database inspection failed ({type(exc).__name__}).")
        return 2
    finally:
        db.close()

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if report.get("status") != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
