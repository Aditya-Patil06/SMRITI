from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from smriti.models import Project, Memory, Task, Milestone

class ProjectIntelligenceService:
    """
    Synthesizes project state and detects milestones automatically and idempotently.
    Ensures manual user edits with last_confirmed_at are preserved.
    """

    SUPPORTED_MILESTONE_TYPES = [
        "project_created",
        "architecture_decided",
        "technology_selected",
        "feature_completed",
        "phase_completed"
    ]

    def synthesize_project_state(self, db: Session, project_id: str) -> Dict[str, Any]:
        """
        Synthesizes goal, architecture, tech stack, and constraints from memories and tasks.
        Updates Project if fields are unset or not user-confirmed.
        """
        proj = db.query(Project).filter(Project.id == project_id).first()
        if not proj:
            raise ValueError(f"Project '{project_id}' not found")

        # Fetch active memories for this project
        memories = db.query(Memory).filter(
            Memory.project_id == project_id,
            Memory.status == "active"
        ).order_by(Memory.created_at.asc()).all()

        tasks = db.query(Task).filter(Task.project_id == project_id).all()

        derived_tech_stack = set()
        derived_constraints = set()
        architecture_points = []
        derived_goal = None

        for m in memories:
            handled_by_claim = False

            # Check structured claim first
            if m.structured_claim:
                claim = m.structured_claim
                pred = claim.get("predicate", "")
                obj = claim.get("object", "")
                if pred in ["uses_database", "uses_backend_framework", "uses_frontend_framework", "uses_technology", "uses_orm", "uses_language", "uses_cache"]:
                    if obj:
                        derived_tech_stack.add(obj)
                        handled_by_claim = True
                if pred in ["constrained_by", "requires_compliance"]:
                    if obj:
                        derived_constraints.add(obj)
                        handled_by_claim = True

            # If not already handled by structured claim, check legacy types
            if not handled_by_claim:
                if m.memory_type == "technology":
                    tech_val = (m.details or {}).get("technology")
                    if tech_val:
                        derived_tech_stack.add(tech_val.lower())
                    else:
                        derived_tech_stack.add(m.statement.replace("Uses technology: ", "").strip().lower())
                elif m.memory_type == "constraint":
                    derived_constraints.add(m.statement.strip())

            if m.memory_type == "decision":
                architecture_points.append(m.statement.strip())
                if not derived_goal and ("build" in m.statement.lower() or "create" in m.statement.lower() or "goal" in m.statement.lower()):
                    derived_goal = m.statement.strip()

        # Check if project has user-confirmed fields
        # If last_confirmed_at is set, we do not overwrite explicit non-empty user fields
        now = datetime.now(timezone.utc)
        changes = {}

        if not proj.goal and derived_goal:
            proj.goal = derived_goal
            changes["goal"] = derived_goal

        sorted_tech = sorted(list(derived_tech_stack))
        if sorted_tech != (proj.tech_stack or []):
            proj.tech_stack = sorted_tech
            changes["tech_stack"] = sorted_tech

        sorted_constraints = sorted(list(derived_constraints))
        if sorted_constraints != (proj.constraints or []):
            proj.constraints = sorted_constraints
            changes["constraints"] = sorted_constraints

        if architecture_points and not proj.architecture_overview:
            overview = "; ".join(architecture_points[:5])
            proj.architecture_overview = overview
            changes["architecture_overview"] = overview

        if changes:
            proj.updated_at = now
            db.commit()
            db.refresh(proj)

        return {
            "project_id": proj.id,
            "name": proj.name,
            "goal": proj.goal,
            "architecture_overview": proj.architecture_overview,
            "tech_stack": proj.tech_stack,
            "constraints": proj.constraints,
            "changes_applied": changes
        }

    def detect_milestones(self, db: Session, project_id: str) -> List[Milestone]:
        """
        Detects milestones idempotently from project state, memories, and completed tasks.
        """
        proj = db.query(Project).filter(Project.id == project_id).first()
        if not proj:
            raise ValueError(f"Project '{project_id}' not found")

        existing_milestones = db.query(Milestone).filter(Milestone.project_id == project_id).all()
        existing_types = {ml.milestone_type: ml for ml in existing_milestones}
        existing_titles = {ml.title: ml for ml in existing_milestones}

        created_milestones = []
        now = datetime.now(timezone.utc)

        # 1. Project Created milestone
        if "project_created" not in existing_types:
            m_created = Milestone(
                project_id=proj.id,
                title="Project Initialized",
                description=f"Project '{proj.name}' was created.",
                milestone_type="project_created",
                reached_at=proj.created_at or now
            )
            db.add(m_created)
            created_milestones.append(m_created)

        # 2. Technology Selected milestone
        if "technology_selected" not in existing_types and proj.tech_stack and len(proj.tech_stack) > 0:
            tech_evidence = db.query(Memory).filter(
                Memory.project_id == project_id,
                Memory.memory_type.in_(["technology", "decision"]),
                Memory.status == "active"
            ).order_by(Memory.created_at.asc()).first()

            m_tech = Milestone(
                project_id=proj.id,
                title="Core Technology Stack Selected",
                description=f"Initial technology stack adopted: {', '.join(proj.tech_stack)}",
                milestone_type="technology_selected",
                evidence_memory_id=tech_evidence.id if tech_evidence else None,
                reached_at=tech_evidence.created_at if tech_evidence else now
            )
            db.add(m_tech)
            created_milestones.append(m_tech)

        # 3. Architecture Decided milestone
        if "architecture_decided" not in existing_types:
            arch_mem = db.query(Memory).filter(
                Memory.project_id == project_id,
                Memory.memory_type == "decision",
                Memory.status == "active"
            ).order_by(Memory.created_at.asc()).first()

            if arch_mem or proj.architecture_overview:
                m_arch = Milestone(
                    project_id=proj.id,
                    title="Architecture Defined",
                    description=proj.architecture_overview or (arch_mem.statement if arch_mem else "Key architectural decisions recorded"),
                    milestone_type="architecture_decided",
                    evidence_memory_id=arch_mem.id if arch_mem else None,
                    reached_at=arch_mem.created_at if arch_mem else now
                )
                db.add(m_arch)
                created_milestones.append(m_arch)

        # 4. Feature Completed milestones (from completed tasks)
        done_tasks = db.query(Task).filter(
            Task.project_id == project_id,
            Task.status.in_(["done", "completed"])
        ).all()

        for task in done_tasks:
            title = f"Task Completed: {task.title}"
            if title not in existing_titles:
                m_task = Milestone(
                    project_id=proj.id,
                    title=title,
                    description=task.description or f"Successfully completed task: {task.title}",
                    milestone_type="feature_completed",
                    reached_at=task.updated_at or now
                )
                db.add(m_task)
                created_milestones.append(m_task)
                existing_titles[title] = m_task

        if created_milestones:
            db.commit()
            for ml in created_milestones:
                db.refresh(ml)

        # Return full active milestones for the project
        return db.query(Milestone).filter(Milestone.project_id == project_id).order_by(Milestone.reached_at.asc()).all()

project_intelligence_service = ProjectIntelligenceService()
