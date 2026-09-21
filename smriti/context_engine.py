from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from smriti.models import Project, Task, Memory, Message, Conversation
from smriti.schemas import PortableContextPackage
from smriti.retrieval import retrieval_engine

class ContextEngine:
    """Generates portable, compact, provider-agnostic context packages."""

    def build_context(
        self,
        db: Session,
        query: str,
        project_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        max_tokens_approx: int = 1500
    ) -> PortableContextPackage:
        project = None
        if project_id:
            project = db.query(Project).filter(Project.id == project_id).first()

        # 1. Retrieve relevant memories via Hybrid Retrieval
        search_hits = retrieval_engine.search(
            db=db,
            query=query,
            project_id=project_id,
            workspace_id=workspace_id,
            limit=10,
            include_superseded=False
        )

        relevant_memories_data = []
        decisions_data = []
        constraints_list = []
        sources_list = []

        seen_sources = set()

        for hit in search_hits:
            mem = db.query(Memory).filter(Memory.id == hit.id).first()
            if not mem:
                continue

            mem_dict = {
                "id": mem.id,
                "type": mem.memory_type,
                "statement": mem.statement,
                "rationale": mem.rationale,
                "confidence": mem.confidence,
                "status": mem.status
            }
            relevant_memories_data.append(mem_dict)

            if mem.memory_type == "decision":
                decisions_data.append(mem_dict)
            elif mem.memory_type == "constraint":
                constraints_list.append(mem.statement)

            if mem.source_message_id and mem.source_message_id not in seen_sources:
                seen_sources.add(mem.source_message_id)
                msg = db.query(Message).filter(Message.id == mem.source_message_id).first()
                if msg:
                    conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id).first()
                    sources_list.append({
                        "message_id": msg.id,
                        "conversation_id": msg.conversation_id,
                        "conversation_title": conv.title if conv else "Unknown Conversation",
                        "snippet": msg.content[:200]
                    })

        # 2. Extract Project Tasks and Metadata
        tasks_data = []
        if project:
            for t in project.tasks:
                if t.status in ["todo", "in_progress", "blocked"]:
                    tasks_data.append({
                        "id": t.id,
                        "title": t.title,
                        "status": t.status,
                        "priority": t.priority
                    })
            if project.constraints and isinstance(project.constraints, list):
                constraints_list.extend(project.constraints)

        # 3. Synthesize summary
        proj_name = project.name if project else "General Context"
        summary = f"Synthesized context for {proj_name} based on '{query}'."

        # 4. Formulate formatted markdown prompt for target LLM
        prompt_lines = [
            f"# PROJECT SMRITI CONTEXT PACKAGE",
            f"**Project**: {proj_name}",
            f"**Query**: {query}",
        ]

        if project:
            if project.goal:
                prompt_lines.append(f"**Goal**: {project.goal}")
            if project.architecture_overview:
                prompt_lines.append(f"**Architecture**: {project.architecture_overview}")
            if project.tech_stack:
                prompt_lines.append(f"**Tech Stack**: {', '.join(project.tech_stack)}")

        if constraints_list:
            prompt_lines.append("\n## Active Constraints")
            for c in set(constraints_list):
                prompt_lines.append(f"- {c}")

        if decisions_data:
            prompt_lines.append("\n## Key Decisions & Architecture Choices")
            for d in decisions_data:
                rat = f" (Rationale: {d['rationale']})" if d['rationale'] else ""
                prompt_lines.append(f"- {d['statement']}{rat}")

        if tasks_data:
            prompt_lines.append("\n## Active Tasks & Next Steps")
            for t in tasks_data:
                prompt_lines.append(f"- [{t['status'].upper()}] {t['title']} (Priority: {t['priority']})")

        if relevant_memories_data:
            prompt_lines.append("\n## Supporting Knowledge & Facts")
            for m in relevant_memories_data:
                if m["type"] not in ["decision", "constraint"]:
                    prompt_lines.append(f"- [{m['type'].upper()}] {m['statement']}")

        prompt_lines.append("\n## Instructions for AI")
        prompt_lines.append("Use the verified project state, decisions, and constraints above to maintain exact continuity. Do not contradict established architectural decisions unless requested.")

        formatted_prompt = "\n".join(prompt_lines)

        return PortableContextPackage(
            project_id=project.id if project else None,
            project_name=proj_name,
            query=query,
            summary=summary,
            goal=project.goal if project else None,
            architecture=project.architecture_overview if project else None,
            tech_stack=project.tech_stack if project else [],
            decisions=decisions_data,
            constraints=list(set(constraints_list)),
            tasks=tasks_data,
            relevant_memories=relevant_memories_data,
            sources=sources_list,
            formatted_prompt=formatted_prompt
        )

context_engine = ContextEngine()
