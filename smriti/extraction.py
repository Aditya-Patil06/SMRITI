import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ExtractedCandidate(BaseModel):
    memory_type: str  # decision, task, problem, solution, technology, concept, constraint, status_change, fact
    statement: str
    rationale: Optional[str] = None
    confidence: float = 0.85
    status: str = "active"  # active or review_required
    details: Dict[str, Any] = {}

class MemoryExtractor:
    """Heuristic and pattern-based extractor for structured project memory."""

    DECISION_PATTERNS = [
        re.compile(r"(?:we decided to|let's go with|decision is to|chose to|opted for|will use)\s+([^.\n]+)", re.IGNORECASE),
        re.compile(r"(?:agreed on|decided on|selected)\s+([^.\n]+)", re.IGNORECASE)
    ]

    TASK_PATTERNS = [
        re.compile(r"(?:todo|next step|action item|task|we need to|implement)\s*:\s*([^.\n]+)", re.IGNORECASE),
        re.compile(r"(?:we should implement|need to build|have to create)\s+([^.\n]+)", re.IGNORECASE)
    ]

    PROBLEM_PATTERNS = [
        re.compile(r"(?:issue is|problem is|blocker|error is|failing because|bug)\s*:\s*([^.\n]+)", re.IGNORECASE),
        re.compile(r"(?:struggling with|having trouble with)\s+([^.\n]+)", re.IGNORECASE)
    ]

    SOLUTION_PATTERNS = [
        re.compile(r"(?:solution is|fixed by|workaround is|resolved by)\s*:\s*([^.\n]+)", re.IGNORECASE),
        re.compile(r"(?:to fix this,\s*|the fix is to)\s+([^.\n]+)", re.IGNORECASE)
    ]

    CONSTRAINT_PATTERNS = [
        re.compile(r"(?:constraint|must not|cannot use|budget limit|requirement)\s*:\s*([^.\n]+)", re.IGNORECASE),
        re.compile(r"(?:strictly require|must always)\s+([^.\n]+)", re.IGNORECASE)
    ]

    TECH_PATTERNS = [
        re.compile(r"\b(FastAPI|React|TypeScript|Python|PostgreSQL|SQLite|NetworkX|Neo4j|Docker|Redis|Tailwind|Node\.js|PyTorch|TensorFlow|Next\.js|Vue|GraphQL|Alembic|SQLAlchemy)\b", re.IGNORECASE)
    ]

    def extract_from_message(self, role: str, content: str) -> List[ExtractedCandidate]:
        candidates: List[ExtractedCandidate] = []
        if not content or len(content.strip()) < 5:
            return candidates

        # 1. Decisions
        for pat in self.DECISION_PATTERNS:
            for match in pat.finditer(content):
                stmt = match.group(1).strip()
                if len(stmt) > 6:
                    # Look for rationale like "because..."
                    rationale = None
                    if "because" in stmt.lower():
                        parts = re.split(r"\bbecause\b", stmt, flags=re.IGNORECASE)
                        stmt_clean = parts[0].strip()
                        rationale = parts[1].strip()
                    else:
                        stmt_clean = stmt

                    conf = 0.92 if role == "assistant" else 0.88
                    candidates.append(ExtractedCandidate(
                        memory_type="decision",
                        statement=f"Decided to {stmt_clean}",
                        rationale=rationale,
                        confidence=conf,
                        status="active"
                    ))

        # 2. Tasks
        for pat in self.TASK_PATTERNS:
            for match in pat.finditer(content):
                stmt = match.group(1).strip()
                if len(stmt) > 6:
                    candidates.append(ExtractedCandidate(
                        memory_type="task",
                        statement=stmt,
                        confidence=0.89,
                        status="active"
                    ))

        # 3. Problems
        for pat in self.PROBLEM_PATTERNS:
            for match in pat.finditer(content):
                stmt = match.group(1).strip()
                if len(stmt) > 6:
                    candidates.append(ExtractedCandidate(
                        memory_type="problem",
                        statement=stmt,
                        confidence=0.86,
                        status="active"
                    ))

        # 4. Solutions
        for pat in self.SOLUTION_PATTERNS:
            for match in pat.finditer(content):
                stmt = match.group(1).strip()
                if len(stmt) > 6:
                    candidates.append(ExtractedCandidate(
                        memory_type="solution",
                        statement=stmt,
                        confidence=0.88,
                        status="active"
                    ))

        # 5. Constraints
        for pat in self.CONSTRAINT_PATTERNS:
            for match in pat.finditer(content):
                stmt = match.group(1).strip()
                if len(stmt) > 6:
                    candidates.append(ExtractedCandidate(
                        memory_type="constraint",
                        statement=stmt,
                        confidence=0.90,
                        status="active"
                    ))

        # 6. Technologies
        seen_tech = set()
        for pat in self.TECH_PATTERNS:
            for match in pat.finditer(content):
                tech_name = match.group(1)
                if tech_name.lower() not in seen_tech:
                    seen_tech.add(tech_name.lower())
                    candidates.append(ExtractedCandidate(
                        memory_type="technology",
                        statement=f"Uses technology: {tech_name}",
                        confidence=0.95,
                        status="active",
                        details={"technology": tech_name}
                    ))

        # Mark low confidence items as review_required
        for c in candidates:
            if c.confidence < 0.87:
                c.status = "review_required"

        return candidates
