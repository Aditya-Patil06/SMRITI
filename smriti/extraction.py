import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExtractedCandidate(BaseModel):
    memory_type: str  # decision, task, problem, solution, technology, concept, constraint, status_change, fact
    statement: str
    rationale: Optional[str] = None
    structured_claim: Optional[Dict[str, Any]] = None  # {subject, predicate, object, scope, temporal_context}
    confidence: float = 0.85
    status: str = "active"  # active or review_required
    details: Dict[str, Any] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)  # [{"name": "...", "entity_type": "..."}]
    relationships: List[Dict[str, Any]] = Field(default_factory=list)  # [{"source": "...", "relation": "...", "target": "..."}]

class ClaimNormalizer:
    """Normalizes entities, aliases, and predicates deterministically."""

    SYNONYM_MAP = {
        # Databases
        "postgres": "postgresql",
        "postgre": "postgresql",
        "postgresql": "postgresql",
        "sqlite3": "sqlite",
        "sqlite": "sqlite",
        "mongo": "mongodb",
        "mongodb": "mongodb",
        "redis db": "redis",
        "redis cache": "redis",
        "redis": "redis",
        # Frameworks & Languages
        "fast api": "fastapi",
        "fastapi framework": "fastapi",
        "fastapi": "fastapi",
        "react.js": "react",
        "reactjs": "react",
        "react": "react",
        "nextjs": "next.js",
        "next.js": "next.js",
        "next": "next.js",
        "vuejs": "vue",
        "vue.js": "vue",
        "vue": "vue",
        "python3": "python",
        "python": "python",
        "typescript": "typescript",
        "ts": "typescript",
        "js": "javascript",
        "javascript": "javascript",
        # Architecture / Deployment
        "docker container": "docker",
        "docker": "docker",
        "tailwind css": "tailwindcss",
        "tailwind": "tailwindcss",
        "sqlalchemy orm": "sqlalchemy",
        "sqlalchemy": "sqlalchemy",
        "alembic migrations": "alembic",
        "alembic": "alembic",
    }

    PREDICATE_MAP = {
        "uses database": "uses_database",
        "database": "uses_database",
        "uses db": "uses_database",
        "uses backend": "uses_backend_framework",
        "uses backend framework": "uses_backend_framework",
        "backend": "uses_backend_framework",
        "uses frontend": "uses_frontend_framework",
        "uses frontend framework": "uses_frontend_framework",
        "frontend": "uses_frontend_framework",
        "uses orm": "uses_orm",
        "orm": "uses_orm",
        "uses language": "uses_language",
        "written in": "uses_language",
        "language": "uses_language",
        "runs on": "runs_on",
        "deployed on": "deployed_on",
        "uses cache": "uses_cache",
        "replaces": "replaces",
        "supersedes": "supersedes",
        "uses": "uses",
    }

    @classmethod
    def normalize_token(cls, token: Optional[str]) -> str:
        if not token:
            return ""
        cleaned = token.strip().lower()
        cleaned = re.sub(r"[-_]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cls.SYNONYM_MAP.get(cleaned, cleaned)

    @classmethod
    def normalize_predicate(cls, predicate: Optional[str]) -> str:
        if not predicate:
            return "relates_to"
        cleaned = predicate.strip().lower()
        cleaned = re.sub(r"[-_]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cls.PREDICATE_MAP.get(cleaned, cleaned.replace(" ", "_"))

    @classmethod
    def normalize_claim(cls, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Returns normalized structured claim dictionary."""
        subject = cls.normalize_token(claim.get("subject", ""))
        predicate = cls.normalize_predicate(claim.get("predicate", ""))
        obj = cls.normalize_token(claim.get("object", ""))
        scope = claim.get("scope") or {}
        # Normalize scope keys and string values
        norm_scope = {}
        for k, v in scope.items():
            k_clean = k.strip().lower()
            if isinstance(v, str):
                norm_scope[k_clean] = cls.normalize_token(v)
            else:
                norm_scope[k_clean] = v
        temporal = claim.get("temporal_context")
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "scope": norm_scope,
            "temporal_context": temporal
        }

class BaseLLMExtractor(ABC):
    @abstractmethod
    def extract(self, role: str, content: str, context: Optional[Dict[str, Any]] = None) -> List[ExtractedCandidate]:
        pass

class MockLLMExtractor(BaseLLMExtractor):
    """
    Deterministic Mock LLM Extractor supporting 9 testing fixtures:
    1. normal: Standard extraction
    2. structured_claim: Explicit subject/predicate/object
    3. duplicate: Paraphrased duplicate of an existing claim
    4. conflict: Incompatible claim with identical subject/predicate
    5. supersession: Explicit replacement statement
    6. clear_coexistence: Different explicit scopes (e.g. dev vs prod)
    7. ambiguous_coexistence: Vague scope distinction
    8. malformed: Garbage or unparseable output handled gracefully
    9. invalid_source: Empty or non-attributable content handled safely
    """

    def __init__(self, mode: str = "normal", custom_fixtures: Optional[Dict[str, List[ExtractedCandidate]]] = None):
        self.mode = mode
        self.custom_fixtures = custom_fixtures or {}

    def set_mode(self, mode: str):
        self.mode = mode

    def extract(self, role: str, content: str, context: Optional[Dict[str, Any]] = None) -> List[ExtractedCandidate]:
        if self.mode in self.custom_fixtures:
            return self.custom_fixtures[self.mode]

        # 9. Invalid source handling
        if self.mode == "invalid_source" or not content or len(content.strip()) < 3:
            return []

        # 8. Malformed handling (simulates LLM returning unparseable or dirty text that parser cleans or discards)
        if self.mode == "malformed":
            return []

        # 1. Normal extraction
        if self.mode == "normal":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="Use PostgreSQL as primary database",
                    rationale="Relational consistency and ACID guarantees",
                    confidence=0.94,
                    status="active",
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_database",
                        "object": "postgresql",
                        "scope": {"component": "backend"},
                        "temporal_context": "current"
                    },
                    entities=[
                        {"name": "backend", "entity_type": "component"},
                        {"name": "postgresql", "entity_type": "technology"}
                    ],
                    relationships=[
                        {"source": "backend", "relation": "USES_DATABASE", "target": "postgresql"}
                    ]
                )
            ]

        # 2. Structured claim extraction
        if self.mode == "structured_claim":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="Backend uses FastAPI framework",
                    confidence=0.96,
                    status="active",
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_backend_framework",
                        "object": "fastapi",
                        "scope": {"component": "backend"},
                        "temporal_context": "current"
                    },
                    entities=[
                        {"name": "backend", "entity_type": "component"},
                        {"name": "fastapi", "entity_type": "technology"}
                    ],
                    relationships=[
                        {"source": "backend", "relation": "USES_FRAMEWORK", "target": "fastapi"}
                    ]
                )
            ]

        # 3. Duplicate / Unchanged (e.g. paraphrased version of PostgreSQL)
        if self.mode == "duplicate":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="We are using Postgres for our database layer",
                    rationale="Postgres ACID guarantees",
                    confidence=0.93,
                    status="active",
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_database",
                        "object": "postgres",
                        "scope": {"component": "backend"},
                        "temporal_context": "current"
                    }
                )
            ]

        # 4. Genuine Conflict (e.g. backend uses MongoDB without superseding rationale)
        if self.mode == "conflict":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="Backend uses MongoDB as the primary database",
                    rationale="Document flexibility",
                    confidence=0.91,
                    status="active",
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_database",
                        "object": "mongodb",
                        "scope": {"component": "backend"},
                        "temporal_context": "current"
                    }
                )
            ]

        # 5. Supersession (explicit replacement language)
        if self.mode == "supersession":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="Switched to SQLite instead of PostgreSQL for local development simplicity",
                    rationale="Zero setup overhead for lightweight environments",
                    confidence=0.95,
                    status="active",
                    details={"replaces": "postgresql"},
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_database",
                        "object": "sqlite",
                        "scope": {"component": "backend"},
                        "temporal_context": "current"
                    }
                )
            ]

        # 6. Clear-scope coexistence (e.g. SQLite for local dev vs Postgres for production)
        if self.mode == "clear_coexistence":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="Use SQLite for dev and test environments",
                    rationale="Fast in-memory testing",
                    confidence=0.95,
                    status="active",
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_database",
                        "object": "sqlite",
                        "scope": {"env": "dev"},
                        "temporal_context": "current"
                    }
                )
            ]

        # 7. Ambiguous-scope coexistence
        if self.mode == "ambiguous_coexistence":
            return [
                ExtractedCandidate(
                    memory_type="decision",
                    statement="We also use SQLite sometimes",
                    rationale="For quick checks",
                    confidence=0.78,
                    status="review_required",
                    structured_claim={
                        "subject": "backend",
                        "predicate": "uses_database",
                        "object": "sqlite",
                        "scope": {"env": "unspecified_or_experimental"},
                        "temporal_context": "occasional"
                    }
                )
            ]

        return []

class MemoryExtractor:
    """Heuristic and pattern-based extractor for structured project memory."""

    DECISION_PATTERNS = [
        re.compile(r"(?:we decided to|let's go with|decision is to|chose to|opted for|will use|switched to|migrated to)\s+([^.\n]+)", re.IGNORECASE),
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
        re.compile(r"\b(FastAPI|React|TypeScript|Python|PostgreSQL|SQLite|NetworkX|Neo4j|Docker|Redis|Tailwind|Node\.js|PyTorch|TensorFlow|Next\.js|Vue|GraphQL|Alembic|SQLAlchemy|MongoDB)\b", re.IGNORECASE)
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
                    rationale = None
                    if "because" in stmt.lower():
                        parts = re.split(r"\bbecause\b", stmt, flags=re.IGNORECASE)
                        stmt_clean = parts[0].strip()
                        rationale = parts[1].strip()
                    else:
                        stmt_clean = stmt

                    conf = 0.92 if role == "assistant" else 0.88
                    # Check for replacement clause
                    details: Dict[str, Any] = {}
                    if "instead of" in stmt_clean.lower():
                        replaces_match = re.search(r"instead of\s+([A-Za-z0-9_\-]+)", stmt_clean, re.IGNORECASE)
                        if replaces_match:
                            details["replaces"] = ClaimNormalizer.normalize_token(replaces_match.group(1))

                    # Heuristic claim generation if tech is mentioned
                    claim = None
                    for tech_match in self.TECH_PATTERNS[0].finditer(stmt_clean):
                        tech = ClaimNormalizer.normalize_token(tech_match.group(1))
                        if tech == "redis":
                            pred = "uses_cache"
                        elif tech in ["postgresql", "sqlite", "mongodb"]:
                            pred = "uses_database"
                        else:
                            pred = "uses_technology"
                        claim = {
                            "subject": "project",
                            "predicate": pred,
                            "object": tech,
                            "scope": {"component": "core"},
                            "temporal_context": "current"
                        }
                        break

                    candidates.append(ExtractedCandidate(
                        memory_type="decision",
                        statement=f"Decided to {stmt_clean}",
                        rationale=rationale,
                        confidence=conf,
                        status="active",
                        structured_claim=claim,
                        details=details
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
                raw_tech = match.group(1)
                tech_name = ClaimNormalizer.normalize_token(raw_tech)
                if tech_name not in seen_tech:
                    seen_tech.add(tech_name)
                    if tech_name == "redis":
                        pred = "uses_cache"
                    elif tech_name in ["postgresql", "sqlite", "mongodb"]:
                        pred = "uses_database"
                    else:
                        pred = "uses_technology"
                    candidates.append(ExtractedCandidate(
                        memory_type="technology",
                        statement=f"Uses technology: {raw_tech}",
                        confidence=0.95,
                        status="active",
                        structured_claim={
                            "subject": "project",
                            "predicate": pred,
                            "object": tech_name,
                            "scope": {"component": "core"},
                            "temporal_context": "current"
                        },
                        details={"technology": raw_tech}
                    ))

        # Mark low confidence items as review_required
        for c in candidates:
            if c.confidence < 0.87:
                c.status = "review_required"

        return candidates

class HybridExtractionEngine:
    """
    Combines deterministic rule heuristics and LLM extraction.
    Normalizes claims, verifies confidence gating, and derives entities and relationships.
    """

    def __init__(self, heuristic_extractor: Optional[MemoryExtractor] = None, llm_extractor: Optional[BaseLLMExtractor] = None):
        self.heuristic_extractor = heuristic_extractor or MemoryExtractor()
        self.llm_extractor = llm_extractor

    def extract(self, role: str, content: str, use_llm: bool = True, context: Optional[Dict[str, Any]] = None) -> List[ExtractedCandidate]:
        results: List[ExtractedCandidate] = []
        seen_statements = set()

        # 1. Heuristic extraction
        heuristics = self.heuristic_extractor.extract_from_message(role, content)
        for h in heuristics:
            stmt_key = h.statement.strip().lower()
            if stmt_key not in seen_statements:
                seen_statements.add(stmt_key)
                if h.structured_claim:
                    h.structured_claim = ClaimNormalizer.normalize_claim(h.structured_claim)
                results.append(h)

        # 2. LLM extraction
        if use_llm and self.llm_extractor:
            try:
                llm_candidates = self.llm_extractor.extract(role, content, context=context)
                for cand in llm_candidates:
                    stmt_key = cand.statement.strip().lower()
                    if stmt_key not in seen_statements:
                        seen_statements.add(stmt_key)
                        if cand.structured_claim:
                            cand.structured_claim = ClaimNormalizer.normalize_claim(cand.structured_claim)
                        # Confidence gating: items below 0.87 threshold need human review
                        if cand.confidence < 0.87 and cand.status == "active":
                            cand.status = "review_required"
                        results.append(cand)
            except Exception:
                # LLM extraction failure must be non-fatal and fall back to heuristics
                pass

        return results
