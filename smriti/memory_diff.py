from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from smriti.models import Memory
from smriti.extraction import ClaimNormalizer, ExtractedCandidate

class DiffResult(BaseModel):
    change_type: str  # "NEW", "UNCHANGED", "SUPERSEDED", "CONFLICTING", "COEXISTING"
    candidate_statement: str
    existing_memory_id: Optional[str] = None
    existing_statement: Optional[str] = None
    reason: str
    confidence: float
    review_required: bool = False
    evidence: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)

class MemoryDiffEngine:
    """
    Fine-grained 5-way memory diff engine:
    1. NEW: No overlapping or matching claims found.
    2. UNCHANGED (DUPLICATE): Identical or semantically matching statement / normalized claim.
    3. SUPERSEDED: Explicit replacement language ("instead of", "switched to", "migrated from").
    4. COEXISTING: Same subject/predicate, different object, but distinct non-overlapping scopes (e.g. dev vs prod).
       - High confidence (>= 0.90) and explicit distinct scope -> auto-accepted (review_required=False).
       - Ambiguous or weak scope -> review_required=True.
    5. CONFLICTING: Incompatible claims with identical/overlapping scope and no supersession evidence.
    """

    REPLACEMENT_INDICATORS = [
        "instead of", "switched to", "migrated to", "migrated from",
        "replaced by", "replaces", "deprecated in favor of", "supersedes", "superseded"
    ]

    MULTI_VALUED_PREDICATES = {
        "uses",
        "uses_technology",
        "relates_to",
        "constrained_by",
        "uses_cache",
        "requires_compliance"
    }

    def classify_diff(
        self,
        candidate: ExtractedCandidate,
        existing_memories: List[Memory],
        workspace_id: str,
        project_id: Optional[str] = None
    ) -> DiffResult:
        cand_claim = ClaimNormalizer.normalize_claim(candidate.structured_claim) if candidate.structured_claim else None
        cand_stmt_lower = candidate.statement.strip().lower()

        # Pre-filter existing candidates: must match workspace and project
        scoped_memories = [
            m for m in existing_memories
            if m.workspace_id == workspace_id and m.project_id == project_id and m.status != "forgotten"
        ]

        # 1. Check for UNCHANGED / DUPLICATE
        for old in scoped_memories:
            old_stmt_lower = old.statement.strip().lower()
            if cand_stmt_lower == old_stmt_lower:
                return DiffResult(
                    change_type="UNCHANGED",
                    candidate_statement=candidate.statement,
                    existing_memory_id=old.id,
                    existing_statement=old.statement,
                    reason="Exact statement match already exists",
                    confidence=1.0,
                    review_required=False
                )

            # Check normalized claim duplicate
            if cand_claim and old.structured_claim:
                old_claim = ClaimNormalizer.normalize_claim(old.structured_claim)
                if (
                    cand_claim.get("subject") == old_claim.get("subject")
                    and cand_claim.get("predicate") == old_claim.get("predicate")
                    and cand_claim.get("object") == old_claim.get("object")
                    and cand_claim.get("scope") == old_claim.get("scope")
                ):
                    return DiffResult(
                        change_type="UNCHANGED",
                        candidate_statement=candidate.statement,
                        existing_memory_id=old.id,
                        existing_statement=old.statement,
                        reason="Normalized structured claim identical to existing memory",
                        confidence=0.98,
                        review_required=False
                    )

        # 2. Check for SUPERSEDED (Explicit replacement language or details["replaces"])
        cand_replaces = (candidate.details or {}).get("replaces")
        has_replacement_kw = any(kw in cand_stmt_lower for kw in self.REPLACEMENT_INDICATORS)

        for old in scoped_memories:
            if old.status not in ["active", "review_required"]:
                continue

            old_claim = ClaimNormalizer.normalize_claim(old.structured_claim) if old.structured_claim else None

            # If explicit replaces target matches
            if cand_replaces:
                if (old_claim and cand_replaces == old_claim.get("object")) or (cand_replaces in old.statement.lower()):
                    return DiffResult(
                        change_type="SUPERSEDED",
                        candidate_statement=candidate.statement,
                        existing_memory_id=old.id,
                        existing_statement=old.statement,
                        reason=f"Candidate explicitly supersedes {cand_replaces}",
                        confidence=0.95,
                        review_required=False,
                        evidence={"replaces": cand_replaces, "matched_field": "explicit_replaces"}
                    )

            if has_replacement_kw:
                # If old object is in candidate statement and subject/predicate align
                if old_claim and old_claim.get("object") and old_claim.get("object") in cand_stmt_lower:
                    if not cand_claim or (cand_claim.get("subject") == old_claim.get("subject") and cand_claim.get("predicate") == old_claim.get("predicate")):
                        return DiffResult(
                            change_type="SUPERSEDED",
                            candidate_statement=candidate.statement,
                            existing_memory_id=old.id,
                            existing_statement=old.statement,
                            reason=f"Claim replacement indicated for subject '{old_claim.get('subject')}'",
                            confidence=0.94,
                            review_required=False,
                            evidence={"superseded_object": old_claim.get("object")}
                        )
                # If old statement content is mentioned as replaced
                elif old.statement.lower() in cand_stmt_lower:
                    return DiffResult(
                        change_type="SUPERSEDED",
                        candidate_statement=candidate.statement,
                        existing_memory_id=old.id,
                        existing_statement=old.statement,
                        reason="Replacement keyword detected referencing previous statement",
                        confidence=0.93,
                        review_required=False,
                        evidence={"indicator": "replacement_keyword"}
                    )

        # 3. Check for SAME SUBJECT + SAME PREDICATE + DIFFERENT OBJECT
        if cand_claim:
            for old in scoped_memories:
                if old.status not in ["active", "review_required"]:
                    continue
                old_claim = ClaimNormalizer.normalize_claim(old.structured_claim) if old.structured_claim else None
                if not old_claim:
                    continue

                if (
                    cand_claim.get("subject") == old_claim.get("subject")
                    and cand_claim.get("predicate") == old_claim.get("predicate")
                    and cand_claim.get("object") != old_claim.get("object")
                ):
                    # Check if the predicate naturally supports multiple coexisting values
                    predicate = cand_claim.get("predicate")
                    if predicate in self.MULTI_VALUED_PREDICATES:
                        return DiffResult(
                            change_type="COEXISTING",
                            candidate_statement=candidate.statement,
                            existing_memory_id=old.id,
                            existing_statement=old.statement,
                            reason=f"Predicate '{predicate}' naturally supports multiple coexisting values",
                            confidence=candidate.confidence,
                            review_required=False,
                            evidence={"multi_valued_predicate": predicate}
                        )
                    cand_scope = cand_claim.get("scope") or {}
                    old_scope = old_claim.get("scope") or {}

                    # Check scope difference
                    scopes_distinct = False
                    # Look for distinct keys or differing non-empty values
                    for k in set(cand_scope.keys()).union(set(old_scope.keys())):
                        v1 = cand_scope.get(k)
                        v2 = old_scope.get(k)
                        if v1 and v2 and v1 != v2:
                            scopes_distinct = True
                            break

                    if scopes_distinct:
                        # Clear coexistence vs Ambiguous coexistence based on confidence
                        if candidate.confidence >= 0.90:
                            return DiffResult(
                                change_type="COEXISTING",
                                candidate_statement=candidate.statement,
                                existing_memory_id=old.id,
                                existing_statement=old.statement,
                                reason=f"Claims share subject and predicate but serve distinct scopes ({cand_scope} vs {old_scope})",
                                confidence=candidate.confidence,
                                review_required=False,
                                evidence={"scope_a": old_scope, "scope_b": cand_scope}
                            )
                        else:
                            return DiffResult(
                                change_type="COEXISTING",
                                candidate_statement=candidate.statement,
                                existing_memory_id=old.id,
                                existing_statement=old.statement,
                                reason=f"Coexisting claims detected with low scope confidence ({candidate.confidence:.2f})",
                                confidence=candidate.confidence,
                                review_required=True,
                                evidence={"scope_a": old_scope, "scope_b": cand_scope}
                            )

                    # No distinct scope and different object -> Genuine conflict
                    return DiffResult(
                        change_type="CONFLICTING",
                        candidate_statement=candidate.statement,
                        existing_memory_id=old.id,
                        existing_statement=old.statement,
                        reason=f"Direct conflict on {cand_claim.get('subject')}.{cand_claim.get('predicate')}: '{old_claim.get('object')}' vs '{cand_claim.get('object')}'",
                        confidence=0.92,
                        review_required=True,
                        evidence={"claim_a": old_claim, "claim_b": cand_claim}
                    )

        # 4. Default: NEW memory
        return DiffResult(
            change_type="NEW",
            candidate_statement=candidate.statement,
            reason="No existing matching or conflicting claims found",
            confidence=candidate.confidence,
            review_required=False
        )

    def diff_batch(
        self,
        db: Session,
        candidates: List[ExtractedCandidate],
        workspace_id: str,
        project_id: Optional[str] = None
    ) -> List[DiffResult]:
        existing_memories = db.query(Memory).filter(
            Memory.workspace_id == workspace_id
        ).all()
        return [
            self.classify_diff(cand, existing_memories, workspace_id, project_id)
            for cand in candidates
        ]

memory_diff_engine = MemoryDiffEngine()
