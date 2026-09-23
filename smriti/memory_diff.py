import re
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

    def _match_structured_supersession(
        self,
        candidate: ExtractedCandidate,
        cand_claim: Optional[Dict[str, Any]],
        cand_replaces_norm: Optional[str],
        has_replacement_kw: bool,
        old: Memory,
        old_claim: Dict[str, Any]
    ) -> Optional[DiffResult]:
        """
        Evaluates whether candidate supersedes an existing memory with a structured claim.
        Strictly compares subject, predicate, object, and scope.
        Statement substring searching against old.statement is NOT permitted.
        """
        old_obj = old_claim.get("object")
        old_subj = old_claim.get("subject")
        old_pred = old_claim.get("predicate")

        # 1. Explicit replacement field
        if cand_replaces_norm and cand_replaces_norm == old_obj:
            if not cand_claim or (
                cand_claim.get("subject") == old_subj and cand_claim.get("predicate") == old_pred
            ):
                return DiffResult(
                    change_type="SUPERSEDED",
                    candidate_statement=candidate.statement,
                    existing_memory_id=old.id,
                    existing_statement=old.statement,
                    reason=f"Candidate explicitly supersedes object '{old_obj}'",
                    confidence=0.95,
                    review_required=False,
                    evidence={"replaces": cand_replaces_norm, "matched_field": "explicit_replaces"}
                )

        # 2. Replacement keyword in candidate statement referencing old object
        if has_replacement_kw and old_obj:
            cand_stmt_lower = candidate.statement.lower()
            cand_words = set(re.findall(r"\b\w+\b", cand_stmt_lower))
            if old_obj in cand_words or old_obj in cand_stmt_lower:
                if not cand_claim or (
                    cand_claim.get("subject") == old_subj and cand_claim.get("predicate") == old_pred
                ):
                    return DiffResult(
                        change_type="SUPERSEDED",
                        candidate_statement=candidate.statement,
                        existing_memory_id=old.id,
                        existing_statement=old.statement,
                        reason=f"Claim replacement indicated for subject '{old_subj}'",
                        confidence=0.94,
                        review_required=False,
                        evidence={"superseded_object": old_obj}
                    )

        return None

    def _match_legacy_supersession(
        self,
        candidate: ExtractedCandidate,
        cand_replaces: Optional[str],
        cand_stmt_lower: str,
        has_replacement_kw: bool,
        old: Memory
    ) -> Optional[DiffResult]:
        """
        Fallback for legacy memories lacking structured claim.
        Requires compatible memory type and explicit replacement reference.
        """
        if old.memory_type != candidate.memory_type:
            return None

        old_stmt_lower = old.statement.strip().lower()

        if cand_replaces and (cand_replaces in old_stmt_lower):
            return DiffResult(
                change_type="SUPERSEDED",
                candidate_statement=candidate.statement,
                existing_memory_id=old.id,
                existing_statement=old.statement,
                reason=f"Candidate explicitly supersedes legacy memory target '{cand_replaces}'",
                confidence=0.93,
                review_required=False,
                evidence={"replaces": cand_replaces, "matched_field": "legacy_explicit_replaces"}
            )

        if has_replacement_kw and (old_stmt_lower in cand_stmt_lower):
            return DiffResult(
                change_type="SUPERSEDED",
                candidate_statement=candidate.statement,
                existing_memory_id=old.id,
                existing_statement=old.statement,
                reason="Replacement keyword detected referencing legacy statement",
                confidence=0.91,
                review_required=False,
                evidence={"indicator": "legacy_replacement_keyword"}
            )

        return None

    def _evaluate_competing_claim(
        self,
        candidate: ExtractedCandidate,
        cand_claim: Dict[str, Any],
        old: Memory,
        old_claim: Dict[str, Any]
    ) -> Optional[DiffResult]:
        """
        Evaluates relationship between candidate and existing memory when both have claims
        with same subject and predicate but different object.
        """
        if (
            cand_claim.get("subject") != old_claim.get("subject")
            or cand_claim.get("predicate") != old_claim.get("predicate")
            or cand_claim.get("object") == old_claim.get("object")
        ):
            return None

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

        scopes_distinct = False
        for k in set(cand_scope.keys()).union(set(old_scope.keys())):
            v1 = cand_scope.get(k)
            v2 = old_scope.get(k)
            if v1 and v2 and v1 != v2:
                scopes_distinct = True
                break

        if scopes_distinct:
            is_high_conf = candidate.confidence >= 0.90
            return DiffResult(
                change_type="COEXISTING",
                candidate_statement=candidate.statement,
                existing_memory_id=old.id,
                existing_statement=old.statement,
                reason=(
                    f"Claims share subject and predicate but serve distinct scopes ({cand_scope} vs {old_scope})"
                    if is_high_conf else
                    f"Coexisting claims detected with low scope confidence ({candidate.confidence:.2f})"
                ),
                confidence=candidate.confidence,
                review_required=not is_high_conf,
                evidence={"scope_a": old_scope, "scope_b": cand_scope}
            )

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

        # 2. Check for SUPERSEDED
        raw_replaces = (candidate.details or {}).get("replaces")
        cand_replaces_norm = ClaimNormalizer.normalize_token(raw_replaces) if raw_replaces else None
        has_replacement_kw = any(kw in cand_stmt_lower for kw in self.REPLACEMENT_INDICATORS)

        for old in scoped_memories:
            if old.status not in ["active", "review_required"]:
                continue

            old_claim = ClaimNormalizer.normalize_claim(old.structured_claim) if old.structured_claim else None

            if old_claim:
                super_res = self._match_structured_supersession(
                    candidate=candidate,
                    cand_claim=cand_claim,
                    cand_replaces_norm=cand_replaces_norm,
                    has_replacement_kw=has_replacement_kw,
                    old=old,
                    old_claim=old_claim
                )
                if super_res:
                    return super_res
            else:
                legacy_res = self._match_legacy_supersession(
                    candidate=candidate,
                    cand_replaces=raw_replaces.lower() if raw_replaces else None,
                    cand_stmt_lower=cand_stmt_lower,
                    has_replacement_kw=has_replacement_kw,
                    old=old
                )
                if legacy_res:
                    return legacy_res

        # 3. Check for SAME SUBJECT + SAME PREDICATE + DIFFERENT OBJECT
        if cand_claim:
            for old in scoped_memories:
                if old.status not in ["active", "review_required"]:
                    continue
                old_claim = ClaimNormalizer.normalize_claim(old.structured_claim) if old.structured_claim else None
                if not old_claim:
                    continue

                comp_res = self._evaluate_competing_claim(candidate, cand_claim, old, old_claim)
                if comp_res:
                    return comp_res

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
