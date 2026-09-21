import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from smriti.models import Memory, Project, Message
from smriti.schemas import SearchResultItem
from smriti.vector_store import vector_store
from smriti.graph import graph_service

class HybridRetrievalEngine:
    """Combines lexical keyword matching, vector embeddings, and knowledge graph expansion."""

    def search(
        self,
        db: Session,
        query: str,
        project_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 15,
        include_superseded: bool = False
    ) -> List[SearchResultItem]:
        query_clean = query.strip()
        if not query_clean:
            return []

        results_map: Dict[str, SearchResultItem] = {}

        # 1. Lexical / Keyword Search in Database
        mem_q = db.query(Memory)
        if not include_superseded:
            mem_q = mem_q.filter(Memory.status.notin_(["superseded", "forgotten"]))
        else:
            mem_q = mem_q.filter(Memory.status != "forgotten")

        if project_id:
            mem_q = mem_q.filter(Memory.project_id == project_id)
        if workspace_id:
            mem_q = mem_q.filter(Memory.workspace_id == workspace_id)

        tokens = re.findall(r"\w+", query_clean.lower())
        if tokens:
            filters = [
                or_(
                    Memory.statement.ilike(f"%{t}%"),
                    Memory.rationale.ilike(f"%{t}%"),
                    Memory.memory_type.ilike(f"%{t}%")
                )
                for t in tokens if len(t) > 2
            ]
            if filters:
                lexical_mems = mem_q.filter(or_(*filters)).limit(limit * 2).all()
                for m in lexical_mems:
                    score = 0.5 + (0.1 * m.confidence)
                    results_map[f"memory:{m.id}"] = SearchResultItem(
                        id=m.id,
                        type="memory",
                        title=f"[{m.memory_type.upper()}] {m.statement[:60]}",
                        content=m.statement + (f" (Rationale: {m.rationale})" if m.rationale else ""),
                        score=score,
                        metadata={
                            "memory_type": m.memory_type,
                            "status": m.status,
                            "confidence": m.confidence,
                            "project_id": m.project_id
                        }
                    )

        # 2. Vector Semantic Search
        vec_hits = vector_store.search(
            query=query_clean,
            top_k=limit * 2,
            filter_fn=lambda m: (not project_id or m.get("project_id") == project_id) and
                                (include_superseded or m.get("status") not in ["superseded", "forgotten"])
        )

        for doc_id, sim_score, meta in vec_hits:
            key = f"memory:{doc_id}"
            scaled_sim = float(sim_score) * 0.8
            if key in results_map:
                results_map[key].score += scaled_sim
            else:
                mem = db.query(Memory).filter(Memory.id == doc_id).first()
                if mem and (include_superseded or mem.status not in ["superseded", "forgotten"]):
                    results_map[key] = SearchResultItem(
                        id=mem.id,
                        type="memory",
                        title=f"[{mem.memory_type.upper()}] {mem.statement[:60]}",
                        content=mem.statement + (f" (Rationale: {mem.rationale})" if mem.rationale else ""),
                        score=scaled_sim,
                        metadata={
                            "memory_type": mem.memory_type,
                            "status": mem.status,
                            "confidence": mem.confidence,
                            "project_id": mem.project_id
                        }
                    )

        # 3. Knowledge Graph Expansion
        # If any top memory is strongly hit, find its 1-hop neighbours
        if results_map:
            top_ids = sorted(results_map.keys(), key=lambda k: results_map[k].score, reverse=True)[:3]
            for top_key in top_ids:
                connected_ids = graph_service.find_connected_memories(top_key, max_hops=1)
                for conn_id in connected_ids:
                    c_key = f"memory:{conn_id}"
                    if c_key in results_map:
                        results_map[c_key].score += 0.15
                    else:
                        conn_mem = db.query(Memory).filter(Memory.id == conn_id).first()
                        if conn_mem and (include_superseded or conn_mem.status not in ["superseded", "forgotten"]):
                            results_map[c_key] = SearchResultItem(
                                id=conn_mem.id,
                                type="memory",
                                title=f"[{conn_mem.memory_type.upper()}] {conn_mem.statement[:60]}",
                                content=conn_mem.statement + (f" (Rationale: {conn_mem.rationale})" if conn_mem.rationale else ""),
                                score=0.35,  # graph expansion prior
                                metadata={
                                    "memory_type": conn_mem.memory_type,
                                    "status": conn_mem.status,
                                    "confidence": conn_mem.confidence,
                                    "project_id": conn_mem.project_id
                                }
                            )

        # Sort combined results
        final_list = sorted(results_map.values(), key=lambda x: x.score, reverse=True)
        return final_list[:limit]

retrieval_engine = HybridRetrievalEngine()
