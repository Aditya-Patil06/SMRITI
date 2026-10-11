import math
import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import Counter
import re
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from smriti.config import settings
from smriti.models import engine, SessionLocal
from smriti.embeddings import MockEmbeddingProvider, LocalEmbeddingProvider

class VectorFilter(BaseModel):
    workspace_id: Optional[str] = None
    project_id: Optional[str] = None
    include_superseded: bool = False
    model_name: Optional[str] = None
    model_version: Optional[str] = None

class BaseVectorStore:
    def upsert(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None, auto_save: bool = True):
        raise NotImplementedError

    def delete(self, doc_id: str, auto_save: bool = True):
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[VectorFilter] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    def rebuild_from_db(self, db: Session) -> int:
        raise NotImplementedError

class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, embedding_provider, storage_path: Optional[str] = None):
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path or os.path.join(settings.storage_dir, "vectors.json")
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()

    def _ensure_dir(self):
        dirname = os.path.dirname(self.storage_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

    def save(self):
        self._ensure_dir()
        serialized = {
            doc_id: {
                "vector": self.vectors[doc_id].tolist(),
                "metadata": self.metadata.get(doc_id, {})
            }
            for doc_id in self.vectors
        }
        dirname = os.path.dirname(self.storage_path) or "."
        temp_path = os.path.join(dirname, f".{os.path.basename(self.storage_path)}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.storage_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.vectors = {}
                    self.metadata = {}
                    for doc_id, item in data.items():
                        vec = np.array(item["vector"], dtype=np.float32)
                        if vec.shape[0] == self.embedding_provider.dimension:
                            self.vectors[doc_id] = vec
                            self.metadata[doc_id] = item.get("metadata", {})
            except Exception:
                self.vectors = {}
                self.metadata = {}
        else:
            self.vectors = {}
            self.metadata = {}

    def upsert(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None, auto_save: bool = True):
        vec = np.array(self.embedding_provider.embed(text), dtype=np.float32)
        self.vectors[doc_id] = vec
        self.metadata[doc_id] = meta or {}
        if auto_save:
            self.save()

    def delete(self, doc_id: str, auto_save: bool = True):
        self.vectors.pop(doc_id, None)
        self.metadata.pop(doc_id, None)
        if auto_save:
            self.save()

    def _matches_filters(self, meta: Dict[str, Any], filters: Optional[VectorFilter]) -> bool:
        """Check if metadata satisfies the given VectorFilter."""
        if not filters:
            return True
        if filters.workspace_id and meta.get("workspace_id") != filters.workspace_id:
            return False
        if filters.project_id and meta.get("project_id") != filters.project_id:
            return False
        status = meta.get("status")
        if status == "forgotten" or (not filters.include_superseded and status == "superseded"):
            return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[VectorFilter] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.vectors:
            return []

        q_vec = np.array(self.embedding_provider.embed(query), dtype=np.float32)
        results = []

        for doc_id, doc_vec in self.vectors.items():
            meta = self.metadata.get(doc_id, {})

            if filter_fn and not filter_fn(meta):
                continue

            if not self._matches_filters(meta, filters):
                continue

            score = float(np.dot(q_vec, doc_vec))
            results.append((doc_id, score, meta))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self, auto_save: bool = True):
        self.vectors.clear()
        self.metadata.clear()
        if auto_save:
            self.save()

    def rebuild_from_db(self, db: Session) -> int:
        from smriti.models import Memory
        eligible_memories = db.query(Memory).filter(Memory.status != "forgotten").all()
        staged_vectors: Dict[str, np.ndarray] = {}
        staged_metadata: Dict[str, Dict[str, Any]] = {}

        for mem in eligible_memories:
            meta = {
                "workspace_id": mem.workspace_id,
                "project_id": mem.project_id,
                "memory_type": mem.memory_type,
                "status": mem.status,
                "confidence": mem.confidence
            }
            text_to_embed = f"{mem.statement} {mem.rationale or ''}"
            vec = np.array(self.embedding_provider.embed(text_to_embed), dtype=np.float32)
            staged_vectors[mem.id] = vec
            staged_metadata[mem.id] = meta

        self.vectors = staged_vectors
        self.metadata = staged_metadata
        self.save()
        return len(eligible_memories)

class PgVectorStore(BaseVectorStore):
    def __init__(self, embedding_provider, db_session_factory, storage_path: Optional[str] = None):
        self.embedding_provider = embedding_provider
        self.db_session_factory = db_session_factory
        self.model_name = embedding_provider.model_name
        self.model_version = embedding_provider.model_version
        self.dimension = embedding_provider.dimension
        self.storage_path = storage_path or os.path.join(settings.storage_dir, "vectors.json")
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def upsert(
        self,
        doc_id: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        auto_save: bool = True,
        session: Optional[Session] = None
    ):
        from smriti.models import EmbeddingMetadata
        from sqlalchemy.dialects.postgresql import insert

        # Maintain in-memory compatibility mirrors
        vec = self.embedding_provider.embed(text)
        self.vectors[doc_id] = np.array(vec, dtype=np.float32)
        self.metadata[doc_id] = meta or {}

        db = session if session is not None else self.db_session_factory()
        should_close = session is None
        try:
            # Row-level lock on parent Memory for doc_id to synchronize concurrent upserts
            from smriti.models import Memory
            db.query(Memory).filter_by(id=doc_id).with_for_update().first()

            content_hash = self._hash(text)

            stmt = insert(EmbeddingMetadata).values(
                memory_id=doc_id,
                model_name=self.model_name,
                model_version=self.model_version,
                embedding_dimension=self.dimension,
                content_hash=content_hash,
                embedding=vec
            ).on_conflict_do_nothing(
                index_elements=['memory_id', 'model_name', 'model_version', 'content_hash']
            )

            db.execute(stmt)

            # Clean up stale hashes atomically (if there's a new hash)
            db.query(EmbeddingMetadata).filter(
                EmbeddingMetadata.memory_id == doc_id,
                EmbeddingMetadata.model_name == self.model_name,
                EmbeddingMetadata.model_version == self.model_version,
                EmbeddingMetadata.content_hash != content_hash
            ).delete(synchronize_session=False)

            if auto_save or should_close:
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    def delete(self, doc_id: str, auto_save: bool = True, session: Optional[Session] = None):
        from smriti.models import EmbeddingMetadata

        self.vectors.pop(doc_id, None)
        self.metadata.pop(doc_id, None)

        db = session if session is not None else self.db_session_factory()
        should_close = session is None
        try:
            db.query(EmbeddingMetadata).filter_by(memory_id=doc_id).delete()
            if auto_save or should_close:
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    def save(self):
        """No-op on PostgreSQL as persistence is transactional in the database."""
        pass

    def clear(self, auto_save: bool = True):
        self.vectors.clear()
        self.metadata.clear()
        if auto_save:
            self.save()

    def _build_search_filters(self, filters: Optional[VectorFilter], params: Dict[str, Any]) -> str:
        """Construct WHERE filter clauses and bind parameters for vector search."""
        if not filters:
            return ""

        clause = ""
        if filters.workspace_id:
            clause += " AND m.workspace_id = :w_id"
            params["w_id"] = filters.workspace_id
        if filters.project_id:
            clause += " AND m.project_id = :p_id"
            params["p_id"] = filters.project_id
        if not filters.include_superseded:
            clause += " AND m.status NOT IN ('superseded', 'forgotten')"
        else:
            clause += " AND m.status != 'forgotten'"
        return clause

    def _format_hits(
        self,
        result: List[Any],
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Convert raw database rows to hit tuples, applying post-filtering if provided."""
        hits: List[Tuple[str, float, Dict[str, Any]]] = []
        for row in result:
            meta = {
                "workspace_id": row.workspace_id,
                "project_id": row.project_id,
                "memory_type": row.memory_type,
                "status": row.status,
                "confidence": row.confidence
            }
            if filter_fn and not filter_fn(meta):
                continue
            hits.append((row.memory_id, row.similarity, meta))
        return hits

    def _paginate_filtered_hits(
        self,
        db: Session,
        base_sql: str,
        base_params: Dict[str, Any],
        top_k: int,
        filter_fn: Callable[[Dict[str, Any]], bool]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Execute bounded SQL queries across pages until top_k filtered hits are found or candidates are exhausted.

        Pagination Strategy and Trade-offs:
        - Bounded Queries: Uses SQL LIMIT :limit OFFSET :offset (page_size = max(top_k * 2, 50))
          to bound the number of returned rows per query and application memory usage.
          Note that while LIMIT bounds the returned rows per query, OFFSET may still require
          PostgreSQL to scan earlier rows up to the offset boundary, meaning total database execution
          cost increases with the offset.
        - Deterministic Ordering: base_sql orders by `vector <=> :vec, e.memory_id`, guaranteeing
          a strict total order even when vector similarities tie.
        - Concurrent Mutation Trade-off: Under concurrent inserts/deletes, offset pagination can
          encounter row shifting (phantom reads or skipped rows across pages). Alternative pagination
          strategies (such as distance/keyset cursors) may affect PostgreSQL vector-index query planning
          (e.g., whether HNSW or IVFFlat index scans are selected) and must be carefully validated against
          the actual EXPLAIN query plan across supported pgvector versions. Given typical read-dominated
          memory retrieval workflows, bounded offset pagination with deterministic tie-breaking provides
          a pragmatic balance of predictable query planning, bounded returned row counts, and retrieval correctness.
        """
        page_size = max(top_k * 2, 50)
        hits: List[Tuple[str, float, Dict[str, Any]]] = []
        offset = 0

        paged_sql = f"{base_sql} LIMIT :limit OFFSET :offset"

        while len(hits) < top_k:
            page_params = dict(base_params)
            page_params["limit"] = page_size
            page_params["offset"] = offset

            cursor = db.execute(text(paged_sql), page_params)
            try:
                page_rows = cursor.fetchall()
            finally:
                cursor.close()

            if not page_rows:
                break

            for row in page_rows:
                meta = {
                    "workspace_id": row.workspace_id,
                    "project_id": row.project_id,
                    "memory_type": row.memory_type,
                    "status": row.status,
                    "confidence": row.confidence
                }
                if filter_fn(meta):
                    hits.append((row.memory_id, row.similarity, meta))
                    if len(hits) == top_k:
                        break

            if len(page_rows) < page_size:
                break

            offset += page_size

        return hits

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[VectorFilter] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if top_k <= 0:
            return []

        q_vec = self.embedding_provider.embed(query)
        # Vector literal
        vec_str = "[" + ",".join(map(str, q_vec)) + "]"

        db = self.db_session_factory()
        try:
            params: Dict[str, Any] = {
                "vec": vec_str,
                "m_name": self.model_name,
                "m_version": self.model_version,
                "m_dim": self.dimension
            }

            filter_clauses = self._build_search_filters(filters, params)
            if filters:
                # Enable transaction-local iterative scan for HNSW when filters are present
                try:
                    db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
                except Exception:
                    # Guard for pgvector versions or dialects where iterative_scan is not supported
                    pass

            base_sql = f"""
            SELECT e.memory_id, 1 - (e.embedding::vector({self.dimension}) <=> :vec) as similarity,
                   m.workspace_id, m.project_id, m.memory_type, m.status, m.confidence
            FROM embedding_metadata e
            JOIN memories m ON e.memory_id = m.id
            WHERE e.model_name = :m_name
              AND e.model_version = :m_version
              AND e.embedding_dimension = :m_dim{filter_clauses}
            ORDER BY e.embedding::vector({self.dimension}) <=> :vec, e.memory_id
            """

            if filter_fn is None:
                params["limit"] = top_k
                sql = f"{base_sql} LIMIT :limit"
                cursor = db.execute(text(sql), params)
                try:
                    result = cursor.fetchall()
                    return self._format_hits(result, None)
                finally:
                    cursor.close()

            return self._paginate_filtered_hits(db, base_sql, params, top_k, filter_fn)
        finally:
            db.close()

    def rebuild_from_db(self, db: Session) -> int:
        from smriti.models import Memory
        memories = db.query(Memory).filter(Memory.status != "forgotten").all()
        count = 0
        for mem in memories:
            text_str = f"{mem.statement} {mem.rationale or ''}"
            self.upsert(mem.id, text_str, auto_save=False, session=db)
            count += 1
        db.commit()
        return count

_provider_instance = None
_vector_store_instance = None

def get_embedding_provider():
    global _provider_instance
    if _provider_instance is None:
        provider_name = settings.embedding_provider
        if provider_name == "local":
            _provider_instance = LocalEmbeddingProvider(model_name=settings.embedding_model)
        else:
            _provider_instance = MockEmbeddingProvider()
    return _provider_instance

def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is None:
        p = get_embedding_provider()
        if engine.name == "postgresql":
            _vector_store_instance = PgVectorStore(p, SessionLocal)
        else:
            _vector_store_instance = InMemoryVectorStore(p)
    return _vector_store_instance

class _ProxyVectorStore(BaseVectorStore):
    """Transparent proxy delegating to lazily initialized vector_store singleton."""
    def __getattr__(self, name):
        return getattr(get_vector_store(), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(get_vector_store(), name, value)

    def upsert(self, *args, **kwargs):
        return get_vector_store().upsert(*args, **kwargs)

    def delete(self, *args, **kwargs):
        return get_vector_store().delete(*args, **kwargs)

    def search(self, *args, **kwargs):
        return get_vector_store().search(*args, **kwargs)

    def rebuild_from_db(self, *args, **kwargs):
        return get_vector_store().rebuild_from_db(*args, **kwargs)

vector_store = _ProxyVectorStore()

# Expose Phase 2 compatibility aliases
EmbeddingService = MockEmbeddingProvider
VectorStore = InMemoryVectorStore

