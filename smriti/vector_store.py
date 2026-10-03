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

            if filters:
                if filters.workspace_id and meta.get("workspace_id") != filters.workspace_id:
                    continue
                if filters.project_id and meta.get("project_id") != filters.project_id:
                    continue
                if not filters.include_superseded and meta.get("status") in ["superseded", "forgotten"]:
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
    def __init__(self, embedding_provider, db_session_factory):
        self.embedding_provider = embedding_provider
        self.db_session_factory = db_session_factory
        self.model_name = embedding_provider.model_name
        self.model_version = embedding_provider.model_version
        self.dimension = embedding_provider.dimension

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def upsert(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None, auto_save: bool = True):
        from smriti.models import EmbeddingMetadata
        from sqlalchemy.dialects.postgresql import insert
        db = self.db_session_factory()
        try:
            content_hash = self._hash(text)
            vec = self.embedding_provider.embed(text)

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

            if auto_save:
                db.commit()
        finally:
            db.close()

    def delete(self, doc_id: str, auto_save: bool = True):
        from smriti.models import EmbeddingMetadata
        db = self.db_session_factory()
        try:
            db.query(EmbeddingMetadata).filter_by(memory_id=doc_id).delete()
            if auto_save:
                db.commit()
        finally:
            db.close()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[VectorFilter] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        from smriti.models import EmbeddingMetadata, Memory
        q_vec = self.embedding_provider.embed(query)
        # Vector literal
        vec_str = "[" + ",".join(map(str, q_vec)) + "]"

        db = self.db_session_factory()
        try:
            sql = f"""
            SELECT e.memory_id, 1 - (e.embedding <=> :vec) as similarity,
                   m.workspace_id, m.project_id, m.memory_type, m.status, m.confidence
            FROM embedding_metadata e
            JOIN memories m ON e.memory_id = m.id
            WHERE e.model_name = :m_name
              AND e.model_version = :m_version
              AND e.embedding_dimension = :m_dim
            """
            params = {
                "vec": vec_str,
                "m_name": self.model_name,
                "m_version": self.model_version,
                "m_dim": self.dimension
            }

            if filters:
                if filters.workspace_id:
                    sql += " AND m.workspace_id = :w_id"
                    params["w_id"] = filters.workspace_id
                if filters.project_id:
                    sql += " AND m.project_id = :p_id"
                    params["p_id"] = filters.project_id
                if not filters.include_superseded:
                    sql += " AND m.status NOT IN ('superseded', 'forgotten')"

            sql += " ORDER BY e.embedding <=> :vec LIMIT :limit"
            params["limit"] = top_k

            result = db.execute(text(sql), params).fetchall()

            hits = []
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
        finally:
            db.close()

    def rebuild_from_db(self, db: Session) -> int:
        from smriti.models import Memory
        memories = db.query(Memory).filter(Memory.status != "forgotten").all()
        count = 0
        for mem in memories:
            text_str = f"{mem.statement} {mem.rationale or ''}"
            self.upsert(mem.id, text_str, auto_save=False)
            count += 1
        db.commit()
        return count

provider_name = settings.embedding_provider
if provider_name == "local":
    provider = LocalEmbeddingProvider(model_name=settings.embedding_model)
else:
    provider = MockEmbeddingProvider()

if engine.name == "postgresql":
    vector_store = PgVectorStore(provider, SessionLocal)
else:
    vector_store = InMemoryVectorStore(provider)

# Expose Phase 2 compatibility aliases
EmbeddingService = MockEmbeddingProvider
VectorStore = InMemoryVectorStore
