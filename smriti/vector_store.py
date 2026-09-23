import math
import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import Counter
import re
from sqlalchemy.orm import Session
from smriti.config import settings

class EmbeddingService:
    """Deterministic local vector embedding service using term-frequency and character n-gram hashing."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        tokens = list(words)
        # Add character tri-grams for subword semantic capture
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(word[i:i+3])
        return tokens

    def _stable_hash(self, token: str) -> int:
        """Deterministic integer bucket index across restarts without Python hash randomization."""
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.dimension

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec.tolist()

        counts = Counter(tokens)
        for token, count in counts.items():
            idx = self._stable_hash(token)
            vec[idx] += count

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

class VectorStore:
    """Persistent vector store with on-disk index serialization and full database rebuild capability."""

    def __init__(self, embedding_service: EmbeddingService, storage_path: Optional[str] = None):
        self.embedding_service = embedding_service
        self.storage_path = storage_path or os.path.join(settings.storage_dir, "vectors.json")
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()

    def _ensure_dir(self):
        dirname = os.path.dirname(self.storage_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

    def save(self):
        """Persist vector index and metadata to disk atomically using a temp file."""
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
        """Load vector index and metadata from disk if present."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.vectors = {}
                    self.metadata = {}
                    for doc_id, item in data.items():
                        vec = np.array(item["vector"], dtype=np.float32)
                        if vec.shape[0] == self.embedding_service.dimension:
                            self.vectors[doc_id] = vec
                            self.metadata[doc_id] = item.get("metadata", {})
            except Exception:
                self.vectors = {}
                self.metadata = {}
        else:
            self.vectors = {}
            self.metadata = {}

    def upsert(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None, auto_save: bool = True):
        vec = np.array(self.embedding_service.embed(text), dtype=np.float32)
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
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.vectors:
            return []

        q_vec = np.array(self.embedding_service.embed(query), dtype=np.float32)
        results = []

        for doc_id, doc_vec in self.vectors.items():
            meta = self.metadata.get(doc_id, {})
            if filter_fn and not filter_fn(meta):
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

    def rebuild_from_db(self, db: Session):
        """Rebuild the entire derived vector index from canonical database records safely using staged build."""
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
            vec = np.array(self.embedding_service.embed(text_to_embed), dtype=np.float32)
            staged_vectors[mem.id] = vec
            staged_metadata[mem.id] = meta

        # Only swap once all embeddings and queries succeeded
        self.vectors = staged_vectors
        self.metadata = staged_metadata
        self.save()
        return len(eligible_memories)

# Global instances
embedding_service = EmbeddingService(dimension=128)
vector_store = VectorStore(embedding_service)
