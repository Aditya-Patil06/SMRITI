import math
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

class EmbeddingService:
    """Local vector embedding service using term-frequency and character n-gram hashing."""

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

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec.tolist()

        counts = Counter(tokens)
        for token, count in counts.items():
            # Stable deterministic hash to bucket
            idx = hash(token) % self.dimension
            vec[idx] += count

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

class VectorStore:
    """In-memory replaceable vector store for cosine similarity search."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def upsert(self, doc_id: str, text: str, meta: Dict[str, Any] = None):
        vec = np.array(self.embedding_service.embed(text), dtype=np.float32)
        self.vectors[doc_id] = vec
        self.metadata[doc_id] = meta or {}

    def delete(self, doc_id: str):
        self.vectors.pop(doc_id, None)
        self.metadata.pop(doc_id, None)

    def search(self, query: str, top_k: int = 10, filter_fn=None) -> List[Tuple[str, float, Dict[str, Any]]]:
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

# Global instance
embedding_service = EmbeddingService(dimension=128)
vector_store = VectorStore(embedding_service)
