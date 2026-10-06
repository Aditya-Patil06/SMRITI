import abc
from typing import List
import hashlib
import numpy as np

class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def model_version(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        pass

class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock for Layer A testing, including adversarial fixtures."""
    
    def __init__(self, dimension: int = 128):
        self._dimension = dimension
        
    def embed(self, text: str) -> List[float]:
        # Adversarial Fixture 1: Cross-workspace near-identical embeddings
        if text.startswith("mock_cross_workspace_collide:"):
            vec = np.zeros(self._dimension, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()
            
        # Adversarial Fixture 2: Semantically distinct but vector close
        if text.startswith("mock_vector_close:"):
            vec = np.zeros(self._dimension, dtype=np.float32)
            vec[0] = 0.99
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % self._dimension
            vec[h] = 0.01
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
            
        # Standard deterministic behavior
        vec = np.zeros(self._dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()
            
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % self._dimension
            vec[h] += 1.0
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @property
    def model_name(self) -> str:
        return "mock-deterministic"

    @property
    def model_version(self) -> str:
        return "1.0"

    @property
    def dimension(self) -> int:
        return self._dimension

try:
    import importlib.util
    HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

if HAS_SENTENCE_TRANSFORMERS:
    class LocalEmbeddingProvider(EmbeddingProvider):
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self._model_name = model_name
            self._model = None
            self._dimension = None

        def _get_model(self):
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            return self._model

        def embed(self, text: str) -> List[float]:
            model = self._get_model()
            # Convert to list of floats
            return [float(x) for x in model.encode(text)]

        @property
        def model_name(self) -> str:
            return self._model_name

        @property
        def model_version(self) -> str:
            return "1.0"

        @property
        def dimension(self) -> int:
            if self._dimension is None:
                self._get_model()
            return self._dimension
else:
    class LocalEmbeddingProvider(EmbeddingProvider):  # type: ignore[no-redef]
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddingProvider. "
                "Install it using `pip install sentence-transformers`."
            )

        def embed(self, text: str) -> List[float]:
            raise NotImplementedError

        @property
        def model_name(self) -> str:
            raise NotImplementedError

        @property
        def model_version(self) -> str:
            raise NotImplementedError

        @property
        def dimension(self) -> int:
            raise NotImplementedError


