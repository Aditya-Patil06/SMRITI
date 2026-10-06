import pytest
from smriti.embeddings import MockEmbeddingProvider, LocalEmbeddingProvider
from smriti.vector_store import VectorFilter, InMemoryVectorStore
import numpy as np
import os

def test_mock_embedding_provider():
    provider = MockEmbeddingProvider(dimension=4)
    # Test deterministic behaviour
    v1 = provider.embed("hello world")
    v2 = provider.embed("hello world")
    assert v1 == v2
    assert len(v1) == 4
    
    # Test adversarial cross-workspace collide fixture
    collide1 = provider.embed("mock_cross_workspace_collide: workspace A")
    collide2 = provider.embed("mock_cross_workspace_collide: workspace B")
    assert collide1 == collide2
    assert collide1[0] == 1.0
    
    # Test adversarial vector close fixture
    close1 = provider.embed("mock_vector_close: sentence 1")
    close2 = provider.embed("mock_vector_close: sentence 2")
    # They should be very close but distinct
    assert close1 != close2
    sim = np.dot(close1, close2)
    assert sim > 0.98

def test_in_memory_vector_store_filtering(tmp_path):
    provider = MockEmbeddingProvider(dimension=4)
    storage_file = str(tmp_path / "vectors.json")
    store = InMemoryVectorStore(provider, storage_path=storage_file)
    
    store.upsert("mem1", "apple", meta={"workspace_id": "ws1", "status": "active"}, auto_save=False)
    store.upsert("mem2", "banana", meta={"workspace_id": "ws2", "status": "active"}, auto_save=False)
    store.upsert("mem3", "apple", meta={"workspace_id": "ws1", "status": "superseded"}, auto_save=False)
    store.upsert("mem4", "apple", meta={"workspace_id": "ws1", "status": "forgotten"}, auto_save=False)
    
    # Search without filter
    res = store.search("apple", top_k=10)
    assert len(res) == 4
    
    # Search with workspace filter
    f = VectorFilter(workspace_id="ws1")
    res_ws1 = store.search("apple", top_k=10, filters=f)
    assert len(res_ws1) == 1
    assert res_ws1[0][0] == "mem1"  # superseded and forgotten are excluded by default
    
    # Include superseded: allows superseded but must NEVER allow forgotten
    f_super = VectorFilter(workspace_id="ws1", include_superseded=True)
    res_super = store.search("apple", top_k=10, filters=f_super)
    assert len(res_super) == 2
    assert {r[0] for r in res_super} == {"mem1", "mem3"}
    assert "mem4" not in {r[0] for r in res_super}

def test_local_embedding_provider_defined():
    # LocalEmbeddingProvider must ALWAYS be defined as a class
    assert issubclass(LocalEmbeddingProvider, object)

def test_local_embedding_provider_fallback_import_error(monkeypatch):
    import sys
    import importlib
    import smriti.embeddings
    import smriti.vector_store

    orig_emb = sys.modules.get("smriti.embeddings")
    orig_provider_emb = getattr(smriti.embeddings, "LocalEmbeddingProvider", None)
    orig_provider_vec = getattr(smriti.vector_store, "LocalEmbeddingProvider", None)

    # Simulate absence of sentence_transformers in importlib
    orig_find_spec = importlib.util.find_spec

    def mock_find_spec(name, *args, **kwargs):
        if name == "sentence_transformers":
            return None
        return orig_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    try:
        # Reload smriti.embeddings so the HAS_SENTENCE_TRANSFORMERS=False branch executes
        reloaded_emb = importlib.reload(smriti.embeddings)
        with pytest.raises(ImportError) as exc_info:
            reloaded_emb.LocalEmbeddingProvider()
        assert "sentence-transformers is required" in str(exc_info.value)
    finally:
        # Restore monkeypatch and reload module cleanly to restore original classes
        monkeypatch.undo()
        if orig_emb is not None:
            importlib.reload(orig_emb)
        if orig_provider_emb is not None:
            smriti.embeddings.LocalEmbeddingProvider = orig_provider_emb
        if orig_provider_vec is not None:
            smriti.vector_store.LocalEmbeddingProvider = orig_provider_vec

@pytest.mark.skip(reason="Requires network access and HuggingFace models")
def test_local_embedding_provider():
    # Only test if sentence-transformers is available
    try:
        provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        v = provider.embed("test")
    except Exception:
        pytest.skip("Local embedding provider failing/not installed")
    assert len(v) == 384
