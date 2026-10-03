import pytest
import os
import threading
from sqlalchemy import create_engine, text
from smriti.vector_store import PgVectorStore, VectorFilter
from smriti.embeddings import MockEmbeddingProvider
from smriti.models import Base, EmbeddingMetadata

pytestmark = pytest.mark.skipif(
    not os.environ.get("SMRITI_PG_TESTS"),
    reason="Requires PostgreSQL + pgvector (SMRITI_PG_TESTS=1)"
)

@pytest.fixture(scope="module")
def pg_engine():
    db_url = os.environ.get("SMRITI_DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5432/smriti_test")
    engine = create_engine(db_url)
    
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    
    # Do not drop_all because we want to test the schema from Alembic,
    # but we will clear out the data.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE embedding_metadata CASCADE"))
        conn.execute(text("TRUNCATE TABLE memories CASCADE"))
    
    # Create HNSW partial index for dimension 64 if not exists
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_embedding_hnsw_64 
            ON embedding_metadata 
            USING hnsw ((embedding::vector(64)) vector_cosine_ops) 
            WHERE embedding_dimension = 64
        """))
    
    yield engine
    
    # Cleanup data at the end of the module
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE embedding_metadata CASCADE"))
        conn.execute(text("TRUNCATE TABLE memories CASCADE"))

@pytest.fixture
def pg_session(pg_engine):
    from sqlalchemy.orm import sessionmaker
    from smriti.models import User, Workspace
    Session = sessionmaker(bind=pg_engine)
    db = Session()
    
    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", email="test@test.com", username="test"))
        db.commit()
        
    for w_id in ["w1", "w2", "wplan", "w_all"]:
        if not db.query(Workspace).filter_by(id=w_id).first():
            db.add(Workspace(id=w_id, user_id="u1", name=w_id))
    db.commit()

    yield lambda: db
    db.close()

def test_extension_and_vector_type(pg_engine):
    with pg_engine.connect() as conn:
        result = conn.execute(text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")).fetchone()
        assert result is not None
        assert result[0] == 'vector'
        print(f"\\n[INFO] PostgreSQL pgvector version: {result[1]}")

def test_idempotency_constraint(pg_session):
    db = pg_session()
    from smriti.models import Memory, Workspace
    db.add(Memory(id="m1", workspace_id="w1", statement="text", memory_type="fact"))
    db.commit()
    
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)
    store.upsert("m1", "text", meta={})
    store.upsert("m1", "text", meta={}) # Same hash
    store.upsert("m1", "new text", meta={}) # Different hash
    
    db = pg_session()
    records = db.query(EmbeddingMetadata).filter_by(memory_id="m1").all()
    assert len(records) == 1

def test_dimension_isolation(pg_session):
    prov384 = MockEmbeddingProvider(dimension=384)
    prov768 = MockEmbeddingProvider(dimension=768)
    store384 = PgVectorStore(prov384, pg_session)
    store768 = PgVectorStore(prov768, pg_session)
    
    from smriti.models import Memory, Workspace
    db = pg_session()
    memA = Memory(id="mem384", workspace_id="w1", statement="textA", memory_type="fact")
    memB = Memory(id="mem768", workspace_id="w1", statement="textB", memory_type="fact")
    db.add_all([memA, memB])
    db.commit()

    store384.upsert("mem384", "textA", meta={})
    store768.upsert("mem768", "textB", meta={})
    
    res384 = store384.search("textA")
    assert len(res384) == 1
    assert res384[0][0] == "mem384"
    
    res768 = store768.search("textB")
    assert len(res768) == 1
    assert res768[0][0] == "mem768"

def test_query_plan_hnsw_fallback(pg_session):
    prov64 = MockEmbeddingProvider(dimension=64)
    store64 = PgVectorStore(prov64, pg_session)
    q_vec = prov64.embed("query")
    vec_str = "[" + ",".join(map(str, q_vec)) + "]"
    
    db = pg_session()
    
    # We must insert enough rows so postgres even considers an index
    from smriti.models import Memory, Workspace
    for i in range(50):
        db.add(Memory(id=f"plan_m_{i}", workspace_id="wplan", statement=f"text {i}", memory_type="fact"))
    db.commit()
    for i in range(50):
        store64.upsert(f"plan_m_{i}", f"text {i}", meta={})

    # Force index usage by turning off seqscan
    db.execute(text("SET enable_seqscan = off;"))
    
    explain_rows = db.execute(text(f"EXPLAIN SELECT e.memory_id FROM embedding_metadata e WHERE e.embedding_dimension = 64 ORDER BY e.embedding::vector(64) <=> '{vec_str}' LIMIT 10")).fetchall()
    explain = "\n".join(r[0] for r in explain_rows)
    
    # We allow the planner to use an index scan if forced
    assert "Index Scan" in explain, f"Expected Index Scan in plan, got:\\n{explain}"

def test_workspace_isolation(pg_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=pg_engine)
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, Session)
    db = Session()
    from smriti.models import Memory, Workspace, User
    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", username="test"))
    
    db.add(Workspace(id="w_iso", user_id="u1", name="w_iso"))
    db.commit()
    db.add(Memory(id="w_iso_m", workspace_id="w_iso", statement="secret", memory_type="fact"))
    if not db.query(Workspace).filter_by(id="w2").first():
        db.add(Workspace(id="w2", user_id="u1", name="w2"))
    db.add(Memory(id="w2_m", workspace_id="w2", statement="secret", memory_type="fact"))
    db.commit()
    
    store.upsert("w_iso_m", "secret", meta={})
    store.upsert("w2_m", "secret", meta={})
    
    f = VectorFilter(workspace_id="w_iso")
    res = store.search("secret", filters=f)
    assert len(res) == 1
    assert res[0][0] == "w_iso_m"

def test_concurrency(pg_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=pg_engine)
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, Session)
    
    db = Session()
    from smriti.models import Memory, Workspace, User
    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", username="test"))
    if not db.query(Workspace).filter_by(id="w1").first():
        db.add(Workspace(id="w1", user_id="u1", name="w1"))
    db.commit()
    
    db.add(Memory(id="conc_mem", workspace_id="w1", statement="conc", memory_type="fact"))
    db.commit()

    def worker(text):
        store.upsert("conc_mem", text, meta={})

    threads = []
    for _ in range(10):
        t = threading.Thread(target=worker, args=("concurrent text update",))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
        
    records = db.query(EmbeddingMetadata).filter_by(memory_id="conc_mem").all()
    assert len(records) == 1

def test_hnsw_recall(pg_session):
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)
    db = pg_session()
    from smriti.models import Memory, Workspace
    
    for i in range(100):
        m_id = f"m_{i}"
        db.add(Memory(id=m_id, workspace_id="w_all", statement=f"statement {i}", memory_type="fact"))
    db.commit()
    
    for i in range(100):
        store.upsert(f"m_{i}", f"statement {i}", meta={})
        
    # Check exact recall
    db.execute(text("SET enable_indexscan = off"))
    exact_res = store.search("statement 50", top_k=10, filters=VectorFilter(workspace_id="w_all"))
    
    db.execute(text("SET enable_indexscan = on"))
    db.execute(text("SET enable_seqscan = off"))  # force index scan
    approx_res = store.search("statement 50", top_k=10, filters=VectorFilter(workspace_id="w_all"))
    
    exact_set = {r[0] for r in exact_res}
    approx_set = {r[0] for r in approx_res}
    
    recall = len(exact_set.intersection(approx_set)) / len(exact_set)
    # HNSW approximate recall threshold for this deterministic CI fixture
    assert recall >= 0.5
