import pytest
import os
import threading
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
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
    engine = create_engine(db_url, poolclass=NullPool)
    
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

@pytest.fixture(autouse=True)
def clean_pg_tables(pg_engine):
    """Ensure clean table state for each Phase 3 test to guarantee test isolation."""
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE embedding_metadata CASCADE"))
        conn.execute(text("TRUNCATE TABLE memories CASCADE"))
    yield
    with pg_engine.begin() as conn:
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
    
    db.execute(text("SET enable_seqscan = on;"))
    
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
    
    if not db.query(Workspace).filter_by(id="w_iso").first():
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
    db.close()
    
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
    db.close()
    
    assert len(records) == 1

def test_concurrency_different_text_same_memory(pg_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=pg_engine)
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, Session)
    db = Session()
    from smriti.models import Memory, Workspace, User

    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", email="test@test.com", username="test"))
    if not db.query(Workspace).filter_by(id="w_diff").first():
        db.add(Workspace(id="w_diff", user_id="u1", name="Diff Text Conc"))
    db.commit()

    db.add(Memory(id="conc_diff_mem", workspace_id="w_diff", statement="initial", memory_type="fact"))
    db.commit()

    def worker(i):
        store.upsert("conc_diff_mem", f"concurrent distinct statement {i}", meta={})

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    records = db.query(EmbeddingMetadata).filter_by(memory_id="conc_diff_mem").all()
    assert len(records) == 1

    # Verify search does not return duplicate results
    hits = store.search("concurrent distinct statement", top_k=10, filters=VectorFilter(workspace_id="w_diff"))
    assert len(hits) == 1
    assert hits[0][0] == "conc_diff_mem"
    db.close()

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
        
    # --- Revised exact and approximate queries for reliable recall measurement ---
    # Build the vector literal for the query embedding
    vec = provider.embed("statement 50")
    vec_str = "[" + ",".join(map(str, vec)) + "]"
    sql = """
        SELECT e.memory_id, 1 - (e.embedding::vector({dim}) <=> :vec) AS similarity
        FROM embedding_metadata e
        JOIN memories m ON e.memory_id = m.id
        WHERE e.model_name = :m_name
          AND e.model_version = :m_version
          AND e.embedding_dimension = :m_dim
          AND m.workspace_id = :w_id
        ORDER BY e.embedding::vector({dim}) <=> :vec, e.memory_id
        LIMIT :limit
    """.format(dim=provider.dimension)
    params = {
        "vec": vec_str,
        "m_name": provider.model_name,
        "m_version": provider.model_version,
        "m_dim": provider.dimension,
        "w_id": "w_all",
        "limit": 10,
    }
    # Exact reference: disable index scans to force sequential scan (exact NN)
    db.execute(text("SET enable_indexscan = off"))
    db.execute(text("SET enable_seqscan = on"))
    exact_res = db.execute(text(sql), params).fetchall()
    # Approximate using HNSW: enable index scan and iterative scan
    db.execute(text("SET enable_indexscan = on"))
    db.execute(text("SET enable_seqscan = off"))
    db.execute(text("SET hnsw.iterative_scan = strict_order"))
    # Verify that PostgreSQL chose an Index Scan (HNSW)
    explain_rows = db.execute(text(f"EXPLAIN {sql}"), params).fetchall()
    explain_str = "\n".join(row[0] for row in explain_rows)
    assert "Index Scan" in explain_str, f"Expected Index Scan in plan, got:\n{explain_str}"
    approx_res = db.execute(text(sql), params).fetchall()
    
    exact_set = {r[0] for r in exact_res}
    approx_set = {r[0] for r in approx_res}
    
    db.execute(text("SET enable_indexscan = on"))
    db.execute(text("SET enable_seqscan = on"))
    
    recall = len(exact_set.intersection(approx_set)) / len(exact_set)
    # HNSW approximate recall threshold for this deterministic CI fixture
    assert recall >= 0.8

def test_pgvector_upsert_session_ownership_and_autosave(pg_session):
    db = pg_session()
    from smriti.models import Memory
    db.add(Memory(id="m_save_1", workspace_id="w1", statement="auto_save test", memory_type="fact"))
    db.commit()

    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)

    # 1. Upsert with auto_save=False when store owns session (should persist and commit before close)
    store.upsert("m_save_1", "auto_save test", auto_save=False)
    db2 = pg_session()
    row = db2.query(EmbeddingMetadata).filter_by(memory_id="m_save_1").first()
    assert row is not None

    # 2. Delete with auto_save=False when store owns session
    store.delete("m_save_1", auto_save=False)
    db3 = pg_session()
    assert db3.query(EmbeddingMetadata).filter_by(memory_id="m_save_1").first() is None

def test_pgvector_upsert_with_external_session_and_rebuild(pg_session):
    db = pg_session()
    from smriti.models import Memory
    db.add(Memory(id="m_reb_1", workspace_id="w1", statement="rebuild 1", memory_type="fact", status="active"))
    db.add(Memory(id="m_reb_2", workspace_id="w1", statement="rebuild 2", memory_type="fact", status="forgotten"))
    db.commit()

    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)

    # rebuild_from_db passes db session and should only index non-forgotten
    count = store.rebuild_from_db(db)
    assert count == 1

    db_check = pg_session()
    assert db_check.query(EmbeddingMetadata).filter_by(memory_id="m_reb_1").first() is not None
    assert db_check.query(EmbeddingMetadata).filter_by(memory_id="m_reb_2").first() is None

def test_pgvector_forgotten_memory_search_exclusion(pg_session):
    db = pg_session()
    from smriti.models import Memory
    db.add(Memory(id="m_act", workspace_id="w1", statement="hello world", memory_type="fact", status="active"))
    db.add(Memory(id="m_sup", workspace_id="w1", statement="hello world", memory_type="fact", status="superseded"))
    db.add(Memory(id="m_forg", workspace_id="w1", statement="hello world", memory_type="fact", status="forgotten"))
    db.commit()

    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)
    store.upsert("m_act", "hello world")
    store.upsert("m_sup", "hello world")
    store.upsert("m_forg", "hello world")

    # Default: exclude superseded and forgotten
    res_default = store.search("hello world", filters=VectorFilter(workspace_id="w1", include_superseded=False))
    ids_default = {r[0] for r in res_default}
    assert "m_act" in ids_default
    assert "m_sup" not in ids_default
    assert "m_forg" not in ids_default

    # include_superseded=True: must include superseded but NEVER forgotten
    res_sup = store.search("hello world", filters=VectorFilter(workspace_id="w1", include_superseded=True))
    ids_sup = {r[0] for r in res_sup}
    assert "m_act" in ids_sup
    assert "m_sup" in ids_sup
    assert "m_forg" not in ids_sup


def test_search_filter_fn_does_not_starve_candidates_beyond_sql_limit(pg_session):
    db = pg_session()
    from smriti.models import Memory, Workspace, User
    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", email="test@test.com", username="test"))
    if not db.query(Workspace).filter_by(id="w_filt").first():
        db.add(Workspace(id="w_filt", user_id="u1", name="Filt Workspace"))
    db.commit()

    # Three memories:
    # m_rej_1 and m_rej_2 have high textual similarity ("apple banana cherry") but memory_type="scratchpad"
    # m_acc has slightly lower overlap ("apple dog elephant") but memory_type="decision"
    db.add(Memory(id="m_rej_1", workspace_id="w_filt", statement="apple banana cherry", memory_type="scratchpad", status="active"))
    db.add(Memory(id="m_rej_2", workspace_id="w_filt", statement="apple banana cherry", memory_type="scratchpad", status="active"))
    db.add(Memory(id="m_acc", workspace_id="w_filt", statement="apple dog elephant", memory_type="decision", status="active"))
    db.commit()

    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)
    store.upsert("m_rej_1", "apple banana cherry")
    store.upsert("m_rej_2", "apple banana cherry")
    store.upsert("m_acc", "apple dog elephant")

    # Baseline: without filter_fn, top_k=2 returns the two scratchpad memories
    base_hits = store.search("apple banana cherry", top_k=2, filters=VectorFilter(workspace_id="w_filt"))
    assert len(base_hits) == 2
    base_ids = [h[0] for h in base_hits]
    assert "m_rej_1" in base_ids
    assert "m_rej_2" in base_ids
    assert "m_acc" not in base_ids

    # With filter_fn selecting only "decision" memories:
    # If SQL LIMIT 2 was applied before filter_fn, m_acc would be starved and hits would be empty.
    # With the fix, m_acc is evaluated and returned.
    hits = store.search(
        "apple banana cherry",
        top_k=2,
        filters=VectorFilter(workspace_id="w_filt"),
        filter_fn=lambda m: m.get("memory_type") == "decision",
    )
    assert len(hits) == 1
    assert hits[0][0] == "m_acc"
    assert len(hits) <= 2


def test_search_filter_fn_batches_and_stops_at_top_k(pg_session, monkeypatch):
    db = pg_session()
    from smriti.models import Memory, Workspace, User
    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", email="test@test.com", username="test"))
    if not db.query(Workspace).filter_by(id="w_batch").first():
        db.add(Workspace(id="w_batch", user_id="u1", name="Batch Workspace"))
    db.commit()

    # Create 58 memories total:
    # First 55 are "noise" (memory_type="scratchpad") - guarantees page 1 (50 items) is completely exhausted
    # Next 3 are "match" (memory_type="decision") located on page 2 (offsets 50+)
    # Statements are crafted so noise has high text overlap and match items have lower overlap
    for i in range(55):
        db.add(Memory(id=f"m_noise_{i:03d}", workspace_id="w_batch", statement=f"statement common apple banana {i:03d}", memory_type="scratchpad", status="active"))
    for i in range(3):
        db.add(Memory(id=f"m_match_{i:03d}", workspace_id="w_batch", statement=f"statement common apple dog {i:03d}", memory_type="decision", status="active"))
    db.commit()

    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)
    for i in range(55):
        store.upsert(f"m_noise_{i:03d}", f"statement common apple banana {i:03d}")
    for i in range(3):
        store.upsert(f"m_match_{i:03d}", f"statement common apple dog {i:03d}")

    # Track executed queries to verify bounded SQL LIMIT and OFFSET across pages
    executed_queries = []
    from sqlalchemy.orm import Session as SASession
    orig_execute = SASession.execute

    def tracking_execute(self, statement, params=None, **kwargs):
        executed_queries.append((str(statement), params))
        return orig_execute(self, statement, params, **kwargs)

    monkeypatch.setattr(SASession, "execute", tracking_execute)

    evaluated_metas = []
    def custom_filter(meta):
        evaluated_metas.append(meta)
        return meta.get("memory_type") == "decision"

    hits = store.search(
        "statement common apple banana",
        top_k=2,
        filters=VectorFilter(workspace_id="w_batch"),
        filter_fn=custom_filter,
    )

    # 1. Stops after collecting top_k accepted candidates (top_k=2)
    assert len(hits) == 2
    for mem_id, sim, meta in hits:
        assert meta["memory_type"] == "decision"

    # 2. Existing filter_fn semantics remain correct
    assert all(h[2]["memory_type"] == "decision" for h in hits)
    # 3. Only needed candidates were accepted; third match on page 2 was never appended
    assert len([m for m in evaluated_metas if m.get("memory_type") == "decision"]) == 2

    # 4. Verifies multiple bounded SQL pages actually executed
    sql_pages = [
        (sql, p) for sql, p in executed_queries
        if "LIMIT :limit OFFSET :offset" in sql
    ]
    # Page size for top_k=2 is max(2*2, 50) = 50.
    # Since 55 noise candidates exist, Page 1 (offset 0) has 0 matches.
    # Page 2 (offset 50) contains the remaining 5 noise and the 3 matches.
    # It must execute at least two queries and stop on page 2.
    assert len(sql_pages) == 2
    page1_sql, page1_params = sql_pages[0]
    page2_sql, page2_params = sql_pages[1]

    assert page1_params["limit"] == 50
    assert page1_params["offset"] == 0

    assert page2_params["limit"] == 50
    assert page2_params["offset"] == 50


def test_search_filter_fn_none_retains_sql_limit(pg_session, monkeypatch):
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)

    executed_sqls = []
    executed_params = []

    from sqlalchemy.orm import Session as SASession
    orig_execute = SASession.execute

    def tracking_execute(self, statement, params=None, **kwargs):
        executed_sqls.append(str(statement))
        executed_params.append(params)
        return orig_execute(self, statement, params, **kwargs)

    monkeypatch.setattr(SASession, "execute", tracking_execute)

    store.search("test query", top_k=7, filter_fn=None)

    # Verify SQL query contains LIMIT :limit and params has limit=7
    matching_query = [
        (sql, p) for sql, p in zip(executed_sqls, executed_params)
        if "LIMIT :limit" in sql and p and p.get("limit") == 7 and "OFFSET" not in sql
    ]
    assert len(matching_query) == 1


def test_search_filter_fn_deterministic_tie_breaking_across_page_boundary(pg_session, monkeypatch):
    db = pg_session()
    from smriti.models import Memory, Workspace, User
    if not db.query(User).filter_by(id="u1").first():
        db.add(User(id="u1", email="test@test.com", username="test"))
    if not db.query(Workspace).filter_by(id="w_tie").first():
        db.add(Workspace(id="w_tie", user_id="u1", name="Tie Workspace"))
    db.commit()

    # 1. Create 110 memories all with identical statement and identical vector distances
    # Insert in reverse order to ensure DB natural ordering does not match expected result
    for i in reversed(range(110)):
        mem_type = f"type_{i:03d}"
        db.add(Memory(id=f"m_tie_{i:03d}", workspace_id="w_tie", statement="identical statement", memory_type=mem_type, status="active"))
    db.commit()

    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)
    for i in reversed(range(110)):
        store.upsert(f"m_tie_{i:03d}", "identical statement")

    # Track executed SQL queries
    executed_queries = []
    from sqlalchemy.orm import Session as SASession
    orig_execute = SASession.execute

    def tracking_execute(self, statement, params=None, **kwargs):
        executed_queries.append((str(statement), params))
        return orig_execute(self, statement, params, **kwargs)

    monkeypatch.setattr(SASession, "execute", tracking_execute)

    # 2. top_k = 52 -> page_size = max(52 * 2, 50) = 104
    # 3. filter_fn accepts only IDs m_tie_054 and later (by inspecting memory_type)
    hits = store.search(
        "identical statement",
        top_k=52,
        filters=VectorFilter(workspace_id="w_tie"),
        filter_fn=lambda m: m.get("memory_type", "") >= "type_054",
    )

    # 4. Page 1 (offset 0, limit 104) yields m_tie_000..m_tie_103:
    #    filter_fn accepts m_tie_054..m_tie_103 (exactly 50 accepted records).
    #    Page 2 (offset 104, limit 104) yields m_tie_104..m_tie_109:
    #    filter_fn accepts m_tie_104 and m_tie_105 to reach top_k=52 and stops.
    assert len(hits) == 52

    # 5. Exactly two paginated queries executed, with offsets 0 and 104 and bounded limits 104
    sql_pages = [
        (sql, p) for sql, p in executed_queries
        if "LIMIT :limit OFFSET :offset" in sql
    ]
    assert len(sql_pages) == 2

    p1_sql, p1_params = sql_pages[0]
    assert p1_params["limit"] == 104
    assert p1_params["offset"] == 0

    p2_sql, p2_params = sql_pages[1]
    assert p2_params["limit"] == 104
    assert p2_params["offset"] == 104

    # 6. Returned IDs are exactly m_tie_054 through m_tie_105, in order, with no duplicates
    hit_ids = [h[0] for h in hits]
    expected_ids = [f"m_tie_{i:03d}" for i in range(54, 106)]
    assert hit_ids == expected_ids
    assert len(set(hit_ids)) == 52


def test_search_top_k_non_positive_returns_empty(pg_session):
    provider = MockEmbeddingProvider(dimension=64)
    store = PgVectorStore(provider, pg_session)

    assert store.search("test", top_k=0) == []
    assert store.search("test", top_k=-1) == []
    assert store.search("test", top_k=0, filter_fn=lambda m: True) == []
    assert store.search("test", top_k=-5, filter_fn=lambda m: True) == []

