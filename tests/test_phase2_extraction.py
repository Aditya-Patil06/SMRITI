import pytest
from smriti.extraction import (
    ClaimNormalizer,
    ExtractedCandidate,
    MemoryExtractor,
    MockLLMExtractor,
    HybridExtractionEngine
)

def test_claim_normalizer_tokens_and_aliases():
    assert ClaimNormalizer.normalize_token("PostgreSQL") == "postgresql"
    assert ClaimNormalizer.normalize_token("postgres") == "postgresql"
    assert ClaimNormalizer.normalize_token("POSTGRE") == "postgresql"
    assert ClaimNormalizer.normalize_token("sqlite3") == "sqlite"
    assert ClaimNormalizer.normalize_token("Fast API") == "fastapi"
    assert ClaimNormalizer.normalize_token("FastAPI Framework") == "fastapi"
    assert ClaimNormalizer.normalize_token("React.js") == "react"
    assert ClaimNormalizer.normalize_token("NextJS") == "next.js"
    assert ClaimNormalizer.normalize_token("Docker Container") == "docker"
    assert ClaimNormalizer.normalize_token("Tailwind CSS") == "tailwindcss"

def test_claim_normalizer_predicates():
    assert ClaimNormalizer.normalize_predicate("uses database") == "uses_database"
    assert ClaimNormalizer.normalize_predicate("uses db") == "uses_database"
    assert ClaimNormalizer.normalize_predicate("backend") == "uses_backend_framework"
    assert ClaimNormalizer.normalize_predicate("frontend") == "uses_frontend_framework"
    assert ClaimNormalizer.normalize_predicate("written in") == "uses_language"
    assert ClaimNormalizer.normalize_predicate("replaces") == "replaces"

def test_claim_normalizer_full_claim():
    raw_claim = {
        "subject": "  Backend  ",
        "predicate": "Uses Database",
        "object": "Postgres",
        "scope": {"Env": "PROD", "Component": "CORE"},
        "temporal_context": "2026"
    }
    norm = ClaimNormalizer.normalize_claim(raw_claim)
    assert norm["subject"] == "backend"
    assert norm["predicate"] == "uses_database"
    assert norm["object"] == "postgresql"
    assert norm["scope"]["env"] == "prod"
    assert norm["scope"]["component"] == "core"
    assert norm["temporal_context"] == "2026"

def test_mock_llm_extractor_nine_fixtures():
    # 1. Normal
    llm = MockLLMExtractor(mode="normal")
    res_normal = llm.extract("assistant", "We use PostgreSQL for backend.")
    assert len(res_normal) == 1
    assert res_normal[0].memory_type == "decision"
    assert res_normal[0].structured_claim["object"] == "postgresql"
    assert res_normal[0].confidence >= 0.90

    # 2. Structured claim
    llm.set_mode("structured_claim")
    res_claim = llm.extract("assistant", "Backend uses FastAPI")
    assert len(res_claim) == 1
    assert res_claim[0].structured_claim["predicate"] == "uses_backend_framework"
    assert res_claim[0].structured_claim["object"] == "fastapi"

    # 3. Duplicate
    llm.set_mode("duplicate")
    res_dup = llm.extract("assistant", "We are using Postgres")
    assert len(res_dup) == 1
    assert "postgres" in res_dup[0].structured_claim["object"]

    # 4. Conflict
    llm.set_mode("conflict")
    res_conflict = llm.extract("assistant", "Backend uses MongoDB")
    assert len(res_conflict) == 1
    assert res_conflict[0].structured_claim["object"] == "mongodb"

    # 5. Supersession
    llm.set_mode("supersession")
    res_super = llm.extract("assistant", "Switched to SQLite instead of PostgreSQL")
    assert len(res_super) == 1
    assert res_super[0].details["replaces"] == "postgresql"

    # 6. Clear coexistence
    llm.set_mode("clear_coexistence")
    res_coexist = llm.extract("assistant", "Use SQLite for dev")
    assert len(res_coexist) == 1
    assert res_coexist[0].structured_claim["scope"]["env"] == "dev"
    assert res_coexist[0].confidence >= 0.90

    # 7. Ambiguous coexistence
    llm.set_mode("ambiguous_coexistence")
    res_ambig = llm.extract("assistant", "We also use SQLite sometimes")
    assert len(res_ambig) == 1
    assert res_ambig[0].status == "review_required"
    assert res_ambig[0].confidence < 0.85

    # 8. Malformed output
    llm.set_mode("malformed")
    res_malformed = llm.extract("assistant", "unparseable garbage output")
    assert res_malformed == []

    # 9. Invalid source
    llm.set_mode("invalid_source")
    res_invalid = llm.extract("assistant", "")
    assert res_invalid == []

def test_hybrid_extraction_engine_integration():
    heuristic = MemoryExtractor()
    llm = MockLLMExtractor(mode="normal")
    hybrid = HybridExtractionEngine(heuristic_extractor=heuristic, llm_extractor=llm)

    # Content with decision pattern and technology
    content = "We decided to implement FastAPI because of automatic OpenAPI docs."
    candidates = hybrid.extract("assistant", content, use_llm=True)
    assert len(candidates) >= 1
    assert any(c.memory_type == "decision" for c in candidates)
    # Check claim normalization
    for c in candidates:
        if c.structured_claim:
            assert c.structured_claim["object"] == ClaimNormalizer.normalize_token(c.structured_claim["object"])
