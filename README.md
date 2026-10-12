# Project SMRITI

> **A persistent, portable personal AI memory layer for connecting knowledge across ChatGPT, Claude, Gemini, and future AI platforms.**

---

## 1. Vision & Core Principle

The fundamental principle of **Project SMRITI** (*Smriti* meaning memory or remembrance in Sanskrit) is:

> **"My knowledge belongs to me, not to a particular AI platform."**

AI systems are reasoning interfaces. **SMRITI is the persistent, portable personal memory layer.**

Today, users accumulate deep contextual insights, architectural decisions, and iterative breakthroughs across disparate AI platforms. However, each system operates as an isolated silo. SMRITI acts as a sovereign cognitive layer above these tools, enabling users to carry forward their accumulated knowledge and project state seamlessly across reasoning models without vendor lock-in.

---

## 2. The Problem

Modern technical workflows frequently span multiple AI ecosystems—a user may sketch architectural concepts in Claude, draft implementations in ChatGPT, and run deep document research in Gemini. This fragmented workflow creates friction:

1. **Context Fragmentation**: Critical reasoning, trade-offs, and decisions remain trapped inside vendor-specific conversation histories.
2. **Repetitive Onboarding**: Starting a new session requires manually re-explaining the entire project background, architecture, and previous decisions.
3. **Loss of Provenance**: Users cannot easily track which model suggested a solution, when an architectural pivot was decided, or which message served as the original source.
4. **Context Window Exhaustion**: Blindly dumping raw conversation transcripts into prompts saturates context windows and degrades model attention.
5. **Vendor Enclosure**: Switching models or adopting new AI platforms requires leaving behind years of accumulated reasoning.

---

## 3. The SMRITI Solution

SMRITI does **not** attempt to synchronize raw chat transcripts or mirror active runtime sessions across providers.

Instead, SMRITI extracts, normalizes, and maintains **structured project state and relational knowledge** independent of the underlying AI provider. By distilling conversations into a structured knowledge graph, persistent vector index, and dynamic project state, SMRITI synthesizes compact, goal-driven context packages tailored for immediate consumption by whatever AI platform the user chooses next.

---

## 4. Architecture

SMRITI positions itself as an independent abstraction layer between the user and conversational AI platforms:

```text
                AI PLATFORMS
           (ChatGPT, Claude, Gemini)
                     │
                     ▼
          SMRITI INTEGRATION LAYER
            (Provider Adapters)
                     │
                     ▼
                SMRITI CORE
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Memory       Retrieval      Graph
       DB          Engine       Layer
   (SQLAlchemy)  (Hybrid & Vec) (NetworkX)
        │            │            │
        └────────────┼────────────┘
                     ▼
              CONTEXT ENGINE
                     │
                     ▼
          PORTABLE AI CONTEXT
```

```mermaid
flowchart TD
    User([User / Web Control Center]) --> SMRITI[Project SMRITI Core]
    
    subgraph SMRITI [Project SMRITI Memory Core]
        direction TB
        DB[(Memory DB - SQLAlchemy / Alembic)]
        VecStore[(Persistent Vector Store)]
        Graph[Knowledge Graph - NetworkX]
        CtxEng[Context Engine]
        Retr[Hybrid Retrieval Engine]
        
        DB <--> Retr
        VecStore <--> Retr
        Graph <--> Retr
        Retr --> CtxEng
    end
    
    CtxEng --> Adapters[AI Provider Adapters]
    Adapters -.-> ChatGPT[ChatGPT]
    Adapters -.-> Claude[Claude]
    Adapters -.-> Gemini[Gemini]
```

---

## 5. Canonical Memory Model & Provenance

### Structured Memory Entities
SMRITI represents knowledge through canonical, relational entities:
* **User**: The owner of the memory space.
* **Workspace**: Isolated domain contexts (e.g., Personal, Work, College) preventing accidental cross-context leakage.
* **ProviderAccount**: Specific provider connections (e.g., ChatGPT Personal vs. ChatGPT Work).
* **Conversation**: Normalized container for session transcripts.
* **Message**: Individual conversation turns with sequence indexes, roles, and timestamps.
* **Project**: Structured workspace initiatives tracking goals, tech stacks, and constraints.
* **Task**: Actionable items tracking priorities, statuses, and dependencies.
* **Memory**: Distilled facts, decisions, problems, solutions, constraints, and technologies.
* **MemoryVersion**: Versioned ledger of memory modifications and supersessions.
* **RelationshipEdge**: Persistent graph links (`BELONGS_TO`, `CONTAINS`, `SUPERSEDES`, `DEPENDS_ON`, `RELATED_TO`).

### Complete Provenance (Explain This Memory)
Every persistent memory can be traced back to its origin:
```text
Memory
  ↓ (source_message_id)
Message
  ↓ (conversation_id)
Conversation
  ↓ (provider_account_id)
ProviderAccount
  ↓
Provider (ChatGPT / Claude / Gemini)
```

For manually created memories, provenance explicitly designates them as `user_explicit` without fabricating artificial conversation records. If a source conversation is deleted, the provenance and explain endpoints mark the source status as `source_unavailable` rather than silently purging or breaking the derived memory.

---

## 6. Hybrid Retrieval & Context Engine

SMRITI avoids dumping entire conversation histories into prompts. Instead, it utilizes hybrid retrieval:

1. **Full-Text Lexical Search**: Keyword matching over statements, rationales, and memory types in the database.
2. **Deterministic Vector Embeddings**: Cosine similarity over persistent, deterministic character n-gram and term-frequency vectors that survive restarts.
3. **Knowledge Graph Expansion**: Graph traversal discovering 1-hop connected facts and active constraints.
4. **Context Synthesis**: Compiles a compact, token-budgeted prompt package containing project goals, active tasks, key decisions, and active constraints.

---

## 7. Zero-Loss Canonical Portability

SMRITI provides complete data sovereignty through its versioned canonical export/import engine (`v1.1.0`):

```text
Original Database
       │
       ▼ (Export Canonical JSON)
Canonical Archive (manifest, users, workspaces, provider accounts,
                   projects, tasks, conversations, messages,
                   memories, versions, graph edges, audit logs)
       │
       ▼ (Import Canonical JSON)
Fresh Database
       │
       ▼ (Semantic Equivalence & Index Rebuild)
Identical SMRITI State
```

Exports can be executed globally or scoped to specific workspaces. Imports validate schema integrity before executing atomic database transactions.

---

## 8. Implementation Status

| Subsystem | Status | Implementation Details |
| :--- | :--- | :--- |
| **Database Migrations** | **IMPLEMENTED** | Alembic migration sequence (`84f31e60a4d2` initial schema, `c2b49b035a68` Phase 2, `5032740e5b4c` Phase 3 embeddings) managing all tables and indexes. |
| **PostgreSQL & pgvector Store** | **IMPLEMENTED** | Production `PgVectorStore` with pgvector cosine distance (`<=>`), transparent proxy routing (`InMemoryVectorStore` fallback on SQLite), and index rebuild via `rebuild_from_db()`. |
| **Bounded Filtered Vector Search** | **IMPLEMENTED** | Bounded SQL pagination (`LIMIT :limit OFFSET :offset`, `page_size = max(top_k * 2, 50)`) with deterministic `(distance, memory_id)` tie-breaking and early termination when post-filtering with `filter_fn`. |
| **Embedding Providers** | **IMPLEMENTED** | `MockEmbeddingProvider` (deterministic unit testing & adversarial fixtures) and `LocalEmbeddingProvider` (using optional `sentence-transformers`, defaulting to `all-MiniLM-L6-v2`). |
| **Workspace Isolation** | **IMPLEMENTED** | Multi-workspace boundaries enforced across database queries, vector searches, and context generation. |
| **Complete Provenance** | **IMPLEMENTED** | Full traceability down to message, conversation, and provider account; manual memories tagged `user_explicit`. |
| **Intelligent Memory Extraction** | **IMPLEMENTED** | Hybrid heuristic + LLM entity/relationship extraction, claim normalization, and replacement direction heuristics. |
| **Conflict & Supersession Handling** | **IMPLEMENTED** | 5-way diff engine, structured supersession claim matching, coexistence edge persistence, and interactive conflict resolution. |
| **Project State Synthesis** | **IMPLEMENTED** | Autonomous project state synthesis, milestone detection, and architecture overview tracking across active/superseded memories. |
| **Canonical Portability** | **IMPLEMENTED** | Zero-loss atomic export/import format (`v1.1.0`) with manifest validation, scoped export, and transactional safety. |
| **Local MCP Server (stdio)** | **IMPLEMENTED** | Local read-only Model Context Protocol server (`smriti.mcp_server`, MCP Python SDK v2) exposing `list_workspaces`, `list_projects`, `search_memories`, and `build_context_package`. |
| **Web Control Center** | **IMPLEMENTED** | React + TypeScript + Vite control center providing Dashboard, Memory Bank, Graph Visualizer, and Context Generator. |
| **Provider Importers** | **IMPLEMENTED** | Normalization parsers for ChatGPT, Claude, Gemini, and Generic standard formats. |
| **Provider Adapters** | **PROTOTYPE** | Adapter layer providing prompt formatting and capability matrix discovery. |
| **Real-Time Webhooks** | **PROVIDER-DEPENDENT** | Dependent on official provider webhooks or MCP server support. |
| **Live Session Sync** | **PROVIDER-DEPENDENT** | Dependent on official provider APIs exposing active session manipulation. |

---

## 9. Latest Project Update: Phase 3 Released

**Phase 3: Advanced Retrieval & pgvector** has been merged into `main`.

Key highlights of this release:
- **Persistent PostgreSQL + pgvector Storage**: Introduced `PgVectorStore` for native in-database vector similarity search using the pgvector extension and cosine distance operations (`<=>`). When running on SQLite or environments without PostgreSQL, SMRITI automatically routes to `InMemoryVectorStore` with disk-backed JSON fallback.
- **Bounded SQL-Level Vector Pagination**: Addressed unbounded candidate queries during post-filtering. Vector search with custom callback filtering (`filter_fn`) now executes bounded SQL pages (`LIMIT :limit OFFSET :offset`, with `page_size = max(top_k * 2, 50)`) and deterministic `(distance, e.memory_id)` tie-breaking, stopping early once `top_k` accepted matches are collected.
- **Alembic Embedding Migration**: Migration `5032740e5b4c_phase_3_embeddings.py` creates the `embedding_metadata` table with idempotency constraints (`memory_id`, `model_name`, `model_version`, `content_hash`). On PostgreSQL, it builds production partial HNSW indexes (`vector_cosine_ops`) for supported dimensions (64, 384, 768). On SQLite/non-PostgreSQL databases, it falls back cleanly to JSON embedding storage.
- **Configurable Embedding Providers**: Added `LocalEmbeddingProvider` powered by `sentence-transformers` (defaults to `all-MiniLM-L6-v2`, 384 dimensions), alongside `MockEmbeddingProvider` for deterministic offline testing.
- **Hardened Test Suite**: The pagination regression suite was hardened with mathematically controlled unit-vector embeddings to eliminate hash collision fragility and ensure deterministic cross-page ranking across boundaries.

---

## 10. Capability Roadmap

The development of Project SMRITI follows a capability-based progression:

### Phase 1 — Core Reliability & Portability (Completed)
- Database schema management via authoritative Alembic migrations.
- Persistent vector storage with deterministic embeddings and rebuild capabilities.
- Workspace and provider-account boundary isolation.
- Comprehensive provenance tracking and explainability.
- Atomic, validated canonical export/import engine (`v1.1.0`).
- Complete API integration test suite.

### Phase 2 — Intelligent Memory & Extraction (Completed)
- Hybrid heuristic and LLM-assisted entity and relationship extraction.
- Automatic project state synthesis and milestone detection.
- Fine-grained memory diff detection (superseded, conflicting, new).
- User-directed interactive conflict resolution workflows.

### Phase 3 — Advanced Retrieval & pgvector (Completed)
- Persistent PostgreSQL vector storage via pgvector extension (`PgVectorStore`).
- Production partial HNSW indexing for standard dimensions (64, 384, 768).
- Bounded, paginated SQL vector search with callback post-filtering and deterministic tie-breaking.
- Non-PostgreSQL compatibility using JSON embedding storage and in-memory vector store.
- Local dense embedding models via optional `sentence-transformers` integration.

### Phase 4 — Provider Integrations & MCP (In Progress)
- **Local Read-Only MCP Server (`smriti.mcp_server`)** (Completed in Phase 4.1): Formal Model Context Protocol (MCP) server over local stdio transport for Claude Desktop and compatible MCP clients, exposing `list_workspaces`, `list_projects`, `search_memories`, and `build_context_package`.
- Custom GPT actions and plugin endpoints for authorized ChatGPT access (Planned).
- Official Gemini Extensions and Function Calling integrations (Planned).
- Local model execution adapters (e.g., Ollama, vLLM) (Planned).
- Multi-hop graph-augmented topological retrieval reasoning (Planned).
- Cross-workspace explicit sharing and selective subgraph publishing (Planned).

### Phase 5 — Production Hardening & Encryption (Future Phase)
- Full PostgreSQL backend deployment profile.
- End-to-end local field encryption for sensitive memory rationales and conversation snippets.
- Granular role-based access control and multi-user authentication.
- Automated secret and PII redaction during ingestion.
- Staleness indicators and temporal memory decay algorithms.

---

## 11. Local Setup & Testing

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- *(Optional for Phase 3 PostgreSQL/pgvector)*: PostgreSQL 15+ with the `pgvector` extension installed (e.g., the official `pgvector/pgvector:pg16` Docker image).

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aditya-Patil06/SMRITI.git
   cd SMRITI
   ```

2. **Install Python dependencies**:
   - **Lightweight SQLite Development Workflow** (no PostgreSQL, no ML dependencies):
     ```bash
     pip install -e ".[dev]"
     ```
   - **SQLite with Semantic Local Embeddings** (no PostgreSQL required):
     ```bash
     pip install -e ".[dev]"
     pip install sentence-transformers
     ```
   - **Full Phase 3 Workflow** (PostgreSQL, pgvector, and local `sentence-transformers` embeddings):
     ```bash
     pip install -e ".[dev,phase3]"
     ```
     *Note: The `phase3` extra bundles `pgvector`, `psycopg[binary]`, and `sentence-transformers`.*
   - **Phase 4 MCP Server Workflow** (Model Context Protocol stdio server):
     ```bash
     pip install -e ".[dev,phase4]"
     # Or install all extras together:
     pip install -e ".[dev,phase3,phase4]"
     ```
     *Note: The `phase4` extra installs `mcp>=2,<3`.*

3. **Configure Database & Embeddings**:
   By default, `smriti/config.py` targets SQLite (`sqlite:///./smriti.db`) but configures `SMRITI_EMBEDDING_PROVIDER="local"`. Configure your environment according to your workflow (examples show Bash-compatible syntax; on Windows PowerShell, use `$env:VARIABLE="value"`):
   - **Lightweight SQLite Workflow (Mock Embeddings)**:
     If using only `.[dev]` without `sentence-transformers`, set the embedding provider to `mock`:
     ```bash
     export SMRITI_EMBEDDING_PROVIDER="mock"
     ```
     *Note: The mock provider uses deterministic bag-of-words MD5 hashing. It is designed for lightweight, deterministic offline testing and development, and is not equivalent to semantic dense embeddings.*
   - **SQLite with Local Semantic Embeddings**:
     If you installed `sentence-transformers`, local semantic embeddings run directly against SQLite without requiring PostgreSQL:
     ```bash
     export SMRITI_EMBEDDING_PROVIDER="local"
     export SMRITI_EMBEDDING_MODEL="all-MiniLM-L6-v2"
     ```
   - **Full PostgreSQL + pgvector Workflow**:
     ```bash
     export SMRITI_DATABASE_URL="postgresql://postgres:postgrespassword@localhost:5432/smriti_test"
     export SMRITI_EMBEDDING_PROVIDER="local"
     export SMRITI_EMBEDDING_MODEL="all-MiniLM-L6-v2"
     ```

4. **Apply database migrations**:
   ```bash
   alembic upgrade head
   ```

5. **Run test suite**:
   - **Local Verification (Layer A SQLite)**:
     ```bash
     pytest -v
     ```
     *In local environments without PostgreSQL running (SMRITI_PG_TESTS unset), the PostgreSQL integration suite (16 tests) is skipped automatically along with 1 optional local embedding test. Local validation yields **85 passed, 17 skipped, 1 warning** across the entire repository (MCP tests: **13 passed, 1 warning**; embedding tests: **5 passed, 1 skipped**).*
   - **Continuous Integration (CI — GitHub Actions)**:
     The GitHub Actions CI pipeline tests both Layer A and Layer B:
     1. Runs Layer A SQLite tests including MCP server tests (`pytest -q --ignore=tests/test_phase3_pgvector.py`).
     2. Runs Alembic migrations on a live `pgvector/pgvector:pg16` PostgreSQL container (`alembic upgrade head`).
     3. Runs Layer B PostgreSQL + pgvector tests (`SMRITI_PG_TESTS=1 pytest tests/test_phase3_pgvector.py -v -s`).

6. **Run SMRITI Services**:
   - **Start the REST API server**:
     ```bash
     uvicorn smriti.api:app --reload --port 8000
     ```
   - **Start the Local Read-Only MCP Server (stdio transport)**:
     ```bash
     python -m smriti.mcp_server
     ```
     *The MCP server speaks JSON-RPC 2.0 over standard input/output (stdio). Protocol messages are exchanged on stdout, while diagnostic logs and startup traces are directed to stderr.*

7. **Configure Claude Desktop (MCP Client)**:
   Add the SMRITI MCP server to your `claude_desktop_config.json` (on Windows located at `%APPDATA%\Claude\claude_desktop_config.json`, or on macOS at `~/Library/Application Support/Claude/claude_desktop_config.json`).

   > [!NOTE]
   > Claude Desktop stdio transports launch the server process directly and may ignore nonstandard working-directory properties (such as `cwd`). To ensure reliable module discovery and database access across any working directory, specify the absolute path to your Python interpreter and pass `PYTHONPATH` along with an absolute database URL in `env`.

   **Example Windows configuration** *(substitute your actual Python interpreter and repository paths)*:
   ```json
   {
     "mcpServers": {
       "smriti": {
         "command": "C:\\path\\to\\python.exe",
         "args": [
           "-m",
           "smriti.mcp_server"
         ],
         "env": {
           "PYTHONPATH": "C:\\path\\to\\project-smriti",
           "SMRITI_DATABASE_URL": "sqlite:///C:/path/to/project-smriti/smriti.db",
           "SMRITI_EMBEDDING_PROVIDER": "mock"
         }
       }
     }
   }
   ```
   *(Note: Ensure paths use escaped backslashes `\\\\` in JSON strings. If using a virtual environment, set `"command"` to that venv's Python binary, e.g., `C:\\path\\to\\project-smriti\\.venv\\Scripts\\python.exe`.)*

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   ```
2. Run production build check:
   ```bash
   npm run build
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

---

## 12. Repository Structure

```text
.
├── alembic/                      # Database migrations
│   ├── env.py                    # Alembic migration environment
│   └── versions/                 # Versioned migration revisions
│       ├── 84f31e60a4d2_initial_schema.py
│       ├── c2b49b035a68_phase_2_intelligent_memory.py
│       └── 5032740e5b4c_phase_3_embeddings.py
├── alembic.ini                   # Alembic configuration
├── frontend/                     # Web Control Center (React + TypeScript + Vite)
│   ├── src/
│   │   ├── api.ts                # Typed SMRITI API client
│   │   ├── App.tsx               # Control Center dashboard & views
│   │   └── main.tsx              # Application entry point
│   ├── package.json
│   └── vite.config.ts
├── pyproject.toml                # Python project configuration & dependencies
├── smriti/                       # Core Python package
│   ├── api.py                    # FastAPI REST API endpoints
│   ├── canonical_export.py       # Zero-loss export/import engine
│   ├── config.py                 # Configuration and environment settings
│   ├── context_engine.py         # Portable context package synthesizer
│   ├── embeddings.py             # Mock and Local sentence-transformers embeddings
│   ├── extraction.py             # Memory candidate extraction
│   ├── graph.py                  # NetworkX knowledge graph service
│   ├── importers/                # Multi-provider conversation parsers
│   │   ├── chatgpt.py
│   │   ├── claude.py
│   │   ├── gemini.py
│   │   └── generic.py
│   ├── mcp_server.py             # Local read-only MCP stdio server (Phase 4.1)
│   ├── memory_diff.py            # 5-way memory diff engine
│   ├── memory_manager.py         # Lifecycle, provenance, and conflict resolution
│   ├── models.py                 # SQLAlchemy canonical data models & VectorType
│   ├── project_intelligence.py   # Project state synthesis & milestone detection
│   ├── providers/                # Provider adapters and capabilities
│   ├── retrieval.py              # Hybrid retrieval (lexical + vector + graph)
│   ├── schemas.py                # Pydantic V2 schemas
│   └── vector_store.py           # Persistent vector stores (PgVectorStore & InMemoryVectorStore)
└── tests/                        # Automated test suites
    ├── test_api_integration.py   # API integration, isolation, and portability tests
    ├── test_phase2_commit_28d842b_regression.py
    ├── test_phase2_diff_and_conflicts.py
    ├── test_phase2_extraction.py
    ├── test_phase2_project_intelligence.py
    ├── test_phase3_embeddings.py # Embedding providers and unit pagination tests
    ├── test_phase3_pgvector.py   # PostgreSQL + pgvector integration & pagination tests
    ├── test_phase4_mcp_server.py # Local MCP server tool and transport tests
    └── test_smriti_core.py       # Core unit and round-trip tests
```

---

## 13. Privacy & Security Principles

* **Local-First**: All relational records, vector indices, and knowledge graphs reside on user-controlled hardware.
* **No Credential Ingestion**: SMRITI never asks for, stores, or handles raw AI provider account passwords.
* **Strict Boundary Isolation**: Memories in one workspace never cross into another without explicit user action.
* **Zero Private Commits**: Conversation archives, local databases, and environment secrets are excluded from Git.
