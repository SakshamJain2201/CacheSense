# Architectural Decision Records (ADRs)

## ADR-001: FAISS IndexFlatIP with IndexIDMap2 for Vector Search and Eviction
- **Context**: LLM semantic caching requires vector search over 384-dimensional embeddings produced by `sentence-transformers/all-MiniLM-L6-v2`. We need fast similarity search, accurate cosine similarity scoring, and reliable removal of evicted vectors.
- **Decision**: Use `faiss.IndexFlatIP` wrapped with `faiss.IndexIDMap2`.
- **Rationale**:
  1. Inner Product on L2-normalized unit vectors is mathematically identical to cosine similarity: $\langle u, v \rangle = \cos(\theta)$ when $\|u\|_2 = \|v\|_2 = 1$.
  2. Unlike `IndexHNSWFlat`, `IndexFlatIP` fully supports atomic removal of vectors by ID using `remove_ids()`. This is essential for synchronized LRU and TTL eviction across SQLite and the vector index.
  3. For index sizes up to 50,000 to 100,000 vectors, `IndexFlatIP` on modern multi-core CPUs executes top-1 searches in $<2\text{ms}$, avoiding the indexing overhead and recall approximation of graph-based indexes.

## ADR-002: Cache Partitioning and Multi-Turn Context Strategy
- **Context**: Prompts with identical wording but different system instructions, temperatures, function definitions, or preceding conversation turns should not share responses.
- **Decision**: Partition key format: `{model}|sys:{sys_hash}|{temp_bucket}|tools:{tools_hash}|ns:{namespace}`.
  - Multi-turn conversation context is handled by hashing preceding dialogue turns and embedding `f"[ctx:{context_hash}] {final_user_message}"`.
- **Tradeoff & Discussion**:
  - Hashing preceding turns ensures 100% strict context isolation: if user A and user B have different conversation histories, they will not cross-hit, preventing hallucinatory or irrelevant cached answers.
  - Paraphrased versions of the latest turn within identical conversational contexts will match with high semantic similarity.

## ADR-003: SQLite WAL Mode as Metadata and Key-Value Store
- **Context**: A persistent, thread-safe, and crash-resilient metadata store is required to store full completion payloads, token usage counts, and LRU access timestamps.
- **Decision**: Implement `SQLiteResponseStore` implementing `ResponseStore(ABC)` operating in Write-Ahead Logging (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`).
- **Rationale**:
  - WAL mode enables concurrent readers without locking writers.
  - Zero external service dependencies for local and edge deployments.
  - The abstract `ResponseStore` interface cleanly decouples application logic, allowing a drop-in `RedisResponseStore` in clustered environments.

## ADR-004: In-Flight Request Coalescing (Single-Flight)
- **Context**: Sudden bursts of identical prompts on cache miss can cause redundant upstream API calls, multiplying latency and billing costs (the "thundering herd" problem).
- **Decision**: Implement `SingleFlight` manager using `asyncio.Future` and `asyncio.Event`.
- **Rationale**: The first request becomes the leader and initiates the upstream request; subsequent concurrent identical requests become followers and await the leader's completed payload, cutting upstream duplicate requests to zero.

## ADR-005: Privacy & PII Bypass Filtering
- **Context**: Caching prompts containing sensitive customer data (such as emails, credit cards, or bearer tokens) could risk data leakage across users or persistent disk storage.
- **Decision**: Scan prompts against configurable regex patterns before cache insertion. If a PII pattern matches, the request is flagged with `no-store` and bypassed from FAISS and SQLite persistence.
