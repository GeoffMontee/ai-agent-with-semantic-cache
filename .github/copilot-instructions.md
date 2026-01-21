# GitHub Copilot Instructions

## Project Overview

This is a Python CLI tool that provides an AI agent interface with semantic caching capabilities. The project integrates:
- **Anthropic Claude** (via Autogen) for AI completions
- **ScyllaDB Cloud** OR **PostgreSQL pgvector** for vector-based semantic caching
- **SentenceTransformers** for generating embeddings

The project consists of four main components:
1. **Main AI Agent** (`ai_agent_with_cache.py`) - Queries Claude with optional semantic caching using ScyllaDB or PostgreSQL pgvector
2. **Performance Benchmark** (`benchmark.py`) - Compares cache performance between backends with comprehensive test scenarios
3. **ScyllaDB Cloud Management** (`scylla-cloud/deploy-scylla-cloud.py`) - Manages ScyllaDB Cloud clusters
4. **PostgreSQL pgvector Docker Management** (`postgres-pgvector-docker/deploy-pgvector-docker.py`) - Manages local PostgreSQL instances with pgvector

## Key Architecture Patterns

### 1. Conditional Caching
- The tool supports running with or without semantic caching via the `--with-cache` flag
- When caching is disabled (`--with-cache none`), database and SentenceTransformer dependencies are not initialized
- When enabled (`--with-cache scylla` or `--with-cache pgvector`), the full vector search pipeline is activated
- ScyllaDB uses synchronous Cassandra driver, PostgreSQL uses async psycopg3

### 2. Configuration Priority
All configuration follows this precedence order:
1. Command-line arguments (highest priority)
2. Environment variables
3. Default values (lowest priority)

### 3. Vector Search Implementation
- Embeddings are 384-dimension vectors from SentenceTransformer models
- **ScyllaDB**: Uses cosine similarity for ANN (approximate nearest neighbor) search via custom vector index
- **PostgreSQL pgvector**: Uses HNSW indexes with configurable similarity functions (cosine, l2, inner_product, l1)
- Cache hits return immediately without calling Claude API

## Code Style Guidelines

### Python Conventions
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Use f-strings for string formatting
- Prefer explicit error messages with context

### Async Patterns
- Main async logic is in `async_main()`
- Use `await` for Anthropic client calls and PostgreSQL pgvector operations
- ScyllaDB operations are synchronous (scylla-driver does not support async, like cassandra-driver)
- Keep the synchronous `main()` wrapper minimal

### Database Operations
- **ScyllaDB**: Use prepared statements for inserts, convert numpy arrays to lists before sending to CQL
- **PostgreSQL pgvector**: Use parameterized queries with psycopg3, convert numpy arrays to lists for pgvector
  - **CRITICAL**: All vector parameters MUST be cast to `::vector` type in SQL queries
  - Example: `SELECT ... WHERE embedding <-> %s::vector < %s`
  - Without `::vector`, queries fail with "operator does not exist" error
- Include proper error handling with informative messages

## Component Responsibilities

### `ScyllaDBCache` Class
- **Purpose**: Encapsulates all ScyllaDB operations
- **Driver Choice**: 
  - **CRITICAL**: Uses `scylla-driver` package, NOT `cassandra-driver`
  - `scylla-driver` is a fork of `cassandra-driver` optimized for ScyllaDB
  - Despite using `scylla-driver`, imports use `cassandra` namespace (e.g., `from cassandra.cluster import Cluster`)
  - There are implementation differences between the two drivers
  - **Always verify API compatibility**: Some cassandra-driver APIs may not exist or behave differently in scylla-driver
  - Example: `HostConnectionPool` class from `cassandra.cluster` is not available in scylla-driver
- **Imports**: Uses top-level imports from scylla-driver (Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT, PlainTextAuthProvider, WhiteListRoundRobinPolicy, ConstantReconnectionPolicy)
  - **No longer imports**: `HostConnectionPool` (cassandra-driver specific, not in scylla-driver)
- **Initialization**: Sets up keyspace, table, and vector index
  - Configures connection pool with `pool_size` and `max_requests_per_connection` parameters
  - Uses ExecutionProfile with WhiteListRoundRobinPolicy for load balancing
  - **CRITICAL**: Waits 5 seconds after creating vector index for it to become queryable
  - Cloud deployments may need longer initialization time (10-15 seconds)
  - If vector index queries fail with "ANN ordering by vector requires the column to be indexed", the index isn't ready yet
- **Key Methods**:
  - `get_cached_response()`: ANN search for similar prompts
    - Includes specific error handling for vector index not ready
    - Returns user-friendly message: "Vector index not ready yet. Try again in a few seconds."
  - `cache_response()`: Store new prompt-response pairs
  - `close()`: Clean shutdown of database connection
- **Vector Index Initialization**: 
  - Wait time increased from 2 to 5 seconds (as of 2026-01)
  - Addresses cloud deployment latency and new keyspace creation
  - Benchmark tool adds additional verification step
- **Connection Pool Configuration**:
  - Default pool_size: 10 connections per host (configurable)
  - Default max_requests_per_connection: 1024 (configurable)
  - Uses `session.get_pool_state(host)` API to access pool configuration (compatible with scylla-driver)
  - Configures `pool.core_connections` and `pool.max_connections` per host dynamically
  - Caps at reasonable limits: core_connections ≤ 32, max_connections ≤ 64
  - Benchmark uses dynamic pool sizing (2x max concurrency level) and max_requests_per_connection=2048
- **Prepared Statement Caching**:
  - INSERT statement prepared once at initialization and cached
  - Eliminates re-parsing overhead under high concurrency
  - SELECT queries cannot be prepared (vector ANN limitation)

### `PgVectorCache` Class
- **Purpose**: Encapsulates all PostgreSQL pgvector operations
- **Initialization**: Sets up schema, table, and HNSW index with connection pooling
  - Uses AsyncConnectionPool for efficient concurrent operations
  - Configurable pool size (default: 10, min: 2)
  - Prepared statement caching for both SELECT and INSERT queries
- **Key Methods**:
  - `connect()`: Async connection pool establishment and database setup
  - `get_cached_response()`: HNSW vector search using prepared statements
  - `cache_response()`: Store new prompt-response pairs with prepared statements
  - `close()`: Clean async shutdown of connection pool
- **Vector Type Casting**: All vector parameters must use `::vector` casting in SQL queries
  - Required for distance operators: `<->` (L2), `<=>` (cosine), `<#>` (inner product), `<+>` (L1)
  - Failure to cast causes "operator does not exist: vector <=> double precision[]" errors
- **Connection Pool Configuration**:
  - Default pool_size: 10 connections (configurable)
  - Min size: 2, Max size: pool_size parameter
  - Prepared statements cached at instance level
- **Important**: All methods except `__init__` are async and must be awaited

### `async_main()` Function
- **Purpose**: Main application logic and orchestration
- **Responsibilities**:
  - Parse command-line arguments
  - Validate API keys
  - Conditionally initialize cache (ScyllaDB or pgvector)
  - Query Claude (with or without cache)
  - Display results
  - Handle async operations for pgvector

## Important Implementation Details

### Embedding Consistency
- The embedding model must match the vector dimension in database schemas
- Default: `all-MiniLM-L6-v2` produces 384-dimension vectors
- If changing models, ensure the `embedding_dim` in both cache classes matches

### Similarity Functions
- **Cosine**: Default, best for normalized embeddings (matches ScyllaDB behavior)
- **L2**: Euclidean distance
- **Inner Product**: Negative inner product (pgvector uses `<#>`)
- **L1**: Manhattan distance
- Configurable via `--similarity-function` argument

### Cache Key Generation
- Primary key: SHA256 hash of the prompt text
- Allows for exact duplicate detection
- Vector search provides semantic similarity matching

### Error Handling
- Cache errors should not prevent Claude queries
- Print informative error messages but continue execution
- Only fail fast on missing API keys
- **Vector Index Errors**: ScyllaDBCache includes specific handling for "ANN ordering by vector requires the column to be indexed" errors
  - Provides user-friendly message about waiting for index initialization
  - Common during first connection to new cluster or new keyspace creation

## When Making Changes

### Adding New Command-Line Options
1. Add argument to parser with appropriate `type`, `help`, and `default`
2. Create corresponding environment variable check
3. Ensure CLI arg takes precedence over env var
4. Update README.md with new option

### Modifying Cache Behavior
1. Consider impact on both cache hit and miss paths
2. Test with `--with-cache none` to ensure graceful degradation
3. Verify embedding dimension compatibility
4. Update database schema if needed
5. For pgvector changes, remember to use `await` for async methods
6. **For ScyllaDB**: Consider vector index initialization time (5 seconds default, may need more for cloud)

### Changing AI Models
1. Check model availability in Anthropic API
2. Update default value in argument parser
3. Test with both environment variable and CLI arg
4. Consider token limits and cost implications

### Modifying Vector Index Initialization
1. **Wait Time**: Currently 5 seconds in both AI agent and benchmark
2. **Cloud Deployments**: May need 10-15 seconds total for full initialization
3. **Verification**: Benchmark includes test query to verify index readiness
4. **Error Messages**: Should guide users to wait if index not ready
5. **Trade-offs**: Longer wait = more reliable, but slower startup for cached keyspaces

## Testing Recommendations

### Manual Testing Checklist
- [ ] Run with `--with-cache none` (no database required)
- [ ] Run with `--with-cache scylla` (requires ScyllaDB)
- [ ] Run with `--with-cache pgvector` (requires PostgreSQL)
- [ ] Test cache hit scenario (run same prompt twice with each backend)
- [ ] Test all CLI arguments override env vars
- [ ] Verify missing API key raises clear error
- [ ] Test with different Claude models
- [ ] Test with different SentenceTransformer models
- [ ] Test different similarity functions (cosine, l2, inner_product, l1)
- [ ] Test ScyllaDB vector index initialization (first connection to new cluster/keyspace)
- [ ] Verify vector index error messages are user-friendly

### Edge Cases to Consider
- Empty or very short prompts
- Extremely long prompts (token limits)
- ScyllaDB connection failures during cache write
- Network timeouts
- Invalid model names
- Non-ASCII characters in prompts
- Vector index not ready (first query to new ScyllaDB cluster/keyspace)
- Cloud deployment latency affecting index initialization

## Performance Optimization Guidelines

### When to Avoid Caching
- One-off queries
- Real-time data requests
- Prompts requiring latest information
- Testing/development iterations

### When Caching Shines
- Repeated similar queries
- FAQs or common questions
- Expensive/slow Claude API calls
- High-volume production use

### Optimization Opportunities
- ScyllaDB connection pooling is implemented and configurable (pool_size, max_requests_per_connection)
- Implement batch embedding generation
- Add cache TTL (time-to-live) for stale data
- Add cache statistics/monitoring

## Common Patterns

### Adding a New Cache Backend
1. Create new cache class implementing same interface as `ScyllaDBCache`
2. Add new choice to `--with-cache` argument
3. Add conditional initialization in `async_main()`
4. Ensure consistent embedding format across backends

### Error Message Format
```python
print(f"✓ Success message")  # Green checkmark for success
print(f"✗ Error/miss message")  # Red X for failures/misses
```

### Configuration Variable Pattern
```python
value = (
    args.option_name or 
    os.getenv("ENV_VAR_NAME", "default_value")
)
```

## Dependencies Management

### Core Dependencies
- `autogen-ext`: Anthropic client integration
- `autogen-core`: Message types and interfaces
- `scylla-driver`: ScyllaDB connectivity
  - **Important**: This is NOT `cassandra-driver`, but a fork optimized for ScyllaDB
  - Uses `cassandra` namespace internally (import with `from cassandra.cluster import ...`)
  - Some cassandra-driver APIs may not be available or may behave differently
  - Always verify API compatibility when using advanced features
- `psycopg[binary]`: PostgreSQL async driver
- `psycopg-pool`: Connection pooling for PostgreSQL (added for concurrency)
- `pgvector`: PostgreSQL vector extension support
- `sentence-transformers`: Embedding generation
- `numpy`: Array operations

### Optional Dependencies
- None currently, but consider adding:
  - `python-dotenv` for easier env var management
  - `rich` for better CLI output formatting
  - `click` for more advanced CLI features

## Security Considerations

- Never log or print API keys or passwords
- Use secure connections to ScyllaDB (TLS)
- VScyllaDB Cloud Management Tool

### Location
The `scylla-cloud/` subdirectory contains tooling for managing ScyllaDB Cloud clusters.

### Purpose
Provides a command-line interface for:
- Creating clusters with vector search support
- Destroying clusters
- Checking cluster status
- Retrieving connection information
- Managing local state of deployed clusters

### Architecture
- **Subcommand-based CLI**: Uses `create`, `destroy`, `status`, `info`, `list`, `list-clusters`, `get-account-info` subcommands
- **State Management**: Stores cluster information in `~/.scylla-clusters.json`
- **API Client**: Wraps ScyllaDB Cloud REST API (`https://api.cloud.scylladb.com`)
- **Output Formats**: Supports both human-readable text and JSON output

### Key Components

#### `ScyllaCloudClient` Class
- Handles all REST API interactions
- Methods for cluster CRUD operations
- **ID Lookup Methods**: Translates user-friendly names to API IDs
  - `lookup_cloud_provider_id()`: AWS/GCP → cloud provider ID
  - `lookup_region_id()`: Region name → region ID
  - `lookup_instance_type_id()`: Instance type name → instance type ID
- Manages authentication headers
- **Debug mode**: Full request/response logging for all API methods
  - Shows HTTP method and URL for every request
  - Logs request headers (Authorization header masked)
  - Displays complete response status, headers, and body
  - Helpful for diagnosing API changes or unexpected response formats

#### `StateManager` Class
- Persists cluster information locally
- Maps cluster names to IDs and configuration
- Ensures state consistency across operations

### Configuration Priority
Same as main tool:
1. Command-line arguments (highest)
2. Environment variables
3. Default values (lowest)

### Vector Search Support
- Clusters are created with vector search enabled by default (use `--disable-vector-search` to disable)
- Separate configuration for vector search nodes (count and instance type)
- Required for the semantic caching feature in the main AI agent

### Cluster Creation API Pattern
- **ID Translation Required**: The API requires numeric IDs, not string names
- **Three-Step Lookup Process**:
  1. Look up cloud provider ID from provider name (AWS/GCP)
  2. Look up region ID from region name using cloud provider ID
  3. Look up instance type IDs from instance type names using cloud provider ID + region ID
- **Request Body Structure**: Uses correct field names per API documentation
  - `clusterName` (not name): Cluster name
  - `cloudProviderId`, `regionId`, `instanceId` (not cloudProvider, region, nodeType)
  - `numberOfNodes` (not nodeCount)
  - `vectorSearch.defaultNodes`, `vectorSearch.defaultInstanceTypeId`, `vectorSearch.singleRack` (not nodeCount, nodeType)
  - `broadcastType`: PUBLIC or PRIVATE (default: PUBLIC)
  - `cidrBlock`: Required, defaults to 192.168.1.0/24, must be /16 or larger (only included when broadcastType is PRIVATE)
  - `replicationFactor`: Defaults to 3
  - `tablets`: "enforced" by default, "false" if disabled
  - `allowedIPs`: List of allowed IP addresses (default: ["0.0.0.0/0"]), each must be /16 or larger
  - `scyllaVersion`: ScyllaDB version (default: "2025.4.0")
  - `accountCredentialId`: Account credential ID (default: 3)
  - `alternatorWriteIsolation`: Alternator write isolation (default: "only_rmw_uses_lwt", only included when userApiInterface is ALTERNATOR)
  - `freeTier`: Boolean for free tier (default: false)
  - `promProxy`: Boolean for Prometheus proxy (default: false)
  - `userApiInterface`: User API interface (default: "CQL")
  - `enableDnsAssociation`: Boolean for DNS association (default: true)
  - `provisioning`: Provisioning type (default: "dedicated-vm")
  - `pu`: Processing units (default: 1, only included when freeTier is true)
  - `expiration`: Expiration (default: "0", only included when freeTier is true)
- **CIDR Validation**: Both `cidrBlock` and `allowedIPs` must be /16 or larger (larger prefix = smaller network)
- **Vector Search**: Entire object omitted when disabled (not `enabled: false`)
- **Vector Instance Type Lookup**: Uses `target=VECTOR_SEARCH` query parameter to get restricted set of vector-compatible instance types
- **Response Structure**: Create cluster returns nested structure:
  ```json
  {
    "data": {
      "requestId": 123456,
      "fields": {
        "clusterName": "...",
        "scyllaVersion": {"version": "2025.4.0", ...},
        "cloudProvider": {...},
        "region": {...},
        "instance": {...},
        ...
      }
    }
  }
  ```
  - Extract `requestId` from `data.requestId` (used as cluster identifier)
  - Extract cluster details from `data.fields`
  - State file stores both `cluster_id` (requestId) and full `response` for reference
- **API Endpoints Used**:
  - `/deployment/cloud-providers` - Get cloud provider list
  - `/deployment/cloud-provider/{id}/regions` - Get regions for provider
  - `/deployment/cloud-provider/{id}/region/{id}` - Get instance types for cluster nodes
  - `/deployment/cloud-provider/{id}/region/{id}?target=VECTOR_SEARCH` - Get instance types for vector search nodes
  - `/account/{accountId}/cluster` (POST) - Create cluster (returns requestId, not clusterId)
  - `/account/{accountId}/cluster/request/{requestId}` (GET) - Resolve request ID to cluster ID and details
  - `/account/{accountId}/cluster/{clusterId}` (GET) - Get cluster details by cluster ID
  - `/account/{accountId}/cluster/connect?clusterId={clusterId}` (GET) - Get connection information (credentials, DNS, IPs)
  - `/account/{accountId}/cluster/{clusterId}/delete` (POST) - Delete cluster (POST, not DELETE)
  - `/account/{accountId}/clusters` (GET) - List all clusters for account
  - `/account/default` (GET) - Get default account information
- **Request ID vs Cluster ID**: 
  - Create cluster returns `requestId` in `data.requestId`, not the actual `clusterId`
  - Use `/account/{accountId}/cluster/request/{requestId}` to resolve to actual cluster ID
  - Store both `request_id` and `cluster_id` in state for flexibility
  - Info command automatically resolves request IDs to cluster IDs if needed

### Integration with Main Tool
The deployment tool creates clusters that are then used by `ai_agent_with_cache.py`:
1. Create cluster with `deploy-scylla-cloud.py create --enable-vector-search`
2. Retrieve connection info with `deploy-scylla-cloud.py info`
3. Use connection details in `ai_agent_with_cache.py --scylla-contact-points ...`

### Error Handling
- Failed operations leave resources in place for inspection

### Important Implementation Fixes
- **Delete Cluster**: Use POST to `/account/{accountId}/cluster/{clusterId}/delete`, not DELETE to `/cluster/{clusterId}`
- **Connection Info**: Use dedicated `/account/{accountId}/cluster/connect?clusterId={clusterId}` endpoint
  - Returns connection-focused data: credentials, DNS hostnames, public/private IPs
  - Response structure: `{data: {broadcastType, credentials: {username, password}, connectDataCenters: [{dcName, publicIPs, privateIPs, dns}]}}`
  - Simpler parsing than full cluster details endpoint
- **ID Resolution**: Add `get_cluster_from_request()` method to resolve request IDs to cluster IDs
- **State Management**: Use `update_cluster()` method to update existing cluster state (e.g., after resolving request ID)
- Clear error messages with HTTP response details
- State remains consistent even on failures
- Debug mode (`--debug` flag) provides full request/response details for troubleshooting:
  - All API methods (create, get, list, delete, get_account_info, get_connection_info) include debug logging
  - Shows exact API endpoints called and response data structures
  - Helpful for diagnosing API changes or unexpected response formats

## PostgreSQL pgvector Docker Management Tool

### Location
The `postgres-pgvector-docker/` subdirectory contains tooling for managing local PostgreSQL instances with pgvector.

### Purpose
Provides a command-line interface for:
- Starting PostgreSQL containers with pgvector extension
- Stopping and restarting containers
- Destroying containers (with optional data removal)
- Checking container status
- Retrieving connection information
- Viewing container logs

### Architecture
- **Subcommand-based CLI**: Uses `start`, `stop`, `restart`, `destroy`, `status`, `info`, `logs` subcommands
- **Docker Integration**: Uses Docker SDK (python `docker` package) to query container state directly
- **No State File**: Container state queried from Docker API instead of maintaining separate state
- **Docker Compose**: Uses docker-compose.yml for container configuration
- **Named Volumes**: Data persisted in Docker named volumes (format: `{container-name}-data`)

### Key Components

#### `DockerManager` Class
- Wraps Docker SDK operations
- Methods for container queries and validation
- **get_container()**: Retrieve container by name
- **is_port_in_use()**: Check if port is already bound by another container

#### Container Lifecycle
- **start**: Creates and starts container (or starts existing stopped container)
- **stop**: Stops container but preserves data
- **restart**: Restarts running container
- **destroy**: Removes container, optionally removes data volumes

### Configuration
- **Container Name**: User-specified with default `pgvector-local`
- **Multiple Instances**: Supports multiple instances on different ports
- **PostgreSQL Version**: Configurable (12-18), default 18
- **Port**: Configurable, default 5432 (validates port is not in use)
- **Credentials**: Configurable user/password/database

### Docker Compose Configuration
- **Base file**: `docker-compose.yml` with environment variable placeholders
- **Image**: Uses official `pgvector/pgvector:pg{version}` images
- **Environment**: Passes POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, etc.
- **Health Check**: Built-in pg_isready check with 5s intervals
- **Auto-restart**: Container restarts unless explicitly stopped

### pgvector Extension Initialization
- **init.sql**: SQL script mounted at `/docker-entrypoint-initdb.d/01-init.sql`
- **Auto-execution**: Runs on first container start (PostgreSQL behavior)
- **Extension Creation**: `CREATE EXTENSION IF NOT EXISTS vector;`
- **Idempotent**: Safe to run multiple times

### Port Management
- **Validation**: Checks if port is in use before starting
- **Multi-instance**: Allows multiple containers on different ports
- **Error Handling**: Clear error message if port conflict detected

### Data Persistence
- **Volume Type**: Docker named volumes (not bind mounts)
- **Naming**: `{container-name}-data` (e.g., `pgvector-local-data`)
- **Lifecycle**: Survives container removal unless `--remove-volumes` flag used
- **Mount Point**: `/var/lib/postgresql/data` (PostgreSQL default)

### Integration with Main Tool
Provides local PostgreSQL alternative to ScyllaDB Cloud for development/testing:
1. Start PostgreSQL: `deploy-pgvector-docker.py start`
2. Get connection info: `deploy-pgvector-docker.py info`
3. Use connection string with AI agent for local testing

### Error Handling
- **Docker Connection**: Validates Docker daemon is running at startup
- **Container Conflicts**: Checks for existing containers and port conflicts
- **Health Checks**: Waits for container to be healthy after start (30s timeout)
- **Clear Messages**: User-friendly error messages for common issues

### Testing Recommendations for PostgreSQL pgvector Tool
- [ ] Test start with default configuration
- [ ] Test start with custom port, user, password, database
- [ ] Test start with different PostgreSQL versions (12-18)
- [ ] Test multiple instances on different ports
- [ ] Test port conflict detection
- [ ] Test stop preserves data
- [ ] Test restart existing container
- [ ] Test destroy without --remove-volumes (preserves data)
- [ ] Test destroy with --remove-volumes (removes data)
- [ ] Test status shows correct container info
- [ ] Test info returns correct connection details
- [ ] Test logs with --tail option
- [ ] Test logs with --follow option
- [ ] Verify pgvector extension is installed on first start
- [ ] Verify health check passes
- [ ] Test with Docker not running
- [ ] Test debug mode with --debug flag

## Performance Benchmark Tool

### Location
The `benchmark.py` script in the root directory provides comprehensive performance testing for cache backends.

### Purpose
Compares cache performance between ScyllaDB and PostgreSQL pgvector with multiple test scenarios:
- Cache hit performance (repeated queries)
- Semantic similarity matching (similar prompt detection)
- Cache miss performance (lookup + write)
- **Concurrency testing** (concurrent reads, writes, and mixed workloads)

### Architecture
- **Multi-scenario Testing**: Four distinct test cases covering different usage patterns
- **Concurrency Testing**: asyncio-based concurrent operations with semaphore-controlled concurrency levels
- **Flexible Backends**: Test pgvector, ScyllaDB, or both simultaneously
- **Detailed Metrics**: Reports p50, p95, p99, max, mean latencies, and QPS (queries per second)
- **Configurable Prompts**: Uses `benchmark_prompts.txt` for test data (200+ prompts, easily customizable)
- **Results Export**: Supports JSON and CSV output formats

### Key Components

#### Test Scenarios
1. **Cache Hit Performance**: Queries same prompt 100 times to measure pure retrieval speed
2. **Semantic Similarity Matching**: Tests if semantically similar prompts trigger cache hits (measures actual semantic caching value)
3. **Cache Miss Performance**: Measures lookup latency before Claude query + write latency after
4. **Concurrency Testing** (optional): Tests performance under concurrent load with configurable concurrency levels

#### Benchmark Functions
- `load_prompts()`: Loads prompts from text file, ignoring comments and empty lines
- `benchmark_cache_hits()`: Repeated queries with same embedding to measure consistent performance
- `benchmark_semantic_similarity()`: Tests cache hit rate for similar but not identical prompts
- `benchmark_cache_misses()`: Measures both lookup and write latencies for new prompts
- `benchmark_concurrent_reads()`: Tests concurrent cache lookups with asyncio.Semaphore for concurrency control
- `benchmark_concurrent_writes()`: Tests concurrent cache inserts to measure write contention
- `benchmark_mixed_workload()`: Tests realistic 80% read / 20% write ratio under concurrent load
- `calculate_percentiles()`: Computes p50, p95, p99, max, and mean from latency distributions
- `run_benchmark_suite()`: Orchestrates complete test suite for a single backend (including optional concurrency tests)
- `print_comparison_table()`: Formats side-by-side comparison of multiple backends (including concurrency results)
- `save_results()`: Exports results to JSON or CSV files

#### Concurrency Testing Implementation
- **asyncio-based**: Uses async/await with asyncio.Semaphore for precise concurrency control
- **ScyllaDB Handling**: Wraps synchronous ScyllaDB operations with `asyncio.to_thread()` for compatibility
- **PgVectorCache**: Already async-ready, no wrapping needed
- **Metrics**: Reports QPS, latency percentiles, and success/failure rates
- **Test Types**:
  - Concurrent reads: Multiple simultaneous cache lookups (tests read scalability)
  - Concurrent writes: Multiple simultaneous inserts (tests write contention and locking)
  - Mixed workload: 80% reads / 20% writes (tests real-world scenarios)

#### Prompt File Structure (`benchmark_prompts.txt`)
- Lines starting with `#` are comments (ignored)
- Empty lines are ignored
- **200+ prompts** organized into categories:
  - **Base prompts** (5 prompts): Initial cache population
  - **Similar variants** (5 prompts): Semantic similarity testing
  - **Diverse prompts** (15 prompts): Cache miss testing
  - **Programming concepts** (10 prompts): OOP, functional programming, design patterns
  - **Data structures** (10 prompts): Trees, graphs, algorithms
  - **Database questions** (10 prompts): SQL, NoSQL, ACID, CAP theorem
  - **Web development** (10 prompts): SSR, PWA, WebSocket, GraphQL
  - **Cloud and DevOps** (10 prompts): IaC, CI/CD, containers
  - **Security topics** (10 prompts): XSS, encryption, authentication
  - **Business and product** (10 prompts): Agile, MVP, OKRs
  - **Science and mathematics** (10 prompts): Calculus, probability, physics
  - **General knowledge** (10 prompts): Climate, economics, technology
  - **Short technical queries** (10 prompts): Git, API, HTTPS, JSON
  - **Long-form questions** (3 prompts): Complex multi-part questions
  - **Edge cases** (10 prompts): Tradeoffs, comparisons, optimizations
  - **ML/AI topics** (10 prompts): NLP, computer vision, neural networks

### Configuration
- **Backends**: `--backends pgvector|scylla|both` (default: both)
- **Prompts File**: `--prompts-file` (default: benchmark_prompts.txt)
- **Embedding Model**: `--sentence-transformer-model` (default: all-MiniLM-L6-v2)
- **Output Format**: `--output json|csv|none` (default: none)
- **Concurrency Options**:
  - `--concurrency-test`: Enable concurrency testing (flag)
  - `--concurrency-levels`: Comma-separated levels (default: "1,5,10,25")
  - `--concurrent-operations`: Total operations per test (default: 100)
- **Database Connection**: Same options as main AI agent tool

### Integration with Cache Backends
- **PostgreSQL pgvector**: Uses async connection, awaits all cache operations
- **ScyllaDB**: Uses synchronous connection, wrapped with `asyncio.to_thread()` for concurrency tests
- **Separate Keyspaces/Schemas**: Uses `llm_cache_benchmark` to avoid conflicts with production data

### Results Interpretation
- **Local vs Cloud**: Network latency dominates cloud-based backends (150x+ difference)
- **Semantic Hit Rate**: Indicates how well the cache matches similar prompts
- **p50 vs p99**: Shows consistency - large gaps indicate variable performance
- **Write Latency**: Important for cache miss scenarios (how fast can we populate cache)
- **QPS (Queries Per Second)**: Throughput at given concurrency level
- **Concurrency Scaling**: How latency and throughput change with increased concurrency
- **Mixed Workload Performance**: Realistic performance under typical read-heavy workloads

### Usage Examples
```bash
# Test both backends with comparison
./benchmark.py --backends both

# Test only local PostgreSQL
./benchmark.py --backends pgvector --postgres-password postgres

# Test ScyllaDB Cloud
./benchmark.py --backends scylla \
  --scylla-contact-points "node-0.example.com,node-1.example.com" \
  --scylla-user scylla --scylla-password "password"

# Save results for analysis
./benchmark.py --backends both --output json

# Use custom test prompts
./benchmark.py --prompts-file custom_prompts.txt

# Run concurrency tests with default levels (1, 5, 10, 25)
./benchmark.py --backends both --concurrency-test

# Run concurrency tests with custom levels
./benchmark.py --backends pgvector --concurrency-test \
  --concurrency-levels "1,10,50,100" \
  --concurrent-operations 200
```

### Performance Considerations
- **First Run**: Includes SentenceTransformer model loading time (~1-2 seconds)
- **Warmup**: Cache operations may be slower on first query (connection establishment)
- **Network Latency**: Cloud backends show 100-300ms latencies primarily due to network
- **Local Backends**: Should show <5ms latencies for cache hits
- **Semantic Matching**: Hit rate depends on similarity threshold and index configuration
- **Concurrency Overhead**: asyncio adds minimal overhead for I/O-bound operations
- **Connection Pooling**: Concurrency tests show real-world behavior with pool limits
- **GIL Limitations**: Python GIL doesn't affect I/O-bound database operations significantly

### Testing Recommendations for Benchmark Tool
- [ ] Test with both backends simultaneously
- [ ] Test with each backend individually
- [ ] Test with custom prompt files
- [ ] Verify JSON output format is valid
- [ ] Verify CSV output format is valid
- [ ] Test with different SentenceTransformer models
- [ ] Compare local vs cloud deployment performance
- [ ] Test with large prompt files (200+ prompts now available)
- [ ] Test with empty cache (cold start)
- [ ] Test with pre-populated cache (warm start)
- [ ] Test concurrency with different levels (1, 5, 10, 25, 50, 100)
- [ ] Verify QPS increases with concurrency (up to a point)
- [ ] Test connection pool limits under high concurrency
- [ ] Compare concurrent vs sequential performance

## Future Enhancement Ideas

### Performance Benchmark Tool
- Add memory usage tracking
- Add cache size vs performance analysis
- Support for custom similarity thresholds
- Add percentile graphs/visualizations
- Support for continuous monitoring mode
- Add warm-up phase before measurements
- Support for multiple embedding models comparison
- Add cost analysis (API calls saved, infrastructure cost)

### Main AI Agent
- Support for multiple cache backends (Redis, PostgreSQL + pgvector)
- Streaming responses for long Claude outputs
- Cache analytics and statistics
- Automatic cache warming
- Similarity threshold configuration
- Cache expiration policies
- Multi-turn conversation support
- Prompt templates and variables

### ScyllaDB Cloud Tool
- Wait/poll for cluster to become active
- Cluster backup and restore commands
- Cost estimation before cluster creation
- Bulk operations (create/destroy multiple clusters)
- Configuration templates/presets
- Cluster scaling (add/remove nodes)
- Monitoring and metrics retrieval
- Support for other ScyllaDB Cloud features (VPC peering, etc.)

### PostgreSQL pgvector Docker Tool
- Backup and restore commands
- Database initialization scripts (custom SQL on startup)
- Performance tuning options (shared_buffers, etc.)
- Connection pooling with pgbouncer
- Replica/standby container support
- Automatic schema migration support
- Integration with pg_dump/pg_restore

