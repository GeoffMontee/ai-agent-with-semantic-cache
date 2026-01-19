# GitHub Copilot Instructions

## Project Overview

This is a Python CLI tool that provides an AI agent interface with semantic caching capabilities. The project integrates:
- **Anthropic Claude** (via Autogen) for AI completions
- **ScyllaDB Cloud** for vector-based semantic caching
- **SentenceTransformers** for generating embeddings

The project consists of three main components:
1. **Main AI Agent** (`ai_agent_with_cache.py`) - Queries Claude with optional semantic caching
2. **ScyllaDB Cloud Management** (`scylla-cloud/deploy-scylla-cloud.py`) - Manages ScyllaDB Cloud clusters
3. **PostgreSQL pgvector Docker Management** (`postgres-pgvector-docker/deploy-pgvector-docker.py`) - Manages local PostgreSQL instances with pgvector

## Key Architecture Patterns

### 1. Conditional Caching
- The tool supports running with or without semantic caching via the `--with-cache` flag
- When caching is disabled (`--with-cache none`), ScyllaDB and SentenceTransformer dependencies are not initialized
- When enabled (`--with-cache scylla`), the full vector search pipeline is activated

### 2. Configuration Priority
All configuration follows this precedence order:
1. Command-line arguments (highest priority)
2. Environment variables
3. Default values (lowest priority)

### 3. Vector Search Implementation
- Embeddings are 384-dimension vectors from SentenceTransformer models
- ScyllaDB uses cosine similarity for ANN (approximate nearest neighbor) search
- Cache hits return immediately without calling Claude API

## Code Style Guidelines

### Python Conventions
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Use f-strings for string formatting
- Prefer explicit error messages with context

### Async Patterns
- Main async logic is in `async_main()`
- Use `await` for Anthropic client calls
- Keep the synchronous `main()` wrapper minimal

### Database Operations
- Always use prepared statements for ScyllaDB inserts
- Convert numpy arrays to lists before sending to CQL
- Include proper error handling with informative messages

## Component Responsibilities

### `ScyllaDBCache` Class
- **Purpose**: Encapsulates all ScyllaDB operations
- **Initialization**: Sets up keyspace, table, and vector index
- **Key Methods**:
  - `get_cached_response()`: ANN search for similar prompts
  - `cache_response()`: Store new prompt-response pairs
  - `close()`: Clean shutdown of database connection

### `async_main()` Function
- **Purpose**: Main application logic and orchestration
- **Responsibilities**:
  - Parse command-line arguments
  - Validate API keys
  - Conditionally initialize cache
  - Query Claude (with or without cache)
  - Display results

## Important Implementation Details

### Embedding Consistency
- The embedding model must match the vector dimension in ScyllaDB schema
- Default: `all-MiniLM-L6-v2` produces 384-dimension vectors
- If changing models, ensure the `embedding_dim` in `ScyllaDBCache.__init__` matches

### Cache Key Generation
- Primary key: SHA256 hash of the prompt text
- Allows for exact duplicate detection
- Vector search provides semantic similarity matching

### Error Handling
- Cache errors should not prevent Claude queries
- Print informative error messages but continue execution
- Only fail fast on missing API keys

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

### Changing AI Models
1. Check model availability in Anthropic API
2. Update default value in argument parser
3. Test with both environment variable and CLI arg
4. Consider token limits and cost implications

## Testing Recommendations

### Manual Testing Checklist
- [ ] Run with `--with-cache none` (no database required)
- [ ] Run with `--with-cache scylla` (requires ScyllaDB)
- [ ] Test cache hit scenario (run same prompt twice)
- [ ] Test all CLI arguments override env vars
- [ ] Verify missing API key raises clear error
- [ ] Test with different Claude models
- [ ] Test with different SentenceTransformer models

### Edge Cases to Consider
- Empty or very short prompts
- Extremely long prompts (token limits)
- ScyllaDB connection failures during cache write
- Network timeouts
- Invalid model names
- Non-ASCII characters in prompts

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
- Consider connection pooling for ScyllaDB
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
  - Includes data parsing details for complex responses (e.g., get_connection_info shows raw cluster keys and values)

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
  - `/account/{accountId}/cluster` (POST) - Create cluster
  - `/account/{accountId}/cluster/{clusterId}` (GET) - Get cluster details
  - `/account/{accountId}/cluster/{clusterId}` (DELETE) - Delete cluster
  - `/account/{accountId}/clusters` (GET) - List all clusters for account
  - `/account/default` (GET) - Get default account information

### Integration with Main Tool
The deployment tool creates clusters that are then used by `ai_agent_with_cache.py`:
1. Create cluster with `deploy-scylla-cloud.py create --enable-vector-search`
2. Retrieve connection info with `deploy-scylla-cloud.py info`
3. Use connection details in `ai_agent_with_cache.py --scylla-contact-points ...`

### Error Handling
- Failed operations leave resources in place for inspection
- Clear error messages with HTTP response details
- State remains consistent even on failures
- Debug mode (`--debug` flag) provides full request/response details for troubleshooting:
  - All API methods (create, get, list, delete, get_account_info) include debug logging
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

## Future Enhancement Ideas

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

