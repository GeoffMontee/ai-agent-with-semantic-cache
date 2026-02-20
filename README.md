# AI Agent with Semantic Cache

A command-line utility that uses Anthropic's Claude AI with optional semantic caching powered by ScyllaDB Cloud or PostgreSQL pgvector with vector search. The semantic cache uses SentenceTransformer embeddings to identify similar prompts and return cached responses, reducing API calls and improving response times.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setting Up Cache Backends](#setting-up-cache-backends)
  - [Option 1: PostgreSQL pgvector (Recommended for Development)](#option-1-postgresql-pgvector-recommended-for-development)
  - [Option 2: ScyllaDB Cloud (Recommended for Production)](#option-2-scylladb-cloud-recommended-for-production)
- [Usage](#usage)
  - [Basic Usage (No Caching)](#basic-usage-no-caching)
  - [With PostgreSQL pgvector Caching](#with-postgresql-pgvector-caching)
  - [With ScyllaDB Caching](#with-scylladb-caching)
  - [Advanced: Similarity and Model Configuration](#advanced-similarity-and-model-configuration)
  - [Using Environment Variables](#using-environment-variables)
- [Command-Line Options](#command-line-options)
  - [Required](#required)
  - [Cache Configuration](#cache-configuration)
  - [PostgreSQL Configuration](#postgresql-configuration)
  - [ScyllaDB Configuration](#scylladb-configuration)
  - [Vector Similarity Configuration](#vector-similarity-configuration)
  - [Cache TTL Behavior (Current CLI)](#cache-ttl-behavior-current-cli)
  - [AI Model Configuration](#ai-model-configuration)
- [Environment Variables](#environment-variables)
  - [General Environment Variables](#general-environment-variables)
  - [PostgreSQL Configuration Environment Variables](#postgresql-configuration-environment-variables)
  - [ScyllaDB Configuration Environment Variables](#scylladb-configuration-environment-variables)
- [How Semantic Caching Works](#how-semantic-caching-works)
- [Database Schema](#database-schema)
  - [PostgreSQL pgvector](#postgresql-pgvector)
  - [ScyllaDB](#scylladb)
- [Demo](#demo)
  - [PostgreSQL Demo](#postgresql-demo)
  - [ScyllaDB Demo](#scylladb-demo)
- [Examples](#examples)
  - [Comparing Cache Performance](#comparing-cache-performance)
  - [PostgreSQL Cache Cleanup](#postgresql-cache-cleanup)
  - [Using Different Similarity Functions](#using-different-similarity-functions)
  - [Using Different Models](#using-different-models)
  - [Running Without Cache](#running-without-cache)
- [Performance Considerations](#performance-considerations)
  - [PostgreSQL pgvector](#postgresql-pgvector-1)
  - [ScyllaDB](#scylladb-1)
- [Cache Behavior](#cache-behavior)
  - [TTL (Time-to-Live)](#ttl-time-to-live)
  - [Similarity Threshold](#similarity-threshold)
  - [When Cache Hits Occur](#when-cache-hits-occur)
  - [When Cache Misses Occur](#when-cache-misses-occur)
- [Troubleshooting](#troubleshooting)
  - [API Key Issues](#api-key-issues)
  - [PostgreSQL Connection Issues](#postgresql-connection-issues)
  - [ScyllaDB Connection Issues](#scylladb-connection-issues)
  - [Vector Index Not Ready](#vector-index-not-ready)
- [Benchmarking Cache Performance](#benchmarking-cache-performance)
  - [Benchmark Scenarios](#benchmark-scenarios)
  - [Benchmark Options](#benchmark-options)
  - [Concurrency Testing](#concurrency-testing)
  - [Customizing Test Prompts](#customizing-test-prompts)
  - [Sample Results](#sample-results)
  - [Common Benchmark Issues](#common-benchmark-issues)
  - [Index Initialization Details](#index-initialization-details)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- 🤖 **Claude AI Integration**: Uses Autogen with Anthropic's Claude models
- 🔍 **Semantic Caching**: Vector-based caching with ScyllaDB or PostgreSQL pgvector for similar prompt detection
- ⏱️ **TTL Support**: 1-hour default TTL (auto-expiration in ScyllaDB, query-time filtering in PostgreSQL)
- 🎯 **Similarity Threshold**: Enforced 0.95 default threshold to control cache hit quality
- ⚡ **Fast Retrieval**: Cosine similarity search using HNSW indexes
- 🎛️ **Flexible Configuration**: Command-line arguments or environment variables
- 🔧 **Customizable Models**: Configure both Claude and SentenceTransformer models
- 📊 **Cache Control**: Enable/disable caching on demand
- 🗄️ **Multiple Backends**: Choose between ScyllaDB Cloud or PostgreSQL pgvector
- 🧹 **Automatic Cleanup**: PostgreSQL supports manual cleanup of expired entries

## Project Structure

The project consists of four main components:

1. **AI Agent** (`ai_agent_with_cache.py`) - Main CLI tool for querying Claude with semantic caching
2. **Performance Benchmark** (`benchmark.py`) - Compare cache performance between backends
   - Comprehensive benchmark suite with multiple test scenarios
   - Measures cache hits, semantic similarity, cache misses, and scale
   - Exports results to JSON or CSV format
3. **ScyllaDB Cloud Management** (`scylla-cloud/`) - Deployment tool for managing ScyllaDB Cloud clusters
   - `deploy-scylla-cloud.py` - Create, destroy, and manage clusters with vector search
   - See [scylla-cloud/README.md](scylla-cloud/README.md) for detailed documentation
4. **PostgreSQL pgvector Docker Management** (`postgres-pgvector-docker/`) - Local PostgreSQL with pgvector
   - `deploy-pgvector-docker.py` - Manage local PostgreSQL containers with pgvector
   - See [postgres-pgvector-docker/README.md](postgres-pgvector-docker/README.md) for detailed documentation

## Requirements

- Python 3.8+
- Cache backend (optional, only needed when using semantic caching):
  - ScyllaDB Cloud cluster, OR
  - PostgreSQL with pgvector extension (can use the included Docker tool)
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-agent-with-semantic-cache
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install autogen-ext autogen-core sentence-transformers scylla-driver psycopg[binary] pgvector numpy anthropic
```

**Note on ScyllaDB Driver**: This project uses `scylla-driver` (not `cassandra-driver`). The scylla-driver is a fork of cassandra-driver optimized for ScyllaDB, but it still uses the `cassandra` namespace internally. When importing, use `from cassandra.cluster import ...` even though the package is `scylla-driver`.

3. Set up your environment variables (optional):
```bash
export ANTHROPIC_API_KEY="your-api-key"

# For ScyllaDB:
export SCYLLA_CONTACT_POINTS="your-scylla-host.com"
export SCYLLA_USER="your-username"
export SCYLLA_PASSWORD="your-password"

# For PostgreSQL:
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
```

## Setting Up Cache Backends

### Option 1: PostgreSQL pgvector (Recommended for Development)

Use the included Docker management tool for quick local setup:

```bash
cd postgres-pgvector-docker

# Start PostgreSQL with pgvector
./deploy-pgvector-docker.py start --name pgvector-cache

# Check status
./deploy-pgvector-docker.py status --name pgvector-cache

# Get connection info
./deploy-pgvector-docker.py info --name pgvector-cache
```

For detailed documentation, see [postgres-pgvector-docker/README.md](postgres-pgvector-docker/README.md).

### Option 2: ScyllaDB Cloud (Recommended for Production)

Deploy a ScyllaDB Cloud cluster with vector search:

```bash
cd scylla-cloud

# Set your ScyllaDB Cloud API key
export SCYLLA_CLOUD_API_KEY="your-cloud-api-key"

# Create a cluster with vector search
./deploy-scylla-cloud.py create \
  --name my-vector-cache \
  --cloud-provider AWS \
  --region us-east-1

# Check status
./deploy-scylla-cloud.py status --name my-vector-cache

# Get connection information
./deploy-scylla-cloud.py info --name my-vector-cache --format json
```

For detailed documentation, see [scylla-cloud/README.md](scylla-cloud/README.md).

## Usage

### Basic Usage (No Caching)

```bash
./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache none \
  --anthropic-api-key "your-api-key"
```

### With PostgreSQL pgvector Caching

```bash
./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache pgvector \
  --anthropic-api-key "your-api-key" \
  --postgres-host "localhost" \
  --postgres-port 5432 \
  --postgres-user "postgres" \
  --postgres-password "postgres"
```

### With ScyllaDB Caching

```bash
./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache scylla \
  --anthropic-api-key "your-api-key" \
  --scylla-contact-points "your-host.com" \
  --scylla-user "your-username" \
  --scylla-password "your-password"
```

### Advanced: Similarity and Model Configuration

```bash
# Use L2 distance with pgvector
./ai_agent_with_cache.py \
  --prompt "Explain machine learning" \
  --with-cache pgvector \
  --similarity-function l2

# Use custom Claude and embedding models
./ai_agent_with_cache.py \
  --prompt "Summarize this article" \
  --with-cache scylla \
  --anthropic-api-model "claude-opus-4-20250514" \
  --sentence-transformer-model "paraphrase-MiniLM-L6-v2"
```

**Note**: The current CLI uses a fixed cache TTL of `3600` seconds and a fixed similarity threshold of `0.95`.

### Using Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key"

# For PostgreSQL pgvector:
export POSTGRES_HOST="localhost"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
./ai_agent_with_cache.py --prompt "What is the capital of France?" --with-cache pgvector

# For ScyllaDB:
export SCYLLA_CONTACT_POINTS="your-host.com"
export SCYLLA_USER="your-username"
export SCYLLA_PASSWORD="your-password"
./ai_agent_with_cache.py --prompt "What is the capital of France?" --with-cache scylla
```

## Command-Line Options

### Required
- `--prompt`: The prompt to send to Claude

### Cache Configuration
- `--with-cache {none,scylla,pgvector}`: Type of semantic cache (default: `scylla`)
  - `none`: Disable caching entirely
  - `scylla`: Use ScyllaDB with vector search
  - `pgvector`: Use PostgreSQL with pgvector extension

### PostgreSQL Configuration
- `--postgres-host`: PostgreSQL host (default: `localhost`)
- `--postgres-port`: PostgreSQL port (default: `5432`)
- `--postgres-user`: PostgreSQL username (default: `postgres`)
- `--postgres-password`: PostgreSQL password (default: empty string)
- `--postgres-database`: PostgreSQL database name (default: `postgres`)
- `--postgres-schema`: PostgreSQL schema name (default: `llm_cache`)
- `--postgres-table`: PostgreSQL table name (default: `llm_responses`)

### ScyllaDB Configuration
- `--scylla-contact-points`: Comma-separated list of ScyllaDB hosts (default: `127.0.0.1`)
- `--scylla-user`: ScyllaDB username (default: `scylla`)
- `--scylla-password`: ScyllaDB password (default: empty string)
- `--scylla-keyspace`: Keyspace name (default: `llm_cache`)
- `--scylla-table`: Table name (default: `llm_responses`)

### Vector Similarity Configuration
- `--similarity-function {cosine,l2,inner_product,l1}`: Vector similarity function for PostgreSQL pgvector (default: `cosine`)
  - `cosine`: Cosine distance (default, best for normalized embeddings)
  - `l2`: Euclidean (L2) distance
  - `inner_product`: Negative inner product
  - `l1`: Manhattan (L1) distance
- **ScyllaDB note**: ScyllaDB currently uses cosine similarity for ANN index and threshold checks.

### Cache TTL Behavior (Current CLI)
- **Default TTL**: `3600` seconds (1 hour)
- **ScyllaDB**: Automatic deletion after TTL expires (`USING TTL`)
- **PostgreSQL**: Time-based filtering in queries; expired rows can be removed with `cleanup_expired()`
- **Current limitation**: TTL is not exposed as a CLI flag or environment variable.

### AI Model Configuration
- `--anthropic-api-key`: Anthropic API key (overrides `ANTHROPIC_API_KEY` env var)
- `--anthropic-api-model`: Claude model to use (default: `claude-sonnet-4-5-20250929`)
- `--sentence-transformer-model`: Embedding model (default: `all-MiniLM-L6-v2`)

## Environment Variables

The following environment variables are supported by the CLI:

### General Environment Variables
- `ANTHROPIC_API_KEY`: Anthropic API key
- `ANTHROPIC_API_MODEL`: Claude model name
- `SENTENCE_TRANSFORMER_MODEL`: SentenceTransformer model name
- `SIMILARITY_FUNCTION`: Vector similarity function for PostgreSQL pgvector

### PostgreSQL Configuration Environment Variables
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DATABASE`: PostgreSQL database name
- `POSTGRES_SCHEMA`: PostgreSQL schema name
- `POSTGRES_TABLE`: PostgreSQL table name

### ScyllaDB Configuration Environment Variables
- `SCYLLA_CONTACT_POINTS`: ScyllaDB hosts
- `SCYLLA_USER`: ScyllaDB username
- `SCYLLA_PASSWORD`: ScyllaDB password
- `SCYLLA_KEYSPACE`: ScyllaDB keyspace
- `SCYLLA_TABLE`: ScyllaDB table

**Note**: Command-line arguments always take precedence over environment variables. Similarity threshold and TTL are currently fixed in the CLI.

## How Semantic Caching Works

1. **Embedding Generation**: When you submit a prompt, the tool generates a 384-dimension vector embedding using SentenceTransformer
2. **Vector Search**:
   - **ScyllaDB**: Uses ANN vector search with cosine similarity
   - **PostgreSQL**: Uses the selected similarity function (`--similarity-function` / `SIMILARITY_FUNCTION`)
3. **Similarity Threshold**: Results are filtered by a fixed threshold of `0.95`
   - **ScyllaDB**: Calculates cosine similarity in Python after retrieval
   - **PostgreSQL**: Converts threshold to distance and filters in SQL
4. **TTL Filtering**: Expired entries are excluded using a fixed TTL of `3600` seconds
   - **ScyllaDB**: Uses native TTL (automatic deletion)
   - **PostgreSQL**: Filters by `created_at` timestamp in queries
5. **Cache Hit/Miss**:
   - **Hit**: If a similar prompt above threshold is found, the cached response is returned instantly
   - **Miss**: The prompt is sent to Claude, and both the embedding and response are cached with TTL
6. **Storage**: Cached entries include the prompt text, embedding vector, response, and timestamp

## Database Schema

### PostgreSQL pgvector

The tool automatically creates the following PostgreSQL schema:

```sql
CREATE SCHEMA llm_cache;

CREATE EXTENSION vector;

CREATE TABLE llm_cache.llm_responses (
    prompt_hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    embedding vector(384),
    response TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX llm_responses_embedding_idx
ON llm_cache.llm_responses
USING hnsw (embedding vector_cosine_ops);
```

### ScyllaDB

The tool automatically creates the following ScyllaDB schema:

```cql
CREATE KEYSPACE llm_cache
WITH replication = {'class': 'NetworkTopologyStrategy', 'replication_factor': 3};

CREATE TABLE llm_responses (
    prompt_hash text PRIMARY KEY,
    prompt text,
    embedding vector<float, 384>,
    response text,
    created_at timestamp
);

CREATE CUSTOM INDEX embedding_ann_index
ON llm_cache.llm_responses(embedding)
USING 'vector_index'
WITH OPTIONS = {'similarity_function': 'COSINE'};
```
## Demo

### PostgreSQL Demo

The following demo shows how the semantic cache works with PostgreSQL and pgVector:

1. Create the database in Docker:

```bash
$ ./postgres-pgvector-docker/deploy-pgvector-docker.py start \
    --name pgvector-local
```

2. Setup some environment variables:

```bash
export ANTHROPIC_API_KEY="your-api-key"

# For PostgreSQL:
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"

# Suppresses some unnecessary messages
export TOKENIZERS_PARALLELISM=false
```

3. Ask Anthropic and the cache about the capital of France:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache pgvector
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to PostgreSQL...

Checking cache...
[-] Cache miss - querying Claude...
[+] Response cached successfully (TTL: 3600s)

Claude's response:
--------------------------------------------------------------------------------
The capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache miss, because it was the first time that we asked the question.

4. Ask Anthropic and the cache the same question again:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache pgvector
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to PostgreSQL...

Checking cache...
[+] Cache hit! (similarity: 1.0000, distance: 0.0000)

[Using cached response]

Claude's response:
--------------------------------------------------------------------------------
The capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache hit, because it was an exact match.

5. Ask Anthropic and the cache about the current capital of France, which is a different question with a similar meaning:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the current capital of France?" \
  --with-cache pgvector
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to PostgreSQL...

Checking cache...
[-] Cache miss - querying Claude...
[+] Response cached successfully (TTL: 3600s)

Claude's response:
--------------------------------------------------------------------------------
The current capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache miss, because the similarity (``0.9438``) was lower than the threshold (``0.95``).

6. Ask Anthropic and the cache about the capital of France right now, which is another different question with a similar meaning:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the capital of France right now?" \
  --with-cache pgvector
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to PostgreSQL...

Checking cache...
[+] Cache hit! (similarity: 0.9570, distance: 0.0430)

[Using cached response]

Claude's response:
--------------------------------------------------------------------------------
The current capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache hit, because the similarity (``0.9570``) was higher than the threshold (``0.95``).

7. Stop database container:

```bash
$ ./postgres-pgvector-docker/deploy-pgvector-docker.py stop \
    --name pgvector-local
```

8. Cleanup database container:

```bash
$ ./postgres-pgvector-docker/deploy-pgvector-docker.py destroy \
    --name pgvector-local \
    --remove-volumes
```

### ScyllaDB Demo

The following demo shows how the semantic cache works with ScyllaDB Cloud with Vector Search:

1. Create the cluster in ScyllaDB Cloud:

```bash
$ export SCYLLA_CLOUD_API_KEY="your-api-key"
$ ./scylla-cloud/deploy-scylla-cloud.py create \
   --name my-vector-cache \
   --allowed-ips "MY_IP"
```

2. Wait for the cluster to become available:

```bash
$ ./scylla-cloud/deploy-scylla-cloud.py status \
   --name my-vector-cache
```

3. Wait for the vector search nodes to become available.

The vector search nodes are provisioned after the cluster itself is ready, 
so wait a little longer for them to be ready.

4. Obtain connection information:

```bash
$ ./scylla-cloud/deploy-scylla-cloud.py info \
   --name my-vector-cache
```

5. Setup some environment variables:

```bash
export ANTHROPIC_API_KEY="your-api-key"

# For ScyllaDB:
export SCYLLA_CONTACT_POINTS="your-host.com"
export SCYLLA_USER="your-username"
export SCYLLA_PASSWORD="your-password"

# Suppresses some unnecessary messages
export TOKENIZERS_PARALLELISM=false
```

6. Ask Anthropic and the cache about the capital of France:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache scylla
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to ScyllaDB...
Waiting for vector index to initialize...

Checking cache...
[-] Cache miss - querying Claude...
[+] Response cached successfully (TTL: 3600s)

Claude's response:
--------------------------------------------------------------------------------
The capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache miss, because it was the first time that we asked the question.

7. Ask Anthropic and the cache the same question again:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache scylla
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to ScyllaDB...
Waiting for vector index to initialize...

Checking cache...
[+] Cache hit! (similarity: 1.0000)

[Using cached response]

Claude's response:
--------------------------------------------------------------------------------
The capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache hit, because it was an exact match.

8. Ask Anthropic and the cache about the current capital of France, which is a different question with a similar meaning:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the current capital of France?" \
  --with-cache scylla
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to ScyllaDB...
Waiting for vector index to initialize...

Checking cache...
[-] Similarity too low (0.9438 < 0.95)
[-] Cache miss - querying Claude...
[+] Response cached successfully (TTL: 3600s)

Claude's response:
--------------------------------------------------------------------------------
The current capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache miss, because the similarity (``0.9438``) was lower than the threshold (``0.95``).

9. Ask Anthropic and the cache about the capital of France right now, which is another different question with a similar meaning:

```bash
$ ./ai_agent_with_cache.py \
  --prompt "What is the capital of France right now?" \
  --with-cache scylla
Loading SentenceTransformer model...
Generating embedding for prompt...
Embedding dimension: 384
Connecting to ScyllaDB...
Waiting for vector index to initialize...

Checking cache...
[+] Cache hit! (similarity: 0.9570)

[Using cached response]

Claude's response:
--------------------------------------------------------------------------------
The current capital of France is Paris.
--------------------------------------------------------------------------------
```

This triggered a cache hit, because the similarity (``0.9570``) was higher than the threshold (``0.95``).

10. Cleanup cluster:

```bash
$ ./scylla-cloud/deploy-scylla-cloud.py destroy \
   --name my-vector-cache
```

## Examples

### Comparing Cache Performance

First run (cache miss):
```bash
./ai_agent_with_cache.py --prompt "Explain quantum computing"
# Output: [-] Cache miss - querying Claude...
# Response time: ~2-3 seconds
```

Second run (cache hit):
```bash
./ai_agent_with_cache.py --prompt "Explain quantum computing"
# Output: [+] Cache hit! (similarity: 1.0000)
# Response time: ~100-200ms
```

### PostgreSQL Cache Cleanup

PostgreSQL uses time-based filtering for TTL, so expired entries remain in the database until cleaned up. To remove expired entries:

```python
from ai_agent_with_cache import PgVectorCache

# Create cache instance
cache = PgVectorCache(
    host="localhost",
    user="postgres",
    password="postgres",
    ttl_seconds=3600  # 1 hour
)

# Connect and cleanup
await cache.connect()
await cache.cleanup_expired()  # Logs how many rows were removed (if any)
await cache.close()
```

You can schedule this as a periodic task (e.g., cron job) or run manually when needed. ScyllaDB does not need this as it automatically deletes expired entries.

### Using Different Similarity Functions

```bash
# Use L2 distance instead of cosine
./ai_agent_with_cache.py \
  --prompt "Explain machine learning" \
  --with-cache pgvector \
  --similarity-function l2

# Use inner product for normalized vectors
./ai_agent_with_cache.py \
  --prompt "Explain machine learning" \
  --with-cache pgvector \
  --similarity-function inner_product
```

### Using Different Models

```bash
# Use Claude Opus
./ai_agent_with_cache.py \
  --prompt "Write a poem" \
  --anthropic-api-model "claude-opus-4-20250514"

# Use different embedding model
./ai_agent_with_cache.py \
  --prompt "Summarize this text" \
  --sentence-transformer-model "paraphrase-MiniLM-L6-v2"
```

### Running Without Cache

```bash
./ai_agent_with_cache.py \
  --prompt "What's the weather like?" \
  --with-cache none
```

## Performance Considerations

### PostgreSQL pgvector
- **First Request**: Includes model loading time (~1-2 seconds for SentenceTransformer)
- **Cache Hit**: Typically 50-150ms (local PostgreSQL)
- **Cache Miss**: Depends on Claude API response time (~2-5 seconds)
- **Index Type**: Uses HNSW for better query performance
- **Embedding Dimension**: 384 for default model (all-MiniLM-L6-v2)
- **TTL Overhead**: Minimal - timestamp filtering in WHERE clause
- **Cleanup**: Manual via `cleanup_expired()` method (scheduled or on-demand)

### ScyllaDB
- **First Request**: Includes model loading time (~1-2 seconds for SentenceTransformer)
- **Cache Hit**: Typically 100-200ms (depends on ScyllaDB latency)
- **Cache Miss**: Depends on Claude API response time (~2-5 seconds)
- **Embedding Dimension**: 384 for default model (all-MiniLM-L6-v2)
- **TTL Overhead**: None - native TTL with automatic deletion
- **Cleanup**: Automatic - no maintenance required

## Cache Behavior

### TTL (Time-to-Live)
- **Default**: 3600 seconds (1 hour)
- **Current CLI behavior**: TTL is fixed at 3600 seconds
- **ScyllaDB**: Uses native `USING TTL` clause - entries automatically deleted after expiration
- **PostgreSQL**: Uses time-based filtering in queries - expired entries remain until cleanup

### Similarity Threshold
- **Default**: 0.95 (95% similarity for cosine)
- **Current CLI behavior**: Threshold is fixed at 0.95
- **ScyllaDB**: Calculates cosine similarity in Python after retrieval, filters results
- **PostgreSQL**: Converts to distance threshold, filters in SQL WHERE clause
- **Cache Output**: Shows similarity score on cache hits: `[+] Cache hit! (similarity: 0.9876)`

### When Cache Hits Occur
A cached response is returned when:
1. A semantically similar prompt is found (via vector search)
2. Similarity meets or exceeds the threshold (default: 0.95)
3. The entry has not expired (TTL check passes)

### When Cache Misses Occur
A new Claude query is made when:
1. No similar prompts found in cache
2. Similar prompts exist but similarity < threshold
3. All similar prompts have expired (past TTL)

## Troubleshooting

### API Key Issues
```
ValueError: ANTHROPIC_API_KEY must be set either via --anthropic-api-key or as an environment variable
```
**Solution**: Set the API key via command-line or environment variable.

### PostgreSQL Connection Issues
```
psycopg.OperationalError: connection failed
```
**Solution**: 
- Verify PostgreSQL is running: `./postgres-pgvector-docker/deploy-pgvector-docker.py status --name your-container`
- Check connection parameters (host, port, username, password)
- Ensure pgvector extension is installed

### ScyllaDB Connection Issues
```
NoHostAvailable: Unable to connect to any servers
```
**Solution**: Verify your ScyllaDB contact points, credentials, and network connectivity. If using ScyllaDB Cloud, ensure your cluster is active using `./scylla-cloud/deploy-scylla-cloud.py status --name your-cluster`.

### Vector Index Not Ready
```
✗ Vector index not ready yet. Try again in a few seconds.
```
or
```
Cache lookup error: Error from server: code=2200 [Invalid query] message="ANN ordering by vector requires the column to be indexed using 'vector_index'"
```
**Solution**: The vector index is still initializing. This is most common when:
- First connecting to a new ScyllaDB Cloud cluster
- Creating a new keyspace for the first time
- Cloud deployments with higher network latency

The tool automatically waits 5 seconds for index initialization. If you still see this error:
1. Wait 10-15 seconds and try your query again
2. For cloud deployments, initialization may take longer
3. Verify the cluster has vector search enabled: `./scylla-cloud/deploy-scylla-cloud.py info --name your-cluster`

## Benchmarking Cache Performance

Compare the performance of ScyllaDB and PostgreSQL pgvector backends using the included benchmark script:

**With local PostgreSQL:**
```bash
./benchmark.py --backends both \
  --postgres-password postgres \
  --scylla-contact-points "your-host.com" \
  --scylla-user scylla \
  --scylla-password "your-password"
```

**With remote PostgreSQL:**
```bash
./benchmark.py --backends both \
  --postgres-host db.example.com \
  --postgres-port 5432 \
  --postgres-user myuser \
  --postgres-password mypassword \
  --postgres-database mydb \
  --scylla-contact-points "node-0.scylla.cloud,node-1.scylla.cloud" \
  --scylla-user scylla \
  --scylla-password "your-password"
```

### Benchmark Scenarios

The benchmark tests four key scenarios:

1. **Cache Hit Performance**: Queries the same cached prompt 100 times to measure pure retrieval speed
2. **Semantic Similarity Matching**: Tests whether semantically similar prompts trigger cache hits
3. **Cache Miss Performance**: Measures lookup + write latency for new prompts
4. **Concurrency Testing** (optional): Tests read/write performance under concurrent load

### Benchmark Options

```bash
# Test only PostgreSQL (local)
./benchmark.py --backends pgvector

# Test PostgreSQL with remote instance
./benchmark.py --backends pgvector \
  --postgres-host db.example.com \
  --postgres-port 5432 \
  --postgres-user myuser \
  --postgres-password mypassword \
  --postgres-database mydb

# Test only ScyllaDB
./benchmark.py --backends scylla

# Save results to JSON
./benchmark.py --backends both --output json

# Use custom prompts
./benchmark.py --prompts-file my_prompts.txt
```

### Concurrency Testing

Test cache performance under concurrent load to measure throughput (QPS) and latency at different concurrency levels:

```bash
# Run concurrency tests with default levels (1, 5, 10, 25)
./benchmark.py --backends both \
  --concurrency-test \
  --postgres-password postgres \
  --scylla-contact-points "your-host.com" \
  --scylla-user scylla \
  --scylla-password "your-password"

# Custom concurrency levels and operation count
./benchmark.py --backends pgvector \
  --concurrency-test \
  --concurrency-levels "1,10,50,100" \
  --concurrent-operations 200
```

Concurrency tests measure:
- **Concurrent Reads**: Multiple simultaneous cache lookups (tests read scalability)
- **Concurrent Writes**: Multiple simultaneous cache inserts (tests write contention)
- **Mixed Workload**: Realistic 80% read / 20% write ratio (tests real-world performance)

Each test reports:
- **QPS (Queries Per Second)**: Throughput at the given concurrency level
- **Latency Percentiles**: p50, p95, p99 latencies under concurrent load
- **Success/Failure Rates**: Error rate under load

**Use Cases:**
- Comparing local vs remote database performance under load
- Determining optimal concurrency for your workload
- Identifying bottlenecks and scaling limits
- Testing connection pool configurations

### Customizing Test Prompts

Edit [benchmark_prompts.txt](benchmark_prompts.txt) to customize the prompts used in benchmarks. The file contains 1,189 prompts (excluding comments and blank lines) organized into categories:
- Base prompts (for cache population)
- Semantically similar variants (for similarity testing)
- Programming concepts, data structures, algorithms (200 prompts)
- Database, web development, cloud/DevOps (300 prompts)
- Security, ML/AI, business topics (250 prompts)
- Science, general knowledge, and miscellaneous (400+ prompts)
- Short queries, long-form questions, and edge cases (50+ prompts)

### Sample Results

Performance comparison (local PostgreSQL vs ScyllaDB Cloud):

| Metric | PostgreSQL pgvector | ScyllaDB Cloud |
|--------|---------------------|----------------|
| Cache Hit (p50) | 1.15ms | 169.39ms |
| Semantic Hit Rate | 100% | Varies |
| Cache Miss Write (p50) | ~0ms | 307.98ms |

**Note**: Network latency significantly impacts cloud-based backends. For fair comparisons, deploy both backends in the same environment (both local or both cloud).

### Common Benchmark Issues

**ScyllaDB shows 0% semantic similarity hit rate:**
- The vector index needs time to initialize (5-10 seconds)
- The benchmark automatically waits and verifies the index before testing
- If you see errors about "ANN ordering requires indexed column", wait longer and retry

**Different results between runs:**
- First run includes model loading time (~1-2 seconds for SentenceTransformer)
- Cold vs warm cache affects initial query performance
- Network conditions vary for cloud backends

### Index Initialization Details
**ScyllaDB**: The tool waits 5 seconds for index initialization. This is sufficient for most scenarios, but cloud deployments or large existing caches may require additional time. The ScyllaDB cache includes:
- Configurable connection pooling (default: 10 connections per host, 1024 max requests per connection)
- Prepared statement caching for INSERT operations
- Proper core and max connection pool configuration for optimal concurrent performance

**PostgreSQL**: HNSW indexes are created automatically and are immediately usable. For large datasets, you may want to create indexes after loading initial data for better performance. The PostgreSQL cache includes:
- AsyncConnectionPool with configurable size (default: 10 connections)
- Parameterized SQL queries for lookup and insert operations
- Connection pooling for efficient concurrent operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude AI
- [ScyllaDB](https://scylladb.com) for high-performance vector search
- [pgvector](https://github.com/pgvector/pgvector) for PostgreSQL vector similarity search
- [SentenceTransformers](https://www.sbert.net) for semantic embeddings
- [Autogen](https://microsoft.github.io/autogen/) for AI agent framework
