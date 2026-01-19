# AI Agent with Semantic Cache

A command-line utility that uses Anthropic's Claude AI with optional semantic caching powered by ScyllaDB Cloud or PostgreSQL pgvector with vector search. The semantic cache uses SentenceTransformer embeddings to identify similar prompts and return cached responses, reducing API calls and improving response times.

## Features

- 🤖 **Claude AI Integration**: Uses Autogen with Anthropic's Claude models
- 🔍 **Semantic Caching**: Vector-based caching with ScyllaDB or PostgreSQL pgvector for similar prompt detection
- ⚡ **Fast Retrieval**: Cosine similarity search using HNSW indexes
- 🎛️ **Flexible Configuration**: Command-line arguments or environment variables
- 🔧 **Customizable Models**: Configure both Claude and SentenceTransformer models
- 📊 **Cache Control**: Enable/disable caching on demand
- 🗄️ **Multiple Backends**: Choose between ScyllaDB Cloud or PostgreSQL pgvector

## Project Structure

The project consists of three main components:

1. **AI Agent** (`ai_agent_with_cache.py`) - Main CLI tool for querying Claude with semantic caching
2. **ScyllaDB Cloud Management** (`scylla-cloud/`) - Deployment tool for managing ScyllaDB Cloud clusters
   - `deploy-scylla-cloud.py` - Create, destroy, and manage clusters with vector search
   - See [scylla-cloud/README.md](scylla-cloud/README.md) for detailed documentation
3. **PostgreSQL pgvector Docker Management** (`postgres-pgvector-docker/`) - Local PostgreSQL with pgvector
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
  --enable-vector-search \
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
- `--similarity-function {cosine,l2,inner_product,l1}`: Vector similarity function (default: `cosine`)
  - `cosine`: Cosine distance (default, best for normalized embeddings)
  - `l2`: Euclidean (L2) distance
  - `inner_product`: Negative inner product
  - `l1`: Manhattan (L1) distance

### AI Model Configuration
- `--anthropic-api-key`: Anthropic API key (overrides `ANTHROPIC_API_KEY` env var)
- `--anthropic-api-model`: Claude model to use (default: `claude-sonnet-4-5-20250929`)
- `--sentence-transformer-model`: Embedding model (default: `all-MiniLM-L6-v2`)

## Environment Variables

All command-line options have corresponding environment variables:

### General
- `ANTHROPIC_API_KEY`: Anthropic API key
- `ANTHROPIC_API_MODEL`: Claude model name
- `SENTENCE_TRANSFORMER_MODEL`: SentenceTransformer model name
- `SIMILARITY_FUNCTION`: Vector similarity function

### PostgreSQL Configuration
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DATABASE`: PostgreSQL database name
- `POSTGRES_SCHEMA`: PostgreSQL schema name
- `POSTGRES_TABLE`: PostgreSQL table name

### ScyllaDB Configuration
- `SCYLLA_CONTACT_POINTS`: ScyllaDB hosts
- `SCYLLA_USER`: ScyllaDB username
- `SCYLLA_PASSWORD`: ScyllaDB password
- `SCYLLA_KEYSPACE`: ScyllaDB keyspace
- `SCYLLA_TABLE`: ScyllaDB table

**Note**: Command-line arguments always take precedence over environment variables.

## How Semantic Caching Works

1. **Embedding Generation**: When you submit a prompt, the tool generates a 384-dimension vector embedding using SentenceTransformer
2. **Vector Search**: The embedding is compared against cached embeddings using the selected similarity function (default: cosine distance)
3. **Cache Hit/Miss**:
   - **Hit**: If a similar prompt is found, the cached response is returned instantly
   - **Miss**: The prompt is sent to Claude, and both the embedding and response are cached
4. **Storage**: Cached entries include the prompt text, embedding vector, response, and timestamp

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

## Examples

### Comparing Cache Performance

First run (cache miss):
```bash
./ai_agent_with_cache.py --prompt "Explain quantum computing"
# Output: ✗ Cache miss - querying Claude...
# Response time: ~2-3 seconds
```

Second run (cache hit):
```bash
./ai_agent_with_cache.py --prompt "Explain quantum computing"
# Output: ✓ Cache hit!
# Response time: ~100-200ms
```

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

### ScyllaDB
- **First Request**: Includes model loading time (~1-2 seconds for SentenceTransformer)
- **Cache Hit**: Typically 100-200ms (depends on ScyllaDB latency)
- **Cache Miss**: Depends on Claude API response time (~2-5 seconds)
- **Embedding Dimension**: 384 for default model (all-MiniLM-L6-v2)

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
**ScyllaDB**: The tool waits 2 seconds for index initialization. For larger databases, you may need to increase this delay in the code.

**PostgreSQL**: HNSW indexes are created automatically. For large datasets, you may want to create indexes after loading initial data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude AI
- [ScyllaDB](https://scylladb.com) for high-performance vector search
- [pgvector](https://github.com/pgvector/pgvector) for PostgreSQL vector similarity search
- [SentenceTransformers](https://www.sbert.net) for semantic embeddings
- [Autogen](https://microsoft.github.io/autogen/) for AI agent framework
