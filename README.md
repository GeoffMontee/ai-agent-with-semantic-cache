# AI Agent with Semantic Cache

A command-line utility that uses Anthropic's Claude AI with optional semantic caching powered by ScyllaDB Cloud and vector search. The semantic cache uses SentenceTransformer embeddings to identify similar prompts and return cached responses, reducing API calls and improving response times.

## Features

- 🤖 **Claude AI Integration**: Uses Autogen with Anthropic's Claude models
- 🔍 **Semantic Caching**: Vector-based caching with ScyllaDB for similar prompt detection
- ⚡ **Fast Retrieval**: Cosine similarity search using ScyllaDB's vector indexing
- 🎛️ **Flexible Configuration**: Command-line arguments or environment variables
- 🔧 **Customizable Models**: Configure both Claude and SentenceTransformer models
- 📊 **Cache Control**: Enable/disable caching on demand

## Requirements

- Python 3.8+
- ScyllaDB Cloud cluster (optional, only needed when using semantic caching)
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-agent-with-semantic-cache
```

2. Install dependencies:
```bash
pip install autogen-ext autogen-core sentence-transformers cassandra-driver numpy anthropic
```

3. Set up your environment variables (optional):
```bash
export ANTHROPIC_API_KEY="your-api-key"
export SCYLLA_CONTACT_POINTS="your-scylla-host.com"
export SCYLLA_USER="your-username"
export SCYLLA_PASSWORD="your-password"
```

## Usage

### Basic Usage (No Caching)

```bash
./ai_agent_with_cache.py \
  --prompt "What is the capital of France?" \
  --with-cache none \
  --anthropic-api-key "your-api-key"
```

### With Semantic Caching

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
export SCYLLA_CONTACT_POINTS="your-host.com"
export SCYLLA_USER="your-username"
export SCYLLA_PASSWORD="your-password"

./ai_agent_with_cache.py --prompt "What is the capital of France?"
```

## Command-Line Options

### Required
- `--prompt`: The prompt to send to Claude

### Cache Configuration
- `--with-cache {none,scylla}`: Type of semantic cache (default: `scylla`)
  - `none`: Disable caching entirely
  - `scylla`: Use ScyllaDB with vector search

### ScyllaDB Configuration
- `--scylla-contact-points`: Comma-separated list of ScyllaDB hosts (default: `127.0.0.1`)
- `--scylla-user`: ScyllaDB username (default: `scylla`)
- `--scylla-password`: ScyllaDB password (default: empty string)
- `--scylla-keyspace`: Keyspace name (default: `llm_cache`)
- `--scylla-table`: Table name (default: `llm_responses`)

### AI Model Configuration
- `--anthropic-api-key`: Anthropic API key (overrides `ANTHROPIC_API_KEY` env var)
- `--anthropic-api-model`: Claude model to use (default: `claude-sonnet-4-5-20250929`)
- `--sentence-transformer-model`: Embedding model (default: `all-MiniLM-L6-v2`)

## Environment Variables

All command-line options have corresponding environment variables:

- `ANTHROPIC_API_KEY`: Anthropic API key
- `ANTHROPIC_API_MODEL`: Claude model name
- `SENTENCE_TRANSFORMER_MODEL`: SentenceTransformer model name
- `SCYLLA_CONTACT_POINTS`: ScyllaDB hosts
- `SCYLLA_USER`: ScyllaDB username
- `SCYLLA_PASSWORD`: ScyllaDB password
- `SCYLLA_KEYSPACE`: ScyllaDB keyspace
- `SCYLLA_TABLE`: ScyllaDB table

**Note**: Command-line arguments always take precedence over environment variables.

## How Semantic Caching Works

1. **Embedding Generation**: When you submit a prompt, the tool generates a 384-dimension vector embedding using SentenceTransformer
2. **Vector Search**: The embedding is compared against cached embeddings using cosine similarity in ScyllaDB
3. **Cache Hit/Miss**:
   - **Hit**: If a similar prompt is found, the cached response is returned instantly
   - **Miss**: The prompt is sent to Claude, and both the embedding and response are cached
4. **Storage**: Cached entries include the prompt text, embedding vector, response, and timestamp

## Database Schema

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

### ScyllaDB Connection Issues
```
NoHostAvailable: Unable to connect to any servers
```
**Solution**: Verify your ScyllaDB contact points, credentials, and network connectivity.

### Vector Index Not Ready
If you see errors about the vector index, the tool waits 2 seconds for index initialization. For larger databases, you may need to increase this delay in the code.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude AI
- [ScyllaDB](https://scylladb.com) for high-performance vector search
- [SentenceTransformers](https://www.sbert.net) for semantic embeddings
- [Autogen](https://microsoft.github.io/autogen/) for AI agent framework
