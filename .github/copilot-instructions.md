# GitHub Copilot Instructions

## Project Overview

This is a Python CLI tool that provides an AI agent interface with semantic caching capabilities. The project integrates:
- **Anthropic Claude** (via Autogen) for AI completions
- **ScyllaDB Cloud** for vector-based semantic caching
- **SentenceTransformers** for generating embeddings

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
- `cassandra-driver`: ScyllaDB/Cassandra connectivity
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
- Validate and sanitize user inputs
- Consider implementing rate limiting
- Be cautious with caching sensitive information

## Future Enhancement Ideas

- Support for multiple cache backends (Redis, PostgreSQL + pgvector)
- Streaming responses for long Claude outputs
- Cache analytics and statistics
- Automatic cache warming
- Similarity threshold configuration
- Cache expiration policies
- Multi-turn conversation support
- Prompt templates and variables
