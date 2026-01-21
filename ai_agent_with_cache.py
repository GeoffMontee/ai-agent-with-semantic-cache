#!/usr/bin/env python3
"""
Command-line utility using Autogen with Anthropic Claude, SentenceTransformer,
and ScyllaDB Cloud or PostgreSQL pgvector for vector-based caching.
"""

import argparse
import os
import asyncio
import hashlib
import time
from datetime import datetime
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import WhiteListRoundRobinPolicy, ConstantReconnectionPolicy
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import UserMessage
from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg_pool
from pgvector.psycopg import register_vector_async


class ScyllaDBCache:
    def __init__(self, contact_points, username, password, keyspace="llm_cache", table="llm_responses",
                 pool_size=10, max_requests_per_connection=1024):
        """
        Initialize ScyllaDB connection with vector search support.
        
        Args:
            contact_points: List of ScyllaDB nodes
            username: ScyllaDB username
            password: ScyllaDB password
            keyspace: Keyspace name
            table: Table name
            pool_size: Number of connections per host (default: 10, increase for high concurrency)
            max_requests_per_connection: Max concurrent requests per connection (default: 1024)
        """
        auth_provider = PlainTextAuthProvider(username=username, password=password)
        
        # Create execution profile with increased connection pool settings
        profile = ExecutionProfile(
            load_balancing_policy=WhiteListRoundRobinPolicy(contact_points),
            request_timeout=30.0,
        )
        
        # Configure cluster with proper connection pooling for concurrency
        # Set core and max connections per host
        
        self.cluster = Cluster(
            contact_points,
            auth_provider=auth_provider,
            execution_profiles={EXEC_PROFILE_DEFAULT: profile},
            protocol_version=4,
            # Connection pool configuration - core and max connections per host
            # This controls how many connections are established to each host
            # More connections = more concurrent operations
            connect_timeout=10,
            # Reconnection policy
            reconnection_policy=ConstantReconnectionPolicy(delay=1.0, max_attempts=10),
        )
        
        # Set connection pool size per host (affects number of connections created)
        # Default is 2-8, we increase for better concurrency
        self.cluster.connection_class.max_in_flight = max_requests_per_connection
        
        self.session = self.cluster.connect()
        
        # Configure session-level settings
        self.session.default_fetch_size = 5000
        
        # Set connection pool size dynamically after cluster is created
        # Note: In scylla-driver, connection pool configuration is done via ExecutionProfile
        # The pool_size and max_requests_per_connection are already set in the profile
        # Additional per-host configuration would require accessing internal _pools attribute
        # which is not part of the public API and may vary between driver versions
        
        self.keyspace = keyspace
        self.table = table
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.pool_size = pool_size
        self.max_requests_per_connection = max_requests_per_connection
        
        # Prepared statement cache (initialized after database setup)
        self._select_stmt = None
        self._insert_stmt = None
        
        self._setup_database()

    def _setup_database(self):
        """Create keyspace and table with vector column."""
        # Create keyspace
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': 3}}
            AND tablets = {{
                'enabled': false
            }}
        """)
        
        self.session.set_keyspace(self.keyspace)
        
        # Create table with vector column for embeddings
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                prompt_hash text PRIMARY KEY,
                prompt text,
                embedding vector<float, {self.embedding_dim}>,
                response text,
                created_at timestamp
            )
        """)
        
        # Create vector search index with cosine similarity
        self.session.execute(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS embedding_ann_index
            ON {self.keyspace}.{self.table}(embedding)
            USING 'vector_index'
            WITH OPTIONS = {{
                'similarity_function': 'COSINE'
            }}
        """)
        
        # Wait for the index to be ready (especially important for cloud deployments)
        print("Waiting for vector index to initialize...")
        time.sleep(5)
        
        # Prepare statements for better concurrency performance
        # Cache prepared statements at instance level to avoid re-preparing
        self._prepare_statements()
    
    def _prepare_statements(self):
        """Prepare frequently-used statements for better performance under concurrency."""
        # Note: Vector ANN queries cannot be prepared in ScyllaDB as they require
        # the vector to be embedded in the query. We prepare the INSERT statement only.
        
        # Prepare INSERT statement (used in cache_response)
        self._insert_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.{self.table} (prompt_hash, prompt, embedding, response, created_at)
            VALUES (?, ?, ?, ?, ?)
        """)

    def get_cached_response(self, embedding, threshold=0.95):
        """
        Retrieve cached response using vector similarity search.
        
        Args:
            embedding: Vector embedding of the prompt
            threshold: Similarity threshold (0-1) for cache hit
        
        Returns:
            Cached response or None
        """
        # Convert numpy array to list for CQL
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Query using ANN vector search
        query = f"""
            SELECT prompt, response
            FROM {self.keyspace}.{self.table}
            ORDER BY embedding ANN OF {embedding_list}
            LIMIT 1
        """
        
        try:
            result = self.session.execute(query)
            row = result.one()
            
            if row:
                # Since we're using ANN with COSINE similarity, the closest result is most similar
                # For exact match detection, we could check if prompt matches exactly
                print(f"✓ Cache hit!")
                return row.response
        except Exception as e:
            # Check if this is a vector index error
            if "ANN ordering by vector requires the column to be indexed" in str(e):
                print(f"✗ Vector index not ready yet. Try again in a few seconds.")
            else:
                print(f"Cache lookup error: {e}")
        
        return None

    def cache_response(self, prompt, embedding, response):
        """Store prompt, embedding, and response in cache."""
        # Generate hash for primary key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        # Convert numpy array to list for CQL
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Use cached prepared statement for better performance
        try:
            self.session.execute(
                self._insert_stmt,
                (prompt_hash, prompt, embedding_list, response, datetime.now())
            )
            print("✓ Response cached successfully")
        except Exception as e:
            print(f"Cache storage error: {e}")

    def close(self):
        """Close the database connection."""
        self.cluster.shutdown()


class PgVectorCache:
    def __init__(self, host="localhost", port=5432, user="postgres", password="", 
                 database="postgres", schema="llm_cache", table="llm_responses",
                 similarity_function="cosine", pool_size=10):
        """Initialize PostgreSQL connection with pgvector support."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.schema = schema
        self.table = table
        self.similarity_function = similarity_function
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.pool_size = pool_size
        self.conn_pool = None
        
        # Map similarity functions to pgvector operators and index ops
        self.similarity_ops = {
            "cosine": {"operator": "<=>", "index_ops": "vector_cosine_ops"},
            "l2": {"operator": "<->", "index_ops": "vector_l2_ops"},
            "inner_product": {"operator": "<#>", "index_ops": "vector_ip_ops"},
            "l1": {"operator": "<+>", "index_ops": "vector_l1_ops"},
        }
        
        if similarity_function not in self.similarity_ops:
            raise ValueError(f"Unknown similarity function: {similarity_function}. "
                           f"Supported: {list(self.similarity_ops.keys())}")
    
    async def connect(self):
        """Establish async connection pool to PostgreSQL."""
        # Build connection string
        conn_string = f"host={self.host} port={self.port} dbname={self.database} "
        conn_string += f"user={self.user}"
        if self.password:
            conn_string += f" password={self.password}"
        
        # Create connection pool for better concurrency
        # AsyncConnectionPool handles concurrent operations efficiently
        self.conn_pool = psycopg_pool.AsyncConnectionPool(
            conn_string,
            min_size=2,  # Minimum connections to keep open
            max_size=self.pool_size,  # Maximum concurrent connections
            timeout=30.0,
            max_waiting=0,  # Don't queue if pool exhausted (fail fast)
            open=False  # Don't open connections yet
        )
        
        # Open the connection pool
        await self.conn_pool.open()
        
        # Register pgvector extension on one connection
        async with self.conn_pool.connection() as conn:
            await register_vector_async(conn)
        
        await self._setup_database()
    
    async def _setup_database(self):
        """Create schema, table, and HNSW index with vector column."""
        async with self.conn_pool.connection() as conn:
            async with conn.cursor() as cur:
                # Create schema
                await cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                
                # Create pgvector extension if not exists
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create table with vector column for embeddings
                await cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.{self.table} (
                        prompt_hash TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        embedding vector({self.embedding_dim}),
                        response TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create HNSW index for vector similarity search
                index_ops = self.similarity_ops[self.similarity_function]["index_ops"]
                await cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
                    ON {self.schema}.{self.table}
                    USING hnsw (embedding {index_ops})
                """)
                
                await conn.commit()
    
    async def get_cached_response(self, embedding, threshold=0.95):
        """
        Retrieve cached response using vector similarity search.
        
        Args:
            embedding: Vector embedding of the prompt
            threshold: Similarity threshold (0-1) for cache hit (not used with HNSW ANN)
        
        Returns:
            Cached response or None
        """
        try:
            # Get operator for similarity function
            operator = self.similarity_ops[self.similarity_function]["operator"]
            
            # Use connection from pool
            async with self.conn_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Use parameterized query with vector casting
                    await cur.execute(
                        f"""
                        SELECT prompt, response
                        FROM {self.schema}.{self.table}
                        ORDER BY embedding {operator} %s::vector
                        LIMIT 1
                        """,
                        (embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,)
                    )
                    row = await cur.fetchone()
                    
                    if row:
                        print(f"✓ Cache hit!")
                        return row[1]  # Return response
        except Exception as e:
            print(f"Cache lookup error: {e}")
        
        return None
    
    async def cache_response(self, prompt, embedding, response):
        """Store prompt, embedding, and response in cache."""
        # Generate hash for primary key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        try:
            # Use connection from pool
            async with self.conn_pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Use parameterized query with vector casting
                    await cur.execute(
                        f"""
                        INSERT INTO {self.schema}.{self.table} (prompt_hash, prompt, embedding, response, created_at)
                        VALUES (%s, %s, %s::vector, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (prompt_hash) DO NOTHING
                        """,
                        (prompt_hash, prompt, embedding.tolist() if isinstance(embedding, np.ndarray) else embedding, response)
                    )
                    await conn.commit()
            print("✓ Response cached successfully")
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    async def close(self):
        """Close the database connection pool."""
        if self.conn_pool:
            await self.conn_pool.close()


def main():
    asyncio.run(async_main())


async def async_main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="CLI utility using Autogen with Anthropic Claude and ScyllaDB caching"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to send to Claude"
    )
    parser.add_argument(
        "--with-cache",
        type=str,
        choices=["none", "scylla", "pgvector"],
        default="scylla",
        help="Type of semantic cache to use: 'none' for no caching, 'scylla' for ScyllaDB, 'pgvector' for PostgreSQL with pgvector (default: scylla)"
    )
    parser.add_argument(
        "--scylla-contact-points",
        type=str,
        help="Comma-separated list of ScyllaDB contact points (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--scylla-user",
        type=str,
        help="ScyllaDB username (default: scylla)"
    )
    parser.add_argument(
        "--scylla-password",
        type=str,
        help="ScyllaDB password (default: empty string)"
    )
    parser.add_argument(
        "--scylla-keyspace",
        type=str,
        help="ScyllaDB keyspace name (default: llm_cache)"
    )
    parser.add_argument(
        "--scylla-table",
        type=str,
        help="ScyllaDB table name (default: llm_responses)"
    )
    parser.add_argument(
        "--postgres-host",
        type=str,
        help="PostgreSQL host (default: localhost)"
    )
    parser.add_argument(
        "--postgres-port",
        type=int,
        help="PostgreSQL port (default: 5432)"
    )
    parser.add_argument(
        "--postgres-user",
        type=str,
        help="PostgreSQL username (default: postgres)"
    )
    parser.add_argument(
        "--postgres-password",
        type=str,
        help="PostgreSQL password (default: empty string)"
    )
    parser.add_argument(
        "--postgres-database",
        type=str,
        help="PostgreSQL database name (default: postgres)"
    )
    parser.add_argument(
        "--postgres-schema",
        type=str,
        help="PostgreSQL schema name (default: llm_cache)"
    )
    parser.add_argument(
        "--postgres-table",
        type=str,
        help="PostgreSQL table name (default: llm_responses)"
    )
    parser.add_argument(
        "--similarity-function",
        type=str,
        choices=["cosine", "l2", "inner_product", "l1"],
        help="Vector similarity function (default: cosine). Options: cosine, l2, inner_product, l1"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (if not provided, uses ANTHROPIC_API_KEY environment variable)"
    )
    parser.add_argument(
        "--anthropic-api-model",
        type=str,
        help="Anthropic API model to use (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--sentence-transformer-model",
        type=str,
        help="SentenceTransformer model to use for embeddings (default: all-MiniLM-L6-v2)"
    )
    args = parser.parse_args()

    # Handle Anthropic API key
    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
    elif not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY must be set either via --anthropic-api-key or as an environment variable")

    # Get Anthropic model
    anthropic_model = (
        args.anthropic_api_model or 
        os.getenv("ANTHROPIC_API_MODEL", "claude-sonnet-4-5-20250929")
    )

    # Get SentenceTransformer model
    sentence_transformer_model = (
        args.sentence_transformer_model or
        os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    )

    # Get ScyllaDB connection details from command-line args or environment
    scylla_hosts = (
        args.scylla_contact_points or 
        os.getenv("SCYLLA_CONTACT_POINTS", "127.0.0.1")
    ).split(",")
    scylla_user = args.scylla_user or os.getenv("SCYLLA_USER", "scylla")
    scylla_password = args.scylla_password if args.scylla_password is not None else os.getenv("SCYLLA_PASSWORD", "")
    scylla_keyspace = args.scylla_keyspace or os.getenv("SCYLLA_KEYSPACE", "llm_cache")
    scylla_table = args.scylla_table or os.getenv("SCYLLA_TABLE", "llm_responses")

    # Get PostgreSQL connection details from command-line args or environment
    postgres_host = args.postgres_host or os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = args.postgres_port or int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user = args.postgres_user or os.getenv("POSTGRES_USER", "postgres")
    postgres_password = args.postgres_password if args.postgres_password is not None else os.getenv("POSTGRES_PASSWORD", "")
    postgres_database = args.postgres_database or os.getenv("POSTGRES_DATABASE", "postgres")
    postgres_schema = args.postgres_schema or os.getenv("POSTGRES_SCHEMA", "llm_cache")
    postgres_table = args.postgres_table or os.getenv("POSTGRES_TABLE", "llm_responses")
    
    # Get similarity function (applies to both ScyllaDB and pgvector)
    similarity_function = args.similarity_function or os.getenv("SIMILARITY_FUNCTION", "cosine")

    # Initialize cache if requested
    cache = None
    prompt_embedding = None
    
    if args.with_cache == "scylla":
        # Initialize the SentenceTransformer model
        print("Loading SentenceTransformer model...")
        embedder = SentenceTransformer(sentence_transformer_model)
        
        # Generate embedding for the prompt
        print("Generating embedding for prompt...")
        prompt_embedding = embedder.encode(args.prompt)
        print(f"Embedding dimension: {len(prompt_embedding)}")

        # Initialize ScyllaDB cache
        print("Connecting to ScyllaDB...")
        cache = ScyllaDBCache(
            contact_points=scylla_hosts,
            username=scylla_user,
            password=scylla_password,
            keyspace=scylla_keyspace,
            table=scylla_table
        )
    
    elif args.with_cache == "pgvector":
        # Initialize the SentenceTransformer model
        print("Loading SentenceTransformer model...")
        embedder = SentenceTransformer(sentence_transformer_model)
        
        # Generate embedding for the prompt
        print("Generating embedding for prompt...")
        prompt_embedding = embedder.encode(args.prompt)
        print(f"Embedding dimension: {len(prompt_embedding)}")

        # Initialize PostgreSQL pgvector cache
        print("Connecting to PostgreSQL...")
        cache = PgVectorCache(
            host=postgres_host,
            port=postgres_port,
            user=postgres_user,
            password=postgres_password,
            database=postgres_database,
            schema=postgres_schema,
            table=postgres_table,
            similarity_function=similarity_function
        )
        await cache.connect()

    try:
        # Check cache first (if enabled)
        cached_response = None
        if cache:
            print("\nChecking cache...")
            if args.with_cache == "pgvector":
                cached_response = await cache.get_cached_response(prompt_embedding)
            else:
                cached_response = cache.get_cached_response(prompt_embedding)
        
        if cached_response:
            response_text = cached_response
            print("\n[Using cached response]")
        else:
            if cache:
                print("✗ Cache miss - querying Claude...")
            else:
                print("\nQuerying Claude (caching disabled)...")
            
            # Initialize the Anthropic client
            client = AnthropicChatCompletionClient(
                model=anthropic_model
            )

            # Send the prompt to Claude (await the async call)
            response = await client.create(
                messages=[UserMessage(content=args.prompt, source="user")]
            )
            
            # Extract response text
            response_text = response.content
            
            # Cache the result (if caching is enabled)
            if cache:
                if args.with_cache == "pgvector":
                    await cache.cache_response(args.prompt, prompt_embedding, response_text)
                else:
                    cache.cache_response(args.prompt, prompt_embedding, response_text)

        # Print the result
        print("\nClaude's response:")
        print("-" * 80)
        print(response_text)
        print("-" * 80)

    finally:
        if cache:
            if args.with_cache == "pgvector":
                await cache.close()
            else:
                cache.close()


if __name__ == "__main__":
    main()

