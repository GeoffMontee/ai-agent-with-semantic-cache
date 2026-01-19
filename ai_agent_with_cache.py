#!/usr/bin/env python3
"""
Command-line utility using Autogen with Anthropic Claude, SentenceTransformer,
and ScyllaDB Cloud or PostgreSQL pgvector for vector-based caching.
"""

import argparse
import os
import asyncio
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import UserMessage
from sentence_transformers import SentenceTransformer
import numpy as np
import time


class ScyllaDBCache:
    def __init__(self, contact_points, username, password, keyspace="llm_cache", table="llm_responses"):
        """Initialize ScyllaDB connection with vector search support."""
        auth_provider = PlainTextAuthProvider(username=username, password=password)
        self.cluster = Cluster(contact_points, auth_provider=auth_provider)
        self.session = self.cluster.connect()
        self.keyspace = keyspace
        self.table = table
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
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
        import hashlib
        from datetime import datetime
        
        # Generate hash for primary key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        # Convert numpy array to list for CQL
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Use prepared statement to avoid formatting issues
        insert_query = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.{self.table} (prompt_hash, prompt, embedding, response, created_at)
            VALUES (?, ?, ?, ?, ?)
        """)
        
        try:
            self.session.execute(
                insert_query,
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
                 similarity_function="cosine"):
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
        self.conn = None
        
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
        """Establish async connection to PostgreSQL."""
        import psycopg
        from pgvector.psycopg import register_vector_async
        
        # Build connection string
        conn_string = f"host={self.host} port={self.port} dbname={self.database} "
        conn_string += f"user={self.user}"
        if self.password:
            conn_string += f" password={self.password}"
        
        # Connect to PostgreSQL
        self.conn = await psycopg.AsyncConnection.connect(conn_string)
        
        # Register pgvector extension
        await register_vector_async(self.conn)
        
        await self._setup_database()
    
    async def _setup_database(self):
        """Create schema, table, and HNSW index with vector column."""
        async with self.conn.cursor() as cur:
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
            
            await self.conn.commit()
    
    async def get_cached_response(self, embedding, threshold=0.95):
        """
        Retrieve cached response using vector similarity search.
        
        Args:
            embedding: Vector embedding of the prompt
            threshold: Similarity threshold (0-1) for cache hit (not used with HNSW ANN)
        
        Returns:
            Cached response or None
        """
        # Get operator for similarity function
        operator = self.similarity_ops[self.similarity_function]["operator"]
        
        # Query using HNSW vector search (ORDER BY ... LIMIT 1 uses the index)
        query = f"""
            SELECT prompt, response
            FROM {self.schema}.{self.table}
            ORDER BY embedding {operator} %s::vector
            LIMIT 1
        """
        
        try:
            async with self.conn.cursor() as cur:
                await cur.execute(query, (embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,))
                row = await cur.fetchone()
                
                if row:
                    print(f"✓ Cache hit!")
                    return row[1]  # Return response
        except Exception as e:
            print(f"Cache lookup error: {e}")
        
        return None
    
    async def cache_response(self, prompt, embedding, response):
        """Store prompt, embedding, and response in cache."""
        import hashlib
        
        # Generate hash for primary key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        query = f"""
            INSERT INTO {self.schema}.{self.table} (prompt_hash, prompt, embedding, response, created_at)
            VALUES (%s, %s, %s::vector, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (prompt_hash) DO NOTHING
        """
        
        try:
            async with self.conn.cursor() as cur:
                await cur.execute(query, (prompt_hash, prompt, embedding.tolist() if isinstance(embedding, np.ndarray) else embedding, response))
                await self.conn.commit()
            print("✓ Response cached successfully")
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()


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

