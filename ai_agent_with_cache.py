#!/usr/bin/env python3
"""
Command-line utility using Autogen with Anthropic Claude, SentenceTransformer,
and ScyllaDB Cloud for vector-based caching.
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
        
        # Wait a moment for the index to be ready
        print("Waiting for vector index to initialize...")
        time.sleep(2)

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
        choices=["none", "scylla"],
        default="scylla",
        help="Type of semantic cache to use: 'none' for no caching, 'scylla' for ScyllaDB (default: scylla)"
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

    try:
        # Check cache first (if enabled)
        cached_response = None
        if cache:
            print("\nChecking cache...")
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
                cache.cache_response(args.prompt, prompt_embedding, response_text)

        # Print the result
        print("\nClaude's response:")
        print("-" * 80)
        print(response_text)
        print("-" * 80)

    finally:
        if cache:
            cache.close()


if __name__ == "__main__":
    main()

