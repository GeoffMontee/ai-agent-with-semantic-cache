#!/usr/bin/env python3
"""
Benchmark script to compare cache performance between ScyllaDB and PostgreSQL pgvector.

Tests multiple scenarios:
1. Cache hit performance (same prompt repeated)
2. Semantic similarity matching (similar prompts)
3. Cache miss performance (unique prompts)
4. Scale testing (varying cache sizes)
"""

import argparse
import asyncio
import time
import statistics
import json
import csv
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# Import cache classes from main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ai_agent_with_cache import ScyllaDBCache, PgVectorCache


def load_prompts(filepath="benchmark_prompts.txt"):
    """Load prompts from file, ignoring comments and empty lines."""
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


async def benchmark_cache_hits(cache, embeddings, num_iterations=100, is_async=False):
    """Benchmark repeated cache hits with the same embedding."""
    latencies = []
    
    # Use the first embedding for repeated queries
    test_embedding = embeddings[0]
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        if is_async:
            result = await cache.get_cached_response(test_embedding)
        else:
            result = cache.get_cached_response(test_embedding)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to milliseconds
    
    return latencies


async def benchmark_semantic_similarity(cache, base_prompts, similar_prompts, embedder, is_async=False):
    """Benchmark semantic similarity matching."""
    results = {
        'hits': 0,
        'misses': 0,
        'latencies': []
    }
    
    # Generate embeddings for similar prompts
    similar_embeddings = [embedder.encode(p) for p in similar_prompts]
    
    for embedding in similar_embeddings:
        start = time.perf_counter()
        if is_async:
            result = await cache.get_cached_response(embedding)
        else:
            result = cache.get_cached_response(embedding)
        end = time.perf_counter()
        
        latencies_ms = (end - start) * 1000
        results['latencies'].append(latencies_ms)
        
        if result:
            results['hits'] += 1
        else:
            results['misses'] += 1
    
    return results


async def benchmark_cache_misses(cache, prompts, embedder, is_async=False):
    """Benchmark cache misses (lookup + write)."""
    lookup_latencies = []
    write_latencies = []
    
    for prompt in prompts:
        embedding = embedder.encode(prompt)
        
        # Measure lookup time
        start = time.perf_counter()
        if is_async:
            result = await cache.get_cached_response(embedding)
        else:
            result = cache.get_cached_response(embedding)
        end = time.perf_counter()
        lookup_latencies.append((end - start) * 1000)
        
        # Measure write time (if cache miss)
        if not result:
            start = time.perf_counter()
            if is_async:
                await cache.cache_response(prompt, embedding, f"Response for: {prompt}")
            else:
                cache.cache_response(prompt, embedding, f"Response for: {prompt}")
            end = time.perf_counter()
            write_latencies.append((end - start) * 1000)
    
    return {
        'lookup_latencies': lookup_latencies,
        'write_latencies': write_latencies
    }


def calculate_percentiles(latencies):
    """Calculate p50, p95, p99, and max latencies."""
    if not latencies:
        return {'p50': 0, 'p95': 0, 'p99': 0, 'max': 0, 'mean': 0}
    
    sorted_latencies = sorted(latencies)
    return {
        'p50': statistics.median(sorted_latencies),
        'p95': sorted_latencies[int(len(sorted_latencies) * 0.95)],
        'p99': sorted_latencies[int(len(sorted_latencies) * 0.99)],
        'max': max(sorted_latencies),
        'mean': statistics.mean(sorted_latencies)
    }


async def run_benchmark_suite(cache_type, cache, embedder, prompts, is_async=False):
    """Run complete benchmark suite for a cache backend."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {cache_type.upper()} Cache Backend")
    print(f"{'='*80}")
    
    results = {
        'cache_type': cache_type,
        'timestamp': datetime.now().isoformat(),
        'total_prompts': len(prompts)
    }
    
    # Separate prompts into categories
    base_prompts = prompts[:5]  # First 5 prompts
    similar_prompts = prompts[5:10]  # Next 5 (semantically similar)
    diverse_prompts = prompts[10:25]  # Next 15 (diverse topics)
    
    # Pre-populate cache with base prompts
    print("\n[1/4] Pre-populating cache with base prompts...")
    base_embeddings = [embedder.encode(p) for p in base_prompts]
    for prompt, embedding in zip(base_prompts, base_embeddings):
        if is_async:
            await cache.cache_response(prompt, embedding, f"Response for: {prompt}")
        else:
            cache.cache_response(prompt, embedding, f"Response for: {prompt}")
    print(f"✓ Cached {len(base_prompts)} base prompts")
    
    # Test 1: Cache Hit Performance
    print("\n[2/4] Testing cache hit performance (same prompt, 100 iterations)...")
    hit_latencies = await benchmark_cache_hits(cache, base_embeddings, num_iterations=100, is_async=is_async)
    results['cache_hits'] = calculate_percentiles(hit_latencies)
    print(f"✓ Completed 100 cache hit queries")
    print(f"  - p50: {results['cache_hits']['p50']:.2f}ms")
    print(f"  - p95: {results['cache_hits']['p95']:.2f}ms")
    print(f"  - p99: {results['cache_hits']['p99']:.2f}ms")
    
    # Test 2: Semantic Similarity
    print("\n[3/4] Testing semantic similarity matching...")
    similarity_results = await benchmark_semantic_similarity(
        cache, base_prompts, similar_prompts, embedder, is_async=is_async
    )
    results['semantic_similarity'] = {
        'hits': similarity_results['hits'],
        'misses': similarity_results['misses'],
        'hit_rate': similarity_results['hits'] / len(similar_prompts) * 100,
        'latencies': calculate_percentiles(similarity_results['latencies'])
    }
    print(f"✓ Tested {len(similar_prompts)} semantically similar prompts")
    print(f"  - Cache hits: {similarity_results['hits']}/{len(similar_prompts)}")
    print(f"  - Hit rate: {results['semantic_similarity']['hit_rate']:.1f}%")
    print(f"  - p50 latency: {results['semantic_similarity']['latencies']['p50']:.2f}ms")
    
    # Test 3: Cache Miss Performance
    print("\n[4/4] Testing cache miss performance (diverse prompts)...")
    miss_results = await benchmark_cache_misses(cache, diverse_prompts, embedder, is_async=is_async)
    results['cache_misses'] = {
        'lookup': calculate_percentiles(miss_results['lookup_latencies']),
        'write': calculate_percentiles(miss_results['write_latencies'])
    }
    print(f"✓ Tested {len(diverse_prompts)} diverse prompts")
    print(f"  - Lookup p50: {results['cache_misses']['lookup']['p50']:.2f}ms")
    print(f"  - Write p50: {results['cache_misses']['write']['p50']:.2f}ms")
    
    return results


def print_comparison_table(results_list):
    """Print comparison table for all backends."""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS COMPARISON")
    print(f"{'='*80}\n")
    
    # Cache Hit Performance
    print("1. CACHE HIT PERFORMANCE (same prompt, 100 iterations)")
    print("-" * 80)
    print(f"{'Backend':<15} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12} {'Max (ms)':<12}")
    print("-" * 80)
    for r in results_list:
        cache_type = r['cache_type']
        hits = r['cache_hits']
        print(f"{cache_type:<15} {hits['p50']:<12.2f} {hits['p95']:<12.2f} "
              f"{hits['p99']:<12.2f} {hits['max']:<12.2f}")
    
    # Semantic Similarity
    print("\n2. SEMANTIC SIMILARITY MATCHING")
    print("-" * 80)
    print(f"{'Backend':<15} {'Hit Rate':<12} {'Hits/Total':<15} {'p50 (ms)':<12} {'p95 (ms)':<12}")
    print("-" * 80)
    for r in results_list:
        cache_type = r['cache_type']
        sim = r['semantic_similarity']
        hit_rate_str = f"{sim['hit_rate']:.1f}%"
        hits_str = f"{sim['hits']}/{sim['hits'] + sim['misses']}"
        lat = sim['latencies']
        print(f"{cache_type:<15} {hit_rate_str:<12} {hits_str:<15} "
              f"{lat['p50']:<12.2f} {lat['p95']:<12.2f}")
    
    # Cache Miss Performance
    print("\n3. CACHE MISS PERFORMANCE (lookup + write)")
    print("-" * 80)
    print(f"{'Backend':<15} {'Lookup p50':<15} {'Lookup p95':<15} {'Write p50':<15} {'Write p95':<15}")
    print("-" * 80)
    for r in results_list:
        cache_type = r['cache_type']
        miss = r['cache_misses']
        print(f"{cache_type:<15} {miss['lookup']['p50']:<15.2f} {miss['lookup']['p95']:<15.2f} "
              f"{miss['write']['p50']:<15.2f} {miss['write']['p95']:<15.2f}")
    
    print("\n" + "="*80)


def save_results(results_list, output_format='json'):
    """Save results to file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format == 'json':
        filename = f"benchmark_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"\n✓ Results saved to {filename}")
    
    elif output_format == 'csv':
        filename = f"benchmark_results_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Cache Type', 'Metric', 'p50 (ms)', 'p95 (ms)', 'p99 (ms)', 'Max (ms)', 'Mean (ms)'
            ])
            
            # Write data
            for r in results_list:
                cache_type = r['cache_type']
                
                # Cache hits
                hits = r['cache_hits']
                writer.writerow([
                    cache_type, 'Cache Hits',
                    f"{hits['p50']:.2f}", f"{hits['p95']:.2f}",
                    f"{hits['p99']:.2f}", f"{hits['max']:.2f}", f"{hits['mean']:.2f}"
                ])
                
                # Semantic similarity
                sim_lat = r['semantic_similarity']['latencies']
                writer.writerow([
                    cache_type, f"Semantic Similarity ({r['semantic_similarity']['hit_rate']:.1f}% hit rate)",
                    f"{sim_lat['p50']:.2f}", f"{sim_lat['p95']:.2f}",
                    f"{sim_lat['p99']:.2f}", f"{sim_lat['max']:.2f}", f"{sim_lat['mean']:.2f}"
                ])
                
                # Cache miss lookup
                lookup = r['cache_misses']['lookup']
                writer.writerow([
                    cache_type, 'Cache Miss - Lookup',
                    f"{lookup['p50']:.2f}", f"{lookup['p95']:.2f}",
                    f"{lookup['p99']:.2f}", f"{lookup['max']:.2f}", f"{lookup['mean']:.2f}"
                ])
                
                # Cache miss write
                write = r['cache_misses']['write']
                writer.writerow([
                    cache_type, 'Cache Miss - Write',
                    f"{write['p50']:.2f}", f"{write['p95']:.2f}",
                    f"{write['p99']:.2f}", f"{write['max']:.2f}", f"{write['mean']:.2f}"
                ])
        
        print(f"\n✓ Results saved to {filename}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cache performance for ScyllaDB and PostgreSQL pgvector"
    )
    parser.add_argument(
        "--backends",
        nargs='+',
        choices=["scylla", "pgvector", "both"],
        default=["both"],
        help="Cache backends to benchmark (default: both)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="benchmark_prompts.txt",
        help="File containing prompts (default: benchmark_prompts.txt)"
    )
    parser.add_argument(
        "--sentence-transformer-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--output",
        choices=["json", "csv", "none"],
        default="none",
        help="Save results to file (default: none)"
    )
    
    # ScyllaDB options
    parser.add_argument("--scylla-contact-points", type=str, default="127.0.0.1")
    parser.add_argument("--scylla-user", type=str, default="scylla")
    parser.add_argument("--scylla-password", type=str, default="")
    parser.add_argument("--scylla-keyspace", type=str, default="llm_cache_benchmark")
    parser.add_argument("--scylla-table", type=str, default="llm_responses_benchmark")
    
    # PostgreSQL options
    parser.add_argument("--postgres-host", type=str, default="localhost")
    parser.add_argument("--postgres-port", type=int, default=5432)
    parser.add_argument("--postgres-user", type=str, default="postgres")
    parser.add_argument("--postgres-password", type=str, default="postgres")
    parser.add_argument("--postgres-database", type=str, default="postgres")
    parser.add_argument("--postgres-schema", type=str, default="llm_cache_benchmark")
    parser.add_argument("--postgres-table", type=str, default="llm_responses_benchmark")
    
    args = parser.parse_args()
    
    # Determine which backends to test
    backends = args.backends
    if "both" in backends:
        backends = ["pgvector", "scylla"]
    
    # Load prompts
    print(f"Loading prompts from {args.prompts_file}...")
    prompts = load_prompts(args.prompts_file)
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Load SentenceTransformer model
    print(f"\nLoading SentenceTransformer model: {args.sentence_transformer_model}...")
    embedder = SentenceTransformer(args.sentence_transformer_model)
    print("✓ Model loaded")
    
    results_list = []
    
    # Benchmark each backend
    for backend in backends:
        try:
            if backend == "pgvector":
                print(f"\nInitializing PostgreSQL pgvector cache...")
                cache = PgVectorCache(
                    host=args.postgres_host,
                    port=args.postgres_port,
                    user=args.postgres_user,
                    password=args.postgres_password,
                    database=args.postgres_database,
                    schema=args.postgres_schema,
                    table=args.postgres_table,
                    similarity_function="cosine"
                )
                await cache.connect()
                
                results = await run_benchmark_suite(
                    backend, cache, embedder, prompts, is_async=True
                )
                results_list.append(results)
                
                await cache.close()
            
            elif backend == "scylla":
                print(f"\nInitializing ScyllaDB cache...")
                cache = ScyllaDBCache(
                    contact_points=args.scylla_contact_points.split(','),
                    username=args.scylla_user,
                    password=args.scylla_password,
                    keyspace=args.scylla_keyspace,
                    table=args.scylla_table
                )
                
                results = await run_benchmark_suite(
                    backend, cache, embedder, prompts, is_async=False
                )
                results_list.append(results)
                
                cache.close()
        
        except Exception as e:
            print(f"\n✗ Error benchmarking {backend}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison table
    if len(results_list) > 0:
        print_comparison_table(results_list)
        
        # Save results if requested
        if args.output != "none":
            save_results(results_list, args.output)
    else:
        print("\n✗ No benchmarks completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
