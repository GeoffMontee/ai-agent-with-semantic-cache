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
    
    for i, embedding in enumerate(similar_embeddings):
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
            # Debug: print which prompt missed
            print(f"  DEBUG: Prompt '{similar_prompts[i][:50]}...' missed cache")
    
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


async def benchmark_concurrent_reads(cache, embeddings, concurrency_level=10, total_operations=100, is_async=False):
    """Benchmark concurrent cache read performance."""
    
    async def read_task(embedding):
        start = time.perf_counter()
        if is_async:
            result = await cache.get_cached_response(embedding)
        else:
            result = await asyncio.to_thread(cache.get_cached_response, embedding)
        end = time.perf_counter()
        return (end - start) * 1000, result is not None
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(concurrency_level)
    
    async def limited_task(embedding):
        async with semaphore:
            return await read_task(embedding)
    
    # Create tasks: distribute operations across available embeddings
    tasks = []
    for i in range(total_operations):
        embedding = embeddings[i % len(embeddings)]
        tasks.append(limited_task(embedding))
    
    # Record start time for throughput calculation
    suite_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    suite_end = time.perf_counter()
    
    latencies = [r[0] for r in results]
    successes = sum(1 for r in results if r[1])
    total_time = suite_end - suite_start
    
    return {
        'latencies': latencies,
        'total_operations': len(results),
        'successes': successes,
        'failures': len(results) - successes,
        'total_time_seconds': total_time,
        'qps': len(results) / total_time if total_time > 0 else 0,
        'concurrency_level': concurrency_level
    }


async def benchmark_concurrent_writes(cache, prompts, embedder, concurrency_level=10, total_operations=100, is_async=False):
    """Benchmark concurrent cache write performance."""
    
    # Pre-generate embeddings
    embeddings = [embedder.encode(prompts[i % len(prompts)]) for i in range(total_operations)]
    
    async def write_task(prompt, embedding, idx):
        start = time.perf_counter()
        if is_async:
            await cache.cache_response(f"{prompt}_write_{idx}", embedding, f"Response for: {prompt}")
        else:
            await asyncio.to_thread(cache.cache_response, f"{prompt}_write_{idx}", embedding, f"Response for: {prompt}")
        end = time.perf_counter()
        return (end - start) * 1000
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(concurrency_level)
    
    async def limited_task(prompt, embedding, idx):
        async with semaphore:
            return await write_task(prompt, embedding, idx)
    
    # Create tasks
    tasks = []
    for i in range(total_operations):
        prompt = prompts[i % len(prompts)]
        embedding = embeddings[i]
        tasks.append(limited_task(prompt, embedding, i))
    
    # Record start time for throughput calculation
    suite_start = time.perf_counter()
    latencies = await asyncio.gather(*tasks)
    suite_end = time.perf_counter()
    
    total_time = suite_end - suite_start
    
    return {
        'latencies': latencies,
        'total_operations': len(latencies),
        'total_time_seconds': total_time,
        'qps': len(latencies) / total_time if total_time > 0 else 0,
        'concurrency_level': concurrency_level
    }


async def benchmark_mixed_workload(cache, embeddings, prompts, embedder, concurrency_level=10, total_operations=100, read_ratio=0.8, is_async=False):
    """Benchmark mixed read/write workload (default 80% reads, 20% writes)."""
    
    num_reads = int(total_operations * read_ratio)
    num_writes = total_operations - num_reads
    
    async def read_task(embedding):
        start = time.perf_counter()
        if is_async:
            result = await cache.get_cached_response(embedding)
        else:
            result = await asyncio.to_thread(cache.get_cached_response, embedding)
        end = time.perf_counter()
        return ('read', (end - start) * 1000, result is not None)
    
    async def write_task(prompt, embedding, idx):
        start = time.perf_counter()
        if is_async:
            await cache.cache_response(f"{prompt}_mixed_{idx}", embedding, f"Response for: {prompt}")
        else:
            await asyncio.to_thread(cache.cache_response, f"{prompt}_mixed_{idx}", embedding, f"Response for: {prompt}")
        end = time.perf_counter()
        return ('write', (end - start) * 1000, True)
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(concurrency_level)
    
    async def limited_read(embedding):
        async with semaphore:
            return await read_task(embedding)
    
    async def limited_write(prompt, embedding, idx):
        async with semaphore:
            return await write_task(prompt, embedding, idx)
    
    # Create mixed tasks
    tasks = []
    for i in range(num_reads):
        embedding = embeddings[i % len(embeddings)]
        tasks.append(limited_read(embedding))
    
    for i in range(num_writes):
        prompt = prompts[i % len(prompts)]
        embedding = embedder.encode(f"{prompt}_mixed_{i}")
        tasks.append(limited_write(prompt, embedding, i))
    
    # Shuffle for realistic interleaving
    import random
    random.shuffle(tasks)
    
    # Record start time for throughput calculation
    suite_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    suite_end = time.perf_counter()
    
    read_latencies = [r[1] for r in results if r[0] == 'read']
    write_latencies = [r[1] for r in results if r[0] == 'write']
    total_time = suite_end - suite_start
    
    return {
        'read_latencies': read_latencies,
        'write_latencies': write_latencies,
        'total_operations': len(results),
        'read_operations': len(read_latencies),
        'write_operations': len(write_latencies),
        'total_time_seconds': total_time,
        'qps': len(results) / total_time if total_time > 0 else 0,
        'concurrency_level': concurrency_level,
        'read_ratio': read_ratio
    }


async def run_benchmark_suite(cache_type, cache, embedder, prompts, is_async=False, concurrency_test=False, concurrency_levels=None, concurrent_operations=100):
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
    test_num = 1
    total_tests = 4 + (3 * len(concurrency_levels) if concurrency_test and concurrency_levels else 0)
    
    print(f"\n[{test_num}/{total_tests}] Pre-populating cache with base prompts...")
    test_num += 1
    base_embeddings = [embedder.encode(p) for p in base_prompts]
    for prompt, embedding in zip(base_prompts, base_embeddings):
        if is_async:
            await cache.cache_response(prompt, embedding, f"Response for: {prompt}")
        else:
            cache.cache_response(prompt, embedding, f"Response for: {prompt}")
    print(f"[+] Cached {len(base_prompts)} base prompts")
    
    # Test 1: Cache Hit Performance
    print(f"\n[{test_num}/{total_tests}] Testing cache hit performance (same prompt, 100 iterations)...")
    test_num += 1
    hit_latencies = await benchmark_cache_hits(cache, base_embeddings, num_iterations=100, is_async=is_async)
    results['cache_hits'] = calculate_percentiles(hit_latencies)
    print(f"[+] Completed 100 cache hit queries")
    print(f"  - p50: {results['cache_hits']['p50']:.2f}ms")
    print(f"  - p95: {results['cache_hits']['p95']:.2f}ms")
    print(f"  - p99: {results['cache_hits']['p99']:.2f}ms")
    
    # Test 2: Semantic Similarity
    print(f"\n[{test_num}/{total_tests}] Testing semantic similarity matching...")
    test_num += 1
    similarity_results = await benchmark_semantic_similarity(
        cache, base_prompts, similar_prompts, embedder, is_async=is_async
    )
    results['semantic_similarity'] = {
        'hits': similarity_results['hits'],
        'misses': similarity_results['misses'],
        'hit_rate': similarity_results['hits'] / len(similar_prompts) * 100,
        'latencies': calculate_percentiles(similarity_results['latencies'])
    }
    print(f"[+] Tested {len(similar_prompts)} semantically similar prompts")
    print(f"  - Cache hits: {similarity_results['hits']}/{len(similar_prompts)}")
    print(f"  - Hit rate: {results['semantic_similarity']['hit_rate']:.1f}%")
    print(f"  - p50 latency: {results['semantic_similarity']['latencies']['p50']:.2f}ms")
    
    # Test 3: Cache Miss Performance
    print(f"\n[{test_num}/{total_tests}] Testing cache miss performance (diverse prompts)...")
    test_num += 1
    miss_results = await benchmark_cache_misses(cache, diverse_prompts, embedder, is_async=is_async)
    results['cache_misses'] = {
        'lookup': calculate_percentiles(miss_results['lookup_latencies']),
        'write': calculate_percentiles(miss_results['write_latencies'])
    }
    print(f"[+] Tested {len(diverse_prompts)} diverse prompts")
    print(f"  - Lookup p50: {results['cache_misses']['lookup']['p50']:.2f}ms")
    print(f"  - Write p50: {results['cache_misses']['write']['p50']:.2f}ms")
    
    # Concurrency Tests (if enabled)
    if concurrency_test and concurrency_levels:
        results['concurrency'] = {}
        
        for level in concurrency_levels:
            # Concurrent Reads
            print(f"\n[{test_num}/{total_tests}] Testing concurrent reads (concurrency={level}, operations={concurrent_operations})...")
            test_num += 1
            read_results = await benchmark_concurrent_reads(
                cache, base_embeddings, concurrency_level=level,
                total_operations=concurrent_operations, is_async=is_async
            )
            print(f"[+] Completed {read_results['total_operations']} concurrent read operations")
            print(f"  - QPS: {read_results['qps']:.2f}")
            print(f"  - p50: {calculate_percentiles(read_results['latencies'])['p50']:.2f}ms")
            print(f"  - p95: {calculate_percentiles(read_results['latencies'])['p95']:.2f}ms")
            
            # Concurrent Writes
            print(f"\n[{test_num}/{total_tests}] Testing concurrent writes (concurrency={level}, operations={concurrent_operations})...")
            test_num += 1
            write_results = await benchmark_concurrent_writes(
                cache, diverse_prompts, embedder, concurrency_level=level,
                total_operations=concurrent_operations, is_async=is_async
            )
            print(f"[+] Completed {write_results['total_operations']} concurrent write operations")
            print(f"  - QPS: {write_results['qps']:.2f}")
            print(f"  - p50: {calculate_percentiles(write_results['latencies'])['p50']:.2f}ms")
            print(f"  - p95: {calculate_percentiles(write_results['latencies'])['p95']:.2f}ms")
            
            # Mixed Workload
            print(f"\n[{test_num}/{total_tests}] Testing mixed workload (concurrency={level}, 80% reads / 20% writes)...")
            test_num += 1
            mixed_results = await benchmark_mixed_workload(
                cache, base_embeddings, diverse_prompts, embedder,
                concurrency_level=level, total_operations=concurrent_operations,
                read_ratio=0.8, is_async=is_async
            )
            read_stats = calculate_percentiles(mixed_results['read_latencies'])
            write_stats = calculate_percentiles(mixed_results['write_latencies'])
            print(f"[+] Completed {mixed_results['total_operations']} mixed operations")
            print(f"  - QPS: {mixed_results['qps']:.2f}")
            print(f"  - Read p50: {read_stats['p50']:.2f}ms, Write p50: {write_stats['p50']:.2f}ms")
            
            results['concurrency'][f'level_{level}'] = {
                'reads': {
                    'latencies': calculate_percentiles(read_results['latencies']),
                    'qps': read_results['qps'],
                    'total_operations': read_results['total_operations']
                },
                'writes': {
                    'latencies': calculate_percentiles(write_results['latencies']),
                    'qps': write_results['qps'],
                    'total_operations': write_results['total_operations']
                },
                'mixed': {
                    'read_latencies': read_stats,
                    'write_latencies': write_stats,
                    'qps': mixed_results['qps'],
                    'total_operations': mixed_results['total_operations']
                }
            }
    
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
    
    # Concurrency Performance (if available)
    if any('concurrency' in r for r in results_list):
        print("\n4. CONCURRENCY PERFORMANCE")
        
        # Get all concurrency levels
        concurrency_levels = set()
        for r in results_list:
            if 'concurrency' in r:
                concurrency_levels.update(r['concurrency'].keys())
        
        for level_key in sorted(concurrency_levels):
            level = level_key.split('_')[1]
            
            # Concurrent Reads
            print(f"\n4.1 Concurrent Reads (Level {level})")
            print("-" * 80)
            print(f"{'Backend':<15} {'QPS':<12} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12}")
            print("-" * 80)
            for r in results_list:
                if 'concurrency' in r and level_key in r['concurrency']:
                    cache_type = r['cache_type']
                    reads = r['concurrency'][level_key]['reads']
                    lat = reads['latencies']
                    print(f"{cache_type:<15} {reads['qps']:<12.2f} {lat['p50']:<12.2f} "
                          f"{lat['p95']:<12.2f} {lat['p99']:<12.2f}")
            
            # Concurrent Writes
            print(f"\n4.2 Concurrent Writes (Level {level})")
            print("-" * 80)
            print(f"{'Backend':<15} {'QPS':<12} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12}")
            print("-" * 80)
            for r in results_list:
                if 'concurrency' in r and level_key in r['concurrency']:
                    cache_type = r['cache_type']
                    writes = r['concurrency'][level_key]['writes']
                    lat = writes['latencies']
                    print(f"{cache_type:<15} {writes['qps']:<12.2f} {lat['p50']:<12.2f} "
                          f"{lat['p95']:<12.2f} {lat['p99']:<12.2f}")
            
            # Mixed Workload
            print(f"\n4.3 Mixed Workload (Level {level}, 80% reads / 20% writes)")
            print("-" * 80)
            print(f"{'Backend':<15} {'QPS':<12} {'Read p50':<12} {'Write p50':<12} {'Read p95':<12} {'Write p95':<12}")
            print("-" * 80)
            for r in results_list:
                if 'concurrency' in r and level_key in r['concurrency']:
                    cache_type = r['cache_type']
                    mixed = r['concurrency'][level_key]['mixed']
                    read_lat = mixed['read_latencies']
                    write_lat = mixed['write_latencies']
                    print(f"{cache_type:<15} {mixed['qps']:<12.2f} {read_lat['p50']:<12.2f} "
                          f"{write_lat['p50']:<12.2f} {read_lat['p95']:<12.2f} {write_lat['p95']:<12.2f}")
    
    print("\n" + "="*80)


def save_results(results_list, output_format='json'):
    """Save results to file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format == 'json':
        filename = f"benchmark_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"\n[+] Results saved to {filename}")
    
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
        
        print(f"\n[+] Results saved to {filename}")


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
    
    # Concurrency testing options
    parser.add_argument(
        "--concurrency-test",
        action="store_true",
        help="Run concurrency benchmarks"
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,5,10,25",
        help="Comma-separated concurrency levels (default: 1,5,10,25)"
    )
    parser.add_argument(
        "--concurrent-operations",
        type=int,
        default=100,
        help="Total operations per concurrency test (default: 100)"
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
    print(f"[+] Loaded {len(prompts)} prompts")
    
    # Load SentenceTransformer model
    print(f"\nLoading SentenceTransformer model: {args.sentence_transformer_model}...")
    embedder = SentenceTransformer(args.sentence_transformer_model)
    print("[+] Model loaded")
    
    # Parse concurrency levels
    concurrency_levels = None
    if args.concurrency_test:
        concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(',')]
        print(f"\n[+] Concurrency testing enabled with levels: {concurrency_levels}")
        print(f"  Operations per test: {args.concurrent_operations}")
    
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
                    backend, cache, embedder, prompts, is_async=True,
                    concurrency_test=args.concurrency_test,
                    concurrency_levels=concurrency_levels,
                    concurrent_operations=args.concurrent_operations
                )
                results_list.append(results)
                
                await cache.close()
            
            elif backend == "scylla":
                print(f"\nInitializing ScyllaDB cache...")
                # Calculate optimal pool size based on concurrency levels if testing concurrency
                pool_size = 10  # default
                if args.concurrency_test and concurrency_levels:
                    # Set pool size to handle max concurrency level
                    max_concurrency = max(concurrency_levels)
                    pool_size = max(10, max_concurrency * 2)  # 2x max concurrency for safety
                    print(f"  Configuring connection pool: {pool_size} connections per host, 2048 max requests per connection")
                
                cache = ScyllaDBCache(
                    contact_points=args.scylla_contact_points.split(','),
                    username=args.scylla_user,
                    password=args.scylla_password,
                    keyspace=args.scylla_keyspace,
                    table=args.scylla_table,
                    pool_size=pool_size,
                    max_requests_per_connection=2048
                )
                
                # Give extra time for vector index to be ready in cloud environments
                print("Waiting additional time for vector index to be fully ready...")
                import time as time_module
                time_module.sleep(5)
                
                # Verify vector index is working with a test query
                print("Verifying vector index...")
                try:
                    test_embedding = embedder.encode("test query")
                    cache.get_cached_response(test_embedding)
                    print("[+] Vector index is ready")
                except Exception as e:
                    print(f"[-] Warning: Vector index test failed: {e}")
                    print("  Continuing anyway, but results may be affected...")
                
                results = await run_benchmark_suite(
                    backend, cache, embedder, prompts, is_async=False,
                    concurrency_test=args.concurrency_test,
                    concurrency_levels=concurrency_levels,
                    concurrent_operations=args.concurrent_operations
                )
                results_list.append(results)
                
                cache.close()
        
        except Exception as e:
            print(f"\n[-] Error benchmarking {backend}: {e}")
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
        print("\n[-] No benchmarks completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
