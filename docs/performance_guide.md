# Performance Optimization Guide

## Overview

This guide provides comprehensive information about performance optimization for the Integrated Pattern Discovery System. It covers profiling, caching, algorithm optimization, and best practices for achieving optimal performance.

## Table of Contents

1. [Performance Profiling](#performance-profiling)
2. [Caching Strategies](#caching-strategies)
3. [Algorithm Optimization](#algorithm-optimization)
4. [Performance Benchmarks](#performance-benchmarks)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Performance Profiling

### PerformanceProfiler

The `PerformanceProfiler` class tracks execution time, memory usage, and call counts for functions and operations.

#### Basic Usage

```python
from src.performance_optimizer import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile an operation
profiler.start("data_loading")
# ... your code here ...
elapsed = profiler.stop("data_loading")

# Get metrics
metrics = profiler.get_metrics("data_loading")
print(f"Average time: {metrics.avg_time:.4f}s")

# Generate report
print(profiler.report())
```

#### Using the Profile Decorator

```python
from src.performance_optimizer import profile, get_profiler

@profile(get_profiler())
def expensive_operation(data):
    # Your expensive operation
    return result

result = expensive_operation(data)
```

#### Identifying Bottlenecks

```python
# Get bottlenecks (operations with avg time >= threshold)
bottlenecks = profiler.get_bottlenecks(threshold=1.0)

for bottleneck in bottlenecks:
    print(f"{bottleneck.name}: {bottleneck.avg_time:.4f}s avg")
```

### Global Profiler

A global profiler instance is available for convenience:

```python
from src.performance_optimizer import get_profiler

profiler = get_profiler()
```

---

## Caching Strategies

### CacheManager

The `CacheManager` provides both memory-based and disk-based caching with LRU eviction.

#### Basic Usage

```python
from src.performance_optimizer import CacheManager

cache = CacheManager(cache_dir="data/cache", max_memory_items=100)

# Store a value
cache.set("my_key", {"data": "value"}, ttl=3600)

# Retrieve a value
value = cache.get("my_key")

# Invalidate specific key
cache.invalidate("my_key")

# Invalidate all cache
cache.invalidate()
```

#### Using the Cached Decorator

```python
from src.performance_optimizer import cached, get_cache

@cached(get_cache(), ttl=3600)
def expensive_computation(x, y):
    # Expensive computation
    return x * y

# First call computes and caches
result1 = expensive_computation(5, 10)

# Second call returns cached result
result2 = expensive_computation(5, 10)
```

#### Cache Statistics

```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Memory cache size: {stats['memory_cache_size']}")
print(f"Disk cache files: {stats['disk_cache_files']}")
```

### Cache Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cache_dir` | `"data/cache"` | Directory for disk cache |
| `max_memory_items` | `100` | Maximum items in memory cache |
| `ttl` | `3600` | Default time to live (seconds) |

---

## Algorithm Optimization

### PerformanceOptimizer

The `PerformanceOptimizer` class provides optimization techniques for data processing and algorithm efficiency.

#### DataFrame Optimization

```python
from src.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()

# Optimize DataFrame operations
optimized_df = optimizer.optimize_dataframe_operations(df)
```

This optimization includes:
- Downcasting numeric types (int64 → int32/int16, float64 → float32)
- Converting low-cardinality object columns to category type

#### Feature Selection Optimization

```python
# Select features with low correlation
selected_features = optimizer.optimize_feature_selection(
    features, 
    n_features=50
)
```

#### Pattern Matching Optimization

```python
# Optimized vectorized pattern matching
mask = optimizer.optimize_pattern_matching(conditions, df)
```

#### Parallel Pattern Evaluation

```python
# Evaluate patterns in parallel
results = optimizer.parallel_pattern_evaluation(
    patterns, 
    evaluate_func, 
    n_jobs=4
)
```

---

## Performance Benchmarks

### Running Benchmarks

```python
from src.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()

# Compare baseline vs optimized
result = optimizer.run_benchmark(
    "DataFrame Processing",
    baseline_function,
    optimized_function,
    *args,
    **kwargs
)

print(f"Speedup: {result.speedup_factor:.2f}x")
print(f"Improvement: {result.improvement_pct:.1f}%")
```

### Benchmark Results

```python
# Get benchmark summary
summary = optimizer.get_benchmark_summary()
print(summary)

# Save benchmarks to file
optimizer.save_benchmarks("data/performance_benchmarks.json")
```

### Expected Performance Improvements

| Operation | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| DataFrame Processing | ~0.5s | ~0.3s | 1.67x |
| Feature Selection | ~2.0s | ~1.0s | 2.0x |
| Pattern Matching | ~0.8s | ~0.2s | 4.0x |
| Cache Hit Rate | N/A | ~80% | N/A |

---

## Best Practices

### 1. Profile Before Optimizing

Always profile your code before making optimizations to identify true bottlenecks:

```python
profiler = PerformanceProfiler()

# Profile your workflow
profiler.start("full_pipeline")
# ... your code ...
profiler.stop("full_pipeline")

# Review report
print(profiler.report())
```

### 2. Cache Expensive Operations

Use caching for frequently accessed or computationally expensive data:

```python
@cached(get_cache(), ttl=3600)
def load_features(symbol):
    # Expensive feature loading
    return features
```

### 3. Optimize Data Types

Use appropriate data types to reduce memory usage:

```python
# Instead of int64
df['column'] = df['column'].astype('int32')

# Instead of float64
df['column'] = df['column'].astype('float32')

# For low-cardinality strings
df['category'] = df['category'].astype('category')
```

### 4. Use Vectorized Operations

Avoid loops when possible; use pandas/numpy vectorized operations:

```python
# Slow (loop)
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * df.loc[i, 'col2']

# Fast (vectorized)
df['new_col'] = df['col1'] * df['col2']
```

### 5. Batch Operations

Process data in batches for memory efficiency:

```python
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    process_batch(batch)
```

### 6. Monitor Cache Performance

Regularly check cache hit rates to ensure caching is effective:

```python
stats = cache.get_stats()
if stats['hit_rate'] < 0.5:
    # Consider increasing TTL or cache size
    pass
```

### 7. Clean Up Cache

Periodically clean up old cache entries:

```python
# Invalidate specific keys
cache.invalidate("old_data_key")

# Or invalidate all
cache.invalidate()
```

---

## Troubleshooting

### Low Cache Hit Rate

**Problem**: Cache hit rate is below 50%

**Solutions**:
- Increase TTL for cached items
- Check if cache keys are being generated correctly
- Verify that the same arguments are being passed to cached functions

### Memory Issues

**Problem**: High memory usage

**Solutions**:
- Reduce `max_memory_items` in CacheManager
- Use `optimize_dataframe_operations()` to downcast types
- Process data in batches instead of loading all at once
- Call `cache.invalidate()` periodically to free memory

### Slow Pattern Discovery

**Problem**: Pattern discovery is taking too long

**Solutions**:
- Profile to identify bottlenecks
- Use `optimize_feature_selection()` to reduce feature count
- Enable caching for feature computation
- Consider using `parallel_pattern_evaluation()` for multiple patterns

### Cache Not Working

**Problem**: Cached functions are still being computed

**Solutions**:
- Verify cache directory exists and is writable
- Check that function arguments are hashable
- Ensure TTL hasn't expired
- Review cache key generation logic

---

## Integration with Integrated System

The performance optimizer can be integrated with the `IntegratedPatternSystem`:

```python
from src.integrated_system import IntegratedPatternSystem
from src.performance_optimizer import get_profiler, get_cache

profiler = get_profiler()
cache = get_cache()

system = IntegratedPatternSystem(
    config=SystemConfig(
        use_cache=True,
        cache_manager=cache,
        profiler=profiler
    )
)

# Run with profiling
profiler.start("full_pipeline")
patterns = system.run_full_pipeline(data_path)
profiler.stop("full_pipeline")

# Check performance
print(profiler.report())
print(f"Cache hit rate: {cache.get_stats()['hit_rate']:.1%}")
```

---

## API Reference

### PerformanceProfiler

| Method | Description |
|--------|-------------|
| `start(name)` | Start profiling an operation |
| `stop(name)` | Stop profiling and record metrics |
| `get_metrics(name=None)` | Get performance metrics |
| `get_bottlenecks(threshold)` | Identify bottlenecks |
| `reset()` | Reset all profiling data |
| `report()` | Generate performance report |

### CacheManager

| Method | Description |
|--------|-------------|
| `get(key)` | Get value from cache |
| `set(key, value, ttl)` | Set value in cache |
| `invalidate(key=None)` | Invalidate cache entries |
| `get_stats()` | Get cache statistics |

### PerformanceOptimizer

| Method | Description |
|--------|-------------|
| `optimize_dataframe_operations(df)` | Optimize DataFrame |
| `optimize_feature_selection(features, n_features)` | Select features |
| `parallel_pattern_evaluation(patterns, func, n_jobs)` | Parallel evaluation |
| `optimize_pattern_matching(conditions, df)` | Vectorized matching |
| `run_benchmark(...)` | Run performance benchmark |
| `generate_performance_report()` | Generate report |

---

## Conclusion

Performance optimization is an iterative process. Always profile first, identify bottlenecks, apply targeted optimizations, and measure improvements. The tools provided in this guide will help you achieve optimal performance for your pattern discovery system.

For more information, see:
- [Integration Guide](integration_guide.md)
- [ML Pattern Discovery](ml_pattern_discovery.md)
- [Rule-Based Patterns](rule_based_patterns.md)