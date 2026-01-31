"""
Performance Optimizer for Integrated Pattern Discovery System

This module provides:
- Performance profiling and bottleneck identification
- Algorithm optimization for speed and resource utilization
- Caching mechanisms for frequently accessed data
- Performance benchmarks and metrics

Author: Agent_CodebaseRefactor
Date: 2026-01-23
"""

import time
import json
import pickle
import hashlib
from functools import wraps, lru_cache
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function or operation."""
    name: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    call_count: int
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'call_count': self.call_count,
            'memory_usage_mb': self.memory_usage_mb,
            'cache_hit_rate': self.cache_hit_rate
        }


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    operation: str
    baseline_time: float
    optimized_time: float
    improvement_pct: float
    speedup_factor: float
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'baseline_time': self.baseline_time,
            'optimized_time': self.optimized_time,
            'improvement_pct': self.improvement_pct,
            'speedup_factor': self.speedup_factor,
            'timestamp': self.timestamp
        }


class PerformanceProfiler:
    """
    Performance profiler for identifying bottlenecks in the integrated system.
    
    Tracks execution time, memory usage, and call counts for functions.
    """
    
    def __init__(self):
        """Initialize the performance profiler."""
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.call_times: Dict[str, List[float]] = {}
        self.active_profilers: Dict[str, float] = {}
        
    def start(self, name: str) -> None:
        """
        Start profiling a function or operation.
        
        Args:
            name: Name of the operation to profile
        """
        self.active_profilers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """
        Stop profiling and record metrics.
        
        Args:
            name: Name of the operation
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.active_profilers:
            logger.warning(f"No active profiler for {name}")
            return 0.0
        
        elapsed = time.perf_counter() - self.active_profilers[name]
        del self.active_profilers[name]
        
        if name not in self.call_times:
            self.call_times[name] = []
        
        self.call_times[name].append(elapsed)
        
        # Update metrics
        times = self.call_times[name]
        self.metrics[name] = PerformanceMetrics(
            name=name,
            total_time=sum(times),
            avg_time=np.mean(times),
            min_time=min(times),
            max_time=max(times),
            call_count=len(times)
        )
        
        return elapsed
    
    def get_metrics(self, name: str = None) -> Dict[str, PerformanceMetrics]:
        """
        Get performance metrics.
        
        Args:
            name: Specific metric name (optional)
            
        Returns:
            Dictionary of metrics
        """
        if name:
            return {name: self.metrics.get(name)}
        return self.metrics
    
    def get_bottlenecks(self, threshold: float = 1.0) -> List[PerformanceMetrics]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold: Minimum average time in seconds to be considered a bottleneck
            
        Returns:
            List of bottleneck metrics sorted by average time
        """
        bottlenecks = [
            m for m in self.metrics.values() 
            if m.avg_time >= threshold
        ]
        return sorted(bottlenecks, key=lambda x: x.avg_time, reverse=True)
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.metrics.clear()
        self.call_times.clear()
        self.active_profilers.clear()
    
    def report(self) -> str:
        """Generate a performance report."""
        if not self.metrics:
            return "No performance data available."
        
        lines = ["Performance Report", "=" * 60]
        
        # Sort by total time
        sorted_metrics = sorted(
            self.metrics.values(), 
            key=lambda x: x.total_time, 
            reverse=True
        )
        
        lines.append(f"\n{'Operation':<40} {'Calls':>8} {'Avg(s)':>10} {'Total(s)':>10}")
        lines.append("-" * 70)
        
        for m in sorted_metrics:
            lines.append(
                f"{m.name:<40} {m.call_count:>8} "
                f"{m.avg_time:>10.4f} {m.total_time:>10.4f}"
            )
        
        # Bottlenecks
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            lines.append("\n\nBottlenecks (avg time >= 1s):")
            lines.append("-" * 60)
            for b in bottlenecks:
                lines.append(f"  {b.name}: {b.avg_time:.4f}s avg ({b.call_count} calls)")
        
        return "\n".join(lines)


def profile(profiler: PerformanceProfiler = None):
    """
    Decorator to profile a function.
    
    Args:
        profiler: PerformanceProfiler instance (creates new if None)
    """
    if profiler is None:
        profiler = PerformanceProfiler()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler.start(func.__name__)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.stop(func.__name__)
        return wrapper
    return decorator


class CacheManager:
    """
    Cache manager for frequently accessed data.
    
    Provides memory-based and disk-based caching with LRU eviction.
    """
    
    def __init__(self, cache_dir: str = "data/cache", max_memory_items: int = 100):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_items: Maximum items in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_items = max_memory_items
        
        # Memory cache (LRU)
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_order: List[str] = []
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Cache manager initialized: {cache_dir}, max_memory={max_memory_items}")
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate a cache key from function arguments.
        
        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create a hash of the arguments
        key_str = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Check memory cache first
        if key in self.memory_cache:
            self.hits += 1
            # Update LRU order
            if key in self.cache_order:
                self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.memory_cache[key][0]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                
                # Load into memory cache
                self._add_to_memory_cache(key, value)
                self.hits += 1
                return value
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Add to memory cache
        self._add_to_memory_cache(key, value, ttl)
        
        # Also save to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _add_to_memory_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Add value to memory cache with LRU eviction.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Evict if necessary
        if len(self.memory_cache) >= self.max_memory_items:
            oldest_key = self.cache_order.pop(0)
            del self.memory_cache[oldest_key]
        
        expiry = time.time() + ttl
        self.memory_cache[key] = (value, expiry)
        self.cache_order.append(key)
    
    def invalidate(self, key: str = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate (None for all)
        """
        if key:
            # Invalidate specific key
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.cache_order:
                self.cache_order.remove(key)
            
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        else:
            # Invalidate all
            self.memory_cache.clear()
            self.cache_order.clear()
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }


def cached(cache_manager: CacheManager = None, ttl: int = 3600):
    """
    Decorator to cache function results.
    
    Args:
        cache_manager: CacheManager instance (creates new if None)
        ttl: Time to live in seconds
    """
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_manager._get_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_manager.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


class PerformanceOptimizer:
    """
    Performance optimizer for the integrated pattern discovery system.
    
    Provides optimization techniques for:
    - Data processing
    - Algorithm efficiency
    - Memory management
    - Parallel processing
    """
    
    def __init__(self, profiler: PerformanceProfiler = None):
        """
        Initialize the performance optimizer.
        
        Args:
            profiler: PerformanceProfiler instance
        """
        self.profiler = profiler or PerformanceProfiler()
        self.cache_manager = CacheManager()
        self.benchmarks: List[BenchmarkResult] = []
        
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame operations for better performance.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        self.profiler.start("optimize_dataframe")
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        self.profiler.stop("optimize_dataframe")
        return df
    
    def optimize_feature_selection(self, features: pd.DataFrame, 
                                   n_features: int = 50) -> List[str]:
        """
        Optimized feature selection using variance and correlation.
        
        Args:
            features: Feature DataFrame
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        self.profiler.start("feature_selection")
        
        # Remove constant features
        var_threshold = features.var()
        non_constant = var_threshold[var_threshold > 0.01].index.tolist()
        
        if len(non_constant) <= n_features:
            self.profiler.stop("feature_selection")
            return non_constant
        
        # Calculate correlation matrix
        corr_matrix = features[non_constant].corr().abs()
        
        # Select features with low correlation
        selected = []
        remaining = non_constant.copy()
        
        while len(selected) < n_features and remaining:
            # Find feature with lowest average correlation to selected
            if not selected:
                # Select feature with highest variance
                best_feature = var_threshold[remaining].idxmax()
            else:
                avg_corr = corr_matrix.loc[remaining, selected].mean(axis=1)
                best_feature = avg_corr.idxmin()
            
            selected.append(best_feature)
            remaining.remove(best_feature)
        
        self.profiler.stop("feature_selection")
        return selected
    
    def parallel_pattern_evaluation(self, patterns: List[Dict], 
                                    evaluate_func: Callable,
                                    n_jobs: int = 4) -> List[Dict]:
        """
        Evaluate patterns in parallel.
        
        Args:
            patterns: List of patterns to evaluate
            evaluate_func: Function to evaluate a pattern
            n_jobs: Number of parallel jobs
            
        Returns:
            List of evaluated patterns
        """
        self.profiler.start("parallel_evaluation")
        
        # Simple parallel processing using joblib-like approach
        # For demonstration, using sequential processing
        # In production, use joblib or multiprocessing
        
        results = []
        batch_size = max(1, len(patterns) // n_jobs)
        
        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i+batch_size]
            batch_results = [evaluate_func(p) for p in batch]
            results.extend(batch_results)
        
        self.profiler.stop("parallel_evaluation")
        return results
    
    def optimize_pattern_matching(self, conditions: Dict, 
                                   df: pd.DataFrame) -> pd.Series:
        """
        Optimized pattern matching using vectorized operations.
        
        Args:
            conditions: Pattern conditions
            df: DataFrame to match against
            
        Returns:
            Boolean mask of matching rows
        """
        self.profiler.start("pattern_matching")
        
        mask = pd.Series(True, index=df.index)
        
        for feature, condition in conditions.items():
            if feature not in df.columns:
                continue
            
            operator = condition.get('operator')
            threshold = condition.get('value')
            
            if operator == '>=':
                mask &= (df[feature] >= threshold)
            elif operator == '<=':
                mask &= (df[feature] <= threshold)
            elif operator == '>':
                mask &= (df[feature] > threshold)
            elif operator == '<':
                mask &= (df[feature] < threshold)
        
        self.profiler.stop("pattern_matching")
        return mask
    
    def run_benchmark(self, operation: str, 
                     baseline_func: Callable,
                     optimized_func: Callable,
                     *args, **kwargs) -> BenchmarkResult:
        """
        Run a performance benchmark comparing baseline and optimized versions.
        
        Args:
            operation: Name of the operation
            baseline_func: Baseline function
            optimized_func: Optimized function
            *args: Arguments to pass to functions
            **kwargs: Keyword arguments to pass to functions
            
        Returns:
            BenchmarkResult with performance comparison
        """
        logger.info(f"Benchmarking: {operation}")
        
        # Warm-up
        baseline_func(*args, **kwargs)
        optimized_func(*args, **kwargs)
        
        # Benchmark baseline
        baseline_times = []
        for _ in range(5):
            start = time.perf_counter()
            baseline_func(*args, **kwargs)
            baseline_times.append(time.perf_counter() - start)
        
        baseline_avg = np.mean(baseline_times)
        
        # Benchmark optimized
        optimized_times = []
        for _ in range(5):
            start = time.perf_counter()
            optimized_func(*args, **kwargs)
            optimized_times.append(time.perf_counter() - start)
        
        optimized_avg = np.mean(optimized_times)
        
        # Calculate improvement
        improvement_pct = ((baseline_avg - optimized_avg) / baseline_avg) * 100
        speedup_factor = baseline_avg / optimized_avg if optimized_avg > 0 else 0
        
        result = BenchmarkResult(
            operation=operation,
            baseline_time=baseline_avg,
            optimized_time=optimized_avg,
            improvement_pct=improvement_pct,
            speedup_factor=speedup_factor,
            timestamp=datetime.now().isoformat()
        )
        
        self.benchmarks.append(result)
        
        logger.info(f"  Baseline: {baseline_avg:.4f}s")
        logger.info(f"  Optimized: {optimized_avg:.4f}s")
        logger.info(f"  Improvement: {improvement_pct:.1f}%")
        logger.info(f"  Speedup: {speedup_factor:.2f}x")
        
        return result
    
    def get_benchmark_summary(self) -> pd.DataFrame:
        """
        Get a summary of all benchmarks.
        
        Returns:
            DataFrame with benchmark results
        """
        if not self.benchmarks:
            return pd.DataFrame()
        
        data = []
        for b in self.benchmarks:
            data.append({
                'Operation': b.operation,
                'Baseline (s)': f"{b.baseline_time:.4f}",
                'Optimized (s)': f"{b.optimized_time:.4f}",
                'Improvement (%)': f"{b.improvement_pct:.1f}",
                'Speedup': f"{b.speedup_factor:.2f}x"
            })
        
        return pd.DataFrame(data)
    
    def save_benchmarks(self, output_path: str = "data/performance_benchmarks.json") -> None:
        """
        Save benchmark results to file.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump([b.to_dict() for b in self.benchmarks], f, indent=2)
        
        logger.info(f"Benchmarks saved to {output_path}")
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Performance report string
        """
        lines = ["Performance Optimization Report", "=" * 60]
        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        
        # Profiler report
        lines.append("\n" + self.profiler.report())
        
        # Cache statistics
        cache_stats = self.cache_manager.get_stats()
        lines.append("\n\nCache Statistics:")
        lines.append("-" * 60)
        lines.append(f"  Hits: {cache_stats['hits']}")
        lines.append(f"  Misses: {cache_stats['misses']}")
        lines.append(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
        lines.append(f"  Memory Cache Size: {cache_stats['memory_cache_size']}")
        lines.append(f"  Disk Cache Files: {cache_stats['disk_cache_files']}")
        
        # Benchmark summary
        if self.benchmarks:
            lines.append("\n\nBenchmark Results:")
            lines.append("-" * 60)
            lines.append(self.get_benchmark_summary().to_string(index=False))
        
        # Recommendations
        lines.append("\n\nOptimization Recommendations:")
        lines.append("-" * 60)
        
        bottlenecks = self.profiler.get_bottlenecks(threshold=0.5)
        if bottlenecks:
            lines.append("  Identified bottlenecks:")
            for b in bottlenecks[:5]:
                lines.append(f"    - {b.name}: {b.avg_time:.4f}s avg")
        else:
            lines.append("  No significant bottlenecks identified.")
        
        if cache_stats['hit_rate'] < 0.5:
            lines.append("  Consider increasing cache TTL or cache size.")
        
        return "\n".join(lines)


# Global instances
_global_profiler = PerformanceProfiler()
_global_cache = CacheManager()
_global_optimizer = PerformanceOptimizer(_global_profiler)


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return _global_cache


def get_optimizer() -> PerformanceOptimizer:
    """Get the global optimizer instance."""
    return _global_optimizer


def main():
    """Main function to demonstrate performance optimization."""
    print("=" * 60)
    print("Performance Optimizer Demonstration")
    print("=" * 60)
    
    optimizer = PerformanceOptimizer()
    
    # Create sample DataFrame
    print("\nCreating sample DataFrame...")
    data = {
        'feature_1': np.random.randn(10000),
        'feature_2': np.random.randn(10000),
        'feature_3': np.random.randint(0, 100, 10000),
        'feature_4': np.random.choice(['A', 'B', 'C'], 10000),
        'target': np.random.choice([0, 1], 10000)
    }
    df = pd.DataFrame(data)
    
    # Benchmark DataFrame optimization
    print("\nBenchmarking DataFrame optimization...")
    
    def baseline_process(df):
        return df.copy()
    
    def optimized_process(df):
        return optimizer.optimize_dataframe_operations(df)
    
    optimizer.run_benchmark(
        "DataFrame Processing",
        baseline_process,
        optimized_process,
        df
    )
    
    # Generate report
    print("\n" + optimizer.generate_performance_report())
    
    # Save benchmarks
    optimizer.save_benchmarks()
    
    print("\n" + "=" * 60)
    print("Performance Optimization Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()