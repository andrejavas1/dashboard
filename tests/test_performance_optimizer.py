"""
Tests for Performance Optimizer Module

Tests for:
- PerformanceProfiler
- CacheManager
- PerformanceOptimizer
- Decorators and utilities

Author: Agent_CodebaseRefactor
Date: 2026-01-23
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from src.performance_optimizer import (
    PerformanceProfiler,
    PerformanceMetrics,
    BenchmarkResult,
    CacheManager,
    profile,
    cached,
    PerformanceOptimizer,
    get_profiler,
    get_cache,
    get_optimizer
)


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        assert profiler.metrics == {}
        assert profiler.call_times == {}
        assert profiler.active_profilers == {}
    
    def test_start_stop_profiling(self):
        """Test start and stop profiling."""
        profiler = PerformanceProfiler()
        
        profiler.start("test_operation")
        time.sleep(0.01)
        elapsed = profiler.stop("test_operation")
        
        assert elapsed > 0
        assert "test_operation" in profiler.metrics
        assert profiler.metrics["test_operation"].call_count == 1
    
    def test_multiple_calls(self):
        """Test profiling multiple calls."""
        profiler = PerformanceProfiler()
        
        for _ in range(3):
            profiler.start("test_operation")
            time.sleep(0.01)
            profiler.stop("test_operation")
        
        metrics = profiler.metrics["test_operation"]
        assert metrics.call_count == 3
        assert metrics.avg_time > 0
        assert metrics.total_time > 0
    
    def test_get_metrics(self):
        """Test getting metrics."""
        profiler = PerformanceProfiler()
        
        profiler.start("op1")
        time.sleep(0.01)
        profiler.stop("op1")
        
        profiler.start("op2")
        time.sleep(0.01)
        profiler.stop("op2")
        
        all_metrics = profiler.get_metrics()
        assert len(all_metrics) == 2
        
        op1_metrics = profiler.get_metrics("op1")
        assert "op1" in op1_metrics
    
    def test_get_bottlenecks(self):
        """Test bottleneck identification."""
        profiler = PerformanceProfiler()
        
        # Fast operation
        profiler.start("fast")
        profiler.stop("fast")
        
        # Slow operation
        profiler.start("slow")
        time.sleep(0.1)
        profiler.stop("slow")
        
        bottlenecks = profiler.get_bottlenecks(threshold=0.05)
        assert len(bottlenecks) == 1
        assert bottlenecks[0].name == "slow"
    
    def test_reset(self):
        """Test resetting profiler."""
        profiler = PerformanceProfiler()
        
        profiler.start("test")
        profiler.stop("test")
        
        profiler.reset()
        
        assert profiler.metrics == {}
        assert profiler.call_times == {}
        assert profiler.active_profilers == {}
    
    def test_report(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler()
        
        profiler.start("op1")
        time.sleep(0.01)
        profiler.stop("op1")
        
        report = profiler.report()
        assert "Performance Report" in report
        assert "op1" in report
    
    def test_profile_decorator(self):
        """Test profile decorator."""
        profiler = PerformanceProfiler()
        
        @profile(profiler)
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        assert result == "result"
        assert "test_function" in profiler.metrics


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def setup_method(self):
        """Setup test cache manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.temp_dir, max_memory_items=5)
    
    def teardown_method(self):
        """Cleanup test cache manager."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.hits == 0
        assert self.cache.misses == 0
        assert self.cache.memory_cache == {}
        assert self.cache.cache_dir.exists()
    
    def test_set_and_get(self):
        """Test setting and getting values."""
        key = "test_key"
        value = {"data": "test_value"}
        
        self.cache.set(key, value)
        result = self.cache.get(key)
        
        assert result == value
        assert self.cache.hits == 1
        assert self.cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss."""
        result = self.cache.get("nonexistent_key")
        
        assert result is None
        assert self.cache.hits == 0
        assert self.cache.misses == 1
    
    def test_lru_eviction(self):
        """Test LRU eviction from memory cache."""
        # Fill cache beyond max_memory_items
        for i in range(6):
            self.cache.set(f"key_{i}", f"value_{i}")
        
        # First key should be evicted from memory cache
        # but still available from disk cache
        assert len(self.cache.memory_cache) == 5  # Max memory items
        assert "key_0" not in self.cache.memory_cache
        assert "key_1" in self.cache.memory_cache
        
        # But key_0 should still be available from disk cache
        assert self.cache.get("key_0") == "value_0"
    
    def test_invalidate_key(self):
        """Test invalidating specific key."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.invalidate("key1")
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is not None
    
    def test_invalidate_all(self):
        """Test invalidating all cache."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.invalidate()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        self.cache.set("key1", "value1")
        self.cache.get("key1")
        self.cache.get("nonexistent")
        
        stats = self.cache.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['total_requests'] == 2
        assert stats['hit_rate'] == 0.5
    
    def test_cached_decorator(self):
        """Test cached decorator."""
        call_count = [0]
        
        @cached(self.cache)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count[0] == 1
        
        # Second call (cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count[0] == 1  # Should not increment


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer."""
    
    def setup_method(self):
        """Setup test optimizer."""
        self.optimizer = PerformanceOptimizer()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.profiler is not None
        assert self.optimizer.cache_manager is not None
        assert self.optimizer.benchmarks == []
    
    def test_optimize_dataframe_operations(self):
        """Test DataFrame optimization."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
            'str_col': ['A', 'B', 'A', 'B', 'A']
        })
        
        optimized = self.optimizer.optimize_dataframe_operations(df)
        
        assert optimized.shape == df.shape
        assert list(optimized.columns) == list(df.columns)
    
    def test_optimize_feature_selection(self):
        """Test feature selection optimization."""
        features = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.rand(100),
            'feature_4': np.random.rand(100) + 0.5,
            'feature_5': np.random.randn(100)
        })
        
        selected = self.optimizer.optimize_feature_selection(features, n_features=3)
        
        assert len(selected) == 3
        assert all(f in features.columns for f in selected)
    
    def test_parallel_pattern_evaluation(self):
        """Test parallel pattern evaluation."""
        patterns = [
            {'id': 1, 'conditions': {}},
            {'id': 2, 'conditions': {}},
            {'id': 3, 'conditions': {}}
        ]
        
        def evaluate_func(p):
            return {**p, 'score': 0.5}
        
        results = self.optimizer.parallel_pattern_evaluation(
            patterns, evaluate_func, n_jobs=2
        )
        
        assert len(results) == 3
        assert all('score' in r for r in results)
    
    def test_optimize_pattern_matching(self):
        """Test pattern matching optimization."""
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50]
        })
        
        conditions = {
            'feature_1': {'operator': '>=', 'value': 3},
            'feature_2': {'operator': '<=', 'value': 40}
        }
        
        mask = self.optimizer.optimize_pattern_matching(conditions, df)
        
        assert isinstance(mask, pd.Series)
        assert len(mask) == len(df)
        assert mask.sum() == 2  # Rows 2 and 3
    
    def test_run_benchmark(self):
        """Test benchmark execution."""
        def baseline_func(x):
            time.sleep(0.01)
            return x
        
        def optimized_func(x):
            return x
        
        result = self.optimizer.run_benchmark(
            "Test Operation",
            baseline_func,
            optimized_func,
            42
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.operation == "Test Operation"
        assert result.baseline_time > 0
        assert result.optimized_time >= 0
        assert len(self.optimizer.benchmarks) == 1
    
    def test_get_benchmark_summary(self):
        """Test benchmark summary."""
        self.optimizer.run_benchmark(
            "Test",
            lambda x: x,
            lambda x: x,
            1
        )
        
        summary = self.optimizer.get_benchmark_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert 'Operation' in summary.columns
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        report = self.optimizer.generate_performance_report()
        
        assert "Performance Optimization Report" in report
        assert "Cache Statistics" in report
    
    def test_save_benchmarks(self):
        """Test saving benchmarks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "benchmarks.json"
            
            self.optimizer.run_benchmark(
                "Test",
                lambda x: x,
                lambda x: x,
                1
            )
            
            self.optimizer.save_benchmarks(str(output_path))
            
            assert output_path.exists()


class TestGlobalInstances:
    """Test cases for global instances."""
    
    def test_get_profiler(self):
        """Test getting global profiler."""
        profiler = get_profiler()
        assert isinstance(profiler, PerformanceProfiler)
    
    def test_get_cache(self):
        """Test getting global cache."""
        cache = get_cache()
        assert isinstance(cache, CacheManager)
    
    def test_get_optimizer(self):
        """Test getting global optimizer."""
        optimizer = get_optimizer()
        assert isinstance(optimizer, PerformanceOptimizer)


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            name="test",
            total_time=1.0,
            avg_time=0.5,
            min_time=0.1,
            max_time=0.9,
            call_count=2
        )
        
        assert metrics.name == "test"
        assert metrics.total_time == 1.0
        assert metrics.call_count == 2
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = PerformanceMetrics(
            name="test",
            total_time=1.0,
            avg_time=0.5,
            min_time=0.1,
            max_time=0.9,
            call_count=2
        )
        
        d = metrics.to_dict()
        
        assert d['name'] == "test"
        assert d['total_time'] == 1.0
        assert d['call_count'] == 2


class TestBenchmarkResult:
    """Test cases for BenchmarkResult dataclass."""
    
    def test_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            operation="test",
            baseline_time=1.0,
            optimized_time=0.5,
            improvement_pct=50.0,
            speedup_factor=2.0
        )
        
        assert result.operation == "test"
        assert result.baseline_time == 1.0
        assert result.speedup_factor == 2.0
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = BenchmarkResult(
            operation="test",
            baseline_time=1.0,
            optimized_time=0.5,
            improvement_pct=50.0,
            speedup_factor=2.0
        )
        
        d = result.to_dict()
        
        assert d['operation'] == "test"
        assert d['baseline_time'] == 1.0
        assert d['speedup_factor'] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])