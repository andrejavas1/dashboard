# Speed Optimizations Guide

## Overview

The system now supports **three run modes** for different speed/quality trade-offs:

| Mode | Use Case | Approx. Speed |
|------|----------|---------------|
| `full` | Production, final results | Baseline (slowest) |
| `quick` | Development, testing | ~3-5x faster |
| `ultra` | Rapid iteration, debugging | ~10-20x faster |

---

## How to Use

Change `run_mode` in `config.yaml`:

```yaml
# Run mode: 'full' (thorough), 'quick' (faster), 'ultra' (fastest, minimal)
run_mode: ultra
```

Or set via environment variable (if supported by your wrapper).

---

## Detailed Optimizations by Phase

### Phase 1: Data Acquisition

| Mode | Data Range | Records (typical) |
|------|-----------|-------------------|
| `full` | 2010-current (~15 years) | ~3,750 days |
| `quick` | Last 2 years | ~500 days |
| `ultra` | Last 6 months | ~125 days |

**Impact**: 30x less data in ultra mode

---

### Phase 3: Feature Engineering

#### Feature Periods

| Feature Type | full | quick | ultra |
|--------------|------|-------|-------|
| MA Periods | 5 (10-200) | 2 (20,50) | 1 (20) |
| ROC Periods | 5 (1-20) | 2 (5,10) | 1 (5) |
| ATR Periods | 4 (5-20) | 1 (14) | 1 (14) |
| RSI Periods | 4 (7-28) | 1 (14) | 1 (14) |
| ADX Periods | 2 (14,20) | 1 (14) | 1 (14) |
| Volume Periods | 4 (5-50) | 2 (10,20) | 1 (10) |
| High/Low Periods | 5 (5-100) | 2 (10,20) | 1 (10) |

#### Skipped Features

| Mode | Enhanced Momentum | Cycle Features |
|------|-------------------|----------------|
| `full` | ✅ | ✅ |
| `quick` | ❌ | ❌ |
| `ultra` | ❌ | ❌ |

---

### Phase 4: Pattern Discovery

#### Parameters

| Parameter | full | quick | ultra |
|-----------|------|-------|-------|
| Min Occurrences | 15 | 5 | 3 |
| Min Success Rate | 53% | 50% | 45% |
| High Confidence Rate | 75% | 60% | 55% |
| High Confidence Occurrences | 30 | 10 | 5 |
| Test Combinations | 1-6 features | 1-3 features | 1-2 features |
| Max Features/Pattern | 6 | 6 | 3 |

#### Features Used

| Mode | Top Features |
|------|--------------|
| `full` | 100 |
| `quick` | 20 |
| `ultra` | 10 |

#### Combinations Tested (max)

| Features | full | quick | ultra |
|----------|------|-------|-------|
| 1-feature | 5,000 | 500 | 100 |
| 2-feature | 5,000 | 200 | 30 |
| 3-feature | 5,000 | 50 | - |
| 4-feature | 2,000 | - | - |
| 5-feature | 500 | - | - |
| 6-feature | 50 | - | - |

#### Thresholds/Windows Tested

| Mode | Thresholds | Windows |
|------|------------|---------|
| `full` | All 5 (1%, 2%, 3%, 5%, 10%) | All 5 (5d, 10d, 15d, 20d, 30d) |
| `quick` | First 2 (1%, 2%) | First 2 (5d, 10d) |
| `ultra` | First 1 (1%) | First 1 (5d) |

#### Discovery Methods

| Method | full | quick | ultra |
|--------|------|-------|-------|
| Rule-based | ✅ | ✅ | ✅ |
| Decision Tree | ✅ | ✅ | ✅ |
| Clustering | ✅ | ❌ | ❌ |
| Sequential | ✅ | ❌ | ❌ |

---

### Phase 5: Pattern Optimization

| Mode | Behavior |
|------|----------|
| `full` | Full optimization on all patterns |
| `quick` | Limited to top 50 patterns |
| `ultra` | **Skipped entirely** - patterns pass through |

---

### Phase 6: Out-of-Sample Validation

#### Pattern Limits

| Mode | Max Patterns Validated |
|------|------------------------|
| `full` | All patterns |
| `quick` | Top 50 patterns |
| `ultra` | Top 30 patterns |

#### Validation Thresholds

| Mode | Robust Threshold | Degraded Threshold |
|------|-----------------|-------------------|
| `full` | 85% | 70% |
| `quick` | 70% | 50% |
| `ultra` | 60% | 40% |

---

## Expected Performance

### Timing Estimates (Approximate)

On a typical modern laptop (i5/i7, 16GB RAM):

| Phase | full | quick | ultra |
|-------|------|-------|-------|
| Phase 1 (Data) | 30-60s | 10-20s | 5-10s |
| Phase 2 (Labeling) | 5s | 3s | 2s |
| Phase 3 (Features) | 30-60s | 10-15s | 3-5s |
| Phase 4 (Discovery) | 10-30 min | 2-5 min | 30-60s |
| Phase 5 (Optimization) | 5-10 min | 2-3 min | 0s (skipped) |
| Phase 6 (Validation) | 2-5 min | 1-2 min | 30s |
| Phase 7 (Portfolio) | 30s | 20s | 10s |
| **Total** | **~45 min** | **~10 min** | **~2 min** |

*Note: Times vary significantly based on data size, CPU, and available RAM*

---

## Use Case Recommendations

### Use `full` mode when:
- Running production analysis
- Generating final reports
- Validating strategies for live trading
- Maximum pattern quality is critical

### Use `quick` mode when:
- Developing new features
- Testing configuration changes
- Daily/weekly batch processing
- Quality vs speed balance needed

### Use `ultra` mode when:
- Rapid prototyping
- Debugging pipeline issues
- Testing code changes
- Quick smoke tests
- CI/CD automated testing

---

## Warnings

⚠️ **Ultra Mode Limitations:**
- Patterns may be lower quality (45% vs 53% threshold)
- Very few occurrences required (3 vs 15)
- No optimization pass
- Minimal validation
- Only 6 months of data

⚠️ **Quick Mode Limitations:**
- Reduced pattern diversity
- Fewer combinations tested
- Less rigorous validation

⚠️ **Never use `ultra` for:**
- Live trading decisions
- Final research results
- Publishing/analyzing pattern quality

---

## Future Optimizations (Not Yet Implemented)

Potential additional speedups:

1. **Parallel Processing**: Multi-core pattern discovery
2. **GPU Acceleration**: CUDA for feature calculations
3. **Caching**: Persistent feature cache across runs
4. **Incremental Updates**: Only process new data
5. **Early Stopping**: Abort pattern discovery if no good patterns found
6. **JIT Compilation**: Numba for hot loops
7. **Database Backend**: PostgreSQL for large datasets
