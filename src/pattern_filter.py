"""
Pattern Filtering Module
Filters duplicate and redundant patterns to prevent wasted resources in validation.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternFilter:
    """
    Filters patterns to remove duplicates and keep only the best variants.
    """

    def __init__(self):
        """Initialize the pattern filter."""
        self.filter_stats = {
            'total_input': 0,
            'exact_duplicates_removed': 0,
            'condition_duplicates_removed': 0,
            'metric_duplicates_removed': 0,
            'subset_patterns_removed': 0,
            'total_output': 0
        }

    def get_conditions_key(self, pattern: Dict) -> str:
        """
        Generate a hashable key from pattern conditions.

        Args:
            pattern: Pattern dictionary

        Returns:
            Hashable string key
        """
        conditions = pattern.get('conditions', {})
        return json.dumps(conditions, sort_keys=True)

    def get_full_key(self, pattern: Dict) -> str:
        """
        Generate a hashable key from pattern conditions and label_col.

        Args:
            pattern: Pattern dictionary

        Returns:
            Hashable string key
        """
        conditions = pattern.get('conditions', {})
        label_col = pattern.get('label_col', '')
        direction = pattern.get('direction', 'unknown')
        return json.dumps({'conditions': conditions, 'label_col': label_col, 'direction': direction}, sort_keys=True)

    def is_subset(self, conditions_a: Dict, conditions_b: Dict) -> bool:
        """
        Check if conditions_a is a subset of conditions_b.

        Args:
            conditions_a: First pattern conditions
            conditions_b: Second pattern conditions

        Returns:
            True if conditions_a is a subset of conditions_b
        """
        if len(conditions_a) >= len(conditions_b):
            return False

        for feature, condition in conditions_a.items():
            if feature not in conditions_b:
                return False
            if conditions_b[feature] != condition:
                return False

        return True

    def filter_exact_duplicates(self, patterns: List[Dict]) -> List[Dict]:
        """
        Remove patterns with identical conditions AND label_col.

        Args:
            patterns: List of patterns to filter

        Returns:
            Filtered patterns
        """
        seen = {}
        filtered = []

        for pattern in patterns:
            key = self.get_full_key(pattern)
            if key not in seen:
                seen[key] = pattern
                filtered.append(pattern)
            else:
                self.filter_stats['exact_duplicates_removed'] += 1
                logger.debug(f"Removed exact duplicate: {key[:50]}...")

        return filtered

    def filter_condition_duplicates(self, patterns: List[Dict]) -> List[Dict]:
        """
        For patterns with identical conditions but different label_col,
        keep only the one with highest success_rate.
        Uses validation_success_rate if available, otherwise training_success_rate.

        Args:
            patterns: List of patterns to filter

        Returns:
            Filtered patterns
        """
        # Group by conditions
        condition_groups = defaultdict(list)
        for pattern in patterns:
            key = self.get_conditions_key(pattern)
            condition_groups[key].append(pattern)

        filtered = []
        for key, group in condition_groups.items():
            if len(group) == 1:
                filtered.append(group[0])
            else:
                # Keep pattern with highest validation_success_rate, or training_success_rate if validation not available
                def get_sort_key(p):
                    # Prefer validation success rate if available
                    if 'validation_success_rate' in p:
                        return (p.get('validation_success_rate', 0), p.get('training_success_rate', 0))
                    else:
                        return (p.get('success_rate', 0), 0)

                best_pattern = max(group, key=get_sort_key)
                filtered.append(best_pattern)
                removed_count = len(group) - 1
                self.filter_stats['condition_duplicates_removed'] += removed_count
                logger.debug(f"Removed {removed_count} condition duplicates, kept with validation_rate={best_pattern.get('validation_success_rate', 'N/A')}, training_rate={best_pattern.get('training_success_rate', best_pattern.get('success_rate', 0)):.2f}%")

        return filtered

    def filter_subset_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        Remove patterns that are subsets of other patterns with same direction.
        Keep the more specific pattern (more conditions).

        Args:
            patterns: List of patterns to filter

        Returns:
            Filtered patterns
        """
        # Group by direction
        direction_groups = defaultdict(list)
        for pattern in patterns:
            direction = pattern.get('direction', 'unknown')
            direction_groups[direction].append(pattern)

        filtered = []

        for direction, group in direction_groups.items():
            # Sort by number of conditions (more specific first)
            group_sorted = sorted(group, key=lambda p: len(p.get('conditions', {})), reverse=True)

            kept = []
            removed_indices = set()

            for i, pattern in enumerate(group_sorted):
                if i in removed_indices:
                    continue

                conditions_i = pattern.get('conditions', {})

                # Check if any subsequent pattern is a subset of this one
                for j in range(i + 1, len(group_sorted)):
                    if j in removed_indices:
                        continue

                    conditions_j = group_sorted[j].get('conditions', {})

                    if self.is_subset(conditions_j, conditions_i):
                        # Pattern j is a subset of pattern i
                        removed_indices.add(j)
                        self.filter_stats['subset_patterns_removed'] += 1
                        logger.debug(f"Removed subset pattern (kept {len(conditions_i)} conditions, removed {len(conditions_j)} conditions)")

                kept.append(pattern)

            filtered.extend(kept)

        return filtered

    def conditions_similar(self, conditions_a: Dict, conditions_b: Dict,
                           value_tolerance: float = 0.1) -> bool:
        """
        Check if two pattern conditions are similar (within tolerance).
        
        Args:
            conditions_a: First pattern conditions
            conditions_b: Second pattern conditions
            value_tolerance: Tolerance for value comparison (percentage)
            
        Returns:
            True if conditions are similar
        """
        # Must have same number of conditions
        if len(conditions_a) != len(conditions_b):
            return False
        
        # Check each condition
        for feature, cond_a in conditions_a.items():
            if feature not in conditions_b:
                return False
            
            cond_b = conditions_b[feature]
            
            # Operators must match
            if cond_a.get('operator') != cond_b.get('operator'):
                return False
            
            # Values must be within tolerance
            val_a = cond_a.get('value', 0)
            val_b = cond_b.get('value', 0)
            
            # Handle special case for very small values
            if abs(val_a) < 0.01 and abs(val_b) < 0.01:
                continue
            
            # Calculate percentage difference
            if abs(val_a) > 0.001:
                diff_pct = abs((val_b - val_a) / val_a)
                if diff_pct > value_tolerance:
                    return False
        
        return True
    
    def metrics_similar(self, pattern_a: Dict, pattern_b: Dict,
                       success_rate_tol: float = 0.01,
                       avg_move_tol: float = 0.1) -> bool:
        """
        Check if two patterns have similar metrics.
        
        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            success_rate_tol: Tolerance for success rate difference
            avg_move_tol: Tolerance for average move difference
            
        Returns:
            True if metrics are similar
        """
        # Occurrences must be exactly the same
        if pattern_a.get('occurrences') != pattern_b.get('occurrences'):
            return False
        
        # Success rate must be within tolerance
        sr_a = pattern_a.get('success_rate', 0)
        sr_b = pattern_b.get('success_rate', 0)
        if abs(sr_a - sr_b) > success_rate_tol:
            return False
        
        # Average move must be within tolerance
        avg_a = pattern_a.get('avg_move', 0)
        avg_b = pattern_b.get('avg_move', 0)
        if abs(avg_a) > 0.001:
            diff_pct = abs((avg_b - avg_a) / avg_a)
            if diff_pct > avg_move_tol:
                return False
        
        return True
    
    def get_pattern_score(self, pattern: Dict) -> float:
        """
        Calculate a score for a pattern to determine which is better.
        Higher score = better pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Score value
        """
        score = 0.0
        
        # Validation success rate (most important)
        if 'validation_success_rate' in pattern:
            score += pattern['validation_success_rate'] * 2
        elif 'success_rate' in pattern:
            score += pattern['success_rate'] * 2
        
        # Training success rate
        if 'training_success_rate' in pattern:
            score += pattern['training_success_rate']
        elif 'success_rate' in pattern:
            score += pattern['success_rate']
        
        # Statistical significance
        sig = pattern.get('statistically_significant')
        if sig == 'True' or sig is True:
            score += 20
        elif sig == 'False' or sig is False:
            score -= 10
        
        # Classification bonus
        classification = pattern.get('classification')
        if classification == 'ROBUST':
            score += 15
        elif classification == 'PROMISING':
            score += 10
        elif classification == 'WEAK':
            score -= 5
        
        # Occurrence count (more occurrences = better, up to a point)
        occ = pattern.get('occurrences', 0)
        if occ >= 50:
            score += 10
        elif occ >= 30:
            score += 5
        
        return score
    
    def filter_metric_duplicates(self, patterns: List[Dict]) -> List[Dict]:
        """
        Remove patterns with similar metrics (occurrences, success rate, avg move).
        Same occurrences = same dates = same trading opportunities.
        Keeps only the better pattern based on validation metrics.
        
        Args:
            patterns: List of patterns to filter
            
        Returns:
            Filtered patterns
        """
        filtered = []
        removed_indices = set()
        
        for i, pattern_a in enumerate(patterns):
            if i in removed_indices:
                continue
            
            best_pattern = pattern_a
            best_score = self.get_pattern_score(pattern_a)
            
            # Compare with all subsequent patterns
            for j in range(i + 1, len(patterns)):
                if j in removed_indices:
                    continue
                
                pattern_b = patterns[j]
                
                # Check if metrics are similar (same occurrences = same dates = same trading opportunities)
                if self.metrics_similar(pattern_a, pattern_b):
                    # Found a potential duplicate - same metrics means same trading opportunities
                    score_b = self.get_pattern_score(pattern_b)
                    
                    if score_b > best_score:
                        # Pattern B is better
                        removed_indices.add(i)
                        removed_indices.add(j)
                        filtered.append(pattern_b)
                        self.filter_stats['metric_duplicates_removed'] += 1
                        logger.debug(
                            f"Removed metric duplicate (kept better pattern): "
                            f"occ={pattern_a.get('occurrences')}, "
                            f"score_a={best_score:.1f}, score_b={score_b:.1f}"
                        )
                        best_pattern = None
                        break
                    else:
                        # Pattern A is better
                        removed_indices.add(j)
                        self.filter_stats['metric_duplicates_removed'] += 1
                        logger.debug(
                            f"Removed metric duplicate (kept better pattern): "
                            f"occ={pattern_a.get('occurrences')}, "
                            f"score_a={best_score:.1f}, score_b={score_b:.1f}"
                        )
            
            if best_pattern is not None and i not in removed_indices:
                filtered.append(best_pattern)
        
        return filtered

    def filter_patterns(self, patterns: List[Dict],
                        remove_exact_duplicates: bool = True,
                        remove_condition_duplicates: bool = True,
                        remove_subsets: bool = True,
                        remove_metric_duplicates: bool = True) -> List[Dict]:
        """
        Apply all filtering methods to patterns.

        Args:
            patterns: List of patterns to filter
            remove_exact_duplicates: Remove patterns with identical conditions and label_col
            remove_condition_duplicates: Remove patterns with identical conditions (keep best)
            remove_subsets: Remove patterns that are subsets of others
            remove_metric_duplicates: Remove patterns with similar metrics (same occurrences = same dates)

        Returns:
            Filtered patterns
        """
        self.filter_stats['total_input'] = len(patterns)
        logger.info(f"Starting pattern filtering with {len(patterns)} patterns")

        filtered = patterns.copy()

        if remove_exact_duplicates:
            logger.info("Filtering exact duplicates...")
            filtered = self.filter_exact_duplicates(filtered)

        if remove_condition_duplicates:
            logger.info("Filtering condition duplicates...")
            filtered = self.filter_condition_duplicates(filtered)

        if remove_metric_duplicates:
            logger.info("Filtering metric duplicates...")
            self.filter_stats['metric_duplicates_removed'] = 0
            filtered = self.filter_metric_duplicates(filtered)

        if remove_subsets:
            logger.info("Filtering subset patterns...")
            filtered = self.filter_subset_patterns(filtered)

        self.filter_stats['total_output'] = len(filtered)

        logger.info(f"Pattern filtering complete:")
        logger.info(f"  Input: {self.filter_stats['total_input']}")
        logger.info(f"  Exact duplicates removed: {self.filter_stats['exact_duplicates_removed']}")
        logger.info(f"  Condition duplicates removed: {self.filter_stats['condition_duplicates_removed']}")
        logger.info(f"  Metric duplicates removed: {self.filter_stats['metric_duplicates_removed']}")
        logger.info(f"  Subset patterns removed: {self.filter_stats['subset_patterns_removed']}")
        logger.info(f"  Output: {self.filter_stats['total_output']}")
        logger.info(f"  Reduction: {(1 - self.filter_stats['total_output'] / self.filter_stats['total_input']) * 100:.1f}%")

        return filtered

    def get_filter_stats(self) -> Dict:
        """Return filtering statistics."""
        return self.filter_stats.copy()


if __name__ == "__main__":
    # Test the filter
    test_patterns = [
        {
            'conditions': {'OBV_ROC_20d': {'operator': '>=', 'value': -2.13}},
            'label_col': 'Label_3pct_5d',
            'success_rate': 85.0,
            'direction': 'long'
        },
        {
            'conditions': {'OBV_ROC_20d': {'operator': '>=', 'value': -2.13}},
            'label_col': 'Label_3pct_10d',
            'success_rate': 80.0,
            'direction': 'long'
        },
        {
            'conditions': {'OBV_ROC_20d': {'operator': '>=', 'value': -2.13}, 'Dist_MA_100': {'operator': '>=', 'value': 10}},
            'label_col': 'Label_3pct_5d',
            'success_rate': 90.0,
            'direction': 'long'
        },
    ]

    filter = PatternFilter()
    filtered = filter.filter_patterns(test_patterns)

    print(f"\nTest Results:")
    print(f"Input: {len(test_patterns)} patterns")
    print(f"Output: {len(filtered)} patterns")
    print(f"\nFiltered patterns:")
    for i, p in enumerate(filtered):
        print(f"  {i+1}. {p['conditions']} - {p['success_rate']}%")