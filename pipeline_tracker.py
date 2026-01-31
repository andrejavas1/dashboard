#!/usr/bin/env python3
"""
Pipeline Run Tracker - Tracks and compares pipeline results across runs
"""

import json
import os
from datetime import datetime
import hashlib

class PipelineTracker:
    """Track pipeline runs and compare results"""
    
    def __init__(self, track_file="data/pipeline_runs.json"):
        self.track_file = track_file
        self.ensure_directory()
        self.runs = self.load_runs()
    
    def ensure_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.track_file), exist_ok=True)
    
    def load_runs(self):
        """Load previous runs from file"""
        if os.path.exists(self.track_file):
            with open(self.track_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_runs(self):
        """Save runs to file"""
        with open(self.track_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def record_run(self, metrics):
        """Record a new pipeline run"""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate hash of key metrics to detect changes
        metrics_str = json.dumps(metrics, sort_keys=True)
        metrics_hash = hashlib.md5(metrics_str.encode()).hexdigest()
        
        run_record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics_hash": metrics_hash,
            "metrics": metrics
        }
        
        self.runs.append(run_record)
        self.save_runs()
        
        # Compare with previous run
        comparison = self.compare_with_previous(run_record)
        
        return run_id, comparison
    
    def compare_with_previous(self, current_run):
        """Compare current run with previous run"""
        if len(self.runs) < 2:
            return {"status": "first_run", "message": "First run - no previous data to compare"}
        
        previous_run = self.runs[-2]
        
        comparison = {
            "status": "improved" if current_run["metrics_hash"] != previous_run["metrics_hash"] else "unchanged",
            "changes": {}
        }
        
        # Compare key metrics
        key_metrics = [
            "data_confidence",
            "patterns_discovered",
            "patterns_validated",
            "portfolio_size",
            "avg_validation_rate",
            "avg_success_rate",
            "total_occurrences"
        ]
        
        for metric in key_metrics:
            current_val = current_run["metrics"].get(metric, 0)
            previous_val = previous_run["metrics"].get(metric, 0)
            
            if current_val != previous_val:
                change = current_val - previous_val
                pct_change = (change / previous_val * 100) if previous_val != 0 else 0
                
                comparison["changes"][metric] = {
                    "previous": previous_val,
                    "current": current_val,
                    "change": change,
                    "pct_change": pct_change,
                    "direction": "better" if self.is_better(metric, change) else "worse"
                }
        
        # Overall assessment
        better_count = sum(1 for c in comparison["changes"].values() if c["direction"] == "better")
        worse_count = sum(1 for c in comparison["changes"].values() if c["direction"] == "worse")
        
        if better_count > worse_count:
            comparison["overall"] = "better"
        elif worse_count > better_count:
            comparison["overall"] = "worse"
        else:
            comparison["overall"] = "mixed"
        
        return comparison
    
    def is_better(self, metric, change):
        """Determine if a change is better or worse for a given metric"""
        # Metrics where higher is better
        higher_is_better = [
            "data_confidence",
            "patterns_discovered",
            "patterns_validated",
            "portfolio_size",
            "avg_validation_rate",
            "avg_success_rate",
            "total_occurrences"
        ]
        
        if metric in higher_is_better:
            return change > 0
        else:
            return change < 0
    
    def get_latest_run(self):
        """Get the most recent run"""
        if self.runs:
            return self.runs[-1]
        return None
    
    def get_best_run(self):
        """Get the best run based on key metrics"""
        if not self.runs:
            return None
        
        # Score each run
        scored_runs = []
        for run in self.runs:
            metrics = run["metrics"]
            score = (
                metrics.get("data_confidence", 0) * 0.2 +
                metrics.get("avg_validation_rate", 0) * 0.3 +
                metrics.get("avg_success_rate", 0) * 0.3 +
                metrics.get("portfolio_size", 0) * 0.1 +
                metrics.get("patterns_validated", 0) * 0.1
            )
            scored_runs.append((score, run))
        
        # Return run with highest score (sort by score only)
        scored_runs.sort(key=lambda x: x[0], reverse=True)
        return scored_runs[0][1]
    
    def print_comparison(self, comparison):
        """Print comparison results"""
        print("\n" + "=" * 60)
        print("PIPELINE RUN COMPARISON")
        print("=" * 60)
        
        if comparison["status"] == "first_run":
            print("First run - no previous data to compare")
            return
        
        print(f"Status: {comparison['status'].upper()}")
        print(f"Overall: {comparison['overall'].upper()}")
        
        if comparison["changes"]:
            print("\nChanges:")
            for metric, change_data in comparison["changes"].items():
                arrow = "UP" if change_data["change"] > 0 else "DOWN"
                direction = change_data["direction"].upper()
                print(f"  {metric}: {change_data['previous']:.2f} -> {change_data['current']:.2f} ({arrow}, {direction})")
        else:
            print("\nNo changes detected")
        
        print("=" * 60)
    
    def print_summary(self):
        """Print summary of all runs"""
        print("\n" + "=" * 60)
        print("PIPELINE RUNS SUMMARY")
        print("=" * 60)
        print(f"Total runs: {len(self.runs)}")
        
        if self.runs:
            latest = self.get_latest_run()
            best = self.get_best_run()
            
            print(f"\nLatest run: {latest['run_id']}")
            print(f"  Patterns discovered: {latest['metrics'].get('patterns_discovered', 0)}")
            print(f"  Patterns validated: {latest['metrics'].get('patterns_validated', 0)}")
            print(f"  Portfolio size: {latest['metrics'].get('portfolio_size', 0)}")
            print(f"  Avg validation rate: {latest['metrics'].get('avg_validation_rate', 0):.2f}%")
            
            if best['run_id'] != latest['run_id']:
                print(f"\nBest run: {best['run_id']}")
                print(f"  Patterns discovered: {best['metrics'].get('patterns_discovered', 0)}")
                print(f"  Patterns validated: {best['metrics'].get('patterns_validated', 0)}")
                print(f"  Portfolio size: {best['metrics'].get('portfolio_size', 0)}")
                print(f"  Avg validation rate: {best['metrics'].get('avg_validation_rate', 0):.2f}%")
        
        print("=" * 60)


if __name__ == "__main__":
    tracker = PipelineTracker()
    tracker.print_summary()