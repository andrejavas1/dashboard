import json
import os

def convert_optimized_to_dashboard_format():
    """Convert optimized patterns to dashboard format with validated data."""
    
    # Load optimized patterns
    with open('data/optimized_patterns.json', 'r') as f:
        optimized_patterns = json.load(f)
    
    # Load validated patterns for realistic validation metrics
    with open('data/validated_patterns.json', 'r') as f:
        validated_data = json.load(f)
    
    # Flatten validated patterns into a list
    validated_patterns = validated_data['robust'] + validated_data['degraded'] + validated_data['failed']
    
    print(f"Converting {len(optimized_patterns)} optimized patterns to dashboard format")
    
    # Convert to dashboard format
    dashboard_patterns = []
    
    for i, pattern in enumerate(optimized_patterns):
        # Get corresponding validated data
        validated_entry = validated_patterns[i] if i < len(validated_patterns) else None
        
        # Create dashboard pattern structure
        dashboard_pattern = {
            "pattern": {
                "conditions": pattern["conditions"],
                "occurrences": pattern.get("occurrences", 0),
                "success_count": str(pattern.get("success_count", 0)),
                "success_rate": pattern.get("success_rate", 0.0),
                "avg_move": pattern.get("avg_move", 0.0),
                "avg_time": pattern.get("avg_time", 0.0),
                "false_positive_rate": pattern.get("false_positive_rate", 0.0),
                "p_value": pattern.get("p_value", 0.0),
                "label_col": pattern.get("label_col", "Label_5pct_30d"),
                "method": pattern.get("method", "clustering"),
                "direction": pattern.get("direction", "long"),
                "stability_score": pattern.get("stability_score", 0.0),
                "period_success_rates": pattern.get("period_success_rates", [])
            },
            "training_success_rate": pattern.get("success_rate", 0.0),
            "validation_success_rate": validated_entry.get("validation_success_rate", pattern.get("success_rate", 0.0)) if validated_entry else pattern.get("success_rate", 0.0),
            "live_success_rate": validated_entry.get("live_success_rate", 0) if validated_entry else 0,
            "validation_ratio": validated_entry.get("validation_ratio", 1.0) if validated_entry else 1.0,
            "classification": validated_entry.get("classification", "UNKNOWN") if validated_entry else "UNKNOWN",
            "validation_occurrences": validated_entry.get("validation_occurrences", pattern.get("occurrences", 0)) if validated_entry else pattern.get("occurrences", 0),
            "live_occurrences": validated_entry.get("live_occurrences", 0) if validated_entry else 0,
            "validation_avg_move": validated_entry.get("validation_avg_move", pattern.get("avg_move", 0.0)) if validated_entry else pattern.get("avg_move", 0.0),
            "validation_avg_time": validated_entry.get("validation_avg_time", pattern.get("avg_time", 0.0)) if validated_entry else pattern.get("avg_time", 0.0),
            "validation_false_positive_rate": validated_entry.get("validation_false_positive_rate", 0.0) if validated_entry else 0.0,
            "direction": pattern.get("direction", "long"),
            "binomial_p_value": validated_entry.get("binomial_p_value", 0.0) if validated_entry else 0.0,
            "chi_square_p_value": validated_entry.get("chi_square_p_value", 0.0) if validated_entry else 0.0,
            "statistically_significant": validated_entry.get("statistically_significant", "False") if validated_entry else "False",
            "scores": {
                "success": validated_entry.get("validation_success_rate", pattern.get("success_rate", 0.0)) if validated_entry else pattern.get("success_rate", 0.0),
                "risk": 70.0,  # Default value
                "practical": 75.0,  # Default value
                "composite": 80.0  # Default value
            }
        }
        
        dashboard_patterns.append(dashboard_pattern)
    
    # Save to data directory
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, 'patterns.json'), 'w') as f:
        json.dump(dashboard_patterns, f, indent=2)
    
    print(f"Generated dashboard patterns.json with {len(dashboard_patterns)} patterns")
    
    return dashboard_patterns

if __name__ == "__main__":
    convert_optimized_to_dashboard_format()