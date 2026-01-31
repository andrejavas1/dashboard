"""
Price Movement Probability Discovery System - Main Orchestrator
Runs all 10 phases of the analysis pipeline.
"""

import os
import sys
import logging
from datetime import datetime
import argparse
from pipeline_tracker import PipelineTracker

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import phase modules (lazy loading to handle optional dependencies)
def get_phase_module(phase_number):
    """Get phase module dynamically to handle optional dependencies."""
    modules = {
        1: ('src.data_acquisition', 'DataAcquisition'),
        2: ('src.phase2_movement_labeling', 'MovementLabeling'),
        3: ('src.phase3_feature_engineering', 'FeatureEngineering'),
        4: ('src.phase4_pattern_discovery', 'PatternDiscovery'),
        5: ('src.phase5_pattern_optimization', 'PatternOptimization'),
        6: ('src.phase6_validation', 'OutOfSampleValidation'),
        7: ('src.phase7_portfolio_construction', 'PortfolioConstruction'),
        8: ('src.phase8_visualization', 'PatternVisualization'),
        9: ('src.phase9_realtime_detection', 'RealTimePatternDetection'),
        10: ('src.phase10_final_report', 'FinalReportGenerator')
    }
    
    if phase_number not in modules:
        raise ValueError(f"Invalid phase number: {phase_number}")
    
    module_name, class_name = modules[phase_number]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories for output."""
    directories = ['data', 'results', 'reports', 'charts']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Created output directories")


def run_full_pipeline():
    """
    Run the complete 10-phase pipeline.
    """
    logger.info("=" * 80)
    logger.info("PRICE MOVEMENT PROBABILITY DISCOVERY SYSTEM")
    logger.info("Starting Full Pipeline Analysis")
    logger.info("=" * 80)
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    
    try:
        # Create directories
        create_directories()
        
        # Phase 1: Data Acquisition & Validation
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: DATA ACQUISITION & VALIDATION")
        logger.info("=" * 80)
        DataAcquisition = get_phase_module(1)
        da = DataAcquisition()
        verified_data, verification_report = da.run_phase1()
        
        if verification_report.get('confidence_score', 0) < 98:
            logger.warning("Data confidence below 98%, but proceeding...")
        
        # Phase 2: Forward-Looking Price Movement Labeling
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: MOVEMENT LABELING")
        logger.info("=" * 80)
        MovementLabeling = get_phase_module(2)
        ml = MovementLabeling()
        movement_data = ml.run_phase2()
        
        # Phase 3: Market State Feature Engineering
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        FeatureEngineering = get_phase_module(3)
        fe = FeatureEngineering()
        features = fe.run_phase3()
        
        # Phase 4: Pattern Discovery
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: PATTERN DISCOVERY")
        logger.info("=" * 80)
        PatternDiscovery = get_phase_module(4)
        pd_obj = PatternDiscovery()
        classified = pd_obj.run_phase4()
        
        # Phase 5: Pattern Optimization
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5: PATTERN OPTIMIZATION")
        logger.info("=" * 80)
        PatternOptimization = get_phase_module(5)
        po = PatternOptimization()
        optimized = po.run_phase5()
        
        # Phase 6: Out-of-Sample Validation
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 6: OUT-OF-SAMPLE VALIDATION")
        logger.info("=" * 80)
        OutOfSampleValidation = get_phase_module(6)
        ov = OutOfSampleValidation()
        validated = ov.run_phase6()
        
        # Phase 7: Portfolio Construction
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 7: PORTFOLIO CONSTRUCTION")
        logger.info("=" * 80)
        PortfolioConstruction = get_phase_module(7)
        pc = PortfolioConstruction()
        portfolio = pc.run_phase7()
        
        # Phase 8: Visual Pattern Documentation (skipped - using standalone dashboard)
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 8: VISUALIZATION (SKIPPED - using standalone dashboard)")
        logger.info("=" * 80)
        visualizations = []
        
        # Generate HTML Dashboard with latest pipeline data
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING HTML DASHBOARD")
        logger.info("=" * 80)
        try:
            # Step 1: Convert OHLCV CSV to JSON for dashboard
            logger.info("Converting OHLCV data to JSON format...")
            import pandas as pd
            import json
            ohlcv_df = pd.read_csv('data/XOM_verified_ohlcv.csv')
            ohlcv_data = []
            for _, row in ohlcv_df.iterrows():
                ohlcv_data.append({
                    'Date': row['Date'],
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': int(row['Volume'])
                })
            with open('data/ohlcv.json', 'w') as f:
                json.dump(ohlcv_data, f, indent=2)
            logger.info(f"OHLCV data converted: {len(ohlcv_data)} records")
            
            # Note: patterns.json is now generated by Phase 7 with filter applied
            # (training>=80%, validation>=80%, occurrences>=30)
            logger.info("Dashboard patterns data generated by Phase 7")
        except Exception as e:
            logger.warning(f"Dashboard generation failed: {e}")
        
        # Phase 9: Real-Time Pattern Detection
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 9: REAL-TIME DETECTION")
        logger.info("=" * 80)
        RealTimePatternDetection = get_phase_module(9)
        rtd = RealTimePatternDetection()
        alerts = rtd.run_phase9()
        
        # Phase 10: Comprehensive Final Report
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 10: FINAL REPORT")
        logger.info("=" * 80)
        FinalReportGenerator = get_phase_module(10)
        frg = FinalReportGenerator()
        report = frg.run_phase10()
        
        # Record pipeline run metrics
        tracker = PipelineTracker()
        
        # Calculate average metrics
        robust_patterns = validated.get('robust', [])
        avg_validation_rate = sum(p.get('validation_success_rate', 0) for p in robust_patterns) / len(robust_patterns) if robust_patterns else 0
        # Use training success_rate from the pattern data
        avg_success_rate = sum(p.get('training_success_rate', p.get('success_rate', 0)) for p in robust_patterns) / len(robust_patterns) if robust_patterns else 0
        total_occurrences = sum(p.get('validation_occurrences', p.get('occurrences', 0)) for p in robust_patterns)
        
        metrics = {
            "data_confidence": verification_report.get('confidence_score', 0),
            "patterns_discovered": len(classified.get('high', [])) + len(classified.get('medium', [])),
            "patterns_validated": len(robust_patterns),
            "portfolio_size": len(portfolio),
            "avg_validation_rate": avg_validation_rate,
            "avg_success_rate": avg_success_rate,
            "total_occurrences": total_occurrences
        }
        
        run_id, comparison = tracker.record_run(metrics)
        tracker.print_comparison(comparison)
        
        # Final Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"End Time: {datetime.now().isoformat()}")
        logger.info(f"Run ID: {run_id}")
        
        logger.info("\nFinal Summary:")
        logger.info(f"  Data Confidence: {verification_report.get('confidence_score', 0):.2f}%")
        logger.info(f"  Features Calculated: {len(features.columns)}")
        logger.info(f"  Patterns Discovered: {len(classified.get('high', [])) + len(classified.get('medium', []))}")
        logger.info(f"  Validated Patterns: {len(validated.get('robust', []))}")
        logger.info(f"  Portfolio Size: {len(portfolio)}")
        logger.info(f"  Avg Validation Rate: {avg_validation_rate:.2f}%")
        logger.info(f"  Avg Success Rate: {avg_success_rate:.2f}%")
        
        logger.info("\nOutput Files:")
        logger.info("  Data: data/")
        logger.info("  Reports: reports/")
        logger.info("  Charts: charts/")
        logger.info("  Log: pipeline.log")
        logger.info("  Pipeline runs: data/pipeline_runs.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False


def run_single_phase(phase_number):
    """
    Run a single phase of the pipeline.
    
    Args:
        phase_number: Phase number to run (1-10)
    """
    logger.info(f"Running Phase {phase_number} only...")
    
    phase_map = {
        1: lambda: get_phase_module(1)().run_phase1(),
        2: lambda: get_phase_module(2)().run_phase2(),
        3: lambda: get_phase_module(3)().run_phase3(),
        4: lambda: get_phase_module(4)().run_phase4(),
        5: lambda: get_phase_module(5)().run_phase5(),
        6: lambda: get_phase_module(6)().run_phase6(),
        7: lambda: get_phase_module(7)().run_phase7(),
        8: lambda: get_phase_module(8)().run_phase8(),
        9: lambda: get_phase_module(9)().run_phase9(),
        10: lambda: get_phase_module(10)().run_phase10()
    }
    
    if phase_number not in phase_map:
        logger.error(f"Invalid phase number: {phase_number}")
        return False
    
    try:
        result = phase_map[phase_number]()
        logger.info(f"Phase {phase_number} completed successfully")
        return True
    except Exception as e:
        logger.error(f"Phase {phase_number} failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Price Movement Probability Discovery System'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=range(1, 11),
        help='Run a specific phase (1-10). If not specified, runs all phases.'
    )
    
    args = parser.parse_args()
    
    if args.phase:
        success = run_single_phase(args.phase)
    else:
        success = run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()