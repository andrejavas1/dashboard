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
        
        # CRITICAL: Extract ticker from Data Acquisition config for ticker-aware paths
        ticker = da.config.get('data_sources', {}).get('ticker', 'UNKNOWN')
        logger.info(f"Using ticker: {ticker}")
        
        # Phase 7: Portfolio Construction
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 7: PORTFOLIO CONSTRUCTION")
        logger.info("=" * 80)
        PortfolioConstruction = get_phase_module(7)
        pc = PortfolioConstruction()
        # Pass ticker-specific validated_patterns path
        validated_patterns_path = f"data/tickers/{ticker}/validated_patterns.json"
        portfolio = pc.run_phase7(patterns_path=validated_patterns_path)
        # CRITICAL: Pass ticker to save_portfolio to write to ticker-specific directory
        pc.save_portfolio(ticker=ticker)
        
        # Generate Pattern Occurrences for Dashboard
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PATTERN OCCURRENCES")
        logger.info("=" * 80)
        try:
            import pandas as pd
            import json
            from src.realtime_feature_calculator import RealtimeFeatureCalculator
            
            # CRITICAL: Load from ticker-specific directory
            ticker_data_dir = f"data/tickers/{ticker}"
            ohlcv_path = f'{ticker_data_dir}/ohlcv.json'
            patterns_path = f'{ticker_data_dir}/patterns.json'
            
            logger.info(f"Loading occurrence data from {ticker_data_dir}/")
            with open(ohlcv_path, 'r') as f:
                ohlcv_data = json.load(f)
            with open(patterns_path, 'r') as f:
                patterns = json.load(f)
            
            # Calculate features
            logger.info("Calculating features for occurrence matching...")
            df = pd.DataFrame(ohlcv_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            calc = RealtimeFeatureCalculator()
            features = calc.calculate(df)
            merged = pd.concat([df.set_index('Date'), features], axis=1).reset_index()
            merged['Date_str'] = merged['Date'].dt.strftime('%Y-%m-%d')
            
            def match_conditions(row, conditions):
                """Check if row matches all pattern conditions."""
                for feature, condition in conditions.items():
                    if feature not in row or pd.isna(row[feature]):
                        return False
                    value = row[feature]
                    op = condition.get('operator', '>=')
                    target = condition['value']
                    if op == '>=':
                        if value < target: return False
                    elif op == '<=':
                        if value > target: return False
                    elif op == '>':
                        if value <= target: return False
                    elif op == '<':
                        if value >= target: return False
                return True
            
            # Generate occurrences for each pattern
            logger.info(f"Generating occurrences for {len(patterns)} patterns...")
            for i, pattern in enumerate(patterns):
                conditions = pattern.get('conditions', {})
                direction = pattern.get('direction', 'long')
                label_col = pattern.get('label_col', '')
                
                # Parse time window from label (e.g., "Label_1.0pct_5d" -> 5 days)
                time_window = 5
                if '_5d' in label_col:
                    time_window = 5
                elif '_10d' in label_col:
                    time_window = 10
                elif '_15d' in label_col:
                    time_window = 15
                elif '_20d' in label_col:
                    time_window = 20
                elif '_30d' in label_col:
                    time_window = 30
                
                occurrences = []
                for idx, row in merged.iterrows():
                    if match_conditions(row, conditions):
                        entry_price = row['Close']
                        exit_idx = min(idx + time_window, len(merged) - 1)
                        exit_price = merged.iloc[exit_idx]['Close']
                        actual_move = ((exit_price - entry_price) / entry_price) * 100
                        
                        # Determine outcome
                        threshold = 1.0  # Default 1%
                        if '1.0pct' in label_col:
                            threshold = 1.0
                        elif '2.0pct' in label_col:
                            threshold = 2.0
                        elif '3.0pct' in label_col:
                            threshold = 3.0
                        elif '5.0pct' in label_col:
                            threshold = 5.0
                        
                        if direction == 'long':
                            if actual_move >= threshold:
                                outcome = 'STRONG_UP'
                            elif actual_move > 0:
                                outcome = 'UP'
                            else:
                                outcome = 'DOWN'
                            target_reached = actual_move >= threshold
                        else:  # short
                            if actual_move <= -threshold:
                                outcome = 'STRONG_DOWN'
                            elif actual_move < 0:
                                outcome = 'DOWN'
                            else:
                                outcome = 'UP'
                            target_reached = actual_move <= -threshold
                        
                        occurrences.append({
                            'Date': row['Date_str'],
                            'Open': float(row['Open']),
                            'High': float(row['High']),
                            'Low': float(row['Low']),
                            'Close': float(entry_price),
                            'Volume': int(row['Volume']),
                            'outcome': outcome,
                            'actual_move': float(actual_move),
                            'time_to_target': time_window,
                            'target_reached': target_reached
                        })
                
                # Save occurrences to ticker-specific directory
                occurrences_dir = f'{ticker_data_dir}/occurrences'
                os.makedirs(occurrences_dir, exist_ok=True)
                with open(f'{occurrences_dir}/pattern_{i}_occurrences.json', 'w') as f:
                    json.dump(occurrences, f, indent=2)
                
                # UPDATE the pattern with actual occurrence count (critical for dashboard consistency)
                pattern['occurrences'] = len(occurrences)
                
                if i < 5:
                    logger.info(f"  Pattern #{i}: {len(occurrences)} occurrences")
            
            # Save updated patterns with correct occurrence counts
            with open(patterns_path, 'w') as f:
                json.dump(patterns, f, indent=2)
            logger.info(f"✓ Updated {len(patterns)} patterns with actual occurrence counts")
            
        except Exception as e:
            logger.error(f"Could not generate occurrences: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
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
            # Get ticker from DataAcquisition config
            ticker = da.config.get('data_sources', {}).get('ticker', 'XOM')
            verified_csv_path = f'data/{ticker}_verified_ohlcv.csv'
            logger.info(f"[DEBUG main.py] Loading OHLCV from: {verified_csv_path}")
            ohlcv_df = pd.read_csv(verified_csv_path)
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
        
        # Run pattern occurrence validation test
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING PATTERN OCCURRENCE VALIDATION TEST")
        logger.info("=" * 80)
        
        import subprocess
        try:
            test_result = subprocess.run(
                ['python', 'test_pattern_occurrences.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Log test output
            for line in test_result.stdout.split('\n'):
                if line.strip():
                    logger.info(line)
            
            if test_result.returncode == 0:
                logger.info("\n[TEST PASSED] All pattern occurrences validated successfully")
                test_status = "PASSED"
            else:
                logger.error("\n[TEST FAILED] Pattern occurrence validation failed")
                test_status = "FAILED"
                # Log stderr if there's an error
                if test_result.stderr:
                    for line in test_result.stderr.split('\n'):
                        if line.strip():
                            logger.error(line)
            
            # Save test status for dashboard
            test_status_data = {
                'timestamp': datetime.now().isoformat(),
                'status': test_status,
                'exit_code': test_result.returncode
            }
            with open('data/test_status.json', 'w') as f:
                json.dump(test_status_data, f, indent=2)
            
            # Copy data to per-ticker directory
            logger.info("\n" + "=" * 80)
            logger.info("COPYING DATA TO PER-TICKER DIRECTORY")
            logger.info("=" * 80)
            try:
                import shutil
                # Get ticker from config
                ticker = da.config.get('data_sources', {}).get('ticker', 'XOM')
                ticker_dir = f'data/tickers/{ticker}'
                os.makedirs(ticker_dir, exist_ok=True)
                os.makedirs(f'{ticker_dir}/occurrences', exist_ok=True)
                
                # Copy main files
                files_to_copy = [
                    ('data/ohlcv.json', f'{ticker_dir}/ohlcv.json'),
                    ('data/patterns.json', f'{ticker_dir}/patterns.json'),
                    ('data/validated_patterns.json', f'{ticker_dir}/validated_patterns.json'),
                    ('data/pattern_regime_analysis.json', f'{ticker_dir}/pattern_regime_analysis.json'),
                    ('data/test_status.json', f'{ticker_dir}/test_status.json'),
                    ('data/features_matrix.csv', f'{ticker_dir}/features_matrix.csv'),
                ]
                
                for src, dst in files_to_copy:
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        logger.info(f"  Copied {src} -> {dst}")
                
                # Clear old occurrence files to prevent stale data
                occ_dir = f'{ticker_dir}/occurrences'
                for old_file in os.listdir(occ_dir):
                    if old_file.startswith('pattern_') and old_file.endswith('_occurrences.json'):
                        os.remove(f'{occ_dir}/{old_file}')
                logger.info(f"  Cleared old occurrence files from {occ_dir}")
                
                # Copy occurrence files
                occurrence_count = 0
                for occ_file in os.listdir('data'):
                    if occ_file.startswith('pattern_') and occ_file.endswith('_occurrences.json'):
                        src = f'data/{occ_file}'
                        dst = f'{ticker_dir}/occurrences/{occ_file}'
                        shutil.copy2(src, dst)
                        occurrence_count += 1
                
                logger.info(f"  Copied {occurrence_count} occurrence files")
                logger.info(f"Data saved to {ticker_dir}/")
            except Exception as copy_error:
                logger.error(f"Error copying to ticker directory: {copy_error}")
                
        except Exception as test_error:
            logger.error(f"Could not run validation test: {test_error}")
            test_status = "ERROR"
        
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