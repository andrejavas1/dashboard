"""
Phase 10: Comprehensive Final Report Module
Generates detailed research report with all findings.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalReportGenerator:
    """
    Generates comprehensive final report.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the report generator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Output directories
        self.reports_dir = self.config['output']['reports_dir']
        self.data_dir = self.config['output']['data_dir']
        self.charts_dir = self.config['output']['charts_dir']
        
        # Data storage
        self.data = None
        self.portfolio = None
        self.validation_results = None
        self.portfolio_summary = None
        self.report_content = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'data_sources': {'ticker': 'XOM', 'start_date': '2010-01-01'},
            'output': {'reports_dir': 'reports', 'data_dir': 'data', 'charts_dir': 'charts'}
        }
    
    def load_all_data(self):
        """Load all required data files."""
        logger.info("Loading all data files...")
        
        # Load features matrix
        features_path = os.path.join(self.data_dir, "features_matrix.csv")
        if os.path.exists(features_path):
            self.data = pd.read_csv(features_path, index_col='Date', parse_dates=True)
            logger.info(f"  Features: {len(self.data)} records")
        
        # Load portfolio
        portfolio_path = os.path.join(self.data_dir, "final_portfolio.json")
        if os.path.exists(portfolio_path):
            with open(portfolio_path, 'r') as f:
                self.portfolio = json.load(f)
            logger.info(f"  Portfolio: {len(self.portfolio)} patterns")
        
        # Load validation results
        validation_path = os.path.join(self.data_dir, "validation_results.json")
        if os.path.exists(validation_path):
            with open(validation_path, 'r') as f:
                self.validation_results = json.load(f)
            logger.info(f"  Validation results loaded")
        
        # Load portfolio summary
        summary_path = os.path.join(self.data_dir, "portfolio_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.portfolio_summary = json.load(f)
            logger.info(f"  Portfolio summary loaded")
    
    def generate_executive_summary(self) -> Dict:
        """
        Generate executive summary.
        
        Returns:
            Executive summary dictionary
        """
        logger.info("Generating executive summary...")
        
        # Count patterns by classification
        robust = len(self.validation_results.get('robust', [])) if self.validation_results else 0
        degraded = len(self.validation_results.get('degraded', [])) if self.validation_results else 0
        failed = len(self.validation_results.get('failed', [])) if self.validation_results else 0
        
        # Get top 5 patterns
        top_patterns = []
        if self.portfolio:
            for i, pattern in enumerate(self.portfolio[:5]):
                top_patterns.append({
                    'pattern_id': i,
                    'label': pattern.get('pattern', {}).get('label_col', ''),
                    'validation_success_rate': pattern.get('validation_success_rate', 0),
                    'training_success_rate': pattern.get('training_success_rate', 0),
                    'occurrences': pattern.get('validation_occurrences', 0)
                })
        
        summary = {
            'title': 'Price Movement Probability Discovery System - Executive Summary',
            'date_generated': datetime.now().isoformat(),
            'ticker': self.config['data_sources']['ticker'],
            'analysis_period': f"{self.config['data_sources']['start_date']} to present",
            'patterns_discovered': {
                'total': robust + degraded + failed,
                'high_confidence': robust,
                'medium_confidence': degraded,
                'rejected': failed
            },
            'portfolio_metrics': self.portfolio_summary or {},
            'top_5_patterns': top_patterns
        }
        
        return summary
    
    def generate_methodology_section(self) -> Dict:
        """
        Generate methodology section.
        
        Returns:
            Methodology dictionary
        """
        logger.info("Generating methodology section...")
        
        methodology = {
            'title': 'Methodology',
            'data_sources': {
                'primary': 'Yahoo Finance (yfinance)',
                'verification': ['Alpha Vantage', 'Tiingo', 'EODHD'],
                'date_range': f"{self.config['data_sources']['start_date']} to present",
                'granularity': 'Daily OHLCV'
            },
            'data_validation': {
                'cross_validation': '4 sources compared',
                'discrepancy_threshold': '0.5% for OHLC values',
                'consensus_threshold': '0.2% for 3+ source agreement',
                'volume_threshold': '10% discrepancy tolerance',
                'minimum_confidence': '98%'
            },
            'feature_engineering': {
                'total_features': '100+',
                'categories': [
                    'Price-based (MA distance, ROC, 52-week range)',
                    'Volatility (ATR, Bollinger Bands)',
                    'Momentum (RSI, MACD, Stochastic)',
                    'Volume (ratios, OBV, divergence)',
                    'Trend (ADX, moving averages)',
                    'Regime (volatility, trend classification)',
                    'Pattern recognition (support/resistance, breakouts)',
                    'Temporal (day of week, month, seasonality)'
                ]
            },
            'pattern_discovery': {
                'methods': [
                    'Rule-based pattern mining',
                    'Decision tree extraction',
                    'Clustering analysis',
                    'Sequential pattern mining'
                ],
                'filtering_criteria': {
                    'minimum_occurrences': 30,
                    'minimum_success_rate': '65%',
                    'statistical_significance': 'p-value < 0.05',
                    'temporal_consistency': 'Works across multiple periods'
                }
            },
            'validation': {
                'training_period': '2010-2020',
                'validation_period': '2021-2024',
                'live_period': '2025-present',
                'classification': {
                    'robust': 'Validation ≥85% of training',
                    'degraded': 'Validation 70-85% of training',
                    'failed': 'Validation <70% of training'
                }
            },
            'statistical_tests': [
                'Binomial test for success rate significance',
                'Chi-square test for training vs validation comparison',
                'Time series consistency analysis'
            ]
        }
        
        return methodology
    
    def generate_pattern_pages(self) -> List[Dict]:
        """
        Generate individual pattern pages.
        
        Returns:
            List of pattern page dictionaries
        """
        logger.info("Generating individual pattern pages...")
        
        pattern_pages = []
        
        if not self.portfolio:
            return pattern_pages
        
        for i, pattern in enumerate(self.portfolio):
            page = {
                'pattern_id': i,
                'pattern_name': f"Pattern {i}",
                'label_column': pattern.get('pattern', {}).get('label_col', ''),
                'conditions': pattern.get('pattern', {}).get('conditions', {}),
                'performance_metrics': {
                    'training_success_rate': pattern.get('training_success_rate', 0),
                    'validation_success_rate': pattern.get('validation_success_rate', 0),
                    'live_success_rate': pattern.get('live_success_rate', 0),
                    'total_occurrences': pattern.get('pattern', {}).get('occurrences', 0),
                    'validation_occurrences': pattern.get('validation_occurrences', 0),
                    'average_move': pattern.get('validation_avg_move', 0),
                    'average_time_to_target': pattern.get('validation_avg_time', 0)
                },
                'statistical_validation': {
                    'p_value': pattern.get('binomial_p_value', 1.0),
                    'statistically_significant': pattern.get('statistically_significant', False),
                    'classification': pattern.get('classification', 'N/A')
                },
                'risk_metrics': {
                    'false_positive_rate': pattern.get('validation_false_positive_rate', 0),
                    'validation_ratio': pattern.get('validation_ratio', 0)
                },
                'composite_score': pattern.get('scores', {}).get('composite', 0),
                'charts': {
                    'overview': f"pattern_{i}_overview.png",
                    'occurrences': f"pattern_{i}_occurrences.png",
                    'statistics': f"pattern_{i}_statistics.png",
                    'equity_curve': f"pattern_{i}_equity_curve.png"
                },
                'data_export': f"pattern_{i}_occurrences.csv"
            }
            
            pattern_pages.append(page)
        
        return pattern_pages
    
    def generate_portfolio_analysis(self) -> Dict:
        """
        Generate portfolio analysis section.
        
        Returns:
            Portfolio analysis dictionary
        """
        logger.info("Generating portfolio analysis...")
        
        if not self.portfolio_summary:
            return {}
        
        analysis = {
            'title': 'Portfolio Analysis',
            'summary': self.portfolio_summary,
            'diversification': {
                'total_patterns': len(self.portfolio) if self.portfolio else 0,
                'bullish_patterns': len([p for p in self.portfolio if 'STRONG_UP' in str(p)]),
                'bearish_patterns': len([p for p in self.portfolio if 'STRONG_DOWN' in str(p)]),
                'short_term_patterns': len([p for p in self.portfolio if '_3d' in str(p) or '_5d' in str(p)]),
                'medium_term_patterns': len([p for p in self.portfolio if '_10d' in str(p) or '_20d' in str(p) or '_30d' in str(p)])
            },
            'expected_performance': {
                'average_success_rate': self.portfolio_summary.get('average_validation_success_rate', 0),
                'expected_signals_per_month': self.portfolio_summary.get('expected_signals_per_month', 0),
                'average_composite_score': self.portfolio_summary.get('average_composite_score', 0)
            }
        }
        
        return analysis
    
    def generate_market_regime_analysis(self) -> Dict:
        """
        Generate market regime analysis section.
        
        Returns:
            Market regime analysis dictionary
        """
        logger.info("Generating market regime analysis...")
        
        regime_analysis = {
            'title': 'Market Regime Analysis',
            'regime_types': [
                'Strong Bull (MA20 slope > 0.1%, MA200 slope > 0.1%)',
                'Weak Bull (MA20 slope 0-0.1%, MA200 slope > 0%)',
                'Sideways (ADX < 25)',
                'Weak Bear (MA20 slope -0.1-0%, MA200 slope < 0%)',
                'Strong Bear (MA20 slope < -0.1%, MA200 slope < -0.1%)'
            ],
            'volatility_regimes': [
                'High (ATR in top tercile)',
                'Medium (ATR in middle tercile)',
                'Low (ATR in bottom tercile)'
            ],
            'note': 'Regime-specific patterns available in regime_patterns.json'
        }
        
        return regime_analysis
    
    def generate_backtesting_results(self) -> Dict:
        """
        Generate backtesting results section.
        
        Returns:
            Backtesting results dictionary
        """
        logger.info("Generating backtesting results...")
        
        backtesting = {
            'title': 'Backtesting Results',
            'assumptions': [
                'Trade every pattern signal',
                'No overlapping positions',
                '2% stop loss on failed trades',
                'Full position on each trade'
            ],
            'period': '2010 to present',
            'metrics': {
                'total_patterns': len(self.portfolio) if self.portfolio else 0,
                'average_success_rate': self.portfolio_summary.get('average_validation_success_rate', 0) if self.portfolio_summary else 0,
                'expected_annual_signals': self.portfolio_summary.get('expected_signals_per_month', 0) * 12 if self.portfolio_summary else 0
            },
            'note': 'Detailed equity curves available in charts directory'
        }
        
        return backtesting
    
    def generate_implementation_guide(self) -> Dict:
        """
        Generate implementation guide section.
        
        Returns:
            Implementation guide dictionary
        """
        logger.info("Generating implementation guide...")
        
        guide = {
            'title': 'Implementation Guide',
            'daily_monitoring': {
                'step1': 'Run phase9_realtime_detection.py to scan for patterns',
                'step2': 'Review daily_report.json for triggered patterns',
                'step3': 'Check dashboard.json for pattern status',
                'step4': 'Review actionable_info in alerts for entry details'
            },
            'alert_interpretation': {
                'triggered': 'All conditions met - consider entering trade',
                'almost_triggered': '90%+ conditions met - monitor closely',
                'confidence_score': 'Higher score = more conditions met'
            },
            'position_sizing': {
                'recommendation': 'Risk 1-2% of capital per trade',
                'calculation': 'Position size = (Capital × Risk%) / (Entry - Stop Loss)'
            },
            'risk_management': {
                'stop_loss': '2% below entry price',
                'time_limit': 'Exit if target not reached within time window',
                'max_concurrent': 'Limit to 3-5 concurrent positions'
            },
            'files_to_use': {
                'portfolio': 'data/final_portfolio.json',
                'alerts': 'reports/latest_alerts.json',
                'dashboard': 'reports/dashboard.json'
            }
        }
        
        return guide
    
    def generate_appendices(self) -> Dict:
        """
        Generate appendices section.
        
        Returns:
            Appendices dictionary
        """
        logger.info("Generating appendices...")
        
        appendices = {
            'title': 'Appendices',
            'feature_list': 'See data/feature_list.txt for complete feature definitions',
            'data_quality': 'See data/XOM_verification_report.yaml',
            'trade_logs': 'See charts/pattern_*_occurrences.csv for each pattern',
            'code_snippets': {
                'pattern_detection': 'See src/phase9_realtime_detection.py',
                'feature_calculation': 'See src/phase3_feature_engineering.py',
                'validation': 'See src/phase6_validation.py'
            }
        }
        
        return appendices
    
    def generate_full_report(self) -> Dict:
        """
        Generate complete final report.
        
        Returns:
            Complete report dictionary
        """
        logger.info("=" * 60)
        logger.info("PHASE 10: COMPREHENSIVE FINAL REPORT")
        logger.info("=" * 60)
        
        # Load all data
        self.load_all_data()
        
        # Generate all sections
        self.report_content = {
            'executive_summary': self.generate_executive_summary(),
            'methodology': self.generate_methodology_section(),
            'pattern_pages': self.generate_pattern_pages(),
            'portfolio_analysis': self.generate_portfolio_analysis(),
            'market_regime_analysis': self.generate_market_regime_analysis(),
            'backtesting_results': self.generate_backtesting_results(),
            'implementation_guide': self.generate_implementation_guide(),
            'appendices': self.generate_appendices(),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'ticker': self.config['data_sources']['ticker'],
                'analysis_period': f"{self.config['data_sources']['start_date']} to present",
                'total_patterns': len(self.portfolio) if self.portfolio else 0
            }
        }
        
        # Save report
        self.save_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 10 COMPLETE")
        logger.info("=" * 60)
        
        return self.report_content
    
    def save_report(self):
        """Save report to files."""
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Save full JSON report
        report_path = os.path.join(self.reports_dir, "final_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.report_content, f, indent=2, default=str)
        logger.info(f"Full report saved to {report_path}")
        
        # Save executive summary
        summary_path = os.path.join(self.reports_dir, "executive_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.report_content['executive_summary'], f, indent=2, default=str)
        logger.info(f"Executive summary saved to {summary_path}")
        
        # Save implementation guide
        guide_path = os.path.join(self.reports_dir, "implementation_guide.json")
        with open(guide_path, 'w') as f:
            json.dump(self.report_content['implementation_guide'], f, indent=2, default=str)
        logger.info(f"Implementation guide saved to {guide_path}")
        
        # Generate text summary
        self.generate_text_report()
    
    def generate_text_report(self):
        """Generate human-readable text report."""
        text_path = os.path.join(self.reports_dir, "final_report.txt")
        
        with open(text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PRICE MOVEMENT PROBABILITY DISCOVERY SYSTEM - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = self.report_content['executive_summary']
            f.write(f"Ticker: {summary['ticker']}\n")
            f.write(f"Analysis Period: {summary['analysis_period']}\n")
            f.write(f"Date Generated: {summary['date_generated']}\n\n")
            
            f.write("Patterns Discovered:\n")
            f.write(f"  Total: {summary['patterns_discovered']['total']}\n")
            f.write(f"  High Confidence: {summary['patterns_discovered']['high_confidence']}\n")
            f.write(f"  Medium Confidence: {summary['patterns_discovered']['medium_confidence']}\n")
            f.write(f"  Rejected: {summary['patterns_discovered']['rejected']}\n\n")
            
            if self.portfolio_summary:
                f.write("Portfolio Metrics:\n")
                f.write(f"  Total Patterns: {self.portfolio_summary.get('total_patterns', 0)}\n")
                f.write(f"  Avg Validation Success Rate: {self.portfolio_summary.get('average_validation_success_rate', 0):.2f}%\n")
                f.write(f"  Expected Signals/Month: {self.portfolio_summary.get('expected_signals_per_month', 0):.2f}\n\n")
            
            # Top 5 Patterns
            f.write("Top 5 Patterns:\n")
            for i, pattern in enumerate(summary['top_5_patterns'], 1):
                f.write(f"  {i}. Pattern {pattern['pattern_id']}: {pattern['label']}\n")
                f.write(f"     Validation: {pattern['validation_success_rate']:.1f}% | "
                       f"Training: {pattern['training_success_rate']:.1f}% | "
                       f"Occurrences: {pattern['occurrences']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("For detailed information, see final_report.json\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Text report saved to {text_path}")
    
    def run_phase10(self) -> Dict:
        """
        Run complete Phase 10: Comprehensive Final Report.
        
        Returns:
            Complete report dictionary
        """
        return self.generate_full_report()


if __name__ == "__main__":
    # Run Phase 10
    frg = FinalReportGenerator()
    report = frg.run_phase10()
    
    print(f"\nFinal Results:")
    print(f"  Report Generated: {report['metadata']['generated_at']}")
    print(f"  Total Patterns: {report['metadata']['total_patterns']}")
    print(f"  Report saved to: reports/final_report.json")