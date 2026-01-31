"""
Context7 Pattern Discovery Assistant

This module integrates Context7 to provide documentation and guidance
for the technical indicators and machine learning techniques used in
our high success pattern discovery system.
"""

import subprocess
import json
import time
import threading
from typing import Dict, List, Optional

class Context7PatternAssistant:
    """Assistant that uses Context7 to provide documentation for pattern discovery"""
    
    def __init__(self):
        self.server_process = None
        self.is_running = False
    
    def start_server(self):
        """Start the Context7 MCP server"""
        if self.is_running:
            print("Context7 server is already running")
            return True
            
        try:
            print("Starting Context7 MCP server...")
            # Try different ways to find the executable on Windows
            import shutil
            executable = shutil.which("context7-mcp")
            if executable is None:
                # Try with .cmd extension on Windows
                executable = shutil.which("context7-mcp.cmd")
            
            if executable is None:
                print("Context7 MCP executable not found. Please ensure it's installed and in PATH.")
                print("Try: npm install -g @upstash/context7-mcp")
                return False
            
            self.server_process = subprocess.Popen([
                executable,
                "--transport", "stdio"
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give the server a moment to start
            time.sleep(2)
            
            self.is_running = True
            print("Context7 MCP server started successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to start Context7 server: {e}")
            return False
    
    def stop_server(self):
        """Stop the Context7 MCP server"""
        if self.server_process and self.is_running:
            print("Stopping Context7 MCP server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.is_running = False
            print("Context7 MCP server stopped.")
    
    def get_technical_indicator_info(self, indicator_name: str) -> str:
        """Get information about a technical indicator"""
        queries = {
            "rsi": "Explain Relative Strength Index (RSI) calculation and best practices for trading",
            "stochastic": "Explain Stochastic Oscillator calculation and trading signals",
            "bollinger_bands": "Explain Bollinger Bands calculation and how to use them in trading",
            "macd": "Explain Moving Average Convergence Divergence (MACD) and trading strategies",
            "atr": "Explain Average True Range (ATR) calculation and volatility measurement",
            "roc": "Explain Rate of Change (ROC) indicator and momentum trading",
            "obv": "Explain On-Balance Volume (OBV) and volume analysis techniques",
            "ad": "Explain Accumulation/Distribution indicator and money flow analysis",
            "ma_alignment": "Explain moving average alignment strategies and trend identification",
            "volume_analysis": "Explain volume analysis techniques for pattern confirmation"
        }
        
        query = queries.get(indicator_name.lower(), 
                           f"Explain {indicator_name} technical indicator and its use in trading")
        
        return self.query_context7(query)
    
    def get_machine_learning_guidance(self, technique: str) -> str:
        """Get guidance on machine learning techniques for pattern discovery"""
        queries = {
            "feature_engineering": "Best practices for feature engineering in financial machine learning",
            "pattern_recognition": "Machine learning approaches for financial pattern recognition",
            "time_series": "Time series analysis techniques for financial data",
            "validation": "Best practices for validating financial machine learning models",
            "overfitting": "How to avoid overfitting in financial machine learning models",
            "ensemble": "Ensemble methods for financial pattern recognition and trading signals"
        }
        
        query = queries.get(technique.lower(), 
                           f"Machine learning techniques for {technique} in financial markets")
        
        return self.query_context7(query)
    
    def query_context7(self, question: str) -> str:
        """Query Context7 with a specific question"""
        if not self.is_running:
            if not self.start_server():
                return "Unable to query Context7: server not running"
        
        # In a real implementation, this would communicate with the MCP server
        # For now, we'll return a template response
        return f"Context7 response for: '{question}'\n[Documentation and code examples would appear here]"
    
    def get_pattern_condition_explanation(self, feature: str, operator: str, threshold: float) -> str:
        """Get explanation for a specific pattern condition"""
        explanations = {
            "RSI_7": f"7-day Relative Strength Index - measures speed and change of price movements. Values <=25 indicate oversold conditions, >=75 indicate overbought conditions.",
            "RSI_14": f"14-day Relative Strength Index - standard RSI period. Values <=20 indicate extreme oversold conditions, >=80 indicate extreme overbought conditions.",
            "Stoch_14_K": f"14-day Stochastic %K - momentum indicator. Values <=10 indicate extreme oversold conditions, >=90 indicate extreme overbought conditions.",
            "ATR_14_Percentile": f"14-day ATR Percentile - volatility percentile rank. Values <=20 indicate very low volatility, >=80 indicate very high volatility.",
            "BB_Width_20": f"20-day Bollinger Band Width - measures band width as percentage. Values >=25 indicate wide bands (high volatility), <=10 indicate narrow bands (low volatility).",
            "Volume_MA_ratio_5d": f"5-day Volume vs Moving Average ratio. Values >=3.0 indicate very high volume, <=0.2 indicate very low volume.",
            "Dist_MA_200": f"Distance from 200-day Moving Average as percentage. Values >=20 indicate price is far above MA, <=-20 indicate price is far below MA.",
            "MA_Alignment_Score": f"Moving Average Alignment Score (0-100). Values >=80 indicate strong bullish alignment, <=20 indicate strong bearish alignment.",
            "Dist_100d_Low": f"Distance from 100-day low as percentage. Values >=50 indicate price is in upper half of 100-day range.",
            "Dist_100d_High": f"Distance from 100-day high as percentage. Values <=-50 indicate price is in lower half of 100-day range.",
            "Days_Since_52w_High": f"Days since 52-week high. Values >=250 indicate it's been a long time since making a new high.",
            "Days_Since_52w_Low": f"Days since 52-week low. Values >=250 indicate it's been a long time since making a new low."
        }
        
        base_explanation = explanations.get(feature, f"{feature} - technical indicator condition")
        return f"{base_explanation}\nCondition: {operator} {threshold}"

def create_context7_dashboard_integration():
    """Create integration between Context7 and our pattern dashboards"""
    
    # This would integrate Context7 documentation directly into our dashboards
    integration_code = """
    // In dashboard HTML - JavaScript integration with Context7
    function getContext7Help(feature) {
        // This would query Context7 for documentation about the feature
        fetch('/api/context7/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: `Explain ${feature} technical indicator and its use in trading`
            })
        })
        .then(response => response.json())
        .then(data => {
            // Display Context7 documentation in dashboard
            showDocumentationModal(data.documentation, data.codeExamples);
        });
    }
    """
    
    return integration_code

def main():
    """Main function to demonstrate Context7 integration"""
    print("Context7 Pattern Discovery Assistant")
    print("=" * 40)
    
    # Create assistant
    assistant = Context7PatternAssistant()
    
    # Start server
    if assistant.start_server():
        print("\nContext7 Assistant is ready!")
        print("Available functions:")
        print("1. get_technical_indicator_info(indicator_name)")
        print("2. get_machine_learning_guidance(technique)")
        print("3. get_pattern_condition_explanation(feature, operator, threshold)")
        print("4. query_context7(custom_question)")
        
        # Example usage
        print("\nExample: Getting information about RSI...")
        rsi_info = assistant.get_technical_indicator_info("rsi")
        print(rsi_info[:200] + "..." if len(rsi_info) > 200 else rsi_info)
        
        print("\nExample: Getting information about a pattern condition...")
        condition_info = assistant.get_pattern_condition_explanation("RSI_14", ">=", 80.0)
        print(condition_info)
        
        # Keep server running for Roo to use
        print("\nContext7 server is running and available for Roo to use.")
        print("Press Ctrl+C to stop the server.")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            assistant.stop_server()

if __name__ == "__main__":
    main()