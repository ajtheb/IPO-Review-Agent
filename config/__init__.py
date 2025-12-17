"""
Configuration settings for IPO Review Agent.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class APIConfig:
    """API configuration settings."""
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    news_api_key: str = os.getenv('NEWS_API_KEY', '')
    # Finnhub removed - not suitable for pre-IPO Indian companies
    
    # LLM API keys for enhanced analysis
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    anthropic_api_key: str = os.getenv('ANTHROPIC_API_KEY', '')
    groq_api_key: str = os.getenv('GROQ_API_KEY', '')
    gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')


@dataclass
class AnalysisConfig:
    """Analysis configuration settings."""
    # Financial analysis parameters
    min_years_data: int = 3
    revenue_growth_threshold: float = 0.15  # 15%
    profit_margin_threshold: float = 0.10   # 10%
    
    # News analysis parameters
    news_days_back: int = 30
    max_news_articles: int = 50
    sentiment_threshold: float = 0.1
    
    # Risk assessment parameters
    risk_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.risk_weights is None:
            self.risk_weights = {
                'financial': 0.4,
                'market': 0.3,
                'operational': 0.3
            }


@dataclass
class UIConfig:
    """User interface configuration."""
    app_title: str = "IPO Review Agent"
    page_icon: str = "📈"
    layout: str = "wide"
    
    # Color scheme
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e', 
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff7f0e',
                'info': '#17a2b8'
            }


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.api = APIConfig()
        self.analysis = AnalysisConfig()
        self.ui = UIConfig()
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Indian market industry benchmarks
        self.industry_benchmarks = {
            'Technology': {
                'revenue_growth': 0.25,  # Higher growth expected in Indian tech
                'profit_margin': 0.18,
                'debt_to_equity': 0.2,
                'pe_ratio': 30,
                'ipo_success_rate': 0.7
            },
            'Financial Services': {
                'revenue_growth': 0.15,
                'profit_margin': 0.20,
                'debt_to_equity': 0.6,
                'pe_ratio': 18,
                'ipo_success_rate': 0.8
            },
            'Healthcare': {
                'revenue_growth': 0.18,
                'profit_margin': 0.15,
                'debt_to_equity': 0.3,
                'pe_ratio': 25,
                'ipo_success_rate': 0.75
            },
            'Consumer Goods': {
                'revenue_growth': 0.12,
                'profit_margin': 0.12,
                'debt_to_equity': 0.4,
                'pe_ratio': 22,
                'ipo_success_rate': 0.65
            },
            'Automotive': {
                'revenue_growth': 0.10,
                'profit_margin': 0.08,
                'debt_to_equity': 0.7,
                'pe_ratio': 15,
                'ipo_success_rate': 0.6
            },
            'Pharmaceuticals': {
                'revenue_growth': 0.15,
                'profit_margin': 0.16,
                'debt_to_equity': 0.3,
                'pe_ratio': 24,
                'ipo_success_rate': 0.8
            },
            'Manufacturing': {
                'revenue_growth': 0.12,
                'profit_margin': 0.10,
                'debt_to_equity': 0.6,
                'pe_ratio': 16,
                'ipo_success_rate': 0.65
            },
            'Retail': {
                'revenue_growth': 0.14,
                'profit_margin': 0.06,
                'debt_to_equity': 0.5,
                'pe_ratio': 20,
                'ipo_success_rate': 0.5
            },
            'Real Estate': {
                'revenue_growth': 0.08,
                'profit_margin': 0.12,
                'debt_to_equity': 0.8,
                'pe_ratio': 12,
                'ipo_success_rate': 0.45
            },
            'Telecom': {
                'revenue_growth': 0.06,
                'profit_margin': 0.15,
                'debt_to_equity': 0.9,
                'pe_ratio': 14,
                'ipo_success_rate': 0.6
            },
            'Default': {
                'revenue_growth': 0.12,
                'profit_margin': 0.10,
                'debt_to_equity': 0.5,
                'pe_ratio': 18,
                'ipo_success_rate': 0.6
            }
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check API keys
        if not self.api.alpha_vantage_key:
            issues.append("No financial data API keys configured")
        
        if not self.api.news_api_key:
            issues.append("News API key not configured")
        
        # Check analysis parameters
        if self.analysis.min_years_data < 1:
            issues.append("Minimum years of data must be at least 1")
        
        if not (0 <= self.analysis.sentiment_threshold <= 1):
            issues.append("Sentiment threshold must be between 0 and 1")
        
        return issues


# Global configuration instance
config = Config()
