"""
Data models for IPO Review Agent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class InvestmentRecommendation(Enum):
    """Investment recommendation types."""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    AVOID = "Avoid"
    STRONG_SELL = "Strong Sell"


@dataclass
class CompanyBasics:
    """Basic company information."""
    name: str
    symbol: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None


@dataclass
class FinancialMetrics:
    """Financial metrics and ratios."""
    revenue: Dict[str, float]  # Year -> Revenue
    profit: Dict[str, float]   # Year -> Profit
    assets: Dict[str, float]   # Year -> Total Assets
    liabilities: Dict[str, float]  # Year -> Total Liabilities
    
    # Calculated metrics
    revenue_growth_rate: Optional[float] = None
    profit_margin: Optional[float] = None
    gross_profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    current_ratio: Optional[float] = None


@dataclass
class MarketData:
    """Market-related data."""
    ipo_price_range: Optional[tuple] = None  # (min_price, max_price)
    expected_listing_date: Optional[datetime] = None
    market_sentiment: Optional[str] = None
    sector_performance: Optional[float] = None
    recent_ipos_performance: Optional[List[Dict]] = None


@dataclass
@dataclass
class NewsAnalysis:
    """News sentiment and market intelligence."""
    sentiment_score: float  # -1 to 1 (negative to positive)
    key_themes: List[str]
    news_volume: int = 0  # Number of news articles analyzed
    positive_mentions: int = 0  # Number of positive sentiment articles
    negative_mentions: int = 0  # Number of negative sentiment articles
    # Optional fields for backward compatibility
    news_articles: List[Dict[str, Any]] = None
    market_trends: List[str] = None
    analyst_opinions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize optional fields if not provided."""
        if self.news_articles is None:
            self.news_articles = []
        if self.market_trends is None:
            self.market_trends = []
        if self.analyst_opinions is None:
            self.analyst_opinions = []


@dataclass
class RiskAssessment:
    """Risk analysis results."""
    overall_risk: RiskLevel
    financial_risk: RiskLevel
    market_risk: RiskLevel
    operational_risk: RiskLevel
    risk_factors: List[str]
    risk_mitigation: List[str]


@dataclass
class StrengthsAndWeaknesses:
    """Business strengths and weaknesses analysis."""
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    competitive_advantages: List[str]


@dataclass
class IPOAnalysisReport:
    """Complete IPO analysis report."""
    company: CompanyBasics
    financial_metrics: FinancialMetrics
    market_data: MarketData
    news_analysis: NewsAnalysis
    risk_assessment: RiskAssessment
    strengths_weaknesses: StrengthsAndWeaknesses
    
    # Analysis results
    listing_gain_prediction: Optional[float] = None
    long_term_score: Optional[float] = None  # 0-10 scale
    recommendation: Optional[InvestmentRecommendation] = None
    target_price: Optional[float] = None
    
    # Metadata
    analysis_date: datetime = datetime.now()
    analyst_confidence: Optional[float] = None  # 0-1 scale
