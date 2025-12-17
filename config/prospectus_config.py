"""
Configuration for IPO Prospectus integration.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProspectusConfig:
    """Configuration for prospectus parsing and SEBI integration."""
    
    # SEBI website settings
    sebi_base_url: str = "https://www.sebi.gov.in"
    sebi_search_url: str = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do"
    request_timeout: int = 30
    max_retries: int = 3
    
    # PDF processing settings
    max_pages_to_process: int = 50  # Limit for performance
    pdf_extraction_timeout: int = 120  # 2 minutes max per PDF
    
    # Document search keywords
    ipo_document_keywords: List[str] = None
    financial_section_keywords: List[str] = None
    risk_section_keywords: List[str] = None
    
    # Caching settings
    enable_caching: bool = True
    cache_duration_hours: int = 24
    max_cache_size_mb: int = 100
    
    # Data quality settings
    min_financial_years: int = 2  # Minimum years of financial data required
    max_risk_factors: int = 10
    max_strengths: int = 7
    
    def __post_init__(self):
        if self.ipo_document_keywords is None:
            self.ipo_document_keywords = [
                'DRHP', 'Draft Red Herring Prospectus', 'Prospectus',
                'Red Herring', 'IPO Document', 'Offer Document'
            ]
        
        if self.financial_section_keywords is None:
            self.financial_section_keywords = [
                'Financial Statements', 'Audited Financial Results',
                'Profit and Loss', 'Balance Sheet', 'Cash Flow Statement',
                'Statement of Income', 'Financial Information'
            ]
        
        if self.risk_section_keywords is None:
            self.risk_section_keywords = [
                'Risk Factors', 'Principal Risks', 'Key Risks',
                'Risk Management', 'Material Risks', 'Business Risks'
            ]


# Financial data extraction patterns for Indian companies
INDIAN_FINANCIAL_PATTERNS = {
    'revenue_patterns': [
        r'Total Revenue.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Net Revenue.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Income from Operations.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Revenue from Operations.*?(\d{1,3}(?:,\d{3})*\.?\d*)'
    ],
    'profit_patterns': [
        r'Net Profit.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Profit After Tax.*?(\d{1,3}(?:,\d{3})*\.?\d*)', 
        r'Net Income.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Profit for the year.*?(\d{1,3}(?:,\d{3})*\.?\d*)'
    ],
    'assets_patterns': [
        r'Total Assets.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Total Current Assets.*?(\d{1,3}(?:,\d{3})*\.?\d*)',
        r'Non-current Assets.*?(\d{1,3}(?:,\d{3})*\.?\d*)'
    ],
    'year_patterns': [
        r'FY\s*(\d{4})',
        r'20(\d{2})-(\d{2})',
        r'March\s+(\d{4})',
        r'Year ended.*?(\d{4})'
    ]
}

# Common Indian business terms and their importance weights
INDIAN_BUSINESS_TERMS = {
    'positive_indicators': {
        'market leader': 0.9,
        'brand recognition': 0.8, 
        'competitive advantage': 0.9,
        'strong distribution': 0.7,
        'cost leadership': 0.8,
        'innovation': 0.7,
        'digital transformation': 0.8,
        'government support': 0.6,
        'regulatory compliance': 0.6,
        'experienced management': 0.7
    },
    'risk_indicators': {
        'intense competition': 0.8,
        'regulatory risk': 0.9,
        'economic slowdown': 0.7,
        'raw material price': 0.6,
        'foreign exchange': 0.5,
        'technology disruption': 0.8,
        'customer concentration': 0.7,
        'debt burden': 0.9,
        'working capital': 0.6,
        'key person dependency': 0.7
    }
}

# SEBI document types and their priority
SEBI_DOCUMENT_PRIORITY = {
    'Draft Red Herring Prospectus': 1,
    'DRHP': 1,
    'Red Herring Prospectus': 2,
    'Prospectus': 3,
    'Offer Document': 4,
    'IPO Grading': 5,
    'Due Diligence Certificate': 6
}
