"""
Enhanced configuration for IPO Prospectus Integration with DRHP parsing.
Includes settings for data quality, validation, caching, and processing optimization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import tempfile


@dataclass
class EnhancedProspectusConfig:
    """Enhanced configuration for prospectus parsing and SEBI integration."""
    
    # SEBI Integration Settings
    sebi_base_url: str = "https://www.sebi.gov.in"
    sebi_search_endpoints: List[str] = None
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds between retries
    
    # PDF Processing Settings
    max_pages_to_process: int = 50
    pdf_extraction_timeout: int = 180  # 3 minutes max per PDF
    use_multiple_parsers: bool = True
    enable_table_extraction: bool = True
    
    # Data Quality Settings
    min_quality_threshold: float = 0.3  # Minimum quality score to accept data
    min_financial_years: int = 2  # Minimum years of financial data required
    validate_financial_logic: bool = True
    cross_validate_sources: bool = True
    
    # Caching Settings
    enable_caching: bool = True
    cache_duration_hours: int = 24
    max_cache_size_mb: int = 200
    cache_directory: Optional[str] = None
    
    # Performance Settings
    max_concurrent_downloads: int = 3
    download_chunk_size: int = 8192
    parallel_processing: bool = True
    
    # Data Extraction Limits
    max_risk_factors: int = 15
    max_strengths: int = 10
    max_use_of_funds: int = 8
    max_competitive_advantages: int = 7
    
    # Content Filtering
    min_business_description_length: int = 100
    max_business_description_length: int = 1000
    filter_boilerplate_text: bool = True
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        
        if self.sebi_search_endpoints is None:
            self.sebi_search_endpoints = [
                "/sebiweb/other/OtherAction.do?doRecognition=yes&intmId=13",
                "/sebiweb/action/CorporateDetails.do",
                "/sebiweb/action/CompanyDetails.do",
                "/sebiweb/action/IPOAction.do"
            ]
        
        if self.cache_directory is None:
            self.cache_directory = str(Path(tempfile.gettempdir()) / "ipo_prospectus_cache")
        
        # Validate settings
        if self.min_quality_threshold < 0 or self.min_quality_threshold > 1:
            raise ValueError("min_quality_threshold must be between 0 and 1")
        
        if self.cache_duration_hours <= 0:
            raise ValueError("cache_duration_hours must be positive")


# Enhanced Financial Data Extraction Patterns for Indian Companies
ENHANCED_FINANCIAL_PATTERNS = {
    'revenue_patterns': [
        # More comprehensive revenue patterns
        r'(?:Total\s+)?Revenue(?:\s+from\s+operations)?.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Net\s+Revenue.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Income\s+from\s+operations.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Revenue\s+from\s+sales.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Turnover.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Sales\s+and\s+other\s+income.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'
    ],
    
    'profit_patterns': [
        r'Net\s+Profit(?:\s+after\s+tax)?.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Profit\s+after\s+tax.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'(?:Net\s+)?Income\s+for\s+the\s+(?:year|period).*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Profit\s+attributable.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Net\s+earnings.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'
    ],
    
    'ebitda_patterns': [
        r'EBITDA.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Earnings\s+before\s+interest.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Operating\s+profit.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'EBIDTA.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'  # Common misspelling
    ],
    
    'assets_patterns': [
        r'Total\s+Assets.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Total\s+Current\s+Assets.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Non-current\s+Assets.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Fixed\s+Assets.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'
    ],
    
    'liability_patterns': [
        r'Total\s+Liabilities.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Current\s+Liabilities.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Non-current\s+Liabilities.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Total\s+Debt.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'
    ],
    
    'equity_patterns': [
        r'Total\s+Equity.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Shareholders?\s+Equity.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Net\s+Worth.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Equity\s+Share\s+Capital.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'
    ],
    
    'cash_flow_patterns': [
        r'Cash\s+Flow\s+from\s+Operations.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Operating\s+Cash\s+Flow.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Free\s+Cash\s+Flow.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)',
        r'Net\s+Cash\s+Generated.*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)'
    ],
    
    'year_patterns': [
        r'(?:FY|F\.Y\.?)\s*(\d{4})',
        r'(?:FY|F\.Y\.?)\s*(\d{2})',
        r'(\d{4})-(\d{2})',
        r'March\s+(\d{4})',
        r'Year\s+ended.*?(\d{4})',
        r'For\s+the\s+year\s+ended.*?(\d{4})',
        r'(\d{4})\s*-\s*(\d{4})'
    ]
}

# Section Keywords for Enhanced Document Parsing
ENHANCED_SECTION_KEYWORDS = {
    'financial_statements': [
        'Financial Statements', 'Audited Financial Results', 'Audited Accounts',
        'Profit and Loss', 'Statement of Profit and Loss', 'Income Statement',
        'Balance Sheet', 'Statement of Financial Position',
        'Cash Flow Statement', 'Statement of Cash Flows',
        'Financial Information', 'Financial Performance', 'Financial Highlights'
    ],
    
    'business_description': [
        'Business Overview', 'Company Profile', 'Business Description', 'Our Business',
        'Nature of Business', 'Business Model', 'Business Activities',
        'Company Background', 'Business Operations', 'Industry Overview'
    ],
    
    'risk_factors': [
        'Risk Factors', 'Principal Risks', 'Key Risks', 'Material Risks',
        'Risk Management', 'Business Risks', 'Regulatory Risks',
        'Risk Considerations', 'Factors that may affect', 'Cautionary Statement'
    ],
    
    'use_of_funds': [
        'Use of Funds', 'Objects of the Offer', 'Fund Utilization',
        'Proceeds Utilization', 'Application of Funds', 'Deployment of Funds',
        'Purpose of the Issue', 'Utilisation of Issue Proceeds'
    ],
    
    'company_strengths': [
        'Competitive Strengths', 'Business Strengths', 'Key Strengths',
        'Competitive Advantages', 'Strategic Advantages', 'Core Competencies',
        'Strengths and Advantages', 'Key Success Factors'
    ],
    
    'management': [
        'Management Discussion', 'Board of Directors', 'Key Personnel',
        'Management Team', 'Leadership Team', 'Senior Management'
    ]
}

# Data Quality Scoring Weights
QUALITY_SCORING_WEIGHTS = {
    'revenue_completeness': 0.20,     # 20% weight for revenue data
    'profit_completeness': 0.15,      # 15% weight for profit data
    'balance_sheet_completeness': 0.15, # 15% weight for balance sheet
    'ratios_availability': 0.10,      # 10% weight for ratios
    'business_description': 0.10,     # 10% weight for business info
    'risk_factors': 0.10,             # 10% weight for risks
    'use_of_funds': 0.10,             # 10% weight for fund usage
    'strengths': 0.10                 # 10% weight for strengths
}

# Validation Rules for Financial Data
VALIDATION_RULES = {
    'max_revenue_growth_rate': 10.0,  # 1000% year-over-year growth triggers warning
    'max_profit_margin': 0.8,         # 80% profit margin triggers warning
    'min_asset_coverage': 0.1,        # Assets should be at least 10% of revenue
    'max_debt_to_equity': 10.0,       # 10:1 debt to equity triggers warning
    'revenue_profit_consistency': True, # Revenue should generally be >= Profit
    'year_over_year_consistency': True  # Check for logical year progression
}

# Indian Market Specific Terms and Weights
INDIAN_MARKET_TERMS = {
    'positive_indicators': {
        'market leader': 0.95,
        'brand recognition': 0.85,
        'competitive advantage': 0.90,
        'strong distribution network': 0.80,
        'cost leadership': 0.85,
        'innovation capability': 0.75,
        'digital transformation': 0.80,
        'government support': 0.70,
        'regulatory compliance': 0.65,
        'experienced management': 0.75,
        'pan-india presence': 0.70,
        'scalable business model': 0.85,
        'recurring revenue': 0.90,
        'asset light model': 0.75,
        'strong financials': 0.80
    },
    
    'risk_indicators': {
        'intense competition': 0.80,
        'regulatory risk': 0.90,
        'economic slowdown': 0.75,
        'raw material price volatility': 0.70,
        'foreign exchange risk': 0.60,
        'technology disruption': 0.85,
        'customer concentration': 0.75,
        'high debt burden': 0.90,
        'working capital issues': 0.70,
        'key person dependency': 0.75,
        'cyclical industry': 0.65,
        'environmental regulations': 0.60,
        'geopolitical risks': 0.55,
        'pandemic impact': 0.70,
        'supply chain disruption': 0.65
    }
}

# SEBI Document Types and Processing Priority
SEBI_DOCUMENT_PRIORITIES = {
    'Draft Red Herring Prospectus': 1,
    'DRHP': 1,
    'Red Herring Prospectus': 2,
    'RHP': 2,
    'Prospectus': 3,
    'Offer Document': 4,
    'Information Memorandum': 5,
    'IPO Grading Report': 6,
    'Due Diligence Certificate': 7,
    'Abridged Prospectus': 8
}

# Default Configuration Instance
DEFAULT_CONFIG = EnhancedProspectusConfig()

def get_config() -> EnhancedProspectusConfig:
    """Get the default configuration instance."""
    return DEFAULT_CONFIG

def update_config(**kwargs) -> EnhancedProspectusConfig:
    """Update configuration with custom values."""
    for key, value in kwargs.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return DEFAULT_CONFIG
