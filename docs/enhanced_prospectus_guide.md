# Enhanced IPO Prospectus Integration - Complete Guide

This guide explains how to use the enhanced IPO Prospectus Integration to parse DRHP (Draft Red Herring Prospectus) documents for real financial data from Indian IPOs.

## 🚀 Overview

The Enhanced IPO Prospectus Integration provides:
- **Automated SEBI Filing Discovery**: Searches multiple SEBI endpoints for IPO documents
- **Advanced PDF Parsing**: Uses multiple parsing engines for better accuracy
- **Financial Data Extraction**: Extracts revenue, profit, assets, ratios automatically
- **Data Quality Validation**: Validates extracted data for logical consistency
- **Smart Caching**: Avoids re-processing documents with configurable caching
- **Comprehensive Analysis**: Extracts business descriptions, risks, strengths, use of funds

## 📦 Installation

### Basic Installation
```bash
# Install basic dependencies
pip install PyPDF2 pdfplumber beautifulsoup4

# Install enhanced dependencies for table extraction
pip install tabula-py

# Note: tabula-py requires Java to be installed on your system
```

### Java Installation (for tabula-py)
```bash
# On macOS
brew install openjdk

# On Ubuntu/Debian
sudo apt-get install default-jre

# On Windows
# Download and install Java JRE from Oracle
```

## 🔧 Configuration

### Basic Configuration
```python
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusDataSource

# Initialize with default settings
prospectus_source = EnhancedProspectusDataSource()

# Initialize with custom settings
prospectus_source = EnhancedProspectusDataSource(cache_enabled=True)
```

### Advanced Configuration
```python
from config.enhanced_prospectus_config import EnhancedProspectusConfig, update_config

# Customize configuration
config = update_config(
    min_quality_threshold=0.4,      # Require higher quality data
    cache_duration_hours=48,        # Cache for 2 days
    max_pages_to_process=100,       # Process more pages
    enable_table_extraction=True,   # Use tabula for tables
    parallel_processing=True        # Enable parallel processing
)
```

## 📊 Usage Examples

### 1. Basic Usage - Get Enhanced Financial Data
```python
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusDataSource

# Initialize the source
prospectus_source = EnhancedProspectusDataSource()

# Get financial data for an IPO company
company_name = "Your IPO Company Name"
financial_data = prospectus_source.get_enhanced_ipo_data(company_name)

if financial_data:
    print(f"Quality Score: {financial_data.data_quality_score:.2f}")
    print(f"Revenue Data: {financial_data.revenue_data}")
    print(f"Profit Data: {financial_data.profit_data}")
    print(f"Business Description: {financial_data.business_description[:200]}...")
    print(f"Risk Factors: {len(financial_data.risk_factors)} identified")
else:
    print("No quality financial data found")
```

### 2. Using with DataSourceManager (Recommended)
```python
from src.data_sources import DataSourceManager

# Initialize with enhanced prospectus enabled
manager = DataSourceManager(use_enhanced_prospectus=True)

# Define IPO details
ipo_details = {
    'company_name': 'Your IPO Company',
    'sector': 'Technology',
    'price_range': '100-120',
    'exchange': 'NSE'
}

# Collect comprehensive data
all_data = manager.collect_ipo_data(
    ipo_details['company_name'], 
    ipo_details
)

# Check results
if 'enhanced_prospectus' in all_data:
    enhanced_data = all_data['enhanced_prospectus']
    if enhanced_data:
        print(f"✅ Enhanced data quality: {enhanced_data.data_quality_score:.2f}")
        
        # Access financial metrics
        if enhanced_data.revenue_data:
            print("Revenue Trend:")
            for year, amount in enhanced_data.revenue_data.items():
                print(f"  {year}: ₹{amount:,.0f} Cr")
        
        # Access qualitative data
        print(f"Risk Factors: {len(enhanced_data.risk_factors)}")
        print(f"Use of Funds: {len(enhanced_data.use_of_funds)}")
```

### 3. Data Quality Assessment
```python
# Get data summary before full processing
summary = prospectus_source.get_data_summary(company_name)

print("Data Availability Summary:")
print(f"SEBI Filings Found: {summary.get('sebi_filings_found', 0)}")
print(f"Cached Data: {summary.get('cached', False)}")
print(f"Estimated Processing Time: {summary.get('estimated_processing_time', 'Unknown')}")

# Process only if data is available
if summary.get('sebi_filings_found', 0) > 0:
    financial_data = prospectus_source.get_enhanced_ipo_data(company_name)
    
    if financial_data:
        quality = financial_data.data_quality_score
        
        if quality >= 0.8:
            print("🟢 High quality data extracted")
        elif quality >= 0.5:
            print("🟡 Medium quality data extracted")
        else:
            print("🔴 Low quality data - use with caution")
        
        # Check validation issues
        if financial_data.validation_flags:
            print("⚠️ Data validation issues:")
            for flag in financial_data.validation_flags:
                print(f"  - {flag}")
```

### 4. Cache Management
```python
# Force refresh cached data
refreshed = prospectus_source.get_enhanced_ipo_data(
    company_name, 
    force_refresh=True
)

# Or refresh via DataSourceManager
manager = DataSourceManager(use_enhanced_prospectus=True)
success = manager.refresh_prospectus_cache(company_name)
print(f"Cache refresh: {'✅ Success' if success else '❌ Failed'}")
```

### 5. Batch Processing Multiple Companies
```python
import time
from concurrent.futures import ThreadPoolExecutor

def process_company(company_name):
    """Process a single company's prospectus data."""
    try:
        prospectus_source = EnhancedProspectusDataSource()
        data = prospectus_source.get_enhanced_ipo_data(company_name)
        
        if data:
            return {
                'company': company_name,
                'quality_score': data.data_quality_score,
                'revenue_years': len(data.revenue_data),
                'profit_years': len(data.profit_data),
                'processing_time': time.time()
            }
        else:
            return {'company': company_name, 'error': 'No data found'}
            
    except Exception as e:
        return {'company': company_name, 'error': str(e)}

# Process multiple companies in parallel
companies = ['Company A', 'Company B', 'Company C']

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_company, companies))

# Display results
for result in results:
    if 'error' in result:
        print(f"{result['company']}: ❌ {result['error']}")
    else:
        print(f"{result['company']}: ✅ Quality {result['quality_score']:.2f}")
```

## 📋 Data Structure

### Enhanced Financial Data Object
```python
@dataclass
class EnhancedFinancialData:
    # Financial metrics (all in ₹ Crores)
    revenue_data: Dict[str, float]          # {'FY2023': 1000.0, 'FY2022': 800.0}
    profit_data: Dict[str, float]           # {'FY2023': 100.0, 'FY2022': 80.0}
    ebitda_data: Dict[str, float]           # EBITDA by year
    assets_data: Dict[str, float]           # Total assets by year
    liabilities_data: Dict[str, float]      # Total liabilities
    equity_data: Dict[str, float]           # Shareholder equity
    cash_flow_data: Dict[str, float]        # Operating cash flow
    
    # Calculated metrics
    key_ratios: Dict[str, float]            # {'profit_margin': 10.0, 'asset_turnover': 0.5}
    growth_metrics: Dict[str, float]        # {'revenue_cagr': 25.0, 'profit_growth': 20.0}
    
    # Qualitative information
    business_description: str               # Company business overview
    risk_factors: List[str]                 # Key risk factors (up to 15)
    use_of_funds: List[str]                 # How IPO funds will be used
    company_strengths: List[str]            # Key competitive strengths
    competitive_advantages: List[str]       # Specific competitive advantages
    
    # Quality metadata
    extraction_date: str                    # ISO timestamp of extraction
    data_quality_score: float              # 0.0 to 1.0 quality score
    source_confidence: float               # Parser confidence level
    validation_flags: List[str]            # Any validation issues found
```

### Accessing Financial Data
```python
if financial_data:
    # Financial metrics
    latest_revenue = max(financial_data.revenue_data.values()) if financial_data.revenue_data else 0
    latest_profit = max(financial_data.profit_data.values()) if financial_data.profit_data else 0
    
    # Calculate profit margin if data available
    if financial_data.key_ratios.get('profit_margin'):
        margin = financial_data.key_ratios['profit_margin']
        print(f"Profit Margin: {margin:.1f}%")
    
    # Growth metrics
    if 'revenue_cagr' in financial_data.growth_metrics:
        cagr = financial_data.growth_metrics['revenue_cagr']
        print(f"Revenue CAGR: {cagr:.1f}%")
    
    # Qualitative analysis
    print(f"Business: {financial_data.business_description[:200]}...")
    print(f"\nTop 3 Risk Factors:")
    for i, risk in enumerate(financial_data.risk_factors[:3], 1):
        print(f"{i}. {risk}")
    
    print(f"\nUse of Funds:")
    for i, use in enumerate(financial_data.use_of_funds, 1):
        print(f"{i}. {use}")
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. No SEBI Filings Found
```python
# Possible causes and solutions:
# - Company name variation: Try different name formats
company_variants = [
    "Full Company Name Limited",
    "Short Company Name Ltd",
    "Company Name",
    "Abbreviated Name"
]

for variant in company_variants:
    summary = prospectus_source.get_data_summary(variant)
    if summary.get('sebi_filings_found', 0) > 0:
        print(f"Found filings for: {variant}")
        break
```

#### 2. Low Quality Scores
```python
# Check what data is missing
if financial_data and financial_data.data_quality_score < 0.5:
    print("Quality Analysis:")
    print(f"Revenue years: {len(financial_data.revenue_data)}")
    print(f"Profit years: {len(financial_data.profit_data)}")
    print(f"Business description: {len(financial_data.business_description)} chars")
    print(f"Risk factors: {len(financial_data.risk_factors)}")
    
    # Check validation issues
    if financial_data.validation_flags:
        print("Validation Issues:")
        for flag in financial_data.validation_flags:
            print(f"  - {flag}")
```

#### 3. Java/Tabula Issues
```python
# If tabula-py fails, the parser will fallback to other methods
# Check logs for tabula errors and ensure Java is installed

# Test Java installation
import subprocess
try:
    result = subprocess.run(['java', '-version'], 
                          capture_output=True, text=True)
    print("Java is available")
except FileNotFoundError:
    print("Java not found - install Java JRE for enhanced table extraction")
```

#### 4. PDF Processing Errors
```python
# The parser uses multiple PDF processing methods
# If one fails, it automatically tries others

# To debug PDF issues, check the logs
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed PDF processing information
```

## 🎯 Best Practices

### 1. Production Usage
```python
# Use try-catch for robust error handling
try:
    manager = DataSourceManager(use_enhanced_prospectus=True)
    data = manager.collect_ipo_data(company_name, ipo_details)
    
    if data.get('enhanced_prospectus'):
        # Process enhanced data
        enhanced = data['enhanced_prospectus']
        
        # Only use high-quality data for critical decisions
        if enhanced.data_quality_score >= 0.7:
            # Proceed with analysis
            pass
        else:
            # Use basic analysis or manual review
            pass
            
except Exception as e:
    logger.error(f"Prospectus integration failed: {e}")
    # Fallback to basic IPO analysis
```

### 2. Performance Optimization
```python
# For better performance:
# 1. Enable caching
config = update_config(
    enable_caching=True,
    cache_duration_hours=48
)

# 2. Use parallel processing for multiple companies
config = update_config(parallel_processing=True)

# 3. Limit pages processed for faster parsing
config = update_config(max_pages_to_process=30)

# 4. Get summary before full processing
summary = prospectus_source.get_data_summary(company_name)
if summary.get('sebi_filings_found', 0) == 0:
    # Skip full processing if no filings found
    pass
```

### 3. Data Validation
```python
# Always validate critical financial data
if enhanced_data:
    # Check for reasonable values
    if enhanced_data.revenue_data:
        revenues = list(enhanced_data.revenue_data.values())
        if max(revenues) > 100000:  # > ₹1 lakh crore
            print("⚠️ Revenue values seem very high - verify units")
    
    # Check growth rates
    if 'revenue_growth' in enhanced_data.growth_metrics:
        growth = enhanced_data.growth_metrics['revenue_growth']
        if abs(growth) > 500:  # > 500% growth
            print("⚠️ Unusual growth rate detected")
    
    # Verify profit margins
    if 'profit_margin' in enhanced_data.key_ratios:
        margin = enhanced_data.key_ratios['profit_margin']
        if margin > 50:  # > 50% profit margin
            print("⚠️ Very high profit margin - verify data")
```

## 📊 Integration with Analyzers

### Using with Financial Analyzer
```python
from src.analyzers.financial_analyzer import FinancialAnalyzer

# Get enhanced prospectus data
enhanced_data = prospectus_source.get_enhanced_ipo_data(company_name)

if enhanced_data:
    # Convert to format expected by analyzer
    financial_data = {
        'revenue_data': enhanced_data.revenue_data,
        'profit_data': enhanced_data.profit_data,
        'assets_data': enhanced_data.assets_data,
        'ratios': enhanced_data.key_ratios
    }
    
    # Run financial analysis
    analyzer = FinancialAnalyzer()
    analysis = analyzer.analyze_financials(financial_data)
    
    print(f"Financial Health Score: {analysis.get('health_score', 'N/A')}")
    print(f"Growth Rating: {analysis.get('growth_rating', 'N/A')}")
```

### Using with Business Analyzer
```python
from src.analyzers.business_analyzer import BusinessAnalyzer

if enhanced_data:
    # Extract business information
    business_info = {
        'description': enhanced_data.business_description,
        'strengths': enhanced_data.company_strengths,
        'competitive_advantages': enhanced_data.competitive_advantages,
        'risk_factors': enhanced_data.risk_factors
    }
    
    # Analyze business model
    analyzer = BusinessAnalyzer()
    business_analysis = analyzer.analyze_business_model(business_info)
    
    print(f"Business Model Score: {business_analysis.get('model_score', 'N/A')}")
    print(f"Competitive Position: {business_analysis.get('competitive_position', 'N/A')}")
```

## 🔄 Updates and Maintenance

### Keeping the Integration Updated

1. **SEBI Website Changes**: The SEBI website structure may change. Update search endpoints in configuration if needed.

2. **PDF Format Changes**: New PDF formats may require pattern updates. Monitor extraction quality scores.

3. **Performance Tuning**: Adjust configuration based on your usage patterns and performance requirements.

### Monitoring Data Quality
```python
# Track quality metrics over time
def track_quality_metrics(company_results):
    """Track extraction quality for monitoring."""
    quality_scores = []
    extraction_times = []
    
    for company, data in company_results.items():
        if data and hasattr(data, 'data_quality_score'):
            quality_scores.append(data.data_quality_score)
            
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print(f"Average Quality Score: {avg_quality:.2f}")
    print(f"Companies with High Quality (>0.7): {sum(1 for q in quality_scores if q > 0.7)}")
    
    return {
        'average_quality': avg_quality,
        'high_quality_count': sum(1 for q in quality_scores if q > 0.7),
        'total_processed': len(quality_scores)
    }
```

This enhanced prospectus integration provides a comprehensive solution for extracting real financial data from IPO documents, enabling more accurate and data-driven IPO analysis and investment recommendations.
