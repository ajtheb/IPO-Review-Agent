# Grey Market Premium (GMP) Fetcher - Complete Guide

## Overview

The GMP Fetcher is a powerful module for fetching Grey Market Premium data for Indian IPOs. Grey Market Premium (GMP) represents the premium at which IPO shares trade in the unofficial grey market before official listing, providing valuable insights into expected listing gains and market sentiment.

## What is Grey Market Premium (GMP)?

**Grey Market Premium (GMP)** is the price at which IPO shares are traded in the unofficial market before they are officially listed on the stock exchange. It's a key indicator of:

- **Market Sentiment**: High GMP indicates strong investor interest
- **Expected Listing Gains**: Helps estimate potential first-day returns
- **Demand-Supply Dynamics**: Reflects the appetite for the IPO
- **Risk Assessment**: Can help identify oversubscribed or undersubscribed IPOs

### Important Notes:
⚠️ **GMP is unofficial and not guaranteed** - It's traded in informal markets and can change rapidly
⚠️ **Not regulated by SEBI** - No official oversight or guarantees
⚠️ **Use as an indicator only** - Should be combined with fundamental analysis

## Installation

### Required Dependencies

```bash
pip install beautifulsoup4 requests loguru
```

All dependencies should already be included if you have the project's `requirements.txt` installed.

## Quick Start

### Basic Usage

```python
from src.data_sources.gmp_fetcher import GMPFetcher, fetch_ipo_gmp

# Method 1: Using the convenience function (simplest)
gmp_data = fetch_ipo_gmp("Vidya Wires")
print(f"GMP: ₹{gmp_data['gmp_price']}")
print(f"Expected Listing Gain: {gmp_data['gmp_percentage']:.2f}%")

# Method 2: Using the GMPFetcher class (more control)
fetcher = GMPFetcher(cache_duration_hours=6)
gmp_data = fetcher.get_gmp("Vidya Wires")
report = fetcher.format_gmp_report(gmp_data)
print(report)
```

### Output Example

```
📊 Grey Market Premium Report for Vidya Wires
============================================================
💰 Issue Price: ₹450.00
📈 GMP: ₹150.00
🔥 GMP Percentage: 33.33%
🎯 Expected Listing Price: ₹600.00
💹 Estimated Listing Gain: 33.33%

📅 Last Updated: 2025-01-13 10:30:45
🔗 Source: investorgain

💡 Interpretation:
   ✅ Strong Grey Market Premium - Good listing gains expected

⚠️  Note: GMP is unofficial and subject to change. It's an indicator, not a guarantee.
```

## API Reference

### GMPFetcher Class

#### Constructor

```python
fetcher = GMPFetcher(cache_duration_hours: int = 6)
```

**Parameters:**
- `cache_duration_hours`: How long to cache GMP data (default: 6 hours)

#### Methods

##### 1. get_gmp()

Fetch GMP data for a single company.

```python
gmp_data = fetcher.get_gmp(
    company_name: str,
    use_cache: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `company_name`: Name of the company (e.g., "Vidya Wires")
- `use_cache`: Whether to use cached data if available (default: True)

**Returns:**
Dictionary with the following structure:
```python
{
    'company_name': str,              # Company name
    'gmp_price': float or None,       # GMP in rupees
    'gmp_percentage': float or None,  # GMP as % of issue price
    'issue_price': float or None,     # IPO issue price
    'expected_listing_price': float or None,  # Issue price + GMP
    'estimated_listing_gain': float or None,  # Expected % gain
    'last_updated': datetime,         # When data was fetched
    'source': str,                    # Data source name
    'status': str                     # 'active', 'not_found', or 'error'
}
```

##### 2. get_multiple_gmp()

Fetch GMP data for multiple companies.

```python
results = fetcher.get_multiple_gmp(
    company_names: List[str]
) -> Dict[str, Dict[str, Any]]
```

**Parameters:**
- `company_names`: List of company names

**Returns:**
Dictionary mapping company names to their GMP data

**Example:**
```python
companies = ["Vidya Wires", "Akums Drugs", "DAM Capital"]
results = fetcher.get_multiple_gmp(companies)

for company, data in results.items():
    if data['status'] == 'active':
        print(f"{company}: GMP ₹{data['gmp_price']} ({data['gmp_percentage']:.2f}%)")
```

##### 3. format_gmp_report()

Format GMP data into a readable report.

```python
report = fetcher.format_gmp_report(gmp_data: Dict[str, Any]) -> str
```

**Parameters:**
- `gmp_data`: GMP data dictionary from `get_gmp()`

**Returns:**
Formatted string report

##### 4. clear_cache()

Clear cached GMP data.

```python
fetcher.clear_cache(company_name: Optional[str] = None)
```

**Parameters:**
- `company_name`: Specific company to clear, or None to clear all cache

### Convenience Function

```python
from src.data_sources.gmp_fetcher import fetch_ipo_gmp

gmp_data = fetch_ipo_gmp(company_name: str) -> Dict[str, Any]
```

Quick function for one-off GMP fetching without creating a fetcher instance.

## Data Sources

The GMP Fetcher automatically tries multiple reliable sources:

1. **InvestorGain** (Primary)
   - URL: https://www.investorgain.com/report/live-ipo-gmp/331/
   - Most comprehensive and up-to-date

2. **Chittorgarh** (Fallback)
   - URL: https://www.chittorgarh.com/ipo/ipo_gmp.asp
   - Well-established IPO tracking site

3. **IPOWatch** (Secondary Fallback)
   - URL: https://www.ipowatch.in/p/ipo-grey-market-premium-latest-ipos.html
   - Community-driven IPO tracking

The fetcher automatically tries each source in order until it finds data.

## Advanced Usage

### Custom Cache Duration

```python
# Cache for 24 hours (useful for development/testing)
fetcher = GMPFetcher(cache_duration_hours=24)

# Cache for 1 hour (for real-time monitoring)
fetcher = GMPFetcher(cache_duration_hours=1)

# Disable cache by always forcing fresh fetch
gmp_data = fetcher.get_gmp("Company Name", use_cache=False)
```

### Handling Different Company Name Formats

The fetcher uses fuzzy matching to handle various company name formats:

```python
# All of these will match:
fetcher.get_gmp("Vidya Wires")
fetcher.get_gmp("Vidya Wires Limited")
fetcher.get_gmp("vidya wires ltd")
fetcher.get_gmp("VIDYA WIRES")
```

### Error Handling

```python
gmp_data = fetcher.get_gmp("Company Name")

if gmp_data['status'] == 'active':
    # Data successfully fetched
    print(f"GMP: ₹{gmp_data['gmp_price']}")
    
elif gmp_data['status'] == 'not_found':
    # Company not found in grey market
    print("No GMP data available")
    
elif gmp_data['status'] == 'error':
    # Error occurred during fetching
    print(f"Error: {gmp_data.get('message', 'Unknown error')}")
```

### Integration with IPO Analysis

```python
from src.data_sources.gmp_fetcher import GMPFetcher

def analyze_ipo_with_gmp(company_name: str):
    """Complete IPO analysis including GMP."""
    
    fetcher = GMPFetcher()
    gmp_data = fetcher.get_gmp(company_name)
    
    analysis = {
        'company': company_name,
        'gmp_data': gmp_data
    }
    
    # Add GMP interpretation to analysis
    if gmp_data['status'] == 'active' and gmp_data['gmp_percentage']:
        if gmp_data['gmp_percentage'] > 50:
            analysis['gmp_signal'] = 'VERY_STRONG_BUY'
            analysis['gmp_interpretation'] = 'Exceptionally high demand'
        elif gmp_data['gmp_percentage'] > 30:
            analysis['gmp_signal'] = 'STRONG_BUY'
            analysis['gmp_interpretation'] = 'Strong market interest'
        elif gmp_data['gmp_percentage'] > 10:
            analysis['gmp_signal'] = 'BUY'
            analysis['gmp_interpretation'] = 'Positive sentiment'
        elif gmp_data['gmp_percentage'] > 0:
            analysis['gmp_signal'] = 'NEUTRAL'
            analysis['gmp_interpretation'] = 'Mild positive sentiment'
        else:
            analysis['gmp_signal'] = 'CAUTION'
            analysis['gmp_interpretation'] = 'Weak or negative sentiment'
    else:
        analysis['gmp_signal'] = 'UNKNOWN'
        analysis['gmp_interpretation'] = 'No grey market data available'
    
    return analysis
```

## Testing

### Run Comprehensive Tests

```bash
# Run the test suite
python examples/test_gmp_fetcher.py
```

### Test Coverage

The test suite includes:

1. **Single Company Fetch Test**
   - Tests fetching GMP for individual companies
   - Validates data structure and content

2. **Multiple Companies Test**
   - Tests batch fetching
   - Measures performance

3. **Cache Functionality Test**
   - Verifies caching mechanism
   - Measures cache performance improvement

4. **Fuzzy Matching Test**
   - Tests company name matching
   - Validates different name formats

5. **Error Handling Test**
   - Tests edge cases
   - Validates graceful error handling

6. **Data Validation Test**
   - Checks data types
   - Validates calculations

7. **Convenience Function Test**
   - Tests the quick-access function

## Integration with CLI

Add GMP fetching to your CLI app:

```python
from src.data_sources.gmp_fetcher import GMPFetcher

def cli_get_gmp(company_name: str):
    """CLI command to fetch GMP."""
    fetcher = GMPFetcher()
    gmp_data = fetcher.get_gmp(company_name)
    report = fetcher.format_gmp_report(gmp_data)
    print(report)

# In your CLI
if args.command == 'gmp':
    cli_get_gmp(args.company_name)
```

## Best Practices

### 1. Rate Limiting

Always implement rate limiting when fetching multiple companies:

```python
import time

fetcher = GMPFetcher()
companies = ["Company1", "Company2", "Company3"]

for company in companies:
    gmp_data = fetcher.get_gmp(company)
    # Process data
    time.sleep(2)  # Wait 2 seconds between requests
```

### 2. Use Caching

Enable caching to reduce unnecessary requests:

```python
# Good: Uses cache
fetcher = GMPFetcher(cache_duration_hours=6)
gmp_data = fetcher.get_gmp("Company", use_cache=True)

# Avoid: Always fetches fresh (can overwhelm servers)
gmp_data = fetcher.get_gmp("Company", use_cache=False)
```

### 3. Handle All Status Cases

```python
gmp_data = fetcher.get_gmp("Company")

if gmp_data['status'] == 'active':
    # Process valid data
    pass
elif gmp_data['status'] == 'not_found':
    # Handle missing data gracefully
    pass
else:  # 'error'
    # Log error and continue
    logger.error(f"GMP fetch failed: {gmp_data.get('message')}")
```

### 4. Combine with Fundamental Analysis

Never rely solely on GMP for investment decisions:

```python
def comprehensive_ipo_analysis(company_name: str):
    """Complete IPO analysis."""
    
    # Fetch GMP
    gmp_data = fetch_ipo_gmp(company_name)
    
    # Fetch fundamentals
    # financial_data = get_financial_data(company_name)
    # prospectus_data = get_prospectus_data(company_name)
    
    # Combine all factors
    recommendation = {
        'gmp_signal': gmp_data.get('gmp_percentage', 0),
        # 'fundamentals': analyze_fundamentals(financial_data),
        # 'prospectus': analyze_prospectus(prospectus_data)
    }
    
    return recommendation
```

## Troubleshooting

### Issue: No GMP Data Found

**Possible Causes:**
1. Company name misspelled
2. IPO not yet active in grey market
3. IPO already listed
4. IPO too small/regional (not tracked)

**Solutions:**
- Try different name variations
- Check if IPO dates are current
- Use fuzzy matching

### Issue: Fetch Takes Too Long

**Solutions:**
- Enable caching
- Check internet connection
- Some sources may be temporarily slow

### Issue: Import Error

```python
# Error: ImportError: No module named 'beautifulsoup4'
```

**Solution:**
```bash
pip install beautifulsoup4
```

## Performance Considerations

### Typical Response Times

- **First Fetch**: 2-5 seconds per company (web scraping)
- **Cached Fetch**: < 0.01 seconds
- **Multiple Fetch**: ~3 seconds per company (with rate limiting)

### Optimization Tips

1. **Use Batch Fetching**
   ```python
   # Good: Single batch call
   results = fetcher.get_multiple_gmp(["Co1", "Co2", "Co3"])
   
   # Avoid: Multiple individual calls
   for company in companies:
       gmp_data = fetcher.get_gmp(company)
   ```

2. **Adjust Cache Duration**
   - Development: 24 hours
   - Production: 6 hours (default)
   - Real-time monitoring: 1 hour

3. **Implement Background Fetching**
   ```python
   import threading
   
   def fetch_gmp_background(company_name: str):
       fetcher = GMPFetcher()
       return fetcher.get_gmp(company_name)
   
   # Fetch in background
   thread = threading.Thread(target=fetch_gmp_background, args=("Company",))
   thread.start()
   ```

## Examples

### Example 1: Simple GMP Check

```python
from src.data_sources.gmp_fetcher import fetch_ipo_gmp

# Quick GMP check
gmp = fetch_ipo_gmp("Vidya Wires")
print(f"GMP: ₹{gmp['gmp_price']} ({gmp['gmp_percentage']}%)")
```

### Example 2: Compare Multiple IPOs

```python
from src.data_sources.gmp_fetcher import GMPFetcher

fetcher = GMPFetcher()
companies = ["IPO1", "IPO2", "IPO3"]
results = fetcher.get_multiple_gmp(companies)

# Sort by GMP percentage
sorted_ipos = sorted(
    results.items(),
    key=lambda x: x[1].get('gmp_percentage', 0),
    reverse=True
)

print("Top IPOs by GMP:")
for company, data in sorted_ipos:
    if data['status'] == 'active':
        print(f"{company}: {data['gmp_percentage']:.2f}%")
```

### Example 3: GMP Alert System

```python
from src.data_sources.gmp_fetcher import GMPFetcher
import time

def gmp_monitor(companies: List[str], threshold: float = 30.0):
    """Monitor GMP and alert when threshold is crossed."""
    fetcher = GMPFetcher(cache_duration_hours=1)
    
    while True:
        for company in companies:
            gmp_data = fetcher.get_gmp(company)
            
            if gmp_data['status'] == 'active':
                gmp_pct = gmp_data['gmp_percentage']
                
                if gmp_pct >= threshold:
                    print(f"🔥 ALERT: {company} GMP at {gmp_pct:.2f}%!")
                    # Send notification, email, etc.
        
        time.sleep(3600)  # Check every hour

# Monitor IPOs
gmp_monitor(["Vidya Wires", "Akums Drugs"], threshold=40.0)
```

## Conclusion

The GMP Fetcher provides a robust, reliable way to track Grey Market Premium data for Indian IPOs. Use it as part of a comprehensive IPO analysis strategy, combining it with fundamental analysis, prospectus review, and market research.

### Key Takeaways:
- ✅ Automated GMP fetching from multiple sources
- ✅ Built-in caching for performance
- ✅ Fuzzy matching for flexible company names
- ✅ Comprehensive error handling
- ✅ Easy integration with existing analysis tools

### Remember:
⚠️ **GMP is an indicator, not a guarantee**
⚠️ **Always perform fundamental analysis**
⚠️ **Consider multiple factors before investing**

For more information, see the test suite in `examples/test_gmp_fetcher.py` or the implementation in `src/data_sources/gmp_fetcher.py`.
