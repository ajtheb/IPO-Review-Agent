# 🚀 GMP System - Quick Start Guide

**Ready to extract Grey Market Premium data for IPOs? Start here!**

---

## ⚡ 60-Second Quick Start

```bash
# 1. Verify system is ready
python examples/verify_gmp_system.py

# 2. Run a quick test
python examples/demo_llm_solution.py

# 3. Test with real IPOs
python examples/test_gmp_fetcher.py
```

That's it! You're now extracting GMP data with AI-powered fallback.

---

## 🎯 What Does This System Do?

Extracts **Grey Market Premium** data for Indian IPOs from multiple sources:
- 📊 **Static scraping** from popular IPO websites (fast, free)
- 🤖 **LLM extraction** when static scraping fails (smart, reliable)
- 💾 **Intelligent caching** to minimize API calls (efficient)

**Result:** You get GMP data with ~90% success rate, even from JavaScript-heavy sites!

---

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys
Add to your `.env` file:
```bash
# For Gemini (recommended - cheaper)
GEMINI_API_KEY=your_gemini_api_key_here

# OR for OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

**Get API Keys:**
- **Gemini:** https://makersuite.google.com/app/apikey (Free tier available!)
- **OpenAI:** https://platform.openai.com/api-keys

---

## 🎬 Usage Examples

### Example 1: Get GMP for One Company

```python
from src.data_sources.gmp_fetcher import GMPFetcher

# Create fetcher (LLM fallback enabled by default)
fetcher = GMPFetcher()

# Get GMP data
gmp_data = fetcher.get_gmp("Vidya Wires")

# Check result
if gmp_data['status'] == 'active':
    print(f"GMP: ₹{gmp_data['gmp_price']}")
    print(f"Expected Listing Gain: {gmp_data['estimated_listing_gain']:.2f}%")
    print(f"Source: {gmp_data['source']}")
else:
    print(f"Status: {gmp_data['status']}")
    print(f"Message: {gmp_data.get('message', 'N/A')}")
```

### Example 2: Batch Processing Multiple IPOs

```python
from src.data_sources.gmp_fetcher import GMPFetcher

fetcher = GMPFetcher()

# Get GMP for multiple companies
companies = [
    "Vidya Wires",
    "Akums Drugs", 
    "DAM Capital Advisors",
    "Quadrant Future Tek"
]

results = fetcher.get_multiple_gmp(companies)

# Print results
for company, data in results.items():
    if data['status'] == 'active':
        print(f"{company}: ₹{data['gmp_price']} ({data['gmp_percentage']:.1f}%)")
    else:
        print(f"{company}: {data['status']}")
```

### Example 3: Pretty Formatted Report

```python
from src.data_sources.gmp_fetcher import GMPFetcher

fetcher = GMPFetcher()
gmp_data = fetcher.get_gmp("Akums Drugs")

# Get formatted report
report = fetcher.format_gmp_report(gmp_data)
print(report)
```

**Output:**
```
════════════════════════════════════════════════════════════
  Grey Market Premium Report
════════════════════════════════════════════════════════════

Company: Akums Drugs
Status: ACTIVE ✓

Price Information:
  Issue Price: ₹679.00
  GMP: ₹165.00 (24.30%)
  Expected Listing Price: ₹844.00

Investment Insights:
  Estimated Listing Gain: 24.30%

Data Source: investorgain_llm
Last Updated: 2025-01-31 15:45:23
════════════════════════════════════════════════════════════
```

### Example 4: Cache Management

```python
from src.data_sources.gmp_fetcher import GMPFetcher

# Set cache duration (6 hours default)
fetcher = GMPFetcher(cache_duration_hours=12)

# First call - fetches from source
gmp1 = fetcher.get_gmp("Vidya Wires")  

# Second call - uses cache (instant!)
gmp2 = fetcher.get_gmp("Vidya Wires")

# Force refresh (ignore cache)
gmp3 = fetcher.get_gmp("Vidya Wires", use_cache=False)

# Clear cache
fetcher.clear_cache("Vidya Wires")  # Clear one company
fetcher.clear_cache()                # Clear all
```

---

## 🔧 Configuration Options

### GMPFetcher Configuration

```python
from src.data_sources.gmp_fetcher import GMPFetcher

fetcher = GMPFetcher(
    cache_duration_hours=6,     # Cache duration (default: 6)
    use_llm_fallback=True       # Enable LLM extraction (default: True)
)
```

**When to use:**
- `cache_duration_hours=1`: During active trading hours (frequent updates)
- `cache_duration_hours=12`: For historical analysis
- `use_llm_fallback=False`: Testing static scraping only

### LLM Provider Selection

```python
from src.data_sources.llm_gmp_extractor import LLMGMPExtractor

# Use Gemini (recommended - cheaper)
extractor = LLMGMPExtractor(provider="gemini")

# Or use OpenAI
extractor = LLMGMPExtractor(provider="openai")
```

---

## 📊 Understanding the Output

### Data Structure

```python
{
    'company_name': str,              # Company name
    'gmp_price': float,               # GMP in ₹
    'gmp_percentage': float,          # GMP as % of issue price
    'issue_price': float,             # IPO issue price
    'expected_listing_price': float,  # Issue price + GMP
    'estimated_listing_gain': float,  # Expected % gain
    'last_updated': datetime,         # When data was fetched
    'source': str,                    # Data source
    'status': str                     # 'active', 'not_found', 'error'
}
```

### Status Codes

- **`active`**: GMP data found and extracted successfully ✅
- **`not_found`**: Company not in grey market or data unavailable ⚠️
- **`error`**: Technical error during extraction ❌
- **`not_available`**: Company mentioned but GMP not accessible 🔄

### Source Indicators

- **`investorgain`**: Static scraping from InvestorGain
- **`chittorgarh`**: Static scraping from Chittorgarh
- **`ipowatch`**: Static scraping from IPOWatch
- **`investorgain_llm`**: LLM extraction from InvestorGain
- **`chittorgarh_llm`**: LLM extraction from Chittorgarh

---

## 🧪 Testing & Verification

### 1. Quick System Check (Recommended First Step)

```bash
python examples/verify_gmp_system.py
```

**Checks:**
- ✅ API keys configured
- ✅ All imports working
- ✅ Static scraping functional
- ✅ LLM fallback operational
- ✅ Caching working correctly

### 2. Comprehensive Testing

```bash
python examples/test_gmp_fetcher.py
```

**Tests:**
- Single company fetching
- Batch processing
- Cache functionality
- Error handling
- Performance benchmarks

### 3. LLM-Specific Tests

```bash
python examples/test_llm_gmp_extraction.py
```

**Tests:**
- HTML parsing and chunking
- Company name matching
- Structured data extraction
- Multi-provider support

### 4. Demo with Live Data

```bash
python examples/demo_llm_solution.py
```

**Shows:**
- End-to-end workflow
- Static vs LLM comparison
- Real extraction results
- Pretty-printed reports

---

## 💡 Best Practices

### 1. **Use Caching Wisely**
```python
# During active analysis (frequent updates)
fetcher = GMPFetcher(cache_duration_hours=1)

# For historical analysis (less frequent updates)
fetcher = GMPFetcher(cache_duration_hours=24)
```

### 2. **Handle Errors Gracefully**
```python
gmp_data = fetcher.get_gmp(company_name)

if gmp_data['status'] != 'active':
    # Handle missing data
    logger.warning(f"GMP not available: {gmp_data.get('message')}")
    # Use alternative data source or skip
```

### 3. **Batch Processing for Efficiency**
```python
# Instead of:
for company in companies:
    gmp = fetcher.get_gmp(company)  # Multiple calls

# Do this:
results = fetcher.get_multiple_gmp(companies)  # Single call
```

### 4. **Monitor API Costs**
```python
# Enable logging to track LLM usage
import logging
logging.basicConfig(level=logging.INFO)

# Check which extractions used LLM
if '_llm' in gmp_data['source']:
    print("LLM was used (API cost incurred)")
```

### 5. **Validate Company Names**
```python
# System uses fuzzy matching, but exact names work better
company_name = "Akums Drugs and Pharmaceuticals Ltd"  # Full name
gmp = fetcher.get_gmp(company_name)

# Also works with shortened names
company_name = "Akums Drugs"  # Common name
gmp = fetcher.get_gmp(company_name)
```

---

## 🐛 Troubleshooting

### Issue: "API key not found"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify API key is set
grep GEMINI_API_KEY .env
# or
grep OPENAI_API_KEY .env

# If missing, add it:
echo "GEMINI_API_KEY=your_key_here" >> .env
```

### Issue: "Module not found"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.data_sources.gmp_fetcher import GMPFetcher; print('OK')"
```

### Issue: "GMP data not found"

**Possible Causes:**
1. Company not in grey market yet
2. IPO subscription period ended
3. Company name doesn't match

**Solution:**
```python
# Try different name variations
names_to_try = [
    "Biopol Chemicals",
    "Biopol",
    "Biopol Chemicals Limited"
]

for name in names_to_try:
    result = fetcher.get_gmp(name)
    if result['status'] == 'active':
        print(f"Found with name: {name}")
        break
```

### Issue: "LLM extraction taking too long"

**Solution:**
```python
# Reduce chunk processing limit
from src.data_sources.llm_gmp_extractor import LLMGMPExtractor

extractor = LLMGMPExtractor(
    max_chunks_to_process=5,  # Process fewer chunks
    chunk_size=800            # Smaller chunks
)
```

### Issue: "Too many API calls / High costs"

**Solution:**
```python
# Increase cache duration
fetcher = GMPFetcher(cache_duration_hours=24)  # Cache for 24 hours

# Batch process to minimize calls
results = fetcher.get_multiple_gmp(all_companies)

# Disable LLM for testing
fetcher = GMPFetcher(use_llm_fallback=False)
```

---

## 📈 Performance Tips

### 1. **Response Time Expectations**
- Static scraping: 1-3 seconds
- LLM extraction: 5-15 seconds
- Cached data: <0.1 seconds

### 2. **API Cost Estimates**
- Static scraping: FREE
- LLM extraction (Gemini): ~$0.001 per request
- LLM extraction (OpenAI): ~$0.005 per request

### 3. **Optimization Strategies**
```python
# Use cache aggressively
fetcher = GMPFetcher(cache_duration_hours=12)

# Batch process
results = fetcher.get_multiple_gmp(companies)

# Pre-fetch during off-hours
import schedule
schedule.every().day.at("02:00").do(lambda: fetcher.get_multiple_gmp(active_ipos))
```

---

## 🎯 Integration Examples

### Integration with IPO Analysis Pipeline

```python
from src.data_sources.gmp_fetcher import GMPFetcher

class IPOAnalyzer:
    def __init__(self):
        self.gmp_fetcher = GMPFetcher()
    
    def analyze_ipo(self, company_name: str) -> dict:
        # ... other analysis code ...
        
        # Add GMP analysis
        gmp_data = self.gmp_fetcher.get_gmp(company_name)
        
        analysis_report = {
            'company': company_name,
            'fundamentals': self._analyze_fundamentals(),
            'valuation': self._analyze_valuation(),
            'grey_market_sentiment': {
                'gmp': gmp_data.get('gmp_price'),
                'expected_listing_gain': gmp_data.get('estimated_listing_gain'),
                'status': gmp_data['status']
            }
        }
        
        # Add investment recommendation based on GMP
        if gmp_data['status'] == 'active':
            listing_gain = gmp_data.get('estimated_listing_gain', 0)
            if listing_gain > 30:
                analysis_report['gmp_signal'] = 'STRONG BUY'
            elif listing_gain > 15:
                analysis_report['gmp_signal'] = 'BUY'
            else:
                analysis_report['gmp_signal'] = 'NEUTRAL'
        
        return analysis_report
```

### Integration with Streamlit Dashboard

```python
import streamlit as st
from src.data_sources.gmp_fetcher import GMPFetcher

st.title("IPO Grey Market Premium Tracker")

# Initialize fetcher
if 'gmp_fetcher' not in st.session_state:
    st.session_state.gmp_fetcher = GMPFetcher()

# User input
company_name = st.text_input("Enter Company Name", "Vidya Wires")

if st.button("Fetch GMP"):
    with st.spinner("Fetching GMP data..."):
        gmp_data = st.session_state.gmp_fetcher.get_gmp(company_name)
    
    if gmp_data['status'] == 'active':
        col1, col2, col3 = st.columns(3)
        col1.metric("GMP", f"₹{gmp_data['gmp_price']}")
        col2.metric("Percentage", f"{gmp_data['gmp_percentage']:.2f}%")
        col3.metric("Expected Gain", f"{gmp_data['estimated_listing_gain']:.2f}%")
        
        st.success(f"Data from: {gmp_data['source']}")
    else:
        st.warning(f"GMP data not available: {gmp_data.get('message')}")
```

---

## 📚 Additional Resources

### Documentation
- **[GMP_SYSTEM_STATUS.md](GMP_SYSTEM_STATUS.md)** - Comprehensive system status
- **[LLM_GMP_EXTRACTION.md](LLM_GMP_EXTRACTION.md)** - LLM extraction deep dive
- **[GMP_DYNAMIC_CONTENT_ANALYSIS.md](GMP_DYNAMIC_CONTENT_ANALYSIS.md)** - Technical analysis

### Example Scripts
- **[verify_gmp_system.py](../examples/verify_gmp_system.py)** - System verification
- **[test_gmp_fetcher.py](../examples/test_gmp_fetcher.py)** - Comprehensive tests
- **[demo_llm_solution.py](../examples/demo_llm_solution.py)** - Live demo

### Source Code
- **[gmp_fetcher.py](../src/data_sources/gmp_fetcher.py)** - Main GMP fetcher
- **[llm_gmp_extractor.py](../src/data_sources/llm_gmp_extractor.py)** - LLM extraction engine

---

## 🆘 Need Help?

1. **Run verification:** `python examples/verify_gmp_system.py`
2. **Check logs:** Look for error messages in console output
3. **Review docs:** See [GMP_SYSTEM_STATUS.md](GMP_SYSTEM_STATUS.md)
4. **Test examples:** Run demo scripts to understand usage

---

## ✅ Quick Reference

```python
# Import
from src.data_sources.gmp_fetcher import GMPFetcher

# Initialize
fetcher = GMPFetcher()

# Fetch single
gmp = fetcher.get_gmp("Company Name")

# Fetch multiple
results = fetcher.get_multiple_gmp(["Company1", "Company2"])

# Format report
report = fetcher.format_gmp_report(gmp)

# Clear cache
fetcher.clear_cache()
```

**That's it! You're ready to extract GMP data like a pro! 🚀**
