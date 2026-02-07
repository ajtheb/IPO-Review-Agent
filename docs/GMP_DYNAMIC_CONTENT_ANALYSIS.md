# GMP Fetcher Dynamic Content Analysis

## Problem Summary

The GMP fetcher **cannot extract GMP data** for Biopol Chemicals (or any other IPO) from InvestorGain because the website uses **client-side rendering** with JavaScript, not static HTML.

## Root Cause

### What We Found

1. **Website Architecture**: InvestorGain uses Next.js (React framework)
2. **Content Loading**: GMP data is loaded **after** the initial page load via JavaScript
3. **Static HTML**: The initial HTML only contains:
   - A loading spinner
   - Page skeleton/structure
   - No actual GMP data
4. **Data Source**: GMP data comes from API calls made by JavaScript

### Evidence from gmp.log

```html
<!-- What BeautifulSoup sees: -->
<div style="display:flex;height:700px;justify-content:center;align-items:center">
    <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading...</span>
    </div>
</div>

<!-- Biopol Chemicals mentioned in page text, but no GMP data -->
<p>IPO GMP is trending for <strong>Kanishk Aluminium India, ... Biopol Chemicals IPOs</strong>.</p>
```

## Why BeautifulSoup Cannot Work

```python
# Current approach (DOESN'T WORK):
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
tables = soup.find_all('table')  # Only finds empty loading skeleton
```

**Problem**: `requests.get()` only fetches static HTML, doesn't execute JavaScript

## Solutions

### Solution 1: Use Selenium (Recommended)

Selenium renders JavaScript and waits for dynamic content to load.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

def fetch_gmp_with_selenium(company_name: str) -> dict:
    """Fetch GMP using Selenium to handle dynamic content."""
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--user-agent=Mozilla/5.0...')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        url = 'https://www.investorgain.com/report/live-ipo-gmp/331/'
        driver.get(url)
        
        # Wait for table to load (up to 15 seconds)
        wait = WebDriverWait(driver, 15)
        
        # Wait for spinner to disappear
        wait.until(EC.invisibility_of_element_located(
            (By.CLASS_NAME, "spinner-border")
        ))
        
        # Wait for table with data
        table = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "table[itemprop='about']")
        ))
        
        # Give extra time for JavaScript to populate data
        time.sleep(2)
        
        # Find all table rows
        rows = driver.find_elements(By.TAG_NAME, "tr")
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if not cells:
                continue
            
            row_text = ' '.join([cell.text.strip() for cell in cells])
            
            # Fuzzy match company name
            if company_name.lower() in row_text.lower():
                # Extract GMP data from cells
                # Cell structure: [Company, GMP, Price, Expected Listing, ...]
                data = {
                    'company_name': company_name,
                    'gmp_price': extract_number(cells[1].text) if len(cells) > 1 else None,
                    'issue_price': extract_number(cells[2].text) if len(cells) > 2 else None,
                    'expected_listing_price': extract_number(cells[3].text) if len(cells) > 3 else None,
                    'source': 'investorgain',
                    'status': 'active'
                }
                
                # Calculate GMP percentage
                if data['gmp_price'] and data['issue_price']:
                    data['gmp_percentage'] = (data['gmp_price'] / data['issue_price']) * 100
                
                return data
        
        return {
            'company_name': company_name,
            'status': 'not_found',
            'message': 'Company not found in GMP table',
            'source': 'investorgain'
        }
        
    finally:
        driver.quit()

def extract_number(text: str) -> float:
    """Extract numeric value from text."""
    import re
    match = re.search(r'[₹\$]?\s*(\d+(?:\.\d+)?)', text)
    return float(match.group(1)) if match else None
```

**Installation**:
```bash
pip install selenium webdriver-manager
pip install chromedriver-autoinstaller
```

**Pros**:
- ✅ Renders JavaScript
- ✅ Waits for dynamic content
- ✅ Most reliable solution

**Cons**:
- ⚠️ Slower (2-5 seconds per request)
- ⚠️ Requires Chrome/ChromeDriver
- ⚠️ Higher memory usage

### Solution 2: Use Playwright (Modern Alternative)

```python
from playwright.sync_api import sync_playwright

def fetch_gmp_with_playwright(company_name: str) -> dict:
    """Fetch GMP using Playwright."""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate and wait for network to be idle
        page.goto('https://www.investorgain.com/report/live-ipo-gmp/331/')
        page.wait_for_load_state('networkidle')
        
        # Wait for table to appear
        page.wait_for_selector('table[itemprop="about"]')
        
        # Extract table data
        rows = page.query_selector_all('tr')
        
        for row in rows:
            cells = row.query_selector_all('td')
            if cells:
                row_text = ' '.join([cell.inner_text() for cell in cells])
                if company_name.lower() in row_text.lower():
                    # Extract data
                    # ... similar to Selenium
                    pass
        
        browser.close()
```

**Installation**:
```bash
pip install playwright
playwright install chromium
```

**Pros**:
- ✅ Modern API
- ✅ Better performance than Selenium
- ✅ Built-in auto-waiting

### Solution 3: Find API Endpoint (Best Performance)

Monitor network requests to find the actual API endpoint.

```python
# If we find the API endpoint, we can call it directly
import requests

def fetch_gmp_via_api(company_name: str) -> dict:
    """Fetch GMP by calling API directly."""
    
    # Example API endpoint (need to discover actual URL)
    api_url = 'https://www.investorgain.com/api/ipo/gmp'
    
    response = requests.get(api_url, params={
        'report_id': 331,
        'parameter': 'all'
    }, headers={
        'User-Agent': 'Mozilla/5.0...',
        'Accept': 'application/json'
    })
    
    data = response.json()
    
    # Find company in data
    for ipo in data.get('ipos', []):
        if company_name.lower() in ipo['name'].lower():
            return ipo
    
    return None
```

**How to find API endpoint**:
1. Open InvestorGain in Chrome
2. Open DevTools (F12) → Network tab
3. Filter by "XHR" or "Fetch"
4. Reload page
5. Look for API calls returning JSON with GMP data

**Pros**:
- ✅ Fast (milliseconds)
- ✅ No browser needed
- ✅ Low resource usage

**Cons**:
- ⚠️ Need to discover endpoint
- ⚠️ May require authentication
- ⚠️ API may change

### Solution 4: Use requests-html

```python
from requests_html import HTMLSession

def fetch_gmp_with_requests_html(company_name: str) -> dict:
    """Fetch GMP using requests-html."""
    
    session = HTMLSession()
    response = session.get('https://www.investorgain.com/report/live-ipo-gmp/331/')
    
    # Render JavaScript (requires Chromium)
    response.html.render(sleep=2, timeout=10)
    
    # Now parse rendered HTML
    tables = response.html.find('table')
    for table in tables:
        rows = table.find('tr')
        for row in rows:
            cells = row.find('td')
            if cells:
                row_text = ' '.join([cell.text for cell in cells])
                if company_name.lower() in row_text.lower():
                    # Extract data
                    pass
    
    session.close()
```

**Installation**:
```bash
pip install requests-html
```

**Pros**:
- ✅ Simple API
- ✅ Renders JavaScript

**Cons**:
- ⚠️ Still needs Chromium
- ⚠️ Less maintained

## Implementation Plan

### Phase 1: Quick Fix (Current)
✅ **DONE**: Updated `gmp_fetcher.py` to detect dynamic content and return appropriate status

```python
{
    'company_name': 'Biopol Chemicals',
    'status': 'not_available',
    'message': 'IPO mentioned but GMP data requires JavaScript to load',
    'source': 'investorgain'
}
```

### Phase 2: Add Selenium Support (Recommended)

1. Create `selenium_gmp_fetcher.py` with Selenium-based scraper
2. Update `GMPFetcher` to use Selenium as fallback
3. Add configuration for browser options

```python
class GMPFetcher:
    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium
        # ... existing code ...
    
    def get_gmp(self, company_name: str) -> dict:
        if self.use_selenium:
            return self._fetch_with_selenium(company_name)
        else:
            return self._fetch_static(company_name)
```

### Phase 3: API Discovery (Optimal)

1. Monitor network requests on InvestorGain
2. Reverse engineer API endpoint
3. Implement direct API calls

## Testing

Created test script: `examples/test_gmp_dynamic_content.py`

**Run analysis**:
```bash
python examples/test_gmp_dynamic_content.py
```

**Output**:
- ✅ Detects dynamic content loading
- ✅ Identifies Next.js framework
- ✅ Confirms no GMP data in static HTML
- ✅ Provides solution examples

## Alternative GMP Sources

Since InvestorGain requires JavaScript, consider these alternatives:

### 1. Chittorgarh.com
- **URL**: https://www.chittorgarh.com/ipo
- **Status**: May also use dynamic content
- **Check**: Need to test if static HTML works

### 2. IPOWatch.in
- **URL**: https://www.ipowatch.in
- **Status**: Unknown
- **Check**: Need to test

### 3. Money Control
- **URL**: https://www.moneycontrol.com/ipo/
- **Status**: Large portal, may have static tables
- **Check**: Worth testing

### 4. BSE/NSE Official Sites
- **URL**: https://www.bseindia.com/
- **Status**: Official data, no GMP but has allotment info
- **Check**: For official IPO details

## Conclusion

**Why GMP fetcher fails**:
1. ❌ InvestorGain uses client-side JavaScript rendering
2. ❌ BeautifulSoup only sees initial HTML (loading spinner)
3. ❌ Actual GMP data loaded via JavaScript after page load

**What works**:
1. ✅ Detecting the issue (current implementation)
2. ✅ Returning appropriate error message
3. ✅ Selenium/Playwright for JavaScript rendering
4. ✅ Direct API calls (if endpoint discovered)

**Recommended next steps**:
1. Implement Selenium-based fetcher for InvestorGain
2. Test alternative GMP sources (Chittorgarh, IPOWatch)
3. Discover and use API endpoint if available
4. Add caching to reduce scraping frequency

## Files Updated

1. **src/data_sources/gmp_fetcher.py**
   - Added dynamic content detection
   - Returns informative status for JavaScript-loaded pages

2. **examples/test_gmp_dynamic_content.py**
   - Comprehensive analysis script
   - Demonstrates the problem
   - Shows solutions with code examples

3. **docs/GMP_DYNAMIC_CONTENT_ANALYSIS.md** (this file)
   - Complete documentation
   - Implementation guide
   - Testing instructions
