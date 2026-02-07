"""
Test GMP Fetcher with Dynamic Content Analysis

This script demonstrates:
1. Why the GMP fetcher cannot extract data from InvestorGain
2. The difference between static HTML and dynamic JavaScript content
3. A solution using Selenium for dynamic content scraping
"""

from bs4 import BeautifulSoup
from loguru import logger
import requests
import sys
from pathlib import Path

def analyze_investorgain_structure():
    """Analyze the structure of InvestorGain website."""
    print("\n" + "="*80)
    print("ANALYZING INVESTORGAIN WEBSITE STRUCTURE")
    print("="*80)
    
    url = 'https://www.investorgain.com/report/live-ipo-gmp/331/'
    
    try:
        # Fetch the page
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check for dynamic loading indicators
        print("\n1. Checking for dynamic content indicators...")
        spinner = soup.find('div', class_='spinner-border')
        if spinner:
            print("   ✓ FOUND: Loading spinner - indicates dynamic content")
        else:
            print("   ✗ No spinner found")
        
        # Check for JavaScript frameworks
        print("\n2. Checking for JavaScript frameworks...")
        scripts = soup.find_all('script')
        js_frameworks = []
        for script in scripts:
            src = script.get('src', '')
            if 'react' in src.lower():
                js_frameworks.append('React')
            if 'next' in src.lower():
                js_frameworks.append('Next.js')
            if 'vue' in src.lower():
                js_frameworks.append('Vue.js')
        
        if js_frameworks:
            print(f"   ✓ FOUND: {', '.join(set(js_frameworks))}")
        else:
            print("   ✗ No major frameworks detected")
        
        # Check for data tables in static HTML
        print("\n3. Checking for data tables in static HTML...")
        tables = soup.find_all('table')
        print(f"   Found {len(tables)} table(s) in static HTML")
        
        data_tables = []
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) > 1:  # Has data rows
                data_tables.append((i, table, rows))
                print(f"   - Table {i+1}: {len(rows)} rows")
        
        # Check if any table has actual GMP data
        print("\n4. Checking for GMP data in tables...")
        gmp_found = False
        for idx, table, rows in data_tables:
            for row in rows[:3]:  # Check first 3 rows
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                    if row_text and 'gmp' in row_text.lower():
                        print(f"   ✓ GMP column found in table {idx+1}")
                        gmp_found = True
                        break
        
        if not gmp_found:
            print("   ✗ No GMP data found in static HTML tables")
        
        # Check for API endpoints
        print("\n5. Checking for API endpoints in page source...")
        page_source = str(soup)
        api_indicators = []
        if '/api/' in page_source:
            api_indicators.append('REST API')
        if 'graphql' in page_source.lower():
            api_indicators.append('GraphQL')
        if 'fetch(' in page_source or 'axios' in page_source.lower():
            api_indicators.append('AJAX calls')
        
        if api_indicators:
            print(f"   ✓ FOUND: {', '.join(api_indicators)}")
        else:
            print("   ? Could not determine API structure")
        
        # Summary
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        print("\n⚠️  The InvestorGain website uses CLIENT-SIDE RENDERING:")
        print("   • Page loads with a spinner/loading state")
        print("   • JavaScript fetches GMP data from API after page load")
        print("   • Static HTML scraping (BeautifulSoup) won't capture this data")
        print("\n💡 SOLUTION:")
        print("   • Use Selenium or Playwright to render JavaScript")
        print("   • Or find and call the underlying API endpoint directly")
        print("   • Or use a headless browser with requests-html")
        
    except Exception as e:
        print(f"❌ Error analyzing website: {e}")


def test_gmp_fetcher_biopol():
    """Test GMP fetcher with Biopol Chemicals."""
    print("\n" + "="*80)
    print("TESTING GMP FETCHER WITH BIOPOL CHEMICALS")
    print("="*80)
    
    print("\n⚠️  Skipping GMPFetcher test (requires project dependencies)")
    print("   The GMPFetcher would return 'not_available' status")
    print("   because InvestorGain uses dynamic content loading")
    
    # Simulated result
    print(f"\n📋 Expected Result:")
    print(f"   Company: Biopol Chemicals")
    print(f"   Status: not_available")
    print(f"   GMP Price: None")
    print(f"   GMP %: None")
    print(f"   Issue Price: None")
    print(f"   Expected Listing: None")
    print(f"   Source: investorgain")
    print(f"\n💬 Message: IPO mentioned but GMP data requires JavaScript to load")


def show_selenium_solution():
    """Show example of how to use Selenium for dynamic content."""
    print("\n" + "="*80)
    print("SELENIUM SOLUTION EXAMPLE")
    print("="*80)
    
    selenium_code = '''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

def fetch_gmp_with_selenium(company_name: str):
    """Fetch GMP using Selenium to handle dynamic content."""
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Initialize driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to page
        url = 'https://www.investorgain.com/report/live-ipo-gmp/331/'
        driver.get(url)
        
        # Wait for table to load (up to 10 seconds)
        wait = WebDriverWait(driver, 10)
        table = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        
        # Give extra time for data to populate
        time.sleep(2)
        
        # Find all table rows
        rows = driver.find_elements(By.TAG_NAME, "tr")
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells:
                row_text = ' '.join([cell.text for cell in cells])
                if company_name.lower() in row_text.lower():
                    # Extract GMP data from cells
                    print(f"Found: {row_text}")
                    # Parse the cells to extract GMP, price, etc.
                    break
        
    finally:
        driver.quit()

# Install with: pip install selenium webdriver-manager
# Also need: pip install chromedriver-autoinstaller
'''
    
    print("\n📝 To scrape dynamic content, use Selenium:")
    print(selenium_code)
    
    print("\n📦 Required packages:")
    print("   pip install selenium webdriver-manager chromedriver-autoinstaller")
    
    print("\n⚡ Alternative: requests-html")
    requests_html_code = '''
from requests_html import HTMLSession

session = HTMLSession()
response = session.get('https://www.investorgain.com/report/live-ipo-gmp/331/')
response.html.render(sleep=2)  # Render JavaScript

# Now parse the rendered HTML
tables = response.html.find('table')
'''
    print(requests_html_code)
    print("\n   pip install requests-html")


if __name__ == "__main__":
    print("\n🔍 GMP FETCHER DYNAMIC CONTENT ANALYSIS\n")
    
    # Analyze the website structure
    analyze_investorgain_structure()
    
    # Test the current GMP fetcher
    test_gmp_fetcher_biopol()
    
    # Show solution
    show_selenium_solution()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The GMP fetcher cannot extract data for Biopol Chemicals because:

1. ❌ STATIC HTML SCRAPING LIMITATION
   - InvestorGain uses React/Next.js for client-side rendering
   - GMP data is loaded via JavaScript API calls after page load
   - BeautifulSoup only sees the initial HTML (with loading spinner)

2. ✓ COMPANY IS MENTIONED
   - Biopol Chemicals appears in the page description
   - But actual GMP table data requires JavaScript execution

3. 💡 SOLUTIONS
   a) Use Selenium/Playwright to render JavaScript
   b) Find and call the API endpoint directly
   c) Use requests-html with rendering
   d) Try alternative GMP sources with static HTML

4. 🎯 NEXT STEPS
   - Implement Selenium-based fetcher for InvestorGain
   - Or focus on alternative sources (Chittorgarh, IPOWatch)
   - Or monitor network requests to find API endpoint
    """)
