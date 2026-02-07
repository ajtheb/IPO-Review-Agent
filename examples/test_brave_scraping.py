"""
Test Brave Search with Website Scraping for GMP Extraction

This script demonstrates the complete workflow:
1. Search Brave for GMP data
2. Scrape actual website content from search results
3. Extract GMP values using LLM (Groq)
4. Save all data (search results, HTML, extracted text) for debugging
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.data_sources.llm_gmp_extractor import LLMGMPExtractor

# Load environment variables
load_dotenv()


def test_brave_scraping_for_company(company_name: str):
    """
    Test Brave Search with website scraping for a specific company.
    
    Args:
        company_name: Name of the company to search for
    """
    print("="*80)
    print(f"Testing Brave Search + Website Scraping for: {company_name}")
    print("="*80)
    
    # Initialize extractor with Groq (default)
    try:
        extractor = LLMGMPExtractor(
            provider="groq",
            use_brave_search=True
        )
        print(f"✅ Initialized LLM GMP Extractor with Groq and Brave Search")
    except Exception as e:
        print(f"❌ Error initializing extractor: {e}")
        return
    
    # Extract GMP with Brave Search and website scraping
    print(f"\n{'='*80}")
    print(f"Step 1: Searching Brave API")
    print(f"{'='*80}")
    
    result = extractor.extract_gmp(
        company_name=company_name,
        use_brave=True,
        save_chunks=True,  # Save all scraped content
        print_chunks=True  # Print search results
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"FINAL EXTRACTION RESULTS")
    print(f"{'='*80}")
    
    print(f"\nCompany: {result.get('company_name', 'N/A')}")
    print(f"Status: {result.get('status', 'N/A')}")
    
    if result.get('status') == 'success':
        print(f"\n📊 GMP Data:")
        print(f"  GMP Price: ₹{result.get('gmp_price', 'N/A')}")
        print(f"  GMP Percentage: {result.get('gmp_percentage', 'N/A')}%")
        print(f"  Issue Price: ₹{result.get('issue_price', 'N/A')}")
        print(f"  Expected Listing Price: ₹{result.get('expected_listing_price', 'N/A')}")
        print(f"  Estimated Listing Gain: {result.get('estimated_listing_gain', 'N/A')}%")
        
        print(f"\n📅 IPO Timeline:")
        print(f"  Status: {result.get('ipo_status', 'N/A')}")
        print(f"  Opening Date: {result.get('open_date', 'N/A')}")
        print(f"  Closing Date: {result.get('close_date', 'N/A')}")
        print(f"  Listing Date: {result.get('listing_date', 'N/A')}")
        
        print(f"\n🔍 Extraction Metadata:")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Source: {result.get('source', 'N/A')}")
        print(f"  Scraped URLs: {len(result.get('scraped_urls', []))}")
        if result.get('scraped_urls'):
            for url in result['scraped_urls']:
                print(f"    - {url}")
        print(f"  Search Results: {result.get('search_results_count', 'N/A')}")
        
        if result.get('notes'):
            print(f"\n📝 Notes: {result['notes']}")
    else:
        print(f"\n⚠️  {result.get('message', 'No GMP data found')}")
    
    print(f"\n{'='*80}")
    print(f"✅ All data saved to gmp_chunks/ folder:")
    print(f"  - Brave search results (JSON)")
    print(f"  - Raw HTML files from scraped websites")
    print(f"  - Extracted text content")
    print(f"{'='*80}\n")


def main():
    """Run test for Fractal Analytics."""
    
    # Check API keys
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set in .env file")
        return
    
    if not os.getenv("BRAVE_API_KEY"):
        print("❌ BRAVE_API_KEY not set in .env file")
        return
    
    print("\n🚀 Starting Brave Search + Website Scraping Test")
    print("="*80)
    
    # Test Fractal Analytics
    test_brave_scraping_for_company("Fractal Analytics")
    
    print("\n✅ Test completed!")
    print(f"📁 Check the gmp_chunks/ folder for saved data:")
    print(f"   - Search results (JSON)")
    print(f"   - Raw HTML files")
    print(f"   - Extracted text files")


if __name__ == "__main__":
    main()
