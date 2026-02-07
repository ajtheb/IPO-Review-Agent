#!/usr/bin/env python3
"""
Quick test script to show chunk printing for a single company.

Usage:
    python examples/test_single_company_chunks.py
    
or with a specific company:
    python examples/test_single_company_chunks.py "Unimech Aerospace"
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.data_sources.gmp_fetcher import GMPFetcher
from loguru import logger

# Configure logger for cleaner output
logger.remove()
logger.add(sys.stderr, level="WARNING")


def test_single_company(company_name: str):
    """Test chunk display for a single company."""
    
    print("\n" + "="*80)
    print(f"  🔍 TESTING CHUNK DISPLAY FOR: {company_name}")
    print("="*80 + "\n")
    
    print("📦 Initializing GMP Fetcher with Groq LLM...")
    fetcher = GMPFetcher(cache_duration_hours=0, use_llm_fallback=True)
    print("✅ Fetcher initialized\n")
    
    print(f"🚀 Fetching GMP data for: {company_name}")
    print("    (This will show relevant chunks if LLM extraction is used)\n")
    
    try:
        # Fetch GMP data - this will print chunks during LLM extraction
        gmp_data = fetcher.get_gmp(company_name, use_cache=False)
        
        print("\n" + "="*80)
        print("📊 FINAL RESULT:")
        print("="*80)
        print(f"Status: {gmp_data['status']}")
        
        if gmp_data['status'] == 'active':
            print(f"✅ GMP Data Found!")
            print(f"\n   Company Name: {gmp_data.get('company_name')}")
            print(f"   GMP Price: ₹{gmp_data.get('gmp_price', 'N/A')}")
            print(f"   GMP Percentage: {gmp_data.get('gmp_percentage', 'N/A')}%")
            print(f"   Issue Price: ₹{gmp_data.get('issue_price', 'N/A')}")
            print(f"   Expected Listing: ₹{gmp_data.get('expected_listing_price', 'N/A')}")
            print(f"   Estimated Gain: {gmp_data.get('estimated_listing_gain', 'N/A')}%")
            print(f"   Source: {gmp_data['source']}")
            print(f"   Last Updated: {gmp_data.get('last_updated', 'N/A')}")
        else:
            print(f"⚠️  {gmp_data.get('message', 'No data available')}")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Get company name from command line or use default
    if len(sys.argv) > 1:
        company = " ".join(sys.argv[1:])
    else:
        # Default company - change this to test different companies
        company = "Transrail Lighting"
    
    test_single_company(company)
