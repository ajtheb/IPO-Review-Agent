#!/usr/bin/env python3
"""
Test Script - Print Relevant Chunks

This script tests the GMP extraction and shows the relevant chunks
that are being analyzed by the LLM.
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

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_with_chunks():
    """Test GMP extraction and show relevant chunks."""
    
    print("\n" + "="*80)
    print("  🔍 GMP EXTRACTION - SHOWING RELEVANT CHUNKS")
    print("="*80 + "\n")
    
    # Test companies
    test_companies = [
        "Biopol Chemicals",
        "Vidya Wires",
        "Akums Drugs"
    ]
    
    # Create fetcher with LLM enabled
    print("📦 Initializing GMP Fetcher with Groq LLM...")
    fetcher = GMPFetcher(cache_duration_hours=1, use_llm_fallback=True)
    print("✅ Fetcher initialized\n")
    
    for i, company in enumerate(test_companies, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(test_companies)}] Testing: {company}")
        print('='*80)
        
        try:
            # This will trigger LLM extraction and show chunks
            gmp_data = fetcher.get_gmp(company, use_cache=False)
            
            print(f"\n📊 RESULT:")
            print(f"   Status: {gmp_data['status']}")
            
            if gmp_data['status'] == 'active':
                print(f"   ✅ GMP Found!")
                print(f"   GMP Price: ₹{gmp_data.get('gmp_price', 'N/A')}")
                print(f"   GMP %: {gmp_data.get('gmp_percentage', 'N/A'):.2f}%")
                print(f"   Issue Price: ₹{gmp_data.get('issue_price', 'N/A')}")
                print(f"   Source: {gmp_data['source']}")
            else:
                print(f"   ⚠️  {gmp_data.get('message', 'No data')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.exception(f"Error testing {company}")
        
        if i < len(test_companies):
            print("\n⏳ Waiting 3 seconds before next test...")
            import time
            time.sleep(3)
    
    print("\n" + "="*80)
    print("  ✅ TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        test_with_chunks()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
