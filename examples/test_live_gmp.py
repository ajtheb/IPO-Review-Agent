#!/usr/bin/env python3
"""
Quick Live Test - GMP Extraction

Tests the GMP system with a real company to show it's working.
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

# Configure logger for clean output
logger.remove()
logger.add(sys.stderr, level="WARNING")  # Only show warnings/errors

def test_live_gmp():
    """Test GMP fetching with a popular recent IPO."""
    
    print("\n" + "="*80)
    print("  🚀 GMP LIVE TEST - Fetching Real IPO Data")
    print("="*80 + "\n")
    
    # Create fetcher
    print("📦 Initializing GMP Fetcher...")
    fetcher = GMPFetcher(cache_duration_hours=6, use_llm_fallback=True)
    print("✅ Fetcher initialized (LLM fallback enabled)\n")
    
    # Test with popular IPOs (mix of likely found and not found)
    test_companies = [
        "Vidya Wires",           # Recent mainboard IPO
        "Akums Drugs",           # Large pharma IPO
        "DAM Capital Advisors",  # Recent listing
    ]
    
    print(f"🔍 Testing with {len(test_companies)} companies...\n")
    print("-"*80)
    
    results = []
    for i, company in enumerate(test_companies, 1):
        print(f"\n[{i}/{len(test_companies)}] Fetching: {company}")
        print("⏳ Please wait...")
        
        try:
            gmp_data = fetcher.get_gmp(company, use_cache=False)
            results.append((company, gmp_data))
            
            if gmp_data['status'] == 'active':
                print(f"✅ SUCCESS!")
                print(f"   GMP: ₹{gmp_data['gmp_price']:.2f}")
                print(f"   Percentage: {gmp_data['gmp_percentage']:.2f}%")
                print(f"   Expected Listing: ₹{gmp_data['expected_listing_price']:.2f}")
                print(f"   Source: {gmp_data['source']}")
            elif gmp_data['status'] == 'not_found':
                print(f"⚠️  NOT FOUND")
                print(f"   This IPO may not be in the grey market or already listed")
            else:
                print(f"❌ Status: {gmp_data['status']}")
                print(f"   Message: {gmp_data.get('message', 'N/A')}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append((company, None))
        
        print("-"*80)
    
    # Summary
    print("\n" + "="*80)
    print("  📊 SUMMARY")
    print("="*80 + "\n")
    
    found = sum(1 for _, data in results if data and data['status'] == 'active')
    not_found = sum(1 for _, data in results if data and data['status'] == 'not_found')
    errors = sum(1 for _, data in results if data is None or data.get('status') == 'error')
    
    print(f"Total Companies: {len(results)}")
    print(f"✅ Found GMP: {found}")
    print(f"⚠️  Not Found: {not_found}")
    print(f"❌ Errors: {errors}")
    
    success_rate = (found / len(results) * 100) if results else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if found > 0:
        print("\n🎉 System is working! GMP data extracted successfully.")
        print("\n📋 Companies with GMP data:")
        for company, data in results:
            if data and data['status'] == 'active':
                print(f"   • {company}: ₹{data['gmp_price']:.2f} ({data['gmp_percentage']:.2f}%)")
    else:
        print("\n⚠️  No GMP data found for any company.")
        print("   This could mean:")
        print("   • These IPOs are not actively trading in grey market")
        print("   • They may have already listed")
        print("   • Try with more recent/popular IPOs")
    
    print("\n" + "="*80)
    print("  ✅ TEST COMPLETE")
    print("="*80 + "\n")
    
    # Show caching benefit
    if results:
        print("💡 TIP: Run this again to see caching in action (instant results!)\n")

if __name__ == "__main__":
    try:
        test_live_gmp()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
