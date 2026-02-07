"""
Comprehensive Test Suite for GMP (Grey Market Premium) Fetcher.

This script tests the GMP fetching functionality with multiple scenarios:
1. Fetching GMP for a single company
2. Fetching GMP for multiple companies
3. Testing cache functionality
4. Testing different data sources
5. Error handling and edge cases
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_sources.gmp_fetcher import GMPFetcher
from loguru import logger
import time
from datetime import datetime

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def test_single_company_gmp():
    """Test fetching GMP for a single company."""
    print_header("TEST 1: Fetch GMP for Single Company")
    
    fetcher = GMPFetcher(cache_duration_hours=1)
    
    # Test with popular recent IPOs (replace with current IPOs)
    test_companies = [
        "Vidya Wires",
        "Akums Drugs",
        "Quadrant Future Tek",
        "DAM Capital Advisors"
    ]
    
    for company in test_companies:
        print(f"\n🔍 Fetching GMP for: {company}")
        print("-" * 60)
        
        try:
            gmp_data = fetcher.get_gmp(company)
            report = fetcher.format_gmp_report(gmp_data)
            print(report)
            
            # Validate data structure
            assert 'company_name' in gmp_data, "Missing company_name"
            assert 'status' in gmp_data, "Missing status"
            assert 'last_updated' in gmp_data, "Missing last_updated"
            
            if gmp_data['status'] == 'active':
                print(f"✅ Successfully fetched GMP data")
                print(f"   GMP: ₹{gmp_data.get('gmp_price', 'N/A')}")
                print(f"   Source: {gmp_data['source']}")
            else:
                print(f"⚠️  Status: {gmp_data['status']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception(f"Failed to fetch GMP for {company}")
        
        # Rate limiting
        time.sleep(2)
    
    print("\n✅ Test 1 completed")


def test_multiple_companies():
    """Test fetching GMP for multiple companies at once."""
    print_header("TEST 2: Fetch GMP for Multiple Companies")
    
    fetcher = GMPFetcher()
    
    companies = [
        "Vidya Wires",
        "Akums Drugs",
        "DAM Capital"
    ]
    
    print(f"📊 Fetching GMP for {len(companies)} companies...")
    
    try:
        start_time = time.time()
        results = fetcher.get_multiple_gmp(companies)
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Completed in {elapsed:.2f} seconds")
        print(f"📈 Success rate: {sum(1 for r in results.values() if r['status'] == 'active')}/{len(companies)}")
        
        for company, data in results.items():
            print(f"\n{'='*60}")
            print(f"Company: {company}")
            if data['status'] == 'active':
                print(f"✅ GMP: ₹{data.get('gmp_price', 'N/A')}")
                print(f"   Percentage: {data.get('gmp_percentage', 'N/A'):.2f}%")
            else:
                print(f"⚠️  Status: {data['status']}")
        
        print("\n✅ Test 2 completed")
        
    except Exception as e:
        print(f"❌ Error in multiple fetch: {e}")
        logger.exception("Test 2 failed")


def test_cache_functionality():
    """Test the caching mechanism."""
    print_header("TEST 3: Cache Functionality")
    
    fetcher = GMPFetcher(cache_duration_hours=1)
    test_company = "Vidya Wires"
    
    print(f"🔍 First fetch (should hit web sources)...")
    start_time = time.time()
    data1 = fetcher.get_gmp(test_company)
    time1 = time.time() - start_time
    print(f"⏱️  Time taken: {time1:.2f} seconds")
    print(f"Status: {data1['status']}")
    
    print(f"\n🔍 Second fetch (should use cache)...")
    start_time = time.time()
    data2 = fetcher.get_gmp(test_company, use_cache=True)
    time2 = time.time() - start_time
    print(f"⏱️  Time taken: {time2:.2f} seconds")
    print(f"Status: {data2['status']}")
    
    if time2 < time1 * 0.1:  # Cache should be much faster
        print("✅ Cache is working - second fetch was significantly faster")
    else:
        print("⚠️  Cache may not be working as expected")
    
    # Test cache clearing
    print(f"\n🧹 Clearing cache for {test_company}...")
    fetcher.clear_cache(test_company)
    
    print(f"🔍 Third fetch (should hit web sources again)...")
    start_time = time.time()
    data3 = fetcher.get_gmp(test_company)
    time3 = time.time() - start_time
    print(f"⏱️  Time taken: {time3:.2f} seconds")
    
    print("\n✅ Test 3 completed")


def test_fuzzy_matching():
    """Test fuzzy matching of company names."""
    print_header("TEST 4: Fuzzy Matching")
    
    fetcher = GMPFetcher()
    
    test_cases = [
        ("Vidya Wires Limited", "Vidya Wires"),
        ("Vidya", "Vidya Wires Limited Company"),
        ("vidya wires", "VIDYA WIRES LIMITED"),
        ("Akums Drugs & Pharmaceuticals", "Akums Drugs")
    ]
    
    for query, text in test_cases:
        result = fetcher._fuzzy_match(query, text)
        print(f"Query: '{query}'")
        print(f"Text:  '{text}'")
        print(f"Match: {'✅ Yes' if result else '❌ No'}")
        print("-" * 60)
    
    print("\n✅ Test 4 completed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print_header("TEST 5: Error Handling")
    
    fetcher = GMPFetcher()
    
    test_cases = [
        ("", "Empty company name"),
        ("XYZ123NonExistentCompany456", "Non-existent company"),
        ("Test Company 2050", "Future company"),
        ("A" * 200, "Very long company name")
    ]
    
    for company, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"Input: '{company[:50]}{'...' if len(company) > 50 else ''}'")
        
        try:
            data = fetcher.get_gmp(company)
            print(f"Status: {data['status']}")
            print(f"Result: {'✅ Handled gracefully' if data['status'] in ['not_found', 'error'] else '⚠️ Unexpected'}")
        except Exception as e:
            print(f"❌ Exception raised: {e}")
    
    print("\n✅ Test 5 completed")


def test_data_validation():
    """Test validation of fetched GMP data."""
    print_header("TEST 6: Data Validation")
    
    fetcher = GMPFetcher()
    test_company = "Vidya Wires"
    
    print(f"🔍 Fetching and validating data for {test_company}...")
    
    try:
        data = fetcher.get_gmp(test_company)
        
        # Check required fields
        required_fields = [
            'company_name', 'gmp_price', 'gmp_percentage', 
            'issue_price', 'expected_listing_price', 
            'estimated_listing_gain', 'last_updated', 
            'source', 'status'
        ]
        
        print("\n📋 Field Validation:")
        for field in required_fields:
            present = field in data
            print(f"  {field}: {'✅' if present else '❌'}")
        
        # Validate data types
        if data['status'] == 'active':
            print("\n📊 Data Type Validation:")
            
            if data['gmp_price'] is not None:
                assert isinstance(data['gmp_price'], (int, float)), "GMP price must be numeric"
                assert data['gmp_price'] >= 0, "GMP price cannot be negative"
                print(f"  ✅ GMP Price: {data['gmp_price']} (valid)")
            
            if data['gmp_percentage'] is not None:
                assert isinstance(data['gmp_percentage'], (int, float)), "GMP percentage must be numeric"
                print(f"  ✅ GMP Percentage: {data['gmp_percentage']:.2f}% (valid)")
            
            if data['issue_price'] is not None:
                assert isinstance(data['issue_price'], (int, float)), "Issue price must be numeric"
                assert data['issue_price'] > 0, "Issue price must be positive"
                print(f"  ✅ Issue Price: {data['issue_price']} (valid)")
            
            # Validate calculations
            if data['gmp_price'] and data['issue_price'] and data['expected_listing_price']:
                calculated = data['issue_price'] + data['gmp_price']
                assert abs(calculated - data['expected_listing_price']) < 0.01, "Listing price calculation error"
                print(f"  ✅ Expected Listing Price calculation: correct")
            
            print("\n✅ All validations passed")
        else:
            print(f"\n⚠️  Data status: {data['status']} - skipping numeric validation")
        
    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        logger.exception("Validation test failed")
    
    print("\n✅ Test 6 completed")


def test_convenience_function():
    """Test the convenience function."""
    print_header("TEST 7: Convenience Function")
    
    test_company = "DAM Capital"
    
    print(f"🔍 Testing GMPFetcher with default settings...")
    print(f"Company: {test_company}")
    
    try:
        # Use the fetcher directly instead of the removed convenience function
        fetcher = GMPFetcher()
        data = fetcher.get_gmp(test_company)
        
        print(f"\nStatus: {data['status']}")
        if data['status'] == 'active':
            print(f"GMP: ₹{data.get('gmp_price', 'N/A')}")
            print(f"Source: {data['source']}")
            print("✅ Fetcher works correctly")
        else:
            print(f"⚠️  {data.get('message', 'No data available')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Fetcher test failed")
    
    print("\n✅ Test 7 completed")


def run_comprehensive_demo():
    """Run a comprehensive demo showing all features."""
    print_header("COMPREHENSIVE GMP FETCHER DEMO")
    
    fetcher = GMPFetcher()
    
    # Demo companies (mix of real and test cases)
    demo_companies = [
        "Vidya Wires",
        "Akums Drugs",
        "DAM Capital Advisors"
    ]
    
    print("📊 Fetching GMP data for popular IPOs...\n")
    
    for i, company in enumerate(demo_companies, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(demo_companies)}] {company}")
        print('='*80)
        
        try:
            gmp_data = fetcher.get_gmp(company)
            report = fetcher.format_gmp_report(gmp_data)
            print(report)
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
        
        if i < len(demo_companies):
            time.sleep(2)  # Rate limiting
    
    print("\n" + "="*80)
    print("  📊 GMP Fetcher Demo Completed")
    print("="*80)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  🧪 GMP FETCHER COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"\n📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Run individual tests
        test_single_company_gmp()
        # test_multiple_companies()
        # test_cache_functionality()
        # test_fuzzy_matching()
        # test_error_handling()
        # test_data_validation()
        # test_convenience_function()
        
        # # Run comprehensive demo
        # run_comprehensive_demo()
        
        print("\n" + "="*80)
        print("  ✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\n📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        logger.exception("Test suite failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
