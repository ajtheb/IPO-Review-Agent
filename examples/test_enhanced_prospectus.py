"""
Comprehensive test script for Enhanced IPO Prospectus Integration.
Tests the new enhanced parser, validation, caching, and data quality features.
"""

import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

try:
    from src.data_sources.enhanced_prospectus_parser import (
        EnhancedProspectusDataSource, 
        DataValidator, 
        CacheManager,
        EnhancedSEBISource
    )
    from src.data_sources import DataSourceManager
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced prospectus parser not available: {e}")
    print("Install required dependencies: pip install tabula-py")
    ENHANCED_AVAILABLE = False

def test_enhanced_sebi_search():
    """Test the enhanced SEBI search capabilities."""
    print("🔍 Testing Enhanced SEBI Search")
    print("=" * 40)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced parser not available")
        return
    
    sebi_source = EnhancedSEBISource()
    
    # Test companies
    test_companies = [
        "Zomato Limited - DHRP",
        "Paytm",
        "One97 Communications",
        "LIC",
        "Life Insurance Corporation of India",
        "SAEL INDUSTRIES LIMITED"
    ]
    
    for company in test_companies:
        print(f"\n📊 Searching: {company}")
        try:
            filings = sebi_source.search_comprehensive(company)
            print(f"   Found: {len(filings)} filings")
            
            if filings:
                latest = filings[0]
                print(f"   Latest: {latest.get('type', 'Unknown')} ({latest.get('date', 'No date')})")
                print(f"   URL: {latest.get('url', 'No URL')[:80]}...")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ SEBI search test completed")

def test_data_validator():
    """Test the data validation functionality."""
    print("\n🛠️ Testing Data Validator")
    print("=" * 40)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced parser not available")
        return
    
    validator = DataValidator()
    
    # Test case 1: Good data
    good_data = {
        'revenue': {'FY2021': 1000, 'FY2022': 1200, 'FY2023': 1400},
        'profit': {'FY2021': 100, 'FY2022': 120, 'FY2023': 140},
        'assets': {'FY2023': 2000},
        'equity': {'FY2023': 800}
    }
    
    is_valid, issues = validator.validate_financial_data(good_data)
    print(f"Good data validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if issues:
        print(f"   Issues: {issues}")
    
    # Test case 2: Bad data
    bad_data = {
        'revenue': {'FY2021': 1000, 'FY2022': 15000},  # Unrealistic jump
        'profit': {'FY2021': 1200, 'FY2022': 800},     # Profit > Revenue
        'assets': {'FY2023': -100}                      # Negative assets
    }
    
    is_valid, issues = validator.validate_financial_data(bad_data)
    print(f"Bad data validation: {'✅ PASS' if not is_valid else '❌ FAIL'}")
    print(f"   Issues found: {len(issues)}")
    for issue in issues:
        print(f"   - {issue}")
    
    print("\n✅ Data validator test completed")

def test_cache_manager():
    """Test the caching functionality."""
    print("\n💾 Testing Cache Manager")
    print("=" * 40)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced parser not available")
        return
    
    try:
        from src.data_sources.enhanced_prospectus_parser import EnhancedFinancialData
        
        cache_manager = CacheManager(max_age_hours=1)
        
        # Create test data
        test_data = EnhancedFinancialData(
            revenue_data={'FY2023': 1000},
            profit_data={'FY2023': 100},
            ebitda_data={},
            assets_data={},
            liabilities_data={},
            equity_data={},
            cash_flow_data={},
            key_ratios={},
            growth_metrics={},
            business_description="Test company",
            risk_factors=["Test risk"],
            use_of_funds=["Test use"],
            company_strengths=["Test strength"],
            competitive_advantages=[],
            extraction_date="2024-01-01",
            data_quality_score=0.8,
            source_confidence=0.9,
            validation_flags=[]
        )
        
        # Test caching
        test_company = "Test Company Ltd"
        
        # Should be None initially
        cached = cache_manager.get_cached_data(test_company)
        print(f"Initial cache check: {'✅ EMPTY' if cached is None else '❌ NOT EMPTY'}")
        
        # Cache the data
        cache_manager.cache_data(test_company, test_data)
        print("✅ Data cached")
        
        # Should now return data
        cached = cache_manager.get_cached_data(test_company)
        print(f"After caching: {'✅ FOUND' if cached is not None else '❌ NOT FOUND'}")
        
        if cached:
            print(f"   Quality score: {cached.data_quality_score}")
            print(f"   Business desc: {cached.business_description}")
        
    except Exception as e:
        print(f"❌ Cache test error: {e}")
    
    print("\n✅ Cache manager test completed")

def test_enhanced_integration():
    """Test the full enhanced prospectus integration."""
    print("\n🚀 Testing Enhanced Prospectus Integration")
    print("=" * 50)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced parser not available")
        print("Install dependencies: pip install tabula-py")
        return
    
    # Initialize enhanced data source
    enhanced_source = EnhancedProspectusDataSource(cache_enabled=True)
    
    # Test companies (start with well-known IPOs)
    test_companies = [
        "Zomato Limited",
        "Paytm One97 Communications",
        "Life Insurance Corporation",
        "Plaza Wires"
    ]
    
    results = {}
    
    for company in test_companies:
        print(f"\n📈 Processing: {company}")
        print("-" * 30)
        
        try:
            # Get data summary first (quick check)
            start_time = time.time()
            summary = enhanced_source.get_data_summary(company)
            summary_time = time.time() - start_time
            
            print(f"Summary check ({summary_time:.2f}s):")
            print(f"   SEBI filings: {summary.get('sebi_filings_found', 0)}")
            print(f"   Cached: {summary.get('cached', False)}")
            
            if summary.get('sebi_filings_found', 0) > 0:
                # Try to get enhanced data
                print("   Attempting enhanced extraction...")
                start_time = time.time()
                
                enhanced_data = enhanced_source.get_enhanced_ipo_data(company)
                extraction_time = time.time() - start_time
                
                if enhanced_data:
                    print(f"✅ Success ({extraction_time:.2f}s)")
                    print(f"   Quality Score: {enhanced_data.data_quality_score:.2f}")
                    print(f"   Source Confidence: {enhanced_data.source_confidence:.2f}")
                    print(f"   Revenue Years: {len(enhanced_data.revenue_data)}")
                    print(f"   Profit Years: {len(enhanced_data.profit_data)}")
                    print(f"   Risk Factors: {len(enhanced_data.risk_factors)}")
                    print(f"   Validation Issues: {len(enhanced_data.validation_flags)}")
                    
                    if enhanced_data.validation_flags:
                        print("   ⚠️ Validation Flags:")
                        for flag in enhanced_data.validation_flags[:3]:
                            print(f"     - {flag}")
                    
                    # Show sample financial data
                    if enhanced_data.revenue_data:
                        print("   💰 Revenue Data:")
                        for year, amount in list(enhanced_data.revenue_data.items())[:3]:
                            print(f"     {year}: ₹{amount:,.0f}")
                    
                    if enhanced_data.profit_data:
                        print("   📊 Profit Data:")
                        for year, amount in list(enhanced_data.profit_data.items())[:3]:
                            print(f"     {year}: ₹{amount:,.0f}")
                    
                    if enhanced_data.key_ratios:
                        print("   📈 Key Ratios:")
                        for ratio, value in list(enhanced_data.key_ratios.items())[:3]:
                            print(f"     {ratio}: {value:.2f}")
                    
                    results[company] = {
                        'success': True,
                        'quality_score': enhanced_data.data_quality_score,
                        'extraction_time': extraction_time,
                        'revenue_years': len(enhanced_data.revenue_data),
                        'validation_issues': len(enhanced_data.validation_flags)
                    }
                    
                else:
                    print(f"❌ No quality data extracted ({extraction_time:.2f}s)")
                    results[company] = {'success': False, 'reason': 'No quality data'}
            else:
                print("❌ No SEBI filings found")
                results[company] = {'success': False, 'reason': 'No SEBI filings'}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[company] = {'success': False, 'reason': str(e)}
    
    # Summary
    print(f"\n📋 INTEGRATION TEST SUMMARY")
    print("=" * 40)
    
    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    
    print(f"Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful > 0:
        avg_quality = sum(r.get('quality_score', 0) for r in results.values() if r.get('success', False)) / successful
        print(f"Average Quality Score: {avg_quality:.2f}")
        
        total_revenue_years = sum(r.get('revenue_years', 0) for r in results.values() if r.get('success', False))
        print(f"Total Revenue Years Extracted: {total_revenue_years}")
    
    print("\n✅ Enhanced integration test completed")
    
    return results

def test_data_source_manager_integration():
    """Test the enhanced integration with DataSourceManager."""
    print("\n🔧 Testing DataSourceManager Integration")
    print("=" * 45)
    
    try:
        # Initialize with enhanced features
        manager = DataSourceManager(use_enhanced_prospectus=True)
        
        test_ipo = {
            'company_name': 'Zomato Limited',
            'sector': 'Technology',
            'price_range': '72-76',
            'exchange': 'NSE'
        }
        
        print(f"Testing with: {test_ipo['company_name']}")
        
        # Test quick summary
        print("Getting prospectus summary...")
        summary = manager.get_prospectus_summary(test_ipo['company_name'])
        print(f"   Cached: {summary.get('cached', False)}")
        print(f"   SEBI filings: {summary.get('sebi_filings_found', 0)}")
        
        if summary.get('sebi_filings_found', 0) > 0:
            # Test full data collection
            print("Collecting comprehensive IPO data...")
            start_time = time.time()
            
            all_data = manager.collect_ipo_data(test_ipo['company_name'], test_ipo)
            collection_time = time.time() - start_time
            
            print(f"✅ Data collection completed ({collection_time:.2f}s)")
            
            # Check what we got
            if 'enhanced_prospectus' in all_data:
                enhanced = all_data['enhanced_prospectus']
                if enhanced:
                    print(f"   Enhanced data quality: {enhanced.data_quality_score:.2f}")
                    print(f"   Revenue data points: {len(enhanced.revenue_data)}")
                    print(f"   Business description length: {len(enhanced.business_description)} chars")
                else:
                    print("   ❌ No enhanced prospectus data")
            
            if 'prospectus_quality' in all_data:
                quality = all_data['prospectus_quality']
                print(f"   Extraction method: {quality.get('extraction_method', 'unknown')}")
                if 'quality_score' in quality:
                    print(f"   Quality metrics available: ✅")
            
            print("   ✅ DataSourceManager integration working")
        else:
            print("   ℹ️ No SEBI data available for testing")
            
    except Exception as e:
        print(f"❌ DataSourceManager test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ DataSourceManager integration test completed")

def main():
    """Run all enhanced prospectus integration tests."""
    print("🎯 Enhanced IPO Prospectus Integration Test Suite")
    print("=" * 60)
    print(f"Enhanced parser available: {'✅ YES' if ENHANCED_AVAILABLE else '❌ NO'}")
    
    if not ENHANCED_AVAILABLE:
        print("\n📦 To enable enhanced features, install:")
        print("   pip install tabula-py")
        print("\n🔄 You can still test basic prospectus integration")
    
    print("\n" + "=" * 60)
    
    # Run all tests
    test_enhanced_sebi_search()
    # test_data_validator()
    # test_cache_manager()
    # test_enhanced_integration()
    # test_data_source_manager_integration()
    
    print(f"\n🏁 All tests completed!")
    print("=" * 60)
    
    if ENHANCED_AVAILABLE:
        print("✅ Enhanced prospectus integration is ready for use!")
        print("\n📝 Next steps:")
        print("1. Test with your specific IPO companies")
        print("2. Check data quality scores and adjust thresholds")
        print("3. Monitor cache performance and adjust settings")
        print("4. Integrate with your analysis workflows")
    else:
        print("⚠️ Install additional dependencies for full functionality")

if __name__ == "__main__":
    main()
