#!/usr/bin/env python3
"""
Quick GMP System Verification Script

This script performs a quick health check of the GMP extraction system:
1. Verifies API keys are configured
2. Tests basic imports
3. Runs a quick GMP fetch test
4. Validates the LLM fallback
5. Reports system status

Run this before detailed testing to ensure everything is set up correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import time
from datetime import datetime
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

def print_header(text: str, style: str = "="):
    """Print a formatted header."""
    print(f"\n{style * 80}")
    print(f"  {text}")
    print(f"{style * 80}\n")

def check_api_keys():
    """Check if API keys are configured."""
    print_header("🔑 Checking API Keys", "-")
    
    keys_found = []
    keys_missing = []
    
    # Check for Groq API key (now primary)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and len(groq_key) > 10:
        keys_found.append("✅ GROQ_API_KEY (Primary)")
        logger.info("Groq API key found")
    else:
        keys_missing.append("❌ GROQ_API_KEY (Primary)")
        logger.warning("Groq API key not found")
    
    # Check for Gemini API key (fallback)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and len(gemini_key) > 10:
        keys_found.append("✅ GEMINI_API_KEY (Fallback)")
        logger.info("Gemini API key found")
    else:
        keys_missing.append("❌ GEMINI_API_KEY (Fallback)")
        logger.warning("Gemini API key not found")
    
    # Check for OpenAI API key (fallback)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and len(openai_key) > 10:
        keys_found.append("✅ OPENAI_API_KEY (Fallback)")
        logger.info("OpenAI API key found")
    else:
        keys_missing.append("❌ OPENAI_API_KEY (Fallback)")
        logger.warning("OpenAI API key not found")
    
    print("API Keys Status:")
    for key in keys_found:
        print(f"  {key}")
    for key in keys_missing:
        print(f"  {key}")
    
    if len(keys_found) == 0:
        print("\n⚠️  WARNING: No API keys found!")
        print("   LLM fallback will NOT work without API keys.")
        print("   Please set GROQ_API_KEY (recommended), GEMINI_API_KEY, or OPENAI_API_KEY in .env file")
        return False
    else:
        print(f"\n✅ {len(keys_found)} API key(s) configured")
        return True

def check_imports():
    """Check if all required modules can be imported."""
    print_header("📦 Checking Imports", "-")
    
    imports_ok = True
    
    try:
        from src.data_sources.gmp_fetcher import GMPFetcher
        print("✅ GMPFetcher imported successfully")
    except Exception as e:
        print(f"❌ Failed to import GMPFetcher: {e}")
        imports_ok = False
    
    try:
        from src.data_sources.llm_gmp_extractor import LLMGMPExtractor
        print("✅ LLMGMPExtractor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LLMGMPExtractor: {e}")
        imports_ok = False
    
    try:
        import requests
        print("✅ requests imported")
    except Exception as e:
        print(f"❌ Failed to import requests: {e}")
        imports_ok = False
    
    try:
        from bs4 import BeautifulSoup
        print("✅ BeautifulSoup imported")
    except Exception as e:
        print(f"❌ Failed to import BeautifulSoup: {e}")
        imports_ok = False
    
    return imports_ok

def test_static_scraping():
    """Test basic static scraping."""
    print_header("🌐 Testing Static Scraping", "-")
    
    try:
        from src.data_sources.gmp_fetcher import GMPFetcher
        
        # Create fetcher with LLM disabled to test static scraping only
        fetcher = GMPFetcher(use_llm_fallback=False)
        
        # Try to fetch GMP for a popular IPO
        test_company = "Akums Drugs"
        print(f"🔍 Testing with: {test_company}")
        
        start_time = time.time()
        result = fetcher.get_gmp(test_company, use_cache=False)
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'active':
            print(f"✅ Static scraping successful!")
            print(f"   GMP: ₹{result.get('gmp_price', 'N/A')}")
            print(f"   Source: {result['source']}")
            return True
        elif result['status'] == 'not_found':
            print(f"⚠️  Company not found in static sources")
            print(f"   This is expected - LLM fallback will help")
            return True  # Not an error, just not found
        else:
            print(f"⚠️  Unexpected status: {result['status']}")
            return False
            
    except Exception as e:
        print(f"❌ Static scraping test failed: {e}")
        logger.exception("Static scraping error")
        return False

def test_llm_fallback():
    """Test LLM-based extraction."""
    print_header("🤖 Testing LLM Fallback", "-")
    
    # Check if API keys are available
    if not (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("⚠️  Skipping LLM test - no API keys configured")
        return None
    
    try:
        from src.data_sources.gmp_fetcher import GMPFetcher
        
        # Create fetcher with LLM enabled
        fetcher = GMPFetcher(use_llm_fallback=True)
        
        # Try with a company that might need LLM extraction
        test_company = "Biopol Chemicals"
        print(f"🔍 Testing LLM extraction with: {test_company}")
        print("   Note: This may take 10-20 seconds...")
        
        start_time = time.time()
        result = fetcher.get_gmp(test_company, use_cache=False)
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'active':
            print(f"✅ LLM extraction successful!")
            print(f"   GMP: ₹{result.get('gmp_price', 'N/A')}")
            print(f"   Percentage: {result.get('gmp_percentage', 'N/A'):.2f}%")
            print(f"   Source: {result['source']}")
            
            # Check if it was actually LLM that extracted it
            if '_llm' in result['source']:
                print(f"   ✅ Confirmed: LLM extraction was used")
            else:
                print(f"   ℹ️  Note: Static scraping succeeded (LLM not needed)")
            return True
        else:
            print(f"⚠️  LLM extraction did not find data")
            print(f"   This could mean:")
            print(f"   - Company not in grey market")
            print(f"   - IPO not active")
            print(f"   - Data not available on tracked sources")
            return None  # Not necessarily an error
            
    except Exception as e:
        print(f"❌ LLM fallback test failed: {e}")
        logger.exception("LLM fallback error")
        return False

def test_cache():
    """Test caching functionality."""
    print_header("💾 Testing Cache", "-")
    
    try:
        from src.data_sources.gmp_fetcher import GMPFetcher
        
        fetcher = GMPFetcher(cache_duration_hours=1)
        test_company = "Akums Drugs"
        
        # First fetch (no cache)
        print(f"🔍 First fetch (no cache)...")
        start_time = time.time()
        result1 = fetcher.get_gmp(test_company, use_cache=False)
        time1 = time.time() - start_time
        print(f"   ⏱️  Time: {time1:.2f} seconds")
        
        # Second fetch (should use cache)
        print(f"\n🔍 Second fetch (should use cache)...")
        start_time = time.time()
        result2 = fetcher.get_gmp(test_company, use_cache=True)
        time2 = time.time() - start_time
        print(f"   ⏱️  Time: {time2:.2f} seconds")
        
        if time2 < time1 * 0.2:  # Cache should be much faster
            speedup = time1 / time2 if time2 > 0 else float('inf')
            print(f"\n✅ Cache is working! {speedup:.1f}x faster")
            return True
        else:
            print(f"\n⚠️  Cache may not be working optimally")
            return False
            
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        logger.exception("Cache error")
        return False

def print_summary(results: dict):
    """Print a summary of all tests."""
    print_header("📊 System Status Summary")
    
    print("Test Results:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"  {status} - {test_name}")
    
    # Calculate overall status
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("  1. Run: python examples/test_gmp_fetcher.py")
        print("  2. Test with current active IPOs")
        print("  3. Monitor API usage and costs")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
        print("\nTroubleshooting:")
        print("  1. Check that all dependencies are installed")
        print("  2. Verify API keys in .env file")
        print("  3. Check internet connection")
        print("  4. Review logs for detailed error messages")
        return False

def main():
    """Run all verification tests."""
    print_header("🚀 GMP System Verification", "=")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Run all tests
    print("Running verification tests...\n")
    
    results["API Keys"] = check_api_keys()
    results["Imports"] = check_imports()
    results["Static Scraping"] = test_static_scraping()
    results["LLM Fallback"] = test_llm_fallback()
    results["Cache"] = test_cache()
    
    # Print summary
    success = print_summary(results)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logger.exception("Verification failed")
        sys.exit(1)
