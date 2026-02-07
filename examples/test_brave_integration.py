"""
Test Brave API Integration for Investment Thesis

This script tests the new Brave API integration in the investment thesis generation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

def test_brave_search():
    """Test Brave search functionality."""
    print("=" * 80)
    print("Testing Brave API Integration for Investment Thesis")
    print("=" * 80)
    
    # Check if BRAVE_API_KEY is set
    brave_key = os.getenv("BRAVE_API_KEY")
    if not brave_key:
        print("❌ BRAVE_API_KEY not set")
        print("   Set it in .env file to enable web search context")
        return False
    else:
        print("✅ BRAVE_API_KEY is configured")
    
    # Initialize analyzer
    print("\n📊 Initializing LLM Prospectus Analyzer...")
    try:
        analyzer = LLMProspectusAnalyzer(provider="openai", use_vector_db=False)
        print("✅ Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False
    
    # Test Brave search
    test_company = "Vidya Wires Limited"
    print(f"\n🔍 Searching Brave for: {test_company}")
    
    try:
        results = analyzer.search_brave_for_ipo_context(test_company, max_results=3)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n   {i}. {result.get('title', 'N/A')}")
                print(f"      URL: {result.get('url', 'N/A')}")
                print(f"      Description: {result.get('description', 'N/A')[:100]}...")
                if result.get('age'):
                    print(f"      Age: {result.get('age')}")
            
            # Test context formatting
            print("\n📝 Formatted Context:")
            print("-" * 80)
            context = analyzer._format_brave_context(results)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 80)
            
            return True
        else:
            print("⚠️  No results found (this might be okay if company is not well-known)")
            return True
            
    except Exception as e:
        print(f"❌ Brave search failed: {e}")
        return False

if __name__ == "__main__":
    success = test_brave_search()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Brave API Integration Test: PASSED")
        print("\nNext Steps:")
        print("1. Run the Streamlit app: streamlit run app.py")
        print("2. Analyze an IPO with enhanced LLM analysis enabled")
        print("3. Check the Investment Thesis section for web context")
    else:
        print("❌ Brave API Integration Test: FAILED")
        print("\nTroubleshooting:")
        print("1. Ensure BRAVE_API_KEY is set in .env file")
        print("2. Check that requests library is installed: pip install requests")
        print("3. Verify Brave API key is valid at: https://brave.com/search/api/")
    print("=" * 80)
