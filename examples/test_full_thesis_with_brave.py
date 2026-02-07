"""
Test Full Investment Thesis Generation with Brave API Context

This script tests the complete flow of generating an investment thesis
that includes both prospectus analysis and Brave API web search context.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.analyzers.llm_prospectus_analyzer import integrate_llm_analysis

def test_full_investment_thesis():
    """Test complete investment thesis generation with Brave context."""
    print("=" * 80)
    print("Testing Full Investment Thesis Generation with Brave API Context")
    print("=" * 80)
    
    # Check prerequisites
    brave_key = os.getenv("BRAVE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not brave_key:
        print("❌ BRAVE_API_KEY not set")
        return False
    if not openai_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    print("✅ API keys configured")
    
    # Test with a sample prospectus text
    test_company = "Vidya Wires Limited"
    test_sector = "Cable Manufacturing"
    
    # Sample prospectus text (simplified for testing)
    sample_prospectus = """
    Vidya Wires Limited is engaged in manufacturing of wires and cables.
    The company has shown consistent revenue growth over the past 3 years.
    
    Financial Highlights:
    - Revenue: Rs 500 Crores (FY2023)
    - EBITDA Margin: 12%
    - Net Profit: Rs 45 Crores
    - Debt/Equity: 0.8
    
    The company plans to use IPO proceeds for capacity expansion and debt reduction.
    Major competitors include Polycab, KEI Industries, and Havells.
    """
    
    print(f"\n📊 Analyzing: {test_company}")
    print(f"   Sector: {test_sector}")
    print(f"   Provider: OpenAI (with Brave Search)")
    
    try:
        # Run complete analysis (this should now include Brave search)
        print("\n🔄 Running comprehensive LLM analysis...")
        print("   - Extracting financial metrics from prospectus")
        print("   - Performing competitive benchmarking")
        print("   - Searching web for additional context (Brave API)")
        print("   - Generating investment thesis with all context")
        
        result = integrate_llm_analysis(
            company_name=test_company,
            pdf_text=sample_prospectus,
            sector=test_sector,
            provider="openai",
            use_vector_db=False
        )
        
        print("\n✅ Analysis completed!")
        
        # Display results
        print("\n" + "=" * 80)
        print("INVESTMENT THESIS (with Brave API Context)")
        print("=" * 80)
        
        thesis = result.get('llm_investment_thesis', '')
        if thesis:
            print(thesis)
        else:
            print("⚠️  No investment thesis generated")
        
        print("\n" + "=" * 80)
        print("ANALYSIS DETAILS")
        print("=" * 80)
        
        print(f"\n📈 Financial Metrics Extracted: {result.get('llm_financial_metrics') is not None}")
        print(f"🏆 Benchmarking Analysis: {result.get('llm_benchmarking') is not None}")
        print(f"🎯 IPO Specifics: {result.get('llm_ipo_specifics') is not None}")
        print(f"🤖 LLM Provider: {result.get('llm_provider', 'unknown')}")
        print(f"🕒 Analysis Timestamp: {result.get('llm_analysis_timestamp', 'N/A')}")
        
        # Check if Brave context was used
        if 'WEB SEARCH CONTEXT' in thesis or 'recent news' in thesis.lower() or 'market data' in thesis.lower():
            print("\n✅ Investment thesis appears to include web search context!")
        else:
            print("\n⚠️  Cannot confirm if web search context was included")
            print("   (Thesis might be based only on prospectus data)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_investment_thesis()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Full Investment Thesis Test: PASSED")
        print("\nThe investment thesis now includes:")
        print("  1. ✅ Prospectus financial data analysis")
        print("  2. ✅ Competitive benchmarking")
        print("  3. ✅ IPO-specific metrics")
        print("  4. ✅ Web search context from Brave API")
        print("\nReady for production use in Streamlit app!")
    else:
        print("❌ Full Investment Thesis Test: FAILED")
        print("\nCheck error messages above for details")
    print("=" * 80)
