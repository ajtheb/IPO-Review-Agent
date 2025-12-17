"""
Example usage of IPO Review Agent.
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from app import IPOReviewAgent


def example_analysis():
    """Example IPO analysis for a public company."""
    
    # Initialize the agent
    agent = IPOReviewAgent()
    
    # Example Indian IPO companies to analyze
    examples = [
        {
            "name": "LIC (Life Insurance Corporation)", 
            "ipo_details": {
                "price_range": (902, 949),
                "sector": "Financial Services",
                "exchange": "NSE & BSE"
            }
        },
        {
            "name": "Paytm (One97 Communications)", 
            "ipo_details": {
                "price_range": (2080, 2150),
                "sector": "Financial Services", 
                "exchange": "NSE & BSE"
            }
        },
        {
            "name": "Zomato Limited", 
            "ipo_details": {
                "price_range": (72, 76),
                "sector": "Technology",
                "exchange": "NSE & BSE"
            }
        },
    ]
    
    print("🚀 IPO Review Agent - Example Analysis")
    print("=" * 50)
    
    for company in examples:
        print(f"\n📊 Analyzing {company['name']}...")
        price_range = company['ipo_details']['price_range']
        print(f"💰 IPO Price: ₹{price_range[0]} - ₹{price_range[1]}")
        
        try:
            # Perform analysis
            report = agent.analyze_ipo(company['name'], company['ipo_details'])
            
            # Display key results
            print(f"Company: {report.company.name}")
            print(f"Sector: {report.company.sector}")
            print(f"Long-term Score: {report.long_term_score:.1f}/10")
            print(f"Recommendation: {report.recommendation.value if report.recommendation else 'N/A'}")
            print(f"Risk Level: {report.risk_assessment.overall_risk.value}")
            
            if report.listing_gain_prediction:
                print(f"Predicted Listing Gains: {report.listing_gain_prediction:.1f}%")
            
            print(f"Sentiment Score: {report.news_analysis.sentiment_score:.2f}")
            
            if report.strengths_weaknesses.strengths:
                print("Key Strengths:")
                for strength in report.strengths_weaknesses.strengths[:3]:
                    print(f"  ✅ {strength}")
            
            if report.risk_assessment.risk_factors:
                print("Key Risks:")
                for risk in report.risk_assessment.risk_factors[:3]:
                    print(f"  ⚠️ {risk}")
                    
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
        
        print("-" * 40)


def check_api_configuration():
    """Check if API keys are properly configured."""
    print("🔑 Checking API Configuration...")
    
    required_keys = [
        'ALPHA_VANTAGE_API_KEY',
        'NEWS_API_KEY', 
        'FINNHUB_API_KEY'
    ]
    
    configured_keys = []
    missing_keys = []
    
    for key in required_keys:
        if os.getenv(key):
            configured_keys.append(key)
        else:
            missing_keys.append(key)
    
    if configured_keys:
        print("✅ Configured API keys:")
        for key in configured_keys:
            print(f"   • {key}")
    
    if missing_keys:
        print("❌ Missing API keys:")
        for key in missing_keys:
            print(f"   • {key}")
        print("\n💡 To get API keys:")
        print("   • Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print("   • News API: https://newsapi.org/")
        print("   • Finnhub: https://finnhub.io/")
    
    return len(configured_keys) > 0


if __name__ == "__main__":
    print("IPO Review Agent - Example Usage")
    print("================================")
    
    # Check API configuration first
    if check_api_configuration():
        print("\n" + "="*50)
        example_analysis()
    else:
        print("\n⚠️  Please configure at least one API key to run examples.")
        print("Copy .env.example to .env and add your API keys.")
