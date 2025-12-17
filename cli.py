#!/usr/bin/env python3
"""
IPO Review Agent - Command Line Interface
Run IPO analysis from the command line.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from app import IPOReviewAgent
from src.models import InvestmentRecommendation, RiskLevel


def print_report(report, format_type='text'):
    """Print analysis report in specified format."""
    
    if format_type == 'json':
        # Convert report to JSON (simplified)
        report_dict = {
            'company': {
                'name': report.company.name,
                'symbol': report.company.symbol,
                'sector': report.company.sector,
                'industry': report.company.industry
            },
            'analysis': {
                'long_term_score': report.long_term_score,
                'listing_gain_prediction': report.listing_gain_prediction,
                'recommendation': report.recommendation.value if report.recommendation else None,
                'sentiment_score': report.news_analysis.sentiment_score
            },
            'risk': {
                'overall': report.risk_assessment.overall_risk.value,
                'financial': report.risk_assessment.financial_risk.value,
                'market': report.risk_assessment.market_risk.value,
                'operational': report.risk_assessment.operational_risk.value
            },
            'strengths': report.strengths_weaknesses.strengths,
            'weaknesses': report.strengths_weaknesses.weaknesses,
            'analysis_date': report.analysis_date.isoformat()
        }
        print(json.dumps(report_dict, indent=2))
    else:
        # Text format
        print("=" * 60)
        print(f"📊 IPO ANALYSIS REPORT: {report.company.name}")
        print("=" * 60)
        
        # Basic info
        print(f"Symbol: {report.company.symbol}")
        print(f"Sector: {report.company.sector}")
        print(f"Industry: {report.company.industry}")
        
        # Key metrics
        print("\n📈 KEY METRICS")
        print("-" * 40)
        print(f"Long-term Score: {report.long_term_score:.1f}/10")
        
        if report.listing_gain_prediction is not None:
            print(f"Predicted Listing Gains: {report.listing_gain_prediction:.1f}%")
        
        if report.recommendation:
            rec_emoji = {
                "Strong Buy": "🟢",
                "Buy": "🔵", 
                "Hold": "🟡",
                "Avoid": "🔴",
                "Strong Sell": "⚫"
            }.get(report.recommendation.value, "⚪")
            print(f"Investment Recommendation: {rec_emoji} {report.recommendation.value}")
        
        # Financial metrics
        if report.financial_metrics.revenue_growth_rate is not None:
            print(f"Revenue Growth Rate: {report.financial_metrics.revenue_growth_rate:.1%}")
        
        if report.financial_metrics.profit_margin is not None:
            print(f"Profit Margin: {report.financial_metrics.profit_margin:.1%}")
        
        # Risk assessment
        print("\n⚠️ RISK ASSESSMENT")
        print("-" * 40)
        risk_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴", "Very High": "⚫"}
        
        print(f"Overall Risk: {risk_emoji.get(report.risk_assessment.overall_risk.value, '⚪')} {report.risk_assessment.overall_risk.value}")
        print(f"Financial Risk: {risk_emoji.get(report.risk_assessment.financial_risk.value, '⚪')} {report.risk_assessment.financial_risk.value}")
        print(f"Market Risk: {risk_emoji.get(report.risk_assessment.market_risk.value, '⚪')} {report.risk_assessment.market_risk.value}")
        print(f"Operational Risk: {risk_emoji.get(report.risk_assessment.operational_risk.value, '⚪')} {report.risk_assessment.operational_risk.value}")
        
        # Risk factors
        if report.risk_assessment.risk_factors:
            print("\nRisk Factors:")
            for factor in report.risk_assessment.risk_factors:
                print(f"  • {factor}")
        
        # Strengths and weaknesses
        print("\n💪 STRENGTHS")
        print("-" * 40)
        if report.strengths_weaknesses.strengths:
            for strength in report.strengths_weaknesses.strengths:
                print(f"✅ {strength}")
        else:
            print("No specific strengths identified")
        
        print("\n⚠️ WEAKNESSES")
        print("-" * 40)
        if report.strengths_weaknesses.weaknesses:
            for weakness in report.strengths_weaknesses.weaknesses:
                print(f"❌ {weakness}")
        else:
            print("No major weaknesses identified")
        
        # Market sentiment
        print("\n📰 MARKET SENTIMENT")
        print("-" * 40)
        sentiment = report.news_analysis.sentiment_score
        if sentiment > 0.1:
            print(f"📈 Positive sentiment: {sentiment:.2f}")
        elif sentiment < -0.1:
            print(f"📉 Negative sentiment: {sentiment:.2f}")
        else:
            print(f"😐 Neutral sentiment: {sentiment:.2f}")
        
        if report.news_analysis.key_themes:
            print("Key themes:", ", ".join(report.news_analysis.key_themes[:5]))
        
        print(f"\nAnalysis performed: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="IPO Review Agent - Comprehensive IPO Analysis",
        epilog="Example: python cli.py AAPL 'Apple Inc.' --format json"
    )
    
    parser.add_argument("company_name", help="IPO Company name (e.g., 'Zomato Limited')")
    parser.add_argument("--sector", default="Technology", help="Company sector")
    parser.add_argument("--price-min", type=float, default=100.0, help="IPO price minimum (₹)")
    parser.add_argument("--price-max", type=float, default=120.0, help="IPO price maximum (₹)")
    parser.add_argument("--exchange", default="NSE & BSE", help="Expected listing exchange")
    parser.add_argument("--format", choices=['text', 'json'], default='text',
                       help="Output format (default: text)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        print(f"🔍 Analyzing IPO: {args.company_name}")
        print(f"💰 Price Range: ₹{args.price_min} - ₹{args.price_max}")
        print(f"🏢 Sector: {args.sector}")
        
        # Create IPO details
        ipo_details = {
            'price_range': (args.price_min, args.price_max),
            'sector': args.sector,
            'exchange': args.exchange
        }
        
        # Initialize agent and perform analysis
        agent = IPOReviewAgent()
        report = agent.analyze_ipo(args.company_name, ipo_details)
        
        # Print results
        print_report(report, args.format)
        
        # Return appropriate exit code based on recommendation
        if report.recommendation in [InvestmentRecommendation.STRONG_BUY, InvestmentRecommendation.BUY]:
            return 0  # Success - positive recommendation
        elif report.recommendation == InvestmentRecommendation.HOLD:
            return 1  # Neutral recommendation
        else:
            return 2  # Negative recommendation
    
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
