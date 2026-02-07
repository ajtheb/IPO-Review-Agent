"""
CLI app for IPOReviewAgent

Run:
    python cli_app.py --company "Vidya Wires Limited" --use-llm --llm-provider openai
    python cli_app.py --company "Fractal Analytics" --gmp-analysis --llm-provider groq
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
import re

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path (same as in Streamlit app)
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_sources import DataSourceManager
from src.analyzers import FinancialAnalyzer, SentimentAnalyzer, RiskAnalyzer, BusinessAnalyzer
from src.models import IPOAnalysisReport, CompanyBasics, InvestmentRecommendation

# Import enhanced analyzer if available
try:
    from src.analyzers import EnhancedFinancialAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYZER_AVAILABLE = False
    logger.warning("EnhancedFinancialAnalyzer not available, falling back to basic analyzer")

# Import GMP extractor
try:
    from src.data_sources.llm_gmp_extractor import LLMGMPExtractor
    from groq import Groq
    GMP_EXTRACTOR_AVAILABLE = True
except ImportError:
    GMP_EXTRACTOR_AVAILABLE = False
    logger.warning("GMP extractor not available")


class IPOReviewAgentCLI:
    """IPO Review Agent wrapper for CLI usage (no Streamlit)."""

    def __init__(self, use_llm: bool = True, llm_provider: str = "openai"):
        self.data_manager = DataSourceManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.business_analyzer = BusinessAnalyzer()

        if use_llm and ENHANCED_ANALYZER_AVAILABLE:
            try:
                self.financial_analyzer = EnhancedFinancialAnalyzer(llm_provider=llm_provider)
                self.enhanced_analysis = True
                logger.info("Enhanced LLM-powered financial analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced analyzer: {e}")
                self.financial_analyzer = FinancialAnalyzer()
                self.enhanced_analysis = False
        else:
            self.financial_analyzer = FinancialAnalyzer()
            self.enhanced_analysis = False

    def _estimate_market_cap(self, ipo_details: dict) -> float:
        price_range = ipo_details.get("price_range", (100, 120))
        if isinstance(price_range, str):
            try:
                if "-" in price_range:
                    min_price, max_price = map(float, price_range.split("-"))
                    price_range = (min_price, max_price)
                else:
                    p = float(price_range)
                    price_range = (p, p)
            except (ValueError, AttributeError):
                price_range = (100, 120)
        avg_price = (price_range[0] + price_range[1]) / 2
        estimated_shares = 10_000_000
        return avg_price * estimated_shares

    def _predict_listing_gains(self, financial_metrics, news_analysis, risk_assessment) -> float:
        base_gain = 10.0
        if financial_metrics and getattr(financial_metrics, "revenue_growth_rate", None):
            if financial_metrics.revenue_growth_rate > 0.2:
                base_gain += 15
            elif financial_metrics.revenue_growth_rate < 0:
                base_gain -= 20

        sentiment_multiplier = 1 + (news_analysis.sentiment_score * 0.3)
        base_gain *= sentiment_multiplier

        if risk_assessment.overall_risk.value == "High":
            base_gain *= 0.7
        elif risk_assessment.overall_risk.value == "Low":
            base_gain *= 1.3

        return max(-50, min(100, base_gain))

    def _calculate_long_term_score(self, financial_metrics, risk_assessment, strengths_weaknesses) -> float:
        score = 5.0

        if financial_metrics and getattr(financial_metrics, "profit_margin", None) is not None:
            if financial_metrics.profit_margin > 0.1:
                score += 1.5
            elif financial_metrics.profit_margin < 0:
                score -= 2

        if financial_metrics and getattr(financial_metrics, "revenue_growth_rate", None):
            if financial_metrics.revenue_growth_rate > 0.15:
                score += 1.5

        if risk_assessment.overall_risk.value == "Low":
            score += 1
        elif risk_assessment.overall_risk.value == "High":
            score -= 1.5
        elif risk_assessment.overall_risk.value == "Very High":
            score -= 2.5

        strength_bonus = min(len(strengths_weaknesses.strengths) * 0.3, 1.5)
        weakness_penalty = min(len(strengths_weaknesses.weaknesses) * 0.2, 1.0)
        score += strength_bonus - weakness_penalty

        return max(0, min(10, score))

    def _generate_recommendation(self, listing_gain, long_term_score, risk_assessment) -> InvestmentRecommendation:
        if long_term_score >= 8 and listing_gain > 20:
            return InvestmentRecommendation.STRONG_BUY
        elif long_term_score >= 6.5 and listing_gain > 10:
            return InvestmentRecommendation.BUY
        elif long_term_score >= 4 and risk_assessment.overall_risk.value != "Very High":
            return InvestmentRecommendation.HOLD
        else:
            return InvestmentRecommendation.AVOID

    def analyze_ipo(self, company_name: str, ipo_details: dict) -> IPOAnalysisReport:
        logger.info(f"Starting IPO analysis for {company_name}")

        raw_data = self.data_manager.collect_ipo_data(company_name, ipo_details)
        # print("Collected raw data keys:", list(raw_data.keys()))

        llm_analysis = raw_data.get("llm_analysis", {})
        investment_thesis = llm_analysis.get("llm_investment_thesis")
        if investment_thesis:
            print("\n=== AI‑Generated Investment Thesis ===")
            print(investment_thesis)

        company = CompanyBasics(
            name=company_name,
            symbol="IPO-PENDING",
            sector=ipo_details.get("sector", "Unknown"),
            industry=ipo_details.get("sector", "Unknown"),
            market_cap=self._estimate_market_cap(ipo_details),
            employees=raw_data.get("employees"),
            website=raw_data.get("website"),
            description=raw_data.get("description"),
        )

        if self.enhanced_analysis:
            try:
                comprehensive = self.financial_analyzer.analyze_comprehensive(
                    raw_data, company_name, ipo_details.get("sector", "Unknown")
                )
                enhanced_metrics = comprehensive.get("enhanced_metrics")
                traditional_metrics = comprehensive.get("traditional_metrics")
                if enhanced_metrics:
                    financial_metrics = enhanced_metrics
                    print("Using enhanced LLM-powered financial metrics")
                elif traditional_metrics:
                    financial_metrics = traditional_metrics
                    print("Using traditional financial metrics from comprehensive analysis")
                else:
                    financial_metrics = None

                raw_data["llm_analysis"] = comprehensive.get("llm_analysis", {})
                raw_data["valuation_analysis"] = comprehensive.get("valuation_analysis", {})
                raw_data["peer_analysis"] = comprehensive.get("peer_analysis", {})
                raw_data["analysis_quality"] = comprehensive.get("analysis_quality", {})
            except Exception as e:
                logger.error(f"Enhanced analysis failed: {e}")
                print(f"Enhanced LLM analysis failed: {e}")
                financial_metrics = self.financial_analyzer.calculate_financial_metrics(
                    raw_data.get("financial_statements", {})
                )
                raw_data["llm_analysis"] = {"error": str(e)}
                raw_data["analysis_error"] = str(e)
        else:
            financial_metrics = self.financial_analyzer.calculate_financial_metrics(
                raw_data.get("financial_statements", {})
            )

        news_analysis = self.sentiment_analyzer.analyze_news_sentiment(
            raw_data.get("company_news", []) + raw_data.get("market_news", [])
        )

        company_info = {
            "sector": ipo_details.get("sector", "Unknown"),
            "market_cap": self._estimate_market_cap(ipo_details),
            "ipo_price_range": ipo_details.get("price_range", (100, 120)),
            "exchange": ipo_details.get("exchange", "NSE"),
        }

        risk_assessment = self.risk_analyzer.assess_risks(
            financial_metrics, raw_data, news_analysis, company_info
        )

        strengths_weaknesses = self.business_analyzer.analyze_business_fundamentals(
            company_info, financial_metrics, raw_data
        )

        listing_gain_prediction = self._predict_listing_gains(
            financial_metrics, news_analysis, risk_assessment
        )

        long_term_score = self._calculate_long_term_score(
            financial_metrics, risk_assessment, strengths_weaknesses
        )

        recommendation = self._generate_recommendation(
            listing_gain_prediction, long_term_score, risk_assessment
        )

        print("\n=== Financial Metrics Summary ===")
        if financial_metrics:
            for attr in [
                "revenue_growth_rate",
                "profit_margin",
                "roe",
                "current_ratio",
                "debt_to_equity",
            ]:
                val = getattr(financial_metrics, attr, None)
                print(f"{attr}: {val}")
        else:
            print("No financial metrics available")

        print("\n=== Final Analysis Results ===")
        print(f"Listing Gain Prediction: {listing_gain_prediction:.1f}%")
        print(f"Long-term Score: {long_term_score:.1f}/10")
        print(f"Recommendation: {recommendation.value if recommendation else 'None'}")

        report = IPOAnalysisReport(
            company=company,
            financial_metrics=financial_metrics,
            market_data={},
            news_analysis=news_analysis,
            risk_assessment=risk_assessment,
            strengths_weaknesses=strengths_weaknesses,
            listing_gain_prediction=listing_gain_prediction,
            long_term_score=long_term_score,
            recommendation=recommendation,
            analyst_confidence=0.75,
        )

        report.raw_data = raw_data
        return report

    def analyze_gmp_with_brave_search(self, company_name: str, llm_provider: str = "groq", save_analysis: bool = True):
        """
        Analyze GMP data using Brave Search API and Groq LLM.
        
        Args:
            company_name: Name of the company
            llm_provider: LLM provider to use (default: groq)
            save_analysis: Whether to save analysis to file
            
        Returns:
            Dictionary with GMP data and comprehensive analysis
        """
        if not GMP_EXTRACTOR_AVAILABLE:
            logger.error("GMP extractor not available")
            print("❌ GMP extractor not available. Please install required packages:")
            print("   pip install groq requests beautifulsoup4")
            return None
        
        # Check API keys
        groq_key = os.getenv('GROQ_API_KEY')
        brave_key = os.getenv('BRAVE_API_KEY')
        
        if not groq_key:
            logger.error("GROQ_API_KEY not set")
            print("❌ GROQ_API_KEY not found in .env file")
            return None
        
        if not brave_key:
            logger.error("BRAVE_API_KEY not set")
            print("❌ BRAVE_API_KEY not found in .env file")
            return None
        
        print("\n" + "="*80)
        print(f"🔍 GMP ANALYSIS FOR: {company_name}")
        print("="*80)
        
        try:
            # Initialize extractor and Groq client
            extractor = LLMGMPExtractor(provider=llm_provider, use_brave_search=True)
            groq_client = Groq(api_key=groq_key)
            
            # Search Brave and scrape websites
            print("\n� Searching Brave API and scraping websites...")
            search_results = extractor.search_gmp_with_brave(company_name, max_results=5)
            
            if not search_results:
                print("❌ No search results found")
                return None
            
            # Scrape website content
            scraped_chunks = []
            
            for i, result in enumerate(search_results[:3], 1):
                url = result.get('url')
                # print(f"   [{i}/3] {url}")  # Optional: comment out for cleaner output
                
                html_content = extractor.scrape_url_content(url)
                if html_content:
                    text_content = extractor.extract_text_from_html(html_content)
                    
                    # Save scraped content
                    extractor.save_scraped_content(
                        company_name=company_name,
                        url=url,
                        html_content=html_content,
                        text_content=text_content,
                        folder="gmp_chunks"
                    )
                    
                    # Add to chunks (limit size)
                    max_chunk_size = 5000
                    if len(text_content) > max_chunk_size:
                        text_content = text_content[:max_chunk_size]
                    
                    scraped_chunks.append(f"Source: {url}\n{text_content}")
                    # print(f"       ✅ Scraped {len(text_content)} characters")  # Optional: comment out
                else:
                    pass  # print(f"       ⚠️  Failed to scrape")  # Optional: comment out
            
            if not scraped_chunks:
                print("\n⚠️  No content could be scraped from websites")
                return None
            
            print(f"✅ Found data from {len(scraped_chunks)} sources")
            
            # Extract structured GMP data (silently for use in analysis)
            result = extractor.extract_gmp_from_brave_results(
                company_name=company_name,
                search_results=search_results,
                scrape_websites=False,  # Already scraped
                save_scraped=False
            )
            
            # Generate comprehensive analysis
            print("🤖 Generating comprehensive analysis...")
            
            analysis = self._generate_gmp_analysis(company_name, scraped_chunks, groq_client)
            
            print("\n" + "="*80)
            print("📝 GMP ANALYSIS REPORT")
            print("="*80 + "\n")
            print(analysis)
            
            # Save analysis to file if requested
            if save_analysis:
                os.makedirs("gmp_chunks", exist_ok=True)
                safe_name = re.sub(r'[^\w\s-]', '_', company_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_file = f"gmp_chunks/{safe_name}_analysis_{timestamp}.txt"
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    f.write(f"GMP Analysis for {company_name}\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write("STRUCTURED DATA:\n")
                    f.write(f"  GMP Price: ₹{result.get('gmp_price', 'N/A')}\n")
                    f.write(f"  GMP %: {result.get('gmp_percentage', 'N/A')}%\n")
                    f.write(f"  Issue Price: ₹{result.get('issue_price', 'N/A')}\n")
                    f.write(f"  Expected Listing: ₹{result.get('expected_listing_price', 'N/A')}\n")
                    f.write(f"  IPO Status: {result.get('ipo_status', 'N/A')}\n")
                    f.write("\n" + "="*80 + "\n\n")
                    
                    f.write("COMPREHENSIVE ANALYSIS:\n")
                    f.write(analysis)
                    f.write("\n\n" + "="*80 + "\n")
                    f.write(f"Sources: {len(scraped_chunks)} websites scraped\n")
                    for i, result_item in enumerate(search_results[:3], 1):
                        f.write(f"  {i}. {result_item['url']}\n")
                
                print(f"\n💾 Analysis saved to: {analysis_file}")
            
            return {
                'structured_data': result,
                'analysis': analysis,
                'sources': [r['url'] for r in search_results[:3]]
            }
            
        except Exception as e:
            logger.error(f"Error during GMP analysis: {e}")
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_gmp_analysis(self, company_name: str, context_chunks: list, groq_client) -> str:
        """Generate comprehensive GMP analysis using Groq API."""
        combined_context = "\n\n".join(context_chunks[:5])
        
        analysis_prompt = f"""You are a financial analyst specializing in Indian IPO market analysis. Analyze the Grey Market Premium (GMP) data for {company_name} based on the following information:

CONTEXT FROM WEB SOURCES:
{combined_context}

Please provide a comprehensive GMP analysis covering:

1. **Current GMP Status**
   - Current GMP price and percentage
   - Issue price and expected listing price
   - Whether GMP is positive, negative, or neutral

2. **Market Sentiment Analysis**
   - What does the GMP indicate about market demand?
   - Is the IPO oversubscribed or undersubscribed?
   - Investor confidence level

3. **Listing Gain Potential**
   - Expected listing gains based on GMP
   - Risk-reward assessment
   - Comparison with similar IPOs if mentioned

4. **IPO Timeline & Status**
   - Current IPO status (Open/Upcoming/Closed/Listed)
   - Important dates (opening, closing, listing)
   - Time-sensitive insights

5. **Investment Recommendation**
   - Should investors apply for this IPO?
   - Grey market trends (rising/falling)
   - Risk factors to consider

6. **Key Takeaways**
   - 3-5 bullet points summarizing the analysis
   - Action items for potential investors

Format the response in clear, professional language suitable for investment decision-making. Use ₹ symbol for prices and % for percentages. Be specific with numbers where available.

If the context doesn't contain sufficient GMP data, clearly state what information is missing."""

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in Indian IPO markets and grey market premium analysis."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating analysis: {e}"

    # ...existing code...


def parse_args():
    parser = argparse.ArgumentParser(description="CLI IPO Review Agent")
    parser.add_argument(
        "--company",
        "-c",
        required=True,
        help="IPO / Company name (e.g., 'Vidya Wires Limited')",
    )
    parser.add_argument(
        "--sector",
        "-s",
        default="Unknown",
        help="Sector name (optional, e.g., 'Financial Services')",
    )
    parser.add_argument(
        "--exchange",
        "-e",
        default="NSE",
        help="Exchange (default: NSE)",
    )
    parser.add_argument(
        "--price-range",
        "-p",
        default=None,
        help="IPO price range, e.g. '100-120' or single price '110'",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable enhanced LLM analysis (requires EnhancedFinancialAnalyzer and API key)",
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic", "groq", "gemini"],
        help="LLM provider to use for enhanced analysis",
    )
    parser.add_argument(
        "--gmp-analysis",
        action="store_true",
        help="Run GMP analysis using Brave Search and Groq LLM (requires GROQ_API_KEY and BRAVE_API_KEY)",
    )
    parser.add_argument(
        "--gmp-only",
        action="store_true",
        help="Run only GMP analysis without full IPO analysis",
    )
    return parser.parse_args()


def build_ipo_details(args) -> dict:
    price_range = None
    if args.price_range:
        pr = args.price_range.strip()
        if "-" in pr:
            try:
                lo, hi = pr.split("-", 1)
                price_range = (float(lo), float(hi))
            except ValueError:
                price_range = (100, 120)
        else:
            try:
                p = float(pr)
                price_range = (p, p)
            except ValueError:
                price_range = (100, 120)
    else:
        price_range = (100, 120)

    return {
        "company_name": args.company,
        "sector": args.sector,
        "exchange": args.exchange,
        "price_range": price_range,
    }


def main():
    args = parse_args()
    
    print("=== IPO Review Agent (CLI) ===")
    print(f"Company: {args.company}")
    
    # If GMP-only mode, just run GMP analysis and exit
    if args.gmp_only:
        print(f"Mode: GMP Analysis Only")
        print(f"LLM Provider: {args.llm_provider}\n")
        
        agent = IPOReviewAgentCLI(use_llm=False, llm_provider=args.llm_provider)
        gmp_result = agent.analyze_gmp_with_brave_search(args.company, args.llm_provider, save_analysis=True)
        
        if gmp_result:
            print("\n" + "="*80)
            print("✅ GMP ANALYSIS COMPLETE")
            print("="*80)
            print(f"\n📁 Check gmp_chunks/ folder for:")
            print(f"   - Brave search results")
            print(f"   - Scraped website content (HTML & text)")
            print(f"   - Comprehensive analysis report")
        else:
            print("\n❌ GMP analysis failed")
        
        return
    
    # Standard IPO analysis mode
    ipo_details = build_ipo_details(args)
    
    print(f"Sector: {args.sector}")
    print(f"Exchange: {args.exchange}")
    print(f"Price Range: {ipo_details['price_range'][0]} - {ipo_details['price_range'][1]}")
    print(f"Use LLM: {args.use_llm} (provider: {args.llm_provider})")
    print(f"GMP Analysis: {args.gmp_analysis}\n")

    agent = IPOReviewAgentCLI(use_llm=args.use_llm, llm_provider=args.llm_provider)
    
    # Run GMP analysis if requested
    if args.gmp_analysis:
        print("\n" + "="*80)
        print("📊 RUNNING GMP ANALYSIS FIRST")
        print("="*80)
        gmp_result = agent.analyze_gmp_with_brave_search(args.company, args.llm_provider, save_analysis=True)
        
        if gmp_result:
            print("\n✅ GMP analysis completed successfully")
            # Add GMP data to ipo_details if available
            if gmp_result.get('structured_data', {}).get('status') == 'success':
                gmp_data = gmp_result['structured_data']
                ipo_details['gmp_price'] = gmp_data.get('gmp_price')
                ipo_details['gmp_percentage'] = gmp_data.get('gmp_percentage')
                ipo_details['expected_listing_price'] = gmp_data.get('expected_listing_price')
        else:
            print("\n⚠️  GMP analysis failed, continuing with standard analysis")
        
        print("\n" + "="*80)
        print("📊 RUNNING STANDARD IPO ANALYSIS")
        print("="*80)
    
    # Run standard IPO analysis
    report = agent.analyze_ipo(args.company, ipo_details)

    # Minimal structured summary
    print("\n=== Summary ===")
    print(f"Company: {report.company.name}")
    print(f"Estimated Market Cap: {report.company.market_cap}")
    print(f"Listing Gain Prediction: {report.listing_gain_prediction:.1f}%")
    print(f"Long-term Score: {report.long_term_score:.1f}/10")
    print(f"Recommendation: {report.recommendation.value if report.recommendation else 'None'}")
    
    # Show GMP data if available
    if args.gmp_analysis and ipo_details.get('gmp_price'):
        print("\n=== GMP Data (from Brave Search) ===")
        print(f"GMP Price: ₹{ipo_details['gmp_price']}")
        print(f"GMP Percentage: {ipo_details['gmp_percentage']}%")
        print(f"Expected Listing: ₹{ipo_details['expected_listing_price']}")
        print(f"\n📁 Detailed GMP analysis saved in gmp_chunks/ folder")


if __name__ == "__main__":
    main()
