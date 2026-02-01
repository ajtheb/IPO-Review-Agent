"""
CLI app for IPOReviewAgent

Run:
    python cli_app.py --company "Vidya Wires Limited" --use-llm --llm-provider openai
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

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
    ipo_details = build_ipo_details(args)

    print("=== IPO Review Agent (CLI) ===")
    print(f"Company: {args.company}")
    print(f"Sector: {args.sector}")
    print(f"Exchange: {args.exchange}")
    print(f"Price Range: {ipo_details['price_range'][0]} - {ipo_details['price_range'][1]}")
    print(f"Use LLM: {args.use_llm} (provider: {args.llm_provider})\n")

    agent = IPOReviewAgentCLI(use_llm=args.use_llm, llm_provider=args.llm_provider)
    report = agent.analyze_ipo(args.company, ipo_details)

    # Minimal structured summary
    print("\n=== Summary ===")
    print(f"Company: {report.company.name}")
    print(f"Estimated Market Cap: {report.company.market_cap}")
    print(f"Listing Gain Prediction: {report.listing_gain_prediction:.1f}%")
    print(f"Long-term Score: {report.long_term_score:.1f}/10")
    print(f"Recommendation: {report.recommendation.value if report.recommendation else 'None'}")


if __name__ == "__main__":
    main()
