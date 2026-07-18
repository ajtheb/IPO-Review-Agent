"""
IPO Review Agent - core business logic.

Decoupled from the Streamlit UI (see ../app.py) so it can be imported by the
CLI and test/benchmark scripts without pulling in streamlit/Selenium as an
import-time side effect.
"""

import json
import os
import re
from datetime import datetime

from loguru import logger

from .data_sources import DataSourceManager
from .analyzers import FinancialAnalyzer, SentimentAnalyzer, RiskAnalyzer, BusinessAnalyzer
from .models import IPOAnalysisReport, CompanyBasics, InvestmentRecommendation

# Import enhanced analyzer if available
try:
    from .analyzers import EnhancedFinancialAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYZER_AVAILABLE = False

# Import GMP extractor
try:
    from .data_sources.llm_gmp_extractor import LLMGMPExtractor
    from groq import Groq
    GMP_EXTRACTOR_AVAILABLE = True
except ImportError:
    GMP_EXTRACTOR_AVAILABLE = False
    logger.warning("GMP extractor not available")


# GMP analysis LLM configuration - centralized here so it can be swapped
# without touching analyze_gmp's control flow.
#
# Structured extraction and the narrative analysis are produced by a single
# call sharing one prompt/context, rather than two separate Groq calls each
# re-sending the (large) scraped context. That roughly halves token usage
# per company against Groq's daily token cap.
GMP_ANALYSIS_MODEL = "llama-3.3-70b-versatile"
GMP_ANALYSIS_TEMPERATURE = 0.3
GMP_ANALYSIS_MAX_TOKENS = 2200
GMP_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert financial analyst specializing in Indian IPO markets "
    "and grey market premium analysis. Always respond in the exact two-part "
    "STRUCTURED_DATA / ANALYSIS format requested."
)
GMP_ANALYSIS_PROMPT_TEMPLATE = """You are a financial analyst specializing in Indian IPO market analysis. Analyze the Grey Market Premium (GMP) data for {company_name} based on the following information:

CONTEXT FROM WEB SOURCES:
{combined_context}

Respond in exactly two parts, in this order.

STRUCTURED_DATA:
A single JSON object (no markdown code fences) with these exact keys:
{{
    "gmp_price": <number or null>,
    "gmp_percentage": <number or null>,
    "issue_price": <number or null>,
    "expected_listing_price": <number or null>,
    "ipo_status": "<Open|Upcoming|Closed|Listed|null>",
    "open_date": "<date or null>",
    "close_date": "<date or null>",
    "listing_date": "<date or null>",
    "confidence": "<high|medium|low>"
}}
Use null for missing values. Extract numeric values only (strip ₹, %, commas).

ANALYSIS:
A comprehensive GMP analysis covering:

1. **Current GMP Status** - current GMP price and percentage, issue price and expected listing price, whether GMP is positive, negative, or neutral
2. **Market Sentiment Analysis** - what the GMP indicates about market demand, whether the IPO is oversubscribed or undersubscribed, investor confidence level
3. **Listing Gain Potential** - expected listing gains based on GMP, risk-reward assessment, comparison with similar IPOs if mentioned
4. **IPO Timeline & Status** - current IPO status (Open/Upcoming/Closed/Listed), important dates (opening, closing, listing), time-sensitive insights
5. **Investment Recommendation** - should investors apply, grey market trend (rising/falling), risk factors to consider
6. **Key Takeaways** - 3-5 bullet points summarizing the analysis with action items for potential investors

Format part 2 in clear, professional language suitable for investment decision-making. Use ₹ symbol for prices and % for percentages. Be specific with numbers where available. If the context doesn't contain sufficient GMP data, clearly state what information is missing in both parts."""


class IPOReviewAgent:
    """Main IPO Review Agent class with LLM-powered analysis."""

    def __init__(self, use_llm: bool = True, llm_provider: str = "openai", enable_reflection: bool = True):
        self.data_manager = DataSourceManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.business_analyzer = BusinessAnalyzer()

        # Initialize financial analyzer (enhanced if available and requested)
        if use_llm and ENHANCED_ANALYZER_AVAILABLE:
            try:
                self.financial_analyzer = EnhancedFinancialAnalyzer(llm_provider=llm_provider, enable_reflection=enable_reflection)
                self.enhanced_analysis = True
                logger.info("Enhanced LLM-powered financial analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced analyzer: {e}")
                self.financial_analyzer = FinancialAnalyzer()
                self.enhanced_analysis = False
        else:
            self.financial_analyzer = FinancialAnalyzer()
            self.enhanced_analysis = False

    def analyze_ipo(self, company_name: str, ipo_details: dict) -> IPOAnalysisReport:
        """Perform comprehensive IPO analysis."""
        logger.info(f"Starting IPO analysis for {company_name}")

        # For IPOs, we don't have stock symbols yet, so we collect limited data
        raw_data = self.data_manager.collect_ipo_data(company_name, ipo_details)
        logger.debug(f"Collected raw data keys: {list(raw_data.keys())}")

        # Create company basics from IPO information
        company = CompanyBasics(
            name=company_name,
            symbol="IPO-PENDING",  # Placeholder for IPO companies
            sector=ipo_details.get('sector', 'Unknown'),
            industry=ipo_details.get('sector', 'Unknown'),  # Use sector as industry for now
            market_cap=self._estimate_market_cap(ipo_details),
            employees=raw_data.get('employees'),
            website=raw_data.get('website'),
            description=raw_data.get('description')
        )

        # Financial analysis - use enhanced analysis if available
        if self.enhanced_analysis:
            logger.info(f"Raw data keys: {list(raw_data.keys())}")

            # Extract prospectus text from enhanced_prospectus data
            prospectus_text = raw_data.get('prospectus_text', '')

            # If no direct prospectus_text, try to extract from enhanced_prospectus
            if not prospectus_text and 'enhanced_prospectus' in raw_data:
                enhanced_data = raw_data['enhanced_prospectus']
                if isinstance(enhanced_data, dict):
                    # Try to extract text from various sections
                    sections = []
                    for key, value in enhanced_data.items():
                        if isinstance(value, str) and len(value) > 50:  # Only meaningful text
                            sections.append(f"{key.upper()}: {value}")
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, str) and len(subvalue) > 20:
                                    sections.append(f"{key.upper()}/{subkey}: {subvalue}")

                    if sections:
                        prospectus_text = "\n\n".join(sections)
                        logger.info("Extracted prospectus text from enhanced_prospectus data")

            # Also check prospectus_summary
            if not prospectus_text and 'prospectus_summary' in raw_data:
                prospectus_summary = raw_data['prospectus_summary']
                if isinstance(prospectus_summary, str) and len(prospectus_summary) > 100:
                    prospectus_text = prospectus_summary
                    logger.info("Using prospectus_summary as text source")

            logger.info(f"Prospectus text length: {len(prospectus_text)}")

            try:
                comprehensive_analysis = self.financial_analyzer.analyze_comprehensive(
                    raw_data, company_name, ipo_details.get('sector', 'Unknown')
                )

                enhanced_metrics = comprehensive_analysis.get('enhanced_metrics')
                traditional_metrics = comprehensive_analysis.get('traditional_metrics')

                if enhanced_metrics:
                    logger.info("Using enhanced LLM-powered financial metrics")
                    financial_metrics = enhanced_metrics
                elif traditional_metrics:
                    logger.info("Using traditional financial metrics (LLM analysis failed or not available)")
                    financial_metrics = traditional_metrics
                else:
                    logger.warning("No financial metrics available (both enhanced and traditional failed)")
                    financial_metrics = None

                # Store additional analysis results
                raw_data['llm_analysis'] = comprehensive_analysis.get('llm_analysis', {})
                raw_data['valuation_analysis'] = comprehensive_analysis.get('valuation_analysis', {})
                raw_data['peer_analysis'] = comprehensive_analysis.get('peer_analysis', {})
                raw_data['analysis_quality'] = comprehensive_analysis.get('analysis_quality', {})

                # Debug LLM analysis results
                logger.info(f"LLM analysis keys: {list(raw_data['llm_analysis'].keys()) if raw_data['llm_analysis'] else 'No LLM analysis'}")

                if raw_data['llm_analysis']:
                    llm_financial_metrics = raw_data['llm_analysis'].get('llm_financial_metrics')
                    logger.info(f"LLM Financial Metrics extracted: {llm_financial_metrics is not None}")
                    if llm_financial_metrics:
                        logger.info(f"LLM Financial Metrics type: {type(llm_financial_metrics)}")

            except Exception as e:
                logger.error(f"Enhanced analysis failed, falling back to traditional financial analysis: {e}")
                # Fallback to traditional analysis
                financial_metrics = self.financial_analyzer.calculate_financial_metrics(
                    raw_data.get('financial_statements', {})
                )
                # Add error info for debugging
                raw_data['llm_analysis'] = {'error': str(e)}
                raw_data['analysis_error'] = str(e)
        else:
            logger.info("Enhanced LLM analysis not enabled - using traditional financial analysis")
            financial_metrics = self.financial_analyzer.calculate_financial_metrics(
                raw_data.get('financial_statements', {})
            )

        # News sentiment analysis
        news_analysis = self.sentiment_analyzer.analyze_news_sentiment(
            raw_data.get('company_news', []) + raw_data.get('market_news', [])
        )

        # Risk assessment
        company_info = {
            'sector': ipo_details.get('sector', 'Unknown'),
            'market_cap': self._estimate_market_cap(ipo_details),
            'ipo_price_range': ipo_details.get('price_range', (100, 120)),
            'exchange': ipo_details.get('exchange', 'NSE')
        }

        risk_assessment = self.risk_analyzer.assess_risks(
            financial_metrics, raw_data, news_analysis, company_info
        )

        # Business analysis
        strengths_weaknesses = self.business_analyzer.analyze_business_fundamentals(
            company_info, financial_metrics, raw_data
        )

        # Generate predictions and recommendations
        listing_gain_prediction = self._predict_listing_gains(
            financial_metrics, news_analysis, risk_assessment
        )

        long_term_score = self._calculate_long_term_score(
            financial_metrics, risk_assessment, strengths_weaknesses
        )

        recommendation = self._generate_recommendation(
            listing_gain_prediction, long_term_score, risk_assessment
        )

        # Self-check pass: review the assembled report for internal
        # consistency (e.g. a Buy recommendation paired with Very High risk)
        # before it's finalized.
        reflection = raw_data.get('llm_analysis', {}).get('llm_reflection')
        reflection_issues = reflection.issues if reflection is not None else []
        consistency_check = self._self_check_report(
            company, financial_metrics, risk_assessment, recommendation,
            listing_gain_prediction, long_term_score, reflection_issues
        )
        raw_data['consistency_check'] = consistency_check

        # Log summary of financial metrics used
        if financial_metrics:
            logger.debug(
                "Financial metrics availability: "
                f"revenue_growth_rate={financial_metrics.revenue_growth_rate is not None}, "
                f"profit_margin={financial_metrics.profit_margin is not None}, "
                f"roe={financial_metrics.return_on_equity is not None}, "
                f"current_ratio={getattr(financial_metrics, 'current_ratio', None) is not None}, "
                f"debt_to_equity={getattr(financial_metrics, 'debt_to_equity', None) is not None}"
            )
        else:
            logger.warning("No financial metrics available!")

        logger.info(
            f"Final analysis results for {company_name}: "
            f"listing_gain_prediction={listing_gain_prediction:.1f}%, "
            f"long_term_score={long_term_score:.1f}/10, "
            f"recommendation={recommendation.value if recommendation else 'None'}"
        )

        # Create analysis report
        computed_confidence = consistency_check.get('confidence')
        analyst_confidence = computed_confidence if computed_confidence is not None else 0.75

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
            analyst_confidence=analyst_confidence  # from report self-check, falls back to 0.75
        )

        # Store raw_data with LLM analysis for display
        report.raw_data = raw_data

        return report

    def analyze_gmp(self, company_name: str, output_dir: str = "gmp_chunks") -> dict:
        """
        Analyze GMP data using Brave Search and Groq LLM.

        Args:
            company_name: Name of the company
            output_dir: Directory to save scraped chunks and the analysis file to

        Returns:
            Dictionary with GMP analysis results
        """
        if not GMP_EXTRACTOR_AVAILABLE:
            return {
                'status': 'error',
                'message': 'GMP extractor not available'
            }

        # Check API keys
        groq_key = os.getenv('GROQ_API_KEY')
        brave_key = os.getenv('BRAVE_API_KEY')

        if not groq_key or not brave_key:
            return {
                'status': 'error',
                'message': 'GROQ_API_KEY and BRAVE_API_KEY required'
            }

        try:
            # Initialize extractor
            extractor = LLMGMPExtractor(provider="groq", use_brave_search=True)
            groq_client = Groq(api_key=groq_key)

            # Search and scrape
            search_results = extractor.search_gmp_with_brave(company_name, max_results=5)

            if not search_results:
                return {
                    'status': 'not_found',
                    'message': 'No search results found'
                }

            # Scrape website content. Brave's top hits aren't always about
            # this company (e.g. a page that only name-drops it in a sidebar
            # list of unrelated IPOs), so walk further down the result list
            # until we have 3 pages that actually mention the company, rather
            # than blindly using search_results[:3].
            scraped_chunks = []
            relevant_sources = []
            for result in search_results[:5]:
                if len(scraped_chunks) >= 3:
                    break

                url = result.get('url')
                html_content = extractor.scrape_url_content(url)

                if html_content:
                    text_content = extractor.extract_text_from_html(html_content)

                    # Save scraped content
                    extractor.save_scraped_content(
                        company_name=company_name,
                        url=url,
                        html_content=html_content,
                        text_content=text_content,
                        folder=output_dir
                    )

                    if company_name.lower() not in text_content.lower():
                        logger.debug(f"Skipping {url}: does not mention {company_name}")
                        continue

                    # Keep the window of text around the company's mention
                    # instead of head-truncating, so the relevant row on a
                    # long aggregator page isn't dropped before analysis.
                    max_chunk_size = 5000
                    if len(text_content) > max_chunk_size:
                        text_content = extractor.extract_relevant_window(company_name, text_content, max_chunk_size)

                    scraped_chunks.append(f"Source: {url}\n{text_content}")
                    relevant_sources.append(url)

            if not scraped_chunks:
                return {
                    'status': 'not_found',
                    'message': 'No content could be scraped'
                }

            # Single Groq call produces both structured data and the
            # narrative analysis from the actual scraped page content (not
            # Brave's short search snippets, which almost never contain a
            # specific GMP figure for one company out of the hundreds listed
            # on an aggregator page). Previously this was two separate calls
            # each re-sending the same large context, roughly doubling token
            # usage against Groq's daily cap.
            combined_context = "\n\n".join(scraped_chunks[:5])
            analysis_prompt = GMP_ANALYSIS_PROMPT_TEMPLATE.format(
                company_name=company_name,
                combined_context=combined_context
            )

            response = groq_client.chat.completions.create(
                model=GMP_ANALYSIS_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": GMP_ANALYSIS_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=GMP_ANALYSIS_TEMPERATURE,
                max_tokens=GMP_ANALYSIS_MAX_TOKENS
            )

            response_text = response.choices[0].message.content
            result, analysis = self._parse_gmp_response(company_name, response_text)

            # Save analysis to file
            os.makedirs(output_dir, exist_ok=True)
            safe_name = re.sub(r'[^\w\s-]', '_', company_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = f"{output_dir}/{safe_name}_analysis_{timestamp}.txt"

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
                for i, url in enumerate(relevant_sources, 1):
                    f.write(f"  {i}. {url}\n")

            return {
                'status': 'success',
                'structured_data': result,
                'analysis': analysis,
                'sources': relevant_sources,
                'file_saved': analysis_file
            }

        except Exception as e:
            logger.error(f"GMP analysis error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _parse_gmp_response(self, company_name: str, response_text: str) -> tuple:
        """
        Split a combined GMP LLM response into (structured_data, analysis_text).

        Expects the STRUCTURED_DATA / ANALYSIS two-part format requested by
        GMP_ANALYSIS_PROMPT_TEMPLATE. Falls back to an empty structured dict
        and the full response as the analysis if parsing fails, so a
        malformed response degrades gracefully instead of raising.
        """
        structured_data: dict = {}
        analysis = response_text

        if "STRUCTURED_DATA:" in response_text and "ANALYSIS:" in response_text:
            json_part, _, analysis_part = response_text.partition("ANALYSIS:")
            json_part = json_part.split("STRUCTURED_DATA:", 1)[-1].strip()
            json_part = json_part.strip('`').strip()
            if json_part.lower().startswith("json"):
                json_part = json_part[4:].strip()

            try:
                structured_data = json.loads(json_part)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse structured GMP JSON for {company_name}")

            analysis = analysis_part.strip()

        # Derive fields the same way the extractor used to, in case the LLM
        # only filled in some of them.
        if structured_data.get('gmp_price') and structured_data.get('issue_price'):
            if not structured_data.get('expected_listing_price'):
                structured_data['expected_listing_price'] = (
                    structured_data['gmp_price'] + structured_data['issue_price']
                )
            if not structured_data.get('gmp_percentage'):
                structured_data['gmp_percentage'] = (
                    structured_data['gmp_price'] / structured_data['issue_price'] * 100
                )

        structured_data['company_name'] = company_name
        structured_data['estimated_listing_gain'] = structured_data.get('gmp_percentage')
        structured_data['status'] = 'success' if structured_data.get('gmp_price') is not None else 'not_found'

        return structured_data, analysis

    def _predict_listing_gains(self, financial_metrics, news_analysis, risk_assessment) -> float:
        """Predict potential listing gains percentage."""
        base_gain = 10.0  # Base expected gain
        logger.debug(f"Calculating listing gains prediction with base gain: {base_gain}%")

        # Adjust based on financial performance
        if financial_metrics.revenue_growth_rate:
            logger.debug(f"Using actual revenue growth rate: {financial_metrics.revenue_growth_rate:.1%}")
            if financial_metrics.revenue_growth_rate > 0.2:
                base_gain += 15
                logger.debug(f"High growth bonus: +15% (new total: {base_gain}%)")
            elif financial_metrics.revenue_growth_rate < 0:
                base_gain -= 20
                logger.debug(f"Negative growth penalty: -20% (new total: {base_gain}%)")
        else:
            logger.debug("Revenue growth rate not available - using default base gain only")

        # Adjust based on sentiment
        sentiment_multiplier = 1 + (news_analysis.sentiment_score * 0.3)
        base_gain *= sentiment_multiplier

        # Adjust based on risk
        if risk_assessment.overall_risk.value == "High":
            base_gain *= 0.7
        elif risk_assessment.overall_risk.value == "Low":
            base_gain *= 1.3

        return max(-50, min(100, base_gain))  # Cap between -50% and 100%

    def _calculate_long_term_score(self, financial_metrics, risk_assessment, strengths_weaknesses) -> float:
        """Calculate long-term investment score (0-10)."""
        score = 5.0  # Base score
        logger.debug(f"Calculating long-term score with base score: {score}/10")

        # Financial factors.
        # Thresholds recalibrated for realistic Indian SME/mid-cap IPO issuers
        # (manufacturing/textile-type margins and growth), not just companies
        # exceptional enough to clear a 15%+ margin or 20%+ growth bar - those
        # bars meant most real, moderately-healthy issuers (e.g. a 6-8% margin,
        # 12-15% growth) got zero credit here, which is most of why this score
        # kept landing on the same base+risk-bonus total regardless of how the
        # company actually performed.
        if financial_metrics.profit_margin and financial_metrics.profit_margin > 0.08:
            score += 1.5
            logger.debug(f"High profit margin bonus: +1.5 (new score: {score:.1f}/10)")
        elif financial_metrics.profit_margin and financial_metrics.profit_margin > 0.03:
            score += 0.75
            logger.debug(f"Moderate profit margin bonus: +0.75 (new score: {score:.1f}/10)")
        elif financial_metrics.profit_margin and financial_metrics.profit_margin < 0:
            score -= 2
            logger.debug(f"Negative profit margin penalty: -2.0 (new score: {score:.1f}/10)")
        elif not financial_metrics.profit_margin:
            logger.debug("Profit margin not available - no adjustment made")

        if financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate > 0.12:
            score += 1.5
            logger.debug(f"Strong revenue growth bonus: +1.5 (new score: {score:.1f}/10)")
        elif financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate > 0.05:
            score += 0.75
            logger.debug(f"Moderate revenue growth bonus: +0.75 (new score: {score:.1f}/10)")
        elif not financial_metrics.revenue_growth_rate:
            logger.debug("Revenue growth rate not available - no growth bonus applied")

        # Risk factors
        if risk_assessment.overall_risk.value == "Low":
            score += 1
        elif risk_assessment.overall_risk.value == "High":
            score -= 1.5
        elif risk_assessment.overall_risk.value == "Very High":
            score -= 2.5

        # Business strengths
        strength_bonus = min(len(strengths_weaknesses.strengths) * 0.3, 1.5)
        weakness_penalty = min(len(strengths_weaknesses.weaknesses) * 0.2, 1.0)
        score += strength_bonus - weakness_penalty

        return max(0, min(10, score))

    def _estimate_market_cap(self, ipo_details: dict) -> float:
        """Estimate market cap based on IPO details."""
        price_range = ipo_details.get('price_range', (100, 120))

        # Handle string format like "100-120" or tuple format
        if isinstance(price_range, str):
            try:
                # Parse string format "100-120"
                if '-' in price_range:
                    min_price, max_price = map(float, price_range.split('-'))
                    price_range = (min_price, max_price)
                else:
                    # Single price
                    price = float(price_range)
                    price_range = (price, price)
            except (ValueError, AttributeError):
                price_range = (100, 120)  # Default fallback

        avg_price = (price_range[0] + price_range[1]) / 2

        # Rough estimation based on typical IPO sizes in India
        # This would ideally come from IPO prospectus data
        estimated_shares = 10000000  # 1 crore shares (typical range)

        return avg_price * estimated_shares

    def _generate_recommendation(self, listing_gain, long_term_score, risk_assessment) -> InvestmentRecommendation:
        """Generate investment recommendation."""
        if long_term_score >= 8 and listing_gain > 20:
            return InvestmentRecommendation.STRONG_BUY
        elif long_term_score >= 6.5 and listing_gain > 10:
            return InvestmentRecommendation.BUY
        elif long_term_score >= 4 and risk_assessment.overall_risk.value != "Very High":
            return InvestmentRecommendation.HOLD
        else:
            return InvestmentRecommendation.AVOID

    def _self_check_report(self, company, financial_metrics, risk_assessment,
                            recommendation, listing_gain_prediction, long_term_score,
                            reflection_issues=None) -> dict:
        """
        One LLM pass reviewing the fully assembled report for internal
        consistency (e.g. a Buy/Strong Buy recommendation paired with Very
        High risk, or a positive listing-gain prediction alongside negative
        revenue growth/profit margin) before it's shown to an investor.

        Reuses the LLM client already configured on the enhanced financial
        analyzer instead of creating a new one. Any failure - no LLM
        analyzer available, empty response, unparseable JSON, exception -
        degrades to a no-op result so the pipeline never breaks and
        analyst_confidence falls back to its previous hardcoded default.
        """
        default = {'ran': False, 'consistent': True, 'issues': [], 'confidence': None}

        llm_analyzer = getattr(self.financial_analyzer, 'llm_analyzer', None)
        if not (self.enhanced_analysis and llm_analyzer):
            return default

        try:
            summary = {
                'company': company.name,
                'sector': company.sector,
                'recommendation': recommendation.value if recommendation else None,
                'listing_gain_prediction_pct': listing_gain_prediction,
                'long_term_score_of_10': long_term_score,
                'overall_risk': risk_assessment.overall_risk.value if risk_assessment else None,
                'risk_factors': risk_assessment.risk_factors if risk_assessment else [],
                'revenue_growth_rate': getattr(financial_metrics, 'revenue_growth_rate', None),
                'profit_margin': getattr(financial_metrics, 'profit_margin', None),
                'debt_to_equity': getattr(financial_metrics, 'debt_to_equity', None),
                'reflection_issues_from_extraction': reflection_issues or [],
            }
            prompt = (
                "You are auditing a finished IPO analysis report for internal "
                "consistency BEFORE it is shown to an investor. Report summary "
                f"(JSON): {json.dumps(summary, default=str)}\n\n"
                "Flag any INTERNAL CONTRADICTION between these fields - for example: "
                "a Buy/Strong Buy recommendation paired with Very High overall risk; "
                "a large positive listing_gain_prediction_pct alongside negative "
                "revenue_growth_rate or negative profit_margin; a high long_term_score "
                "alongside multiple severe risk_factors; or any of the "
                "reflection_issues_from_extraction that would materially change the "
                "recommendation if true. Do NOT re-derive the numbers yourself, only "
                "check whether the fields already here are mutually consistent.\n\n"
                "Respond with ONLY valid JSON, no markdown fences: "
                '{"consistent": true/false, "issues": ["..."], "confidence": 0.0-1.0}'
            )

            response = llm_analyzer._call_llm(prompt, max_tokens=600, temperature=0.0)
            if not response:
                return default

            data = llm_analyzer._parse_json_with_fallbacks(response, "report self-check")
            if not data:
                return default

            return {
                'ran': True,
                'consistent': data.get('consistent', True),
                'issues': data.get('issues', []),
                'confidence': data.get('confidence'),
            }
        except Exception as e:
            logger.warning(f"Report self-check failed, skipping: {e}")
            return default
