"""
IPO Review Agent - Main Application
Streamlit web interface for IPO analysis and investment recommendations.
Enhanced with SEBI Draft Offer Documents search functionality.
"""

import streamlit as st
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger
import requests
import time
import re

# Conditional imports for SEBI functionality
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available - SEBI search functionality will be limited")

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_sources import DataSourceManager
from src.analyzers import FinancialAnalyzer, SentimentAnalyzer, RiskAnalyzer, BusinessAnalyzer
from src.models import IPOAnalysisReport, CompanyBasics, InvestmentRecommendation

# Import enhanced analyzer if available
try:
    from src.analyzers import EnhancedFinancialAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYZER_AVAILABLE = False


class IPOReviewAgent:
    """Main IPO Review Agent class with LLM-powered analysis."""
    
    def __init__(self, use_llm: bool = True, llm_provider: str = "openai"):
        self.data_manager = DataSourceManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.business_analyzer = BusinessAnalyzer()
        
        # Initialize financial analyzer (enhanced if available and requested)
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
    
    def analyze_ipo(self, company_name: str, ipo_details: dict) -> IPOAnalysisReport:
        """Perform comprehensive IPO analysis."""
        logger.info(f"Starting IPO analysis for {company_name}")
        
        # For IPOs, we don't have stock symbols yet, so we collect limited data
        raw_data = self.data_manager.collect_ipo_data(company_name, ipo_details)
        print("Collected raw data", raw_data)
        print("Collected raw data keys", raw_data.keys())
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
            # Debug: Check what data we have
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
                        logger.info(f"Extracted prospectus text from enhanced_prospectus data")
            
            # Also check prospectus_summary
            if not prospectus_text and 'prospectus_summary' in raw_data:
                prospectus_summary = raw_data['prospectus_summary']
                if isinstance(prospectus_summary, str) and len(prospectus_summary) > 100:
                    prospectus_text = prospectus_summary
                    logger.info(f"Using prospectus_summary as text source")
            
            logger.info(f"Prospectus text length: {len(prospectus_text)}")
            
            # For demo purposes, if no real prospectus text is found, use sample text
            # if not prospectus_text:
            #     logger.info("No prospectus text found, creating realistic sample data for LLM demo")
            #     print("prospectus not found")
            #     # Create sector-specific financial data
            #     sector_data = {
            #         'Technology': {
            #             'revenue_growth': '35%', 'profit_margin': '15%', 'roe': '22%', 
            #             'pe_ratio': '28x', 'current_ratio': '2.1', 'debt_equity': '0.2'
            #         },
            #         'Healthcare': {
            #             'revenue_growth': '18%', 'profit_margin': '12%', 'roe': '16%',
            #             'pe_ratio': '24x', 'current_ratio': '1.8', 'debt_equity': '0.3'
            #         },
            #         'Financial Services': {
            #             'revenue_growth': '15%', 'profit_margin': '20%', 'roe': '14%',
            #             'pe_ratio': '12x', 'current_ratio': '1.2', 'debt_equity': '3.5'
            #         },
            #         'Manufacturing': {
            #             'revenue_growth': '12%', 'profit_margin': '8%', 'roe': '13%',
            #             'pe_ratio': '16x', 'current_ratio': '1.5', 'debt_equity': '0.6'
            #         }
            #     }
                
            #     sector = ipo_details.get('sector', 'Technology')
            #     financial_data = sector_data.get(sector, sector_data['Technology'])
                
            #     raw_data['prospectus_text'] = f"""
            #     DRAFT RED HERRING PROSPECTUS
                
            #     {company_name}
            #     (A Company incorporated under the Companies Act, 2013)
                
            #     COMPANY OVERVIEW:
            #     {company_name} is a leading company in the {sector} sector with strong market presence and growth potential. The company has demonstrated consistent financial performance and strategic market positioning.
                
            #     KEY FINANCIAL METRICS (Restated Financials):
                
            #     Revenue Performance:
            #     - FY 2023: ₹1,250 Crores (Growth: {financial_data['revenue_growth']})
            #     - FY 2022: ₹1,050 Crores  
            #     - FY 2021: ₹890 Crores
            #     - 3-Year CAGR: {financial_data['revenue_growth']}
                
            #     Profitability Metrics:
            #     - Net Profit Margin: {financial_data['profit_margin']}
            #     - EBITDA Margin: {float(financial_data['profit_margin'].strip('%')) + 5}%
            #     - Return on Equity (ROE): {financial_data['roe']}
            #     - Return on Assets (ROA): {float(financial_data['roe'].strip('%')) - 3}%
            #     - Return on Invested Capital (ROIC): {float(financial_data['roe'].strip('%')) - 1}%
                
            #     Valuation Ratios:
            #     - Price-to-Earnings (P/E) Ratio: {financial_data['pe_ratio']}
            #     - Price-to-Book (P/B) Ratio: {float(financial_data['pe_ratio'].strip('x')) / 10:.1f}x
            #     - Enterprise Value/EBITDA: {float(financial_data['pe_ratio'].strip('x')) - 4}x
            #     - Price-to-Sales Ratio: {float(financial_data['pe_ratio'].strip('x')) / 15:.1f}x
                
            #     Financial Health Ratios:
            #     - Current Ratio: {financial_data['current_ratio']}
            #     - Quick Ratio: {float(financial_data['current_ratio']) - 0.3:.1f}
            #     - Debt-to-Equity Ratio: {financial_data['debt_equity']}
            #     - Interest Coverage Ratio: {15 if sector == 'Technology' else 8}x
                
            #     BUSINESS MODEL & COMPETITIVE POSITIONING:
            #     The company operates with a scalable and sustainable business model in the {sector.lower()} space. Key competitive advantages include:
            #     - Market leadership with {25 if sector == 'Technology' else 15}% market share
            #     - Strong brand recognition and customer loyalty
            #     - Advanced technology platform and R&D capabilities
            #     - Experienced management team with proven track record
                
            #     PEER COMPARISON:
            #     The company's financial metrics compare favorably to industry peers:
            #     - Revenue growth above industry average of {float(financial_data['revenue_growth'].strip('%')) - 5}%
            #     - ROE higher than sector median of {float(financial_data['roe'].strip('%')) - 3}%
            #     - Debt levels lower than industry average
                
            #     USE OF IPO PROCEEDS:
            #     Total Issue Size: ₹800 Crores
            #     - Capacity Expansion & Capex: 40% (₹320 Cr)
            #     - Working Capital Requirements: 25% (₹200 Cr)  
            #     - Debt Repayment: 20% (₹160 Cr)
            #     - General Corporate Purposes: 15% (₹120 Cr)
                
            #     KEY RISK FACTORS:
            #     - Intense market competition and pricing pressure
            #     - Regulatory changes in the {sector.lower()} sector
            #     - Economic slowdown affecting demand
            #     - Raw material price volatility
            #     - Technology disruption risks
                
            #     MANAGEMENT ASSESSMENT:
            #     - Strong promoter background with {20 if sector == 'Technology' else 25} years experience
            #     - Professional management team with relevant expertise
            #     - Good corporate governance practices
            #     - Track record of consistent performance
                
            #     GROWTH STRATEGY:
            #     - Market expansion to Tier-2 and Tier-3 cities
            #     - New product development and innovation
            #     - Strategic partnerships and alliances  
            #     - Technology upgrades and digitalization
            #     """
            
            try:
                comprehensive_analysis = self.financial_analyzer.analyze_comprehensive(
                    raw_data, company_name, ipo_details.get('sector', 'Unknown')
                )
                
                enhanced_metrics = comprehensive_analysis.get('enhanced_metrics')
                traditional_metrics = comprehensive_analysis.get('traditional_metrics')
                
                if enhanced_metrics:
                    print("✅ Using enhanced LLM-powered financial metrics")
                    financial_metrics = enhanced_metrics
                elif traditional_metrics:
                    print("⚠️ Using traditional financial metrics (LLM analysis failed or not available)")
                    financial_metrics = traditional_metrics
                else:
                    print("❌ No financial metrics available (both enhanced and traditional failed)")
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
                logger.error(f"Enhanced analysis failed: {e}")
                print(f"❌ Enhanced LLM analysis failed: {e}")
                print("⚠️ Falling back to traditional financial analysis")
                # Fallback to traditional analysis
                financial_metrics = self.financial_analyzer.calculate_financial_metrics(
                    raw_data.get('financial_statements', {})
                )
                # Add error info for debugging
                raw_data['llm_analysis'] = {'error': str(e)}
                raw_data['analysis_error'] = str(e)
        else:
            print("⚠️ Enhanced LLM analysis not enabled - using traditional financial analysis")
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
        
        # Print summary of financial metrics used
        if financial_metrics:
            print("📋 Financial Metrics Summary:")
            print(f"   Revenue Growth Rate: {'✅ Available' if financial_metrics.revenue_growth_rate is not None else '❌ Not Available'}")
            print(f"   Profit Margin: {'✅ Available' if financial_metrics.profit_margin is not None else '❌ Not Available'}")
            print(f"   ROE: {'✅ Available' if hasattr(financial_metrics, 'roe') and financial_metrics.roe is not None else '❌ Not Available'}")
            print(f"   Current Ratio: {'✅ Available' if hasattr(financial_metrics, 'current_ratio') and financial_metrics.current_ratio is not None else '❌ Not Available'}")
            print(f"   Debt to Equity: {'✅ Available' if hasattr(financial_metrics, 'debt_to_equity') and financial_metrics.debt_to_equity is not None else '❌ Not Available'}")
        else:
            print("❌ No financial metrics available!")
        
        print(f"📊 Final Analysis Results:")
        print(f"   Listing Gain Prediction: {listing_gain_prediction:.1f}%")
        print(f"   Long-term Score: {long_term_score:.1f}/10")
        print(f"   Recommendation: {recommendation.value if recommendation else 'None'}")
        
        # Create analysis report
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
            analyst_confidence=0.75  # Default confidence
        )
        
        # Store raw_data with LLM analysis for display
        report.raw_data = raw_data
        
        return report
    
    def _predict_listing_gains(self, financial_metrics, news_analysis, risk_assessment) -> float:
        """Predict potential listing gains percentage."""
        base_gain = 10.0  # Base expected gain
        print(f"📊 Calculating listing gains prediction with base gain: {base_gain}%")
        
        # Adjust based on financial performance
        if financial_metrics.revenue_growth_rate:
            print(f"✅ Using actual revenue growth rate: {financial_metrics.revenue_growth_rate:.1%}")
            if financial_metrics.revenue_growth_rate > 0.2:
                base_gain += 15
                print(f"📈 High growth bonus: +15% (new total: {base_gain}%)")
            elif financial_metrics.revenue_growth_rate < 0:
                base_gain -= 20
                print(f"📉 Negative growth penalty: -20% (new total: {base_gain}%)")
        else:
            print("⚠️ Revenue growth rate not available - using default base gain only")
        
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
        print(f"📊 Calculating long-term score with base score: {score}/10")
        
        # Financial factors
        if financial_metrics.profit_margin and financial_metrics.profit_margin > 0.1:
            score += 1.5
            print(f"✅ High profit margin bonus: +1.5 (new score: {score:.1f}/10)")
        elif financial_metrics.profit_margin and financial_metrics.profit_margin < 0:
            score -= 2
            print(f"❌ Negative profit margin penalty: -2.0 (new score: {score:.1f}/10)")
        elif not financial_metrics.profit_margin:
            print("⚠️ Profit margin not available - no adjustment made")
        
        if financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate > 0.15:
            score += 1.5
            print(f"✅ Strong revenue growth bonus: +1.5 (new score: {score:.1f}/10)")
        elif not financial_metrics.revenue_growth_rate:
            print("⚠️ Revenue growth rate not available - no growth bonus applied")
        
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="IPO Review Agent",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🚀 Indian IPO Review Agent")
    st.markdown("### 🇮🇳 Comprehensive IPO Analysis for Indian Stock Market")
    st.markdown("🎯 **Specialized for Pre-IPO Analysis using SEBI Draft Offer Documents**")
    
    # Create main navigation tabs
    tab1, tab2, tab3 = st.tabs(["🔍 IPO Analysis", "📄 SEBI Document Search", "ℹ️ About"])
    
    with tab1:
        ipo_analysis_tab()
    
    with tab2:
        sebi_document_search_tab()
    
    with tab3:
        about_tab()

def ipo_analysis_tab():
    """Main IPO analysis functionality."""
    st.header("🔍 IPO Analysis")
    
    # Check if a company was selected from SEBI search
    if 'sebi_company' in st.session_state:
        st.success(f"✅ **Selected from SEBI search:** {st.session_state.sebi_company}")
        if 'sebi_date' in st.session_state:
            st.info(f"📅 **Filing Date:** {st.session_state.sebi_date}")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Analysis Parameters")
        
        # Check for API keys
        st.subheader("🔑 API Configuration")
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        news_key = os.getenv('NEWS_API_KEY')
        
        if not any([alpha_key, news_key]):
            st.error("⚠️ No API keys configured. Please set up your .env file.")
            st.info("Copy .env.example to .env and add your API keys.")
        else:
            st.success("✅ API keys configured")
        
        # LLM Configuration
        st.subheader("🤖 LLM Configuration")
        
        # Check for LLM API keys
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        groq_key = os.getenv('GROQ_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        available_providers = []
        if openai_key and not openai_key.startswith('your_'):
            available_providers.append("OpenAI GPT-4")
        if anthropic_key and not anthropic_key.startswith('your_'):
            available_providers.append("Anthropic Claude")
        if groq_key and not groq_key.startswith('your_'):
            available_providers.append("Groq Mixtral")
        if gemini_key and not gemini_key.startswith('your_'):
            available_providers.append("Google Gemini")
        
        if available_providers:
            st.success(f"✅ Available LLM providers: {len(available_providers)}")
            
            # LLM provider selection
            provider_map = {
                "OpenAI GPT-4": "openai",
                "Anthropic Claude": "anthropic", 
                "Groq Mixtral": "groq",
                "Google Gemini": "gemini"
            }
            
            selected_provider_name = st.selectbox(
                "Select LLM Provider",
                available_providers,
                help="Choose the LLM provider for enhanced prospectus analysis"
            )
            selected_llm_provider = provider_map[selected_provider_name]
            
            use_llm_analysis = st.checkbox(
                "🚀 Enable Enhanced LLM Analysis", 
                value=True,
                help="Use AI to extract P/E ratios, peer analysis, and generate investment thesis"
            )
        else:
            st.warning("⚠️ No LLM API keys configured - basic analysis only")
            st.info("Add LLM API keys to .env for enhanced analysis")
            selected_llm_provider = "openai"
            use_llm_analysis = False
        
        # Company input - simplified to just company name
        st.subheader("🏢 Company Information")
        company_name = st.text_input(
            "IPO/Company Name", 
            placeholder="e.g., Zomato Limited, Paytm, Vidya Wires Limited",
            help="Enter the company name for comprehensive IPO analysis"
        )
        
        analyze_button = st.button("🔍 Analyze IPO", type="primary", disabled=not company_name)
    
    # Main content area
    if analyze_button and company_name:
        with st.spinner(f"🔄 Analyzing {company_name}..."):
            try:
                # Initialize agent with LLM configuration
                agent = IPOReviewAgent(use_llm=use_llm_analysis, llm_provider=selected_llm_provider)
                
                # Create IPO details dictionary - simplified for name-only analysis
                ipo_details = {
                    'company_name': company_name
                }
                
                # Perform analysis
                report = agent.analyze_ipo(company_name, ipo_details)
                
                # Display results
                display_analysis_report(report)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                # Add more detailed error information
                st.error("**Debug Information:**")
                st.error(f"- Error type: {type(e).__name__}")
                st.error(f"- Error message: {str(e)}")
                import traceback
                st.error(f"- Full traceback: {traceback.format_exc()}")
    else:
        # Welcome screen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            ### 🇮🇳 Indian IPO Analysis
            - Company fundamentals review
            - Industry benchmarking (Indian market)
            - Sectoral growth analysis
            - Financial health assessment
            """)
        
        with col2:
            st.info("""
            ### 📊 Listing Predictions
            - Listing day gains prediction
            - Long-term investment scoring
            - Price band analysis
            - Market cap estimation
            """)
        
        with col3:
            st.info("""
            ### 📰 Market Intelligence
            - News sentiment (Indian markets)
            - NSE/BSE market trends
            - Recent IPO performance
            - Sector-specific insights
            """)
        
        # Add information about Indian IPO market
        st.markdown("---")
        st.markdown("""
        ### 🎯 **Specialized for Indian Stock Market**
        
        This tool is designed specifically for analyzing **Indian IPOs** listing on **NSE** and **BSE**. 
        
        **Key Features:**
        - 📈 Analysis without stock symbols (for pre-listing companies)
        - 🏛️ Indian market benchmarks and sectoral analysis  
        - 💹 Historical performance of recent Indian IPOs (Zomato, Paytm, Nykaa, etc.)
        - 📰 India-focused market sentiment and news analysis
        - ₹ INR-based pricing and market cap calculations
        
        **How to use:** Enter the company name, IPO price range, and sector to get comprehensive investment insights!
        """)
    
def about_tab():
    """About tab with application information."""
    st.header("ℹ️ About IPO Review Agent")
    
    st.markdown("""
    ## 🎯 Purpose
    The IPO Review Agent is an AI-powered tool designed specifically for analyzing **Pre-IPO companies** 
    in the Indian stock market using official **SEBI Draft Offer Documents**.
    
    ## 🔍 Key Features
    - **SEBI Integration**: Direct search in SEBI Draft Offer Documents
    - **Pre-IPO Analysis**: Analyze companies before they go public
    - **Real Financial Data**: Extract data from official DRHP documents
    - **Risk Assessment**: Comprehensive risk analysis
    - **Investment Recommendations**: Data-driven investment advice
    
    ## 📊 How It Works
    1. **Search SEBI**: Find companies in Draft Offer Documents section
    2. **Extract Data**: Download and parse DRHP documents
    3. **Analyze Financials**: Calculate ratios, growth rates, and metrics
    4. **Assess Risk**: Evaluate business, financial, and market risks
    5. **Generate Report**: Provide investment recommendation
    
    ## 🎯 Success Story: Vidya Wires Limited
    - **Found**: Jan 16, 2025 filing in SEBI Draft Documents
    - **Status**: Pre-IPO stage (perfect for analysis)
    - **Data**: Real DRHP with complete financial statements
    
    ## 🔧 Technical Stack
    - **Data Sources**: SEBI, Alpha Vantage, NewsAPI
    - **Analysis**: Python, pandas, financial ratios
    - **AI/ML**: Sentiment analysis, risk scoring
    - **Interface**: Streamlit web application
    
    ## ⚡ Getting Started
    1. Go to **SEBI Document Search** tab
    2. Search for a company (e.g., "Vidya Wires")
    3. Download the DRHP document
    4. Run the IPO analysis
    5. Get investment recommendation
    
    ## 🔑 API Configuration
    For full functionality, configure these API keys in `.env`:
    
    **Basic Financial Data:**
    - `ALPHA_VANTAGE_API_KEY` - Financial data
    - `NEWS_API_KEY` - News sentiment analysis
    - `GEMINI_API_KEY` or `OPENAI_API_KEY` - AI-powered analysis (optional)
    
    **Enhanced LLM Analysis (choose one or more):**
    - `OPENAI_API_KEY` - OpenAI GPT-4 models
    - `ANTHROPIC_API_KEY` - Anthropic Claude models
    - `GROQ_API_KEY` - Groq Mixtral models  
    - `GEMINI_API_KEY` - Google Gemini models
    
    ## 📝 Disclaimer
    This tool is for educational and research purposes. Investment decisions should 
    always be made with professional financial advice and thorough due diligence.
    """)

def sebi_document_search_tab():
    """Tab for searching SEBI Draft Offer Documents."""
    st.header("📄 SEBI Draft Offer Documents Search")
    st.markdown("**Search for pre-IPO companies in SEBI Draft Offer Documents**")
    
    # Key insight about data sources
    st.success("""
    🎯 **Correct Data Source Discovery!**
    
    **Previous approach:** ❌ Issues & Listing (completed IPOs, 2004-2016)
    **Correct approach:** ✅ Draft Offer Documents (pre-IPO companies, current filings)
    """)
    
    st.info("""
    🔍 **Why SEBI Draft Documents?**
    - Find companies in **pre-IPO stage** (DRHP - Draft Red Herring Prospectus)
    - Access **real financial data** from official SEBI filings
    - Perfect timing for **investment analysis** before public launch
    - **Current data**: 2024-2025 filings instead of historical data
    - **Success example**: Vidya Wires Limited (Jan 16, 2025 filing)
    """)
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input(
            "🔍 Search Company Name", 
            placeholder="e.g., Vidya Wires, Company Name, Keywords...",
            help="Search in SEBI Draft Offer Documents filed with SEBI"
        )
    
    with col2:
        search_button = st.button("🔍 Search SEBI", type="primary")
    
    if search_button and search_term:
        search_sebi_documents(search_term)
    
    # Show recent successful searches
    st.markdown("---")
    st.subheader("✅ Recently Found Companies")
    
    # Example successful searches
    example_companies = [
        {
            'name': 'Vidya Wires Limited',
            'filing_date': 'Jan 16, 2025',
            'document_type': 'DRHP',
            'status': 'Available for Analysis'
        }
        # Add more as they're discovered
    ]
    
    for company in example_companies:
        with st.expander(f"📄 {company['name']} - {company['filing_date']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Filing Date:** {company['filing_date']}")
            with col2:
                st.write(f"**Document Type:** {company['document_type']}")
            with col3:
                st.write(f"**Status:** {company['status']}")
            
            if st.button(f"📊 Analyze {company['name']}", key=f"analyze_{company['name']}"):
                st.session_state.sebi_company = company['name']
                st.session_state.sebi_date = company['filing_date']
                st.success(f"✅ {company['name']} selected for analysis!")
                st.rerun()


def search_sebi_documents(search_term):
    """Search SEBI Draft Offer Documents for the given term."""
    st.info(f"🔍 Searching SEBI for: **{search_term}**")
    
    if not SELENIUM_AVAILABLE:
        st.error("❌ Selenium not available - cannot perform automated search")
        st.info("""
        🔧 **Manual Alternative:**
        1. Visit: https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10
        2. Use the "Search by Title, Keywords, Entity Name" field
        3. Search for your company name
        4. Look for DRHP documents
        """)
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("⏳ Initializing browser...")
        progress_bar.progress(20)
        
        # Setup Chrome options
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Initialize driver
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=options
            )
        except Exception as e:
            st.error(f"❌ Could not initialize Chrome driver: {str(e)}")
            st.info("Please ensure Chrome is installed and ChromeDriver is available")
            return
        
        try:
            status_text.text("🌐 Loading SEBI page...")
            progress_bar.progress(40)
            
            # Navigate to SEBI Draft Offer Documents
            url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
            driver.get(url)
            time.sleep(3)
            
            status_text.text("🔍 Performing search...")
            progress_bar.progress(60)
            
            # Find and use search box
            search_box = driver.find_element(By.CSS_SELECTOR, "input[placeholder*='Title, Keywords, Entity Name']")
            search_box.clear()
            search_box.send_keys(search_term)
            search_box.send_keys(Keys.RETURN)
            time.sleep(3)
            
            status_text.text("📊 Analyzing results...")
            progress_bar.progress(80)
            
            # Parse results
            results = []
            tables = driver.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    row_text = row.text.strip()
                    if search_term.lower() in row_text.lower() and len(row_text) > 10:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 2:
                            cell_texts = [cell.text.strip() for cell in cells]
                            
                            # Look for document links
                            links = row.find_elements(By.TAG_NAME, "a")
                            document_link = None
                            if links:
                                document_link = links[0].get_attribute('href')
                            
                            results.append({
                                'date': cell_texts[0] if cell_texts else '',
                                'company': cell_texts[1] if len(cell_texts) > 1 else row_text,
                                'full_text': row_text,
                                'link': document_link
                            })
            
            progress_bar.progress(100)
            status_text.text("✅ Search completed!")
            
            # Display results
            if results:
                st.success(f"🎉 Found {len(results)} results for '{search_term}'!")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"📄 Result {i}: {result['company']}"):
                        st.write(f"**Date:** {result['date']}")
                        st.write(f"**Full Details:** {result['full_text']}")
                        
                        if result['link']:
                            st.write(f"**Document Link:** {result['link']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"📥 Download DRHP", key=f"download_{i}"):
                                    download_sebi_document(result['link'], result['company'])
                            
                            with col2:
                                if st.button(f"📊 Analyze Company", key=f"analyze_sebi_{i}"):
                                    # Set up for analysis
                                    st.session_state.sebi_company = result['company']
                                    st.session_state.sebi_date = result['date']
                                    st.success(f"✅ {result['company']} selected for analysis!")
            else:
                st.warning(f"❌ No results found for '{search_term}' in SEBI Draft Offer Documents")
                st.info("""
                💡 **Tips for better search:**
                - Try partial company names (e.g., "Vidya" instead of "Vidya Wires Limited")
                - Use different keywords related to the business
                - Check if the company has filed under a different legal name
                """)
        
        finally:
            driver.quit()
    
    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        st.info("🔧 **Manual Alternative:** Visit the SEBI website directly and search manually")

def download_sebi_document(filing_url, company_name):
    """Download SEBI document from filing URL."""
    try:
        st.info(f"📥 Downloading document for {company_name}...")
        
        # Extract PDF URL from filing page
        response = requests.get(filing_url)
        
        if "sebi_data/attachdocs" in response.text:
            pdf_pattern = r'https://www\.sebi\.gov\.in/sebi_data/attachdocs/[^"]*\.pdf'
            pdf_urls = re.findall(pdf_pattern, response.text)
            
            if pdf_urls:
                pdf_url = pdf_urls[0]
                pdf_response = requests.get(pdf_url, stream=True)
                
                if pdf_response.status_code == 200:
                    filename = f"{company_name.replace(' ', '_')}_DRHP.pdf"
                    
                    st.download_button(
                        label=f"💾 Download {filename}",
                        data=pdf_response.content,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success("✅ Document ready for download!")
                else:
                    st.error(f"❌ Failed to download PDF: HTTP {pdf_response.status_code}")
            else:
                st.error("❌ Could not find PDF URL in filing page")
        else:
            st.error("❌ Could not extract document from filing page")
    
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")


def display_analysis_report(report: IPOAnalysisReport):
    """Display the enhanced analysis report with LLM insights."""
    
    # Header with company info
    st.header(f"📊 Enhanced Analysis Report: {report.company.name}")
    
    # Check if enhanced LLM analysis is available
    if hasattr(report, 'raw_data') and 'llm_analysis' in getattr(report, 'raw_data', {}):
        llm_analysis = report.raw_data['llm_analysis']
        st.success(f"🤖 **Enhanced with LLM-Powered Prospectus Analysis** (Provider: {llm_analysis.get('llm_provider', 'unknown').title()})")
        
        # Show LLM Investment Thesis
        investment_thesis = llm_analysis.get('llm_investment_thesis', '')
        if investment_thesis:
            with st.expander("🎯 AI-Generated Investment Thesis", expanded=True):
                st.markdown(investment_thesis)
        
        # Display LLM Financial Metrics
        llm_financial_metrics = llm_analysis.get('llm_financial_metrics')
        
        # Debug information (can be removed in production)
        # with st.expander("🔧 Debug Information", expanded=False):
        #     st.write(f"LLM Financial Metrics type: {type(llm_financial_metrics)}")
        #     st.write(f"LLM Analysis components: {list(llm_analysis.keys())}")
        #     if hasattr(report, 'raw_data') and 'analysis_error' in report.raw_data:
        #         st.error(f"Analysis Error: {report.raw_data['analysis_error']}")
        
        if llm_financial_metrics:
            display_llm_financial_metrics(llm_financial_metrics)
        else:
            st.warning("⚠️ No LLM financial metrics extracted.")
            
            # Check for specific error information
            if 'error' in llm_analysis:
                st.error(f"LLM Analysis Error: {llm_analysis['error']}")
            
            # Show available analysis components
            available_components = [k for k in llm_analysis.keys() if llm_analysis[k]]
            if available_components:
                st.info(f"✅ Available LLM Analysis Components: {', '.join(available_components)}")
            
            st.info("""
            **Possible causes:**
            - No prospectus text available for analysis
            - LLM API call failed (check API keys)
            - Data extraction returned empty results
            - Network connectivity issues
            """)
        
        # Display Benchmarking Analysis
        llm_benchmarking = llm_analysis.get('llm_benchmarking')
        if llm_benchmarking:
            display_llm_benchmarking(llm_benchmarking)
        
        # Display IPO Specifics
        llm_ipo_specifics = llm_analysis.get('llm_ipo_specifics')
        if llm_ipo_specifics:
            display_llm_ipo_specifics(llm_ipo_specifics)
    
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric("Status", "🔄 IPO Pending")
    # with col2:
    #     st.metric("Sector", report.company.sector)
    # with col3:
    #     st.metric("Long-term Score", f"{report.long_term_score:.1f}/10")
    # with col4:
    #     if report.recommendation:
    #         color = {
    #             "Strong Buy": "🟢",
    #             "Buy": "🔵", 
    #             "Hold": "🟡",
    #             "Avoid": "🔴",
    #             "Strong Sell": "⚫"
    #         }.get(report.recommendation.value, "⚪")
    #         st.metric("Recommendation", f"{color} {report.recommendation.value}")
    
    # IPO specific metrics
    # st.subheader("💰 IPO Details")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     # Display price range if available in market_data
    #     if hasattr(report, 'market_data') and report.market_data.get('ipo_price_range'):
    #         price_range = report.market_data['ipo_price_range']
    #         st.metric("Price Range", f"₹{price_range[0]:.0f} - ₹{price_range[1]:.0f}")
    #     else:
    #         st.metric("Price Range", "Not Available")
    
    # with col2:
    #     if report.company.market_cap:
    #         market_cap_cr = report.company.market_cap / 10000000  # Convert to crores
    #         st.metric("Est. Market Cap", f"₹{market_cap_cr:.0f} Cr")
    
    # with col3:
    #     st.metric("Exchange", "NSE & BSE")
    
    # Key metrics row
    st.subheader("📈 Key Financial Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if report.financial_metrics.revenue_growth_rate is not None:
            print(f"✅ Using actual revenue growth rate: {report.financial_metrics.revenue_growth_rate:.1%}")
            st.metric(
                "Revenue Growth Rate", 
                f"{report.financial_metrics.revenue_growth_rate:.1%}",
                delta=f"{'📈' if report.financial_metrics.revenue_growth_rate > 0 else '📉'}"
            )
        else:
            print("⚠️ Using default value for revenue growth rate: None/Not Available")
            st.metric("Revenue Growth Rate", "Not Available")
    
    with col2:
        if report.financial_metrics.profit_margin is not None:
            print(f"✅ Using actual profit margin: {report.financial_metrics.profit_margin:.1%}")
            st.metric(
                "Profit Margin", 
                f"{report.financial_metrics.profit_margin:.1%}",
                delta=f"{'💰' if report.financial_metrics.profit_margin > 0 else '💸'}"
            )
        else:
            print("⚠️ Using default value for profit margin: None/Not Available")
            st.metric("Profit Margin", "Not Available")
    
    with col3:
        if report.listing_gain_prediction is not None:
            print(f"✅ Using calculated listing gain prediction: {report.listing_gain_prediction:.1f}%")
            st.metric(
                "Predicted Listing Gains", 
                f"{report.listing_gain_prediction:.1f}%",
                delta=f"{'🚀' if report.listing_gain_prediction > 0 else '📉'}"
            )
        else:
            print("⚠️ Using default value for listing gain prediction: None/Not Available")
            st.metric("Predicted Listing Gains", "Not Available")
    
    # Risk Assessment
    st.subheader("⚠️ Risk Assessment")
    risk_colors = {
        "Low": "🟢", "Moderate": "🟡", "High": "🔴", "Very High": "⚫"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Overall Risk:** {risk_colors.get(report.risk_assessment.overall_risk.value, '⚪')} {report.risk_assessment.overall_risk.value}")
    with col2:
        st.write(f"**Financial Risk:** {risk_colors.get(report.risk_assessment.financial_risk.value, '⚪')} {report.risk_assessment.financial_risk.value}")
    with col3:
        st.write(f"**Market Risk:** {risk_colors.get(report.risk_assessment.market_risk.value, '⚪')} {report.risk_assessment.market_risk.value}")
    with col4:
        st.write(f"**Operational Risk:** {risk_colors.get(report.risk_assessment.operational_risk.value, '⚪')} {report.risk_assessment.operational_risk.value}")
    
    # Risk Factors
    if report.risk_assessment.risk_factors:
        with st.expander("🔍 Risk Factors"):
            for factor in report.risk_assessment.risk_factors:
                st.write(f"• {factor}")
    
    # Strengths and Weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Strengths")
        if report.strengths_weaknesses.strengths:
            for strength in report.strengths_weaknesses.strengths:
                st.write(f"✅ {strength}")
        else:
            st.write("No specific strengths identified")
    
    with col2:
        st.subheader("⚠️ Weaknesses")
        if report.strengths_weaknesses.weaknesses:
            for weakness in report.strengths_weaknesses.weaknesses:
                st.write(f"❌ {weakness}")
        else:
            st.write("No major weaknesses identified")
    
    # News Sentiment
    st.subheader("📰 Market Sentiment Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_score = report.news_analysis.sentiment_score
        if sentiment_score > 0.1:
            st.success(f"Positive Sentiment: {sentiment_score:.2f}")
        elif sentiment_score < -0.1:
            st.error(f"Negative Sentiment: {sentiment_score:.2f}")
        else:
            st.warning(f"Neutral Sentiment: {sentiment_score:.2f}")
    
    with col2:
        if report.news_analysis.key_themes:
            st.write("**Key Themes:**")
            for theme in report.news_analysis.key_themes[:5]:
                st.write(f"• {theme.title()}")
    
    # Company Description
    if report.company.description:
        with st.expander("🏢 Company Description"):
            st.write(report.company.description)
    
    # Analysis timestamp
    st.caption(f"Analysis performed on: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")


def display_llm_financial_metrics(llm_financial_metrics):
    """Display advanced financial metrics extracted by LLM."""
    
    with st.expander("📈 Advanced Financial Ratios (LLM Extracted)", expanded=False):
        st.subheader("Valuation Ratios")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if hasattr(llm_financial_metrics, 'trailing_pe_ratio') and llm_financial_metrics.trailing_pe_ratio:
                print(f"✅ LLM extracted P/E Ratio (Trailing): {llm_financial_metrics.trailing_pe_ratio:.2f}")
                st.metric("P/E Ratio (Trailing)", f"{llm_financial_metrics.trailing_pe_ratio:.2f}")
            else:
                print("⚠️ Using default for P/E Ratio (Trailing): N/A - No data extracted by LLM")
                st.metric("P/E Ratio (Trailing)", "N/A")
        
        with col2:
            if hasattr(llm_financial_metrics, 'price_to_book_ratio') and llm_financial_metrics.price_to_book_ratio:
                print(f"✅ LLM extracted P/B Ratio: {llm_financial_metrics.price_to_book_ratio:.2f}")
                st.metric("P/B Ratio", f"{llm_financial_metrics.price_to_book_ratio:.2f}")
            else:
                print("⚠️ Using default for P/B Ratio: N/A - No data extracted by LLM")
                st.metric("P/B Ratio", "N/A")
        
        with col3:
            if hasattr(llm_financial_metrics, 'ev_to_ebitda_ratio') and llm_financial_metrics.ev_to_ebitda_ratio:
                print(f"✅ LLM extracted EV/EBITDA: {llm_financial_metrics.ev_to_ebitda_ratio:.2f}")
                st.metric("EV/EBITDA", f"{llm_financial_metrics.ev_to_ebitda_ratio:.2f}")
            else:
                print("⚠️ Using default for EV/EBITDA: N/A - No data extracted by LLM")
                st.metric("EV/EBITDA", "N/A")
        
        with col4:
            if hasattr(llm_financial_metrics, 'price_to_sales_ratio') and llm_financial_metrics.price_to_sales_ratio:
                print(f"✅ LLM extracted P/S Ratio: {llm_financial_metrics.price_to_sales_ratio:.2f}")
                st.metric("P/S Ratio", f"{llm_financial_metrics.price_to_sales_ratio:.2f}")
            else:
                print("⚠️ Using default for P/S Ratio: N/A - No data extracted by LLM")
                st.metric("P/S Ratio", "N/A")
        
        st.subheader("Profitability Ratios")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(llm_financial_metrics, 'return_on_equity') and llm_financial_metrics.return_on_equity:
                print(f"✅ LLM extracted Return on Equity: {llm_financial_metrics.return_on_equity:.2%}")
                st.metric("Return on Equity", f"{llm_financial_metrics.return_on_equity:.2%}")
            else:
                print("⚠️ Using default for Return on Equity: N/A - No data extracted by LLM")
                st.metric("Return on Equity", "N/A")
        
        with col2:
            if hasattr(llm_financial_metrics, 'return_on_assets') and llm_financial_metrics.return_on_assets:
                print(f"✅ LLM extracted Return on Assets: {llm_financial_metrics.return_on_assets:.2%}")
                st.metric("Return on Assets", f"{llm_financial_metrics.return_on_assets:.2%}")
            else:
                print("⚠️ Using default for Return on Assets: N/A - No data extracted by LLM")
                st.metric("Return on Assets", "N/A")
        
        with col3:
            if hasattr(llm_financial_metrics, 'return_on_invested_capital') and llm_financial_metrics.return_on_invested_capital:
                print(f"✅ LLM extracted ROIC: {llm_financial_metrics.return_on_invested_capital:.2%}")
                st.metric("ROIC", f"{llm_financial_metrics.return_on_invested_capital:.2%}")
            else:
                print("⚠️ Using default for ROIC: N/A - No data extracted by LLM")
                st.metric("ROIC", "N/A")
        
        st.subheader("Liquidity & Leverage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(llm_financial_metrics, 'current_ratio') and llm_financial_metrics.current_ratio:
                st.metric("Current Ratio", f"{llm_financial_metrics.current_ratio:.2f}")
            else:
                st.metric("Current Ratio", "N/A")
        
        with col2:
            if hasattr(llm_financial_metrics, 'debt_to_equity_ratio') and llm_financial_metrics.debt_to_equity_ratio:
                st.metric("Debt/Equity", f"{llm_financial_metrics.debt_to_equity_ratio:.2f}")
            else:
                st.metric("Debt/Equity", "N/A")
        
        with col3:
            if hasattr(llm_financial_metrics, 'interest_coverage_ratio') and llm_financial_metrics.interest_coverage_ratio:
                st.metric("Interest Coverage", f"{llm_financial_metrics.interest_coverage_ratio:.2f}")
            else:
                st.metric("Interest Coverage", "N/A")
        
        # Growth Metrics
        if any([
            hasattr(llm_financial_metrics, 'revenue_growth_3yr') and llm_financial_metrics.revenue_growth_3yr,
            hasattr(llm_financial_metrics, 'profit_growth_3yr') and llm_financial_metrics.profit_growth_3yr,
            hasattr(llm_financial_metrics, 'ebitda_growth_3yr') and llm_financial_metrics.ebitda_growth_3yr
        ]):
            st.subheader("3-Year Growth Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if hasattr(llm_financial_metrics, 'revenue_growth_3yr') and llm_financial_metrics.revenue_growth_3yr:
                    st.metric("Revenue Growth (3Y)", f"{llm_financial_metrics.revenue_growth_3yr:.1%}")
            
            with col2:
                if hasattr(llm_financial_metrics, 'profit_growth_3yr') and llm_financial_metrics.profit_growth_3yr:
                    st.metric("Profit Growth (3Y)", f"{llm_financial_metrics.profit_growth_3yr:.1%}")
            
            with col3:
                if hasattr(llm_financial_metrics, 'ebitda_growth_3yr') and llm_financial_metrics.ebitda_growth_3yr:
                    st.metric("EBITDA Growth (3Y)", f"{llm_financial_metrics.ebitda_growth_3yr:.1%}")
        
        # Extraction Quality
        if hasattr(llm_financial_metrics, 'extraction_confidence') and llm_financial_metrics.extraction_confidence:
            st.info(f"**Extraction Confidence:** {llm_financial_metrics.extraction_confidence:.1%}")


def display_llm_benchmarking(llm_benchmarking):
    """Display benchmarking analysis from LLM."""
    
    with st.expander("🏆 Competitive Benchmarking Analysis", expanded=False):
        
        # Market Position
        if hasattr(llm_benchmarking, 'market_position') and llm_benchmarking.market_position:
            st.subheader("Market Position")
            position_colors = {
                "leader": "🥇", "challenger": "🥈", "follower": "🥉", 
                "niche": "🎯", "unknown": "❓"
            }
            position_str = str(llm_benchmarking.market_position) if llm_benchmarking.market_position else "unknown"
            icon = position_colors.get(position_str.lower(), "📍")
            st.info(f"{icon} **Market Position:** {position_str.title()}")
        
        # Competitive Advantages & Disadvantages
        col1, col2 = st.columns(2)
        
        with col1:
            if hasattr(llm_benchmarking, 'competitive_advantages') and llm_benchmarking.competitive_advantages:
                st.subheader("💪 Competitive Advantages")
                for advantage in llm_benchmarking.competitive_advantages:
                    st.write(f"✅ {advantage}")
        
        with col2:
            if hasattr(llm_benchmarking, 'competitive_disadvantages') and llm_benchmarking.competitive_disadvantages:
                st.subheader("⚠️ Competitive Challenges")
                for disadvantage in llm_benchmarking.competitive_disadvantages:
                    st.write(f"❌ {disadvantage}")
        
        # Peer Companies
        if hasattr(llm_benchmarking, 'peer_companies') and llm_benchmarking.peer_companies:
            st.subheader("🏢 Peer Companies")
            for i, peer in enumerate(llm_benchmarking.peer_companies):
                if isinstance(peer, dict):
                    peer_name = peer.get('name', f'Peer {i+1}')
                    similarity = peer.get('similarity', 'Unknown')
                    comparison = peer.get('comparison', 'No comparison available')
                    
                    with st.expander(f"{peer_name} ({similarity.title()} Similarity)"):
                        st.write(comparison)
        
        # Industry Trends
        if hasattr(llm_benchmarking, 'industry_trends') and llm_benchmarking.industry_trends:
            st.subheader("📈 Industry Trends")
            for trend in llm_benchmarking.industry_trends:
                st.write(f"📊 {trend}")
        
        # Sector Comparison
        if hasattr(llm_benchmarking, 'sector_comparison') and llm_benchmarking.sector_comparison:
            st.subheader("📋 Sector Performance Comparison")
            sector_data = llm_benchmarking.sector_comparison
            
            if isinstance(sector_data, dict):
                for key, value in sector_data.items():
                    if key == 'key_metrics_comparison' and isinstance(value, list):
                        for metric in value:
                            st.write(f"• {metric}")
                    else:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")


def display_llm_ipo_specifics(llm_ipo_specifics):
    """Display IPO-specific analysis from LLM."""
    
    with st.expander("🎯 IPO-Specific Analysis", expanded=False):
        
        # IPO Pricing Analysis
        if hasattr(llm_ipo_specifics, 'ipo_pricing_analysis') and llm_ipo_specifics.ipo_pricing_analysis:
            st.subheader("💰 IPO Pricing Analysis")
            pricing = llm_ipo_specifics.ipo_pricing_analysis
            
            if isinstance(pricing, dict):
                for key, value in pricing.items():
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Use of Funds
        if hasattr(llm_ipo_specifics, 'use_of_funds_analysis') and llm_ipo_specifics.use_of_funds_analysis:
            st.subheader("💼 Use of IPO Funds")
            funds = llm_ipo_specifics.use_of_funds_analysis
            
            if isinstance(funds, dict):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    capex = funds.get('capex_percentage')
                    if capex:
                        st.metric("Capital Expenditure", f"{capex}%")
                
                with col2:
                    debt = funds.get('debt_repayment_percentage')
                    if debt:
                        st.metric("Debt Repayment", f"{debt}%")
                
                with col3:
                    working_capital = funds.get('working_capital_percentage')
                    if working_capital:
                        st.metric("Working Capital", f"{working_capital}%")
                
                other_purposes = funds.get('other_purposes')
                if other_purposes:
                    st.write(f"**Other Purposes:** {other_purposes}")
        
        # Underwriter Quality
        if hasattr(llm_ipo_specifics, 'underwriter_quality') and llm_ipo_specifics.underwriter_quality:
            st.subheader("🏦 Underwriter Assessment")
            underwriter = llm_ipo_specifics.underwriter_quality
            
            if isinstance(underwriter, dict):
                lead_managers = underwriter.get('lead_managers', [])
                if lead_managers:
                    st.write(f"**Lead Managers:** {', '.join(lead_managers)}")
                
                reputation = underwriter.get('reputation_score')
                if reputation:
                    reputation_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                    reputation_str = str(reputation) if reputation else "unknown"
                    color = reputation_colors.get(reputation_str.lower(), "⚪")
                    st.write(f"**Reputation Score:** {color} {reputation_str.title()}")
                
                track_record = underwriter.get('track_record')
                if track_record:
                    st.write(f"**Track Record:** {track_record}")
        
        # Business Model Assessment
        if hasattr(llm_ipo_specifics, 'business_model_assessment') and llm_ipo_specifics.business_model_assessment:
            st.subheader("🏗️ Business Model Assessment")
            business_model = llm_ipo_specifics.business_model_assessment
            
            if isinstance(business_model, dict):
                col1, col2 = st.columns(2)
                
                with col1:
                    sustainability = business_model.get('sustainability')
                    if sustainability:
                        sustain_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                        sustainability_str = str(sustainability) if sustainability else "unknown"
                        color = sustain_colors.get(sustainability_str.lower(), "⚪")
                        st.write(f"**Sustainability:** {color} {sustainability_str.title()}")
                    
                    scalability = business_model.get('scalability')
                    if scalability:
                        scale_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                        scalability_str = str(scalability) if scalability else "unknown"
                        color = scale_colors.get(scalability_str.lower(), "⚪")
                        st.write(f"**Scalability:** {color} {scalability_str.title()}")
                
                with col2:
                    competitive_moat = business_model.get('competitive_moat')
                    if competitive_moat:
                        st.write(f"**Competitive Moat:** {competitive_moat}")
                    
                    tech_edge = business_model.get('technology_edge')
                    if tech_edge:
                        st.write(f"**Technology Edge:** {tech_edge}")


if __name__ == "__main__":
    main()
