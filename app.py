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
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

from src.models import IPOAnalysisReport
from src.agent import IPOReviewAgent, GMP_EXTRACTOR_AVAILABLE


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="IPO Review Agent",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 IPO Review Agent")
    st.caption("Pre-IPO analysis for the Indian market")
    st.caption("📱 On a phone? Tap the **›** arrow at the top-left to open Analysis Parameters and enter a company name.")

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
            
            default_provider_index = (
                available_providers.index("Groq Mixtral")
                if "Groq Mixtral" in available_providers
                else 0
            )
            selected_provider_name = st.selectbox(
                "Select LLM Provider",
                available_providers,
                index=default_provider_index,
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
        
        # GMP Analysis Configuration
        st.subheader("💹 GMP Analysis")
        
        # Check for GMP API keys
        groq_gmp_key = os.getenv('GROQ_API_KEY')
        brave_gmp_key = os.getenv('BRAVE_API_KEY')
        
        if GMP_EXTRACTOR_AVAILABLE and groq_gmp_key and brave_gmp_key:
            use_gmp_analysis = st.checkbox(
                "📊 Enable GMP Analysis",
                value=True,
                help="Analyze Grey Market Premium using Brave Search and Groq LLM"
            )
        else:
            use_gmp_analysis = False
            if not GMP_EXTRACTOR_AVAILABLE:
                st.warning("⚠️ GMP extractor not available")
            elif not groq_gmp_key:
                st.warning("⚠️ GROQ_API_KEY not configured")
            elif not brave_gmp_key:
                st.warning("⚠️ BRAVE_API_KEY not configured")
        
        # Company input - simplified to just company name
        st.subheader("🏢 Company Information")
        company_name = st.text_input(
            "IPO/Company Name",
            value="Aastha Spintex",
            placeholder="e.g., Zomato Limited, Paytm, Vidya Wires Limited",
            help="Enter the company name for comprehensive IPO analysis"
        )
        
        analyze_button = st.button("🔍 Analyze IPO", type="primary", disabled=not company_name)
    
    # Main content area
    if analyze_button and company_name:
        # Store GMP analysis result
        gmp_result = None

        # Single agent instance reused for both the GMP and full-analysis calls
        agent = IPOReviewAgent(use_llm=use_llm_analysis, llm_provider=selected_llm_provider, enable_reflection=False)

        # Run GMP analysis FIRST if enabled
        if use_gmp_analysis:
            with st.spinner(f"🔍 Analyzing GMP for {company_name}..."):
                try:
                    gmp_result = agent.analyze_gmp(company_name)

                    if gmp_result and gmp_result.get('status') == 'success':
                        st.success("✅ GMP Analysis completed successfully!")
                    elif gmp_result and gmp_result.get('status') == 'not_found':
                        st.warning(f"⚠️ {gmp_result.get('message', 'GMP data not found')}")
                    elif gmp_result and gmp_result.get('status') == 'error':
                        st.error(f"❌ GMP Analysis failed: {gmp_result.get('message')}")
                except Exception as e:
                    st.error(f"❌ GMP Analysis error: {str(e)}")
                    logger.error(f"GMP analysis error: {e}")

        # Run standard IPO analysis
        with st.spinner(f"🔄 Analyzing {company_name}..."):
            try:
                # Create IPO details dictionary - simplified for name-only analysis
                ipo_details = {
                    'company_name': company_name
                }

                # Perform analysis
                report = agent.analyze_ipo(company_name, ipo_details)
                
                # Display results WITH GMP data
                display_analysis_report(report, gmp_result)
                
            except Exception as e:
                logger.exception(f"Analysis error for {company_name}: {e}")
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Check the application logs for full error details.")

    else:
        # Welcome screen
        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ### 📄 DRHP-Grounded Report
            - Financial metrics extracted from the company's actual SEBI Draft Offer Document (P/E, margins, ROE/ROA, growth rates)
            - Business, promoter, and risk analysis grounded in filed disclosures
            - Financial health assessment and industry benchmarking
            - Data-driven investment recommendation
            """)

        with col2:
            st.info("""
            ### 💹 GMP Market Sentiment
            - Live Grey Market Premium via Brave Search + Groq LLM
            - Current GMP price, percentage, and estimated listing gain
            - Narrative read on demand and grey market sentiment
            - Runs alongside the DRHP report for a fuller picture
            """)

        # Add information about how the two features fit together
        st.markdown("---")
        st.markdown("""
        ### 🎯 **How It Works**

        This tool combines two data sources for Indian pre-IPO companies:

        1. 📄 **DRHP analysis** — the agent reads the company's Draft Red Herring Prospectus (find it via the **SEBI Document Search** tab) and uses an LLM to extract real financial figures and risk factors, not estimates.
        2. 💹 **GMP sentiment** — in parallel, it searches for the current Grey Market Premium to gauge listing-day demand.

        The two are combined into one report: fundamentals and risk from the DRHP, market sentiment from the GMP.

        **How to use:** Enter the company name and click **Analyze IPO** to get both.
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
    - **DRHP-Grounded Data**: Financial figures and risk factors extracted from official Draft Offer Documents, not estimates
    - **GMP Market Sentiment**: Live Grey Market Premium via Brave Search + Groq LLM
    - **Risk Assessment**: Comprehensive business, financial, and market risk analysis
    - **Investment Recommendations**: Data-driven investment advice combining fundamentals and market sentiment

    ## 📊 How It Works
    1. **Search SEBI**: Find the company's Draft Offer Document in the SEBI Document Search tab
    2. **Extract Financials**: An LLM reads the DRHP and extracts ratios, growth rates, and risk factors
    3. **Check GMP**: In parallel, current Grey Market Premium is fetched for market sentiment
    4. **Assess Risk**: Evaluate business, financial, and market risks
    5. **Generate Report**: Combine both into one investment recommendation

    ## 🔧 Technical Stack
    - **Data Sources**: SEBI Draft Offer Documents, Brave Search (GMP), Alpha Vantage, NewsAPI
    - **Analysis**: LLM-powered financial extraction, Python/pandas ratio calculations
    - **AI/ML**: LLM prospectus analysis (OpenAI/Anthropic/Groq/Gemini), Groq for GMP extraction
    - **Interface**: Streamlit web application

    ## ⚡ Getting Started
    1. Go to **SEBI Document Search** tab
    2. Search for a company (e.g., "Vidya Wires")
    3. Download the DRHP document
    4. Run the IPO analysis
    5. Get investment recommendation with DRHP fundamentals and GMP sentiment

    ## 🔑 API Configuration
    For full functionality, configure these API keys in `.env`:

    **GMP Analysis:**
    - `GROQ_API_KEY` - Groq LLM for GMP extraction
    - `BRAVE_API_KEY` - Brave Search for finding current GMP data

    **Basic Financial Data:**
    - `ALPHA_VANTAGE_API_KEY` - Financial data
    - `NEWS_API_KEY` - News sentiment analysis

    **Enhanced DRHP Analysis (choose one or more):**
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
    
    st.info("""
    🔍 **Why SEBI Draft Documents?**
    - Find companies in **pre-IPO stage** (DRHP - Draft Red Herring Prospectus)
    - Access **real financial data** from official SEBI filings, not estimates
    - Perfect timing for **investment analysis** before public launch
    - Reflects **current filings**, not historical completed listings
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

SEBI_ALLOWED_HOST = "www.sebi.gov.in"


def _is_sebi_url(url: str) -> bool:
    """Only allow fetching URLs hosted on the official SEBI domain."""
    try:
        return urlparse(url).hostname == SEBI_ALLOWED_HOST
    except ValueError:
        return False


def download_sebi_document(filing_url, company_name):
    """Download SEBI document from filing URL."""
    try:
        if not _is_sebi_url(filing_url):
            st.error("❌ Refusing to fetch a document from a non-SEBI URL")
            return

        st.info(f"📥 Downloading document for {company_name}...")

        # Extract PDF URL from filing page
        response = requests.get(filing_url)

        if "sebi_data/attachdocs" in response.text:
            pdf_pattern = r'https://www\.sebi\.gov\.in/sebi_data/attachdocs/[^"]*\.pdf'
            pdf_urls = re.findall(pdf_pattern, response.text)

            if pdf_urls and _is_sebi_url(pdf_urls[0]):
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


def _thesis_sections(thesis_text: str) -> dict:
    """
    Split the LLM-generated investment thesis into its numbered sections
    (matches the "1. EXECUTIVE SUMMARY" / "2. BUSINESS FUNDAMENTALS" ...
    template in generate_investment_thesis) so each part can be routed to the
    right tab instead of rendered as one undifferentiated wall of text.
    Falls back to a single section if the expected headers aren't found, so a
    format drift in the LLM's output never silently hides content.
    """
    pattern = re.compile(r'^\**\s*(\d)\.\s*([A-Z][A-Z &/]+?)\**\s*$', re.MULTILINE)
    matches = list(pattern.finditer(thesis_text))
    if len(matches) < 2:
        return {"Full Analysis": thesis_text.strip()}

    sections = {}
    for i, m in enumerate(matches):
        title = m.group(2).strip().title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(thesis_text)
        content = thesis_text[start:end].strip()
        if content:
            sections[title] = content
    return sections


def _section(sections: dict, *keywords: str) -> str:
    """Find a thesis section whose title contains any of the given keywords."""
    for title, content in sections.items():
        if any(kw.lower() in title.lower() for kw in keywords):
            return content
    return ""


# Recommendation label -> (emoji, Streamlit banner function). Buy-side
# recommendations use the built-in "success" (green) styling, Hold uses
# "warning" (amber), Avoid/Sell uses "error" (red) - reusing Streamlit's
# semantic colors instead of custom CSS.
RECOMMENDATION_STYLE = {
    "Strong Buy": ("🟢", st.success),
    "Buy": ("🟢", st.success),
    "Hold": ("🟡", st.warning),
    "Avoid": ("🔴", st.error),
    "Strong Sell": ("🔴", st.error),
}

RISK_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very High": "🔴"}


def display_analysis_report(report: IPOAnalysisReport, gmp_result: dict = None):
    """Display the enhanced analysis report with LLM insights and GMP analysis."""

    st.header(f"📊 {report.company.name}")
    if report.company.sector and report.company.sector != "Unknown":
        st.caption(report.company.sector)

    # Link back to the actual SEBI filing this analysis was built from, so the
    # user can open the source DRHP and verify any number in this report themselves.
    enhanced_prospectus = getattr(report, 'raw_data', {}).get('enhanced_prospectus')
    source_url = getattr(enhanced_prospectus, 'source_url', None) if enhanced_prospectus else None
    if source_url and _is_sebi_url(source_url):
        st.markdown(f"📄 [View source DRHP on SEBI]({source_url})")
    else:
        st.caption(
            "📄 Source DRHP link not available for this run — search "
            "[SEBI Draft Offer Documents](https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10) "
            f"for \"{report.company.name}\" to verify."
        )

    llm_analysis = None
    if hasattr(report, 'raw_data') and 'llm_analysis' in getattr(report, 'raw_data', {}):
        llm_analysis = report.raw_data['llm_analysis']

    thesis_sections = {}
    if llm_analysis:
        investment_thesis = llm_analysis.get('llm_investment_thesis', '')
        if investment_thesis:
            thesis_sections = _thesis_sections(investment_thesis)

    gmp_percentage = None
    if gmp_result and gmp_result.get('status') == 'success':
        gmp_percentage = gmp_result.get('structured_data', {}).get('gmp_percentage')

    # ── Verdict bar: the agent's bottom-line call, up front ─────────────
    if report.recommendation:
        emoji, banner = RECOMMENDATION_STYLE.get(report.recommendation.value, ("⚪", st.info))
        banner(f"### {emoji} {report.recommendation.value}")
        st.caption(
            "Algorithmic estimate based on available financial data and disclosed risk "
            "factors - not a substitute for professional investment advice."
        )

    metric_specs = [
        ("Long-Term Score", report.long_term_score, lambda v: f"{v:.1f}/10"),
        (
            "Overall Risk",
            report.risk_assessment.overall_risk.value if report.risk_assessment else None,
            lambda v: f"{RISK_EMOJI.get(v, '⚪')} {v}",
        ),
        ("GMP %", gmp_percentage, lambda v: f"{v}%"),
        ("Analyst Confidence", report.analyst_confidence, lambda v: f"{v:.0%}"),
    ]
    available = [(label, fmt(value)) for label, value, fmt in metric_specs if value is not None]
    if available:
        cols = st.columns(len(available))
        for col, (label, display_value) in zip(cols, available):
            with col:
                st.metric(label, display_value)

    # ── AI self-review: surface anything the agent's own reflection/
    # consistency-check passes flagged, so a contradiction isn't silently
    # buried in the numbers above ─────────────────────────────────────────
    consistency_check = getattr(report, 'raw_data', {}).get('consistency_check') or {}
    reflection = (llm_analysis or {}).get('llm_reflection')
    callout_issues = list(consistency_check.get('issues', []))
    if reflection is not None and getattr(reflection, 'issues', None):
        callout_issues.extend(reflection.issues)
    reflection_retries = getattr(reflection, 'iterations_used', 0) if reflection is not None else 0
    if callout_issues or reflection_retries:
        with st.expander(f"⚠️ {len(callout_issues)} consistency flag(s) from AI self-review", expanded=False):
            if reflection_retries:
                st.caption(
                    f"The agent re-extracted financial metrics {reflection_retries} "
                    f"time(s) after its own review flagged issues."
                )
            for issue in callout_issues:
                st.write(f"- {issue}")

    # ── Executive summary, always visible - no click required ──────────
    exec_summary = _section(thesis_sections, "executive summary")
    if exec_summary:
        st.markdown("---")
        st.markdown(exec_summary)

    st.markdown("---")

    # ── Detail tabs instead of a long stack of expanders ────────────────
    tabs = st.tabs([
        "📋 Business & SWOT", "💰 Financials", "🏆 Market & Competitive",
        "🎯 IPO Details", "⚠️ Risks & Outlook",
    ])

    with tabs[0]:
        business_section = _section(thesis_sections, "business fundamentals")
        if business_section:
            st.markdown(business_section)
        _display_swot(report.strengths_weaknesses)

    with tabs[1]:
        financial_section = _section(thesis_sections, "financial health")
        if financial_section:
            st.markdown(financial_section)
        if llm_analysis:
            llm_financial_metrics = llm_analysis.get('llm_financial_metrics')
            if llm_financial_metrics:
                st.markdown("#### Extracted Financial Ratios")
                display_llm_financial_metrics(llm_financial_metrics)
            elif llm_analysis.get('error'):
                st.error(f"LLM Analysis Error: {llm_analysis['error']}")
            else:
                st.caption("No LLM financial metrics extracted.")

    with tabs[2]:
        market_section = _section(thesis_sections, "market context")
        if market_section:
            st.markdown(market_section)
        if llm_analysis:
            llm_benchmarking = llm_analysis.get('llm_benchmarking')
            if llm_benchmarking:
                display_llm_benchmarking(llm_benchmarking)

    with tabs[3]:
        ipo_section = _section(thesis_sections, "ipo characteristics")
        if ipo_section:
            st.markdown(ipo_section)
        _display_gmp(gmp_result)
        if llm_analysis:
            llm_ipo_specifics = llm_analysis.get('llm_ipo_specifics')
            if llm_ipo_specifics:
                display_llm_ipo_specifics(llm_ipo_specifics)

    with tabs[4]:
        risks_section = _section(thesis_sections, "opportunities", "risks")
        if risks_section:
            st.markdown(risks_section)
        _display_risk_assessment(report.risk_assessment)
        perspective = _section(thesis_sections, "investment perspective")
        if perspective:
            st.markdown("#### Investment Perspective")
            st.markdown(perspective)

    # Fallback: if section parsing found nothing usable anywhere above (e.g.
    # the LLM's output didn't match the expected numbered-header format),
    # show the raw thesis rather than silently dropping it.
    if llm_analysis and list(thesis_sections.keys()) == ["Full Analysis"]:
        with tabs[0]:
            st.markdown("---")
            with st.expander("Full AI-Generated Thesis (unparsed)", expanded=True):
                st.markdown(thesis_sections["Full Analysis"])

    st.markdown("---")
    st.caption(f"Analysis performed on: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")


def _display_gmp(gmp_result: dict = None):
    """Display GMP metrics, narrative (if real data was found), and sources."""
    if not gmp_result or gmp_result.get('status') != 'success':
        return

    st.markdown("#### 💹 Grey Market Premium (GMP)")

    structured_data = gmp_result.get('structured_data', {})
    metric_fields = [
        ("GMP Price", structured_data.get('gmp_price'), lambda v: f"₹{v}"),
        ("GMP %", structured_data.get('gmp_percentage'), lambda v: f"{v}%"),
        ("Issue Price", structured_data.get('issue_price'), lambda v: f"₹{v}"),
        ("Expected Listing", structured_data.get('expected_listing_price'), lambda v: f"₹{v}"),
    ]
    available_metrics = [(label, fmt(value)) for label, value, fmt in metric_fields if value]
    has_gmp_data = bool(available_metrics)

    if available_metrics:
        cols = st.columns(len(available_metrics))
        for col, (label, display_value) in zip(cols, available_metrics):
            with col:
                st.metric(label, display_value)

    analysis_text = gmp_result.get('analysis', '')
    if has_gmp_data and analysis_text:
        with st.expander("📝 Comprehensive GMP Analysis", expanded=True):
            st.markdown(analysis_text)

    sources = gmp_result.get('sources', [])
    if sources:
        with st.expander("🔗 Data Sources", expanded=not has_gmp_data):
            for i, source in enumerate(sources, 1):
                st.write(f"{i}. {source}")
    elif not has_gmp_data:
        st.caption("No GMP data or sources found for this company.")


def _display_swot(swot):
    """Display the structured strengths/weaknesses/opportunities/threats analysis."""
    if not swot or not any([swot.strengths, swot.weaknesses, swot.opportunities, swot.threats]):
        return

    st.markdown("#### SWOT Summary")
    col1, col2 = st.columns(2)
    with col1:
        if swot.strengths:
            st.markdown("**💪 Strengths**")
            for s in swot.strengths:
                st.write(f"✅ {s}")
        if swot.opportunities:
            st.markdown("**🚀 Opportunities**")
            for o in swot.opportunities:
                st.write(f"📈 {o}")
    with col2:
        if swot.weaknesses:
            st.markdown("**⚠️ Weaknesses**")
            for w in swot.weaknesses:
                st.write(f"❌ {w}")
        if swot.threats:
            st.markdown("**🌩️ Threats**")
            for t in swot.threats:
                st.write(f"⚡ {t}")


def _display_risk_assessment(risk_assessment):
    """Display the structured risk assessment (overall/financial/market/operational)."""
    if not risk_assessment:
        return

    st.markdown("#### Risk Breakdown")
    cols = st.columns(4)
    risk_items = [
        ("Overall", risk_assessment.overall_risk),
        ("Financial", risk_assessment.financial_risk),
        ("Market", risk_assessment.market_risk),
        ("Operational", risk_assessment.operational_risk),
    ]
    for col, (label, level) in zip(cols, risk_items):
        with col:
            level_label = level.value if level else "Unknown"
            st.metric(label, f"{RISK_EMOJI.get(level_label, '⚪')} {level_label}")

    if risk_assessment.risk_factors:
        st.markdown("**Key Risk Factors**")
        for factor in risk_assessment.risk_factors:
            st.write(f"🔸 {factor}")

    if risk_assessment.risk_mitigation:
        st.markdown("**Risk Mitigation**")
        for mitigation in risk_assessment.risk_mitigation:
            st.write(f"🛡️ {mitigation}")


def display_llm_financial_metrics(llm_financial_metrics):
    """Display advanced financial metrics extracted by LLM."""
    
    with st.expander("📈 Advanced Financial Ratios (LLM Extracted)", expanded=True):
        st.subheader("Profitability Ratios")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(llm_financial_metrics, 'return_on_equity') and llm_financial_metrics.return_on_equity:
                # return_on_equity is already a percentage value (e.g. 20.47 for
                # 20.47%), not a fraction - ":.2%" would multiply it by 100 again
                print(f"✅ LLM extracted Return on Equity: {llm_financial_metrics.return_on_equity:.2f}%")
                st.metric("Return on Equity", f"{llm_financial_metrics.return_on_equity:.2f}%")
            else:
                print("⚠️ Using default for Return on Equity: N/A - No data extracted by LLM")
                st.metric("Return on Equity", "N/A")
        
        with col2:
            if hasattr(llm_financial_metrics, 'return_on_assets') and llm_financial_metrics.return_on_assets:
                print(f"✅ LLM extracted Return on Assets: {llm_financial_metrics.return_on_assets:.2f}%")
                st.metric("Return on Assets", f"{llm_financial_metrics.return_on_assets:.2f}%")
            else:
                print("⚠️ Using default for Return on Assets: N/A - No data extracted by LLM")
                st.metric("Return on Assets", "N/A")
        
        with col3:
            if hasattr(llm_financial_metrics, 'return_on_invested_capital') and llm_financial_metrics.return_on_invested_capital:
                print(f"✅ LLM extracted ROIC: {llm_financial_metrics.return_on_invested_capital:.2f}%")
                st.metric("ROIC", f"{llm_financial_metrics.return_on_invested_capital:.2f}%")
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
                    st.metric("Revenue Growth (3Y)", f"{llm_financial_metrics.revenue_growth_3yr:.1f}%")

            with col2:
                if hasattr(llm_financial_metrics, 'profit_growth_3yr') and llm_financial_metrics.profit_growth_3yr:
                    st.metric("Profit Growth (3Y)", f"{llm_financial_metrics.profit_growth_3yr:.1f}%")

            with col3:
                if hasattr(llm_financial_metrics, 'ebitda_growth_3yr') and llm_financial_metrics.ebitda_growth_3yr:
                    st.metric("EBITDA Growth (3Y)", f"{llm_financial_metrics.ebitda_growth_3yr:.1f}%")
        
        # Extraction Quality
        if hasattr(llm_financial_metrics, 'extraction_confidence') and llm_financial_metrics.extraction_confidence:
            st.info(f"**Extraction Confidence:** {llm_financial_metrics.extraction_confidence:.1%}")


def display_llm_benchmarking(llm_benchmarking):
    """Display benchmarking analysis from LLM."""
    
    with st.expander("🏆 Competitive Benchmarking Analysis", expanded=True):
        
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
    
    with st.expander("🎯 IPO-Specific Analysis", expanded=True):
        
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
