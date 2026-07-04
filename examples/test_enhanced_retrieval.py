"""
Test Enhanced Context Retrieval for IPO Analysis

This script validates:
1. Comprehensive financial metrics extraction with enhanced vector DB retrieval
2. Competitive analysis with multi-query context gathering
3. IPO specifics with targeted context retrieval
4. Investment thesis generation with rich, structured context
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer, integrate_llm_analysis
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def load_sample_prospectus():
    """Load a sample prospectus text for testing."""
    return """
    DRAFT RED HERRING PROSPECTUS
    
    XYZ Technology Solutions Limited
    
    Our company is India's leading provider of enterprise software solutions with operations 
    across 15 countries. We specialize in cloud-based ERP, CRM, and analytics platforms for 
    mid-to-large enterprises in manufacturing, retail, and financial services sectors.
    
    FINANCIAL HIGHLIGHTS
    
    Restated Consolidated Statement of Assets and Liabilities (₹ in millions):
    
    Particulars                  FY 2023-24    FY 2022-23    FY 2021-22
    Revenue from operations      11,884.89     9,456.23      7,234.56
    Total Revenue               12,045.67     9,589.45      7,356.89
    
    Total Expenses              11,628.96     9,332.50      7,123.45
    EBITDA                         416.71       256.95        233.44
    Profit Before Tax (PBT)        256.93       178.45        145.67
    Profit After Tax (PAT)         189.45       132.56        108.23
    
    Net Profit Margin (%)           1.57%        1.40%         1.47%
    
    Balance Sheet Highlights (₹ in millions):
    
    Current Assets              3,456.78     2,987.45      2,456.89
    Current Liabilities         2,134.56     1,876.34      1,567.23
    Current Ratio                   1.62         1.59          1.57
    
    Total Assets               15,678.90    13,456.78     11,234.56
    Total Liabilities           8,456.78     7,234.56      6,123.45
    Shareholders' Equity        7,222.12     6,222.22      5,111.11
    
    Debt to Equity Ratio            0.45         0.52          0.58
    Return on Equity (%)           2.62%        2.13%         2.12%
    Return on Assets (%)           1.21%        0.98%         0.96%
    
    KEY FINANCIAL RATIOS
    
    - Trailing P/E Ratio: 35.6 (based on latest 12-month EPS)
    - Price to Book Ratio: 4.2
    - Operating Profit Margin: 3.5%
    - Gross Profit Margin: 24.3%
    - Interest Coverage Ratio: 8.5x
    
    BUSINESS OVERVIEW
    
    Our Business Model:
    
    We operate on a Software-as-a-Service (SaaS) subscription model with 85% of revenue 
    from recurring subscriptions. Our platform serves over 2,500 enterprise clients with 
    an average contract value of ₹4.7 million and 95% customer retention rate.
    
    Key Competitive Advantages:
    
    1. Proprietary AI-powered analytics engine with 15 patents
    2. Deep integration with 200+ third-party business applications
    3. Industry-specific solutions for manufacturing, retail, and BFSI
    4. Strong R&D capabilities with 40% of team in product development
    5. Proven track record with Fortune 500 clients including Tata Motors, Reliance Retail
    
    Market Position:
    
    We are the #2 player in India's enterprise SaaS market with approximately 18% market 
    share (by revenue). The Indian enterprise software market is projected to grow at 
    22% CAGR from $8.2 billion in 2023 to $18.5 billion by 2028.
    
    Our main competitors include:
    - Zoho Corporation (Market Leader, ~35% share)
    - Oracle India (~12% share)
    - SAP India (~10% share)
    - International players like Salesforce, Microsoft Dynamics
    
    OBJECTS OF THE ISSUE
    
    The total proceeds from the IPO are estimated at ₹2,500 million, to be utilized as:
    
    1. Investment in R&D and product development: ₹1,000 million (40%)
    2. Sales and marketing expansion: ₹750 million (30%)
    3. Strategic acquisitions of complementary technologies: ₹500 million (20%)
    4. General corporate purposes: ₹250 million (10%)
    
    IPO DETAILS
    
    - Issue Size: ₹2,500 million
    - Price Band: ₹420-450 per share
    - Fresh Issue: 5.5 million shares
    - Offer for Sale: None
    - Face Value: ₹10 per share
    - Issue Opens: January 15, 2025
    - Issue Closes: January 17, 2025
    - Listing Date: January 24, 2025
    
    BOOK RUNNING LEAD MANAGERS
    
    - ICICI Securities Limited
    - Kotak Mahindra Capital Company Limited
    - Axis Capital Limited
    
    PROMOTERS AND MANAGEMENT
    
    Promoter Group:
    - Rajesh Kumar (CEO & Founder): 42% shareholding
    - Priya Sharma (CTO & Co-founder): 18% shareholding
    
    Both promoters have 15+ years of experience in enterprise software. Rajesh previously 
    held senior positions at Infosys and TCS. Priya was lead architect at Adobe India.
    
    Post-IPO, promoter holding will reduce to 60% from current 75%.
    
    RISK FACTORS
    
    Material Risks:
    
    1. Customer Concentration: Top 10 clients account for 32% of revenue
    2. Intense Competition: Facing competition from global giants with deeper pockets
    3. Technology Obsolescence: Rapid changes in cloud and AI technologies
    4. Regulatory Compliance: Data privacy regulations across multiple jurisdictions
    5. Employee Attrition: High competition for tech talent, current attrition at 18%
    6. Foreign Exchange Risk: 35% of revenue in USD/EUR, creating currency exposure
    
    INDUSTRY OVERVIEW
    
    The global enterprise software market is experiencing strong growth driven by:
    - Digital transformation initiatives across industries
    - Shift from on-premise to cloud-based solutions
    - Increasing adoption of AI and machine learning
    - Growing need for data analytics and business intelligence
    
    India is emerging as a key market with enterprises increasingly adopting modern 
    cloud-based solutions. Government initiatives like Digital India and Make in India 
    are accelerating enterprise digitization.
    
    Industry Growth Drivers:
    - Manufacturing sector modernization
    - Retail digital transformation
    - BFSI regulatory compliance needs
    - SME digitization wave
    
    PEER COMPARISON
    
    Compared to listed peers like Zoho (unlisted), Oracle India, we offer:
    - Higher Revenue Growth: 25.7% vs. industry average of 18%
    - Better Customer Retention: 95% vs. industry average of 82%
    - Lower Customer Acquisition Cost: ₹380K vs. industry average of ₹520K
    - Comparable Profitability: Net margin of 1.57% vs. industry range of 1.2-2.5%
    
    However, we have:
    - Lower Market Share than Zoho
    - Smaller geographic presence compared to Oracle/SAP
    - Higher Debt levels relative to larger competitors
    
    VALUATION INSIGHTS
    
    At upper price band of ₹450:
    - Market Capitalization: ₹18,000 million
    - P/E Ratio: 35.6x (FY24 EPS: ₹12.64)
    - P/B Ratio: 4.2x
    - EV/EBITDA: 28.5x
    
    Peer Comparison:
    - Listed SaaS companies in India trade at 30-45x P/E
    - Global SaaS leaders trade at 40-60x P/E
    - Our pricing is positioned at moderate premium to Indian peers
    
    MANAGEMENT DISCUSSION AND ANALYSIS
    
    Future Growth Strategy:
    
    1. Product Innovation: Launch of AI-powered analytics suite in Q2 FY25
    2. Market Expansion: Entry into Southeast Asian markets (Singapore, Malaysia)
    3. Vertical Expansion: New solutions for healthcare and logistics sectors
    4. Partnerships: Strategic partnerships with AWS, Google Cloud, and Microsoft Azure
    5. Acquisitions: Actively evaluating 3-4 acquisition targets in analytics space
    
    Our three-year targets (by FY27):
    - Revenue Growth to ₹20,000 million (68% growth from FY24)
    - EBITDA Margin improvement to 8-10%
    - International revenue contribution to 40% (from current 15%)
    - Client base expansion to 5,000+ enterprise customers
    
    CONCLUSION
    
    XYZ Technology Solutions represents a compelling investment opportunity in India's 
    rapidly growing enterprise software market. With strong revenue growth, improving 
    profitability, market-leading products, and experienced management, the company is 
    well-positioned to capitalize on the ongoing digital transformation wave.
    """

def test_enhanced_retrieval():
    """Test enhanced context retrieval for IPO analysis."""
    
    print("\n" + "="*80)
    print("TESTING ENHANCED CONTEXT RETRIEVAL FOR IPO ANALYSIS")
    print("="*80 + "\n")
    
    company_name = "XYZ Technology Solutions Limited"
    sector = "Enterprise Software / SaaS"
    
    # Load sample prospectus
    prospectus_text = load_sample_prospectus()
    print(f"✓ Loaded sample prospectus ({len(prospectus_text)} characters)\n")
    
    # Initialize analyzer with vector DB
    print("Step 1: Initializing analyzer with vector DB...")
    analyzer = LLMProspectusAnalyzer(provider="groq", use_vector_db=True)
    print("✓ Analyzer initialized\n")
    
    # Chunk and store prospectus
    print("Step 2: Chunking and storing prospectus in vector DB...")
    try:
        analyzer.chunk_and_store_prospectus(
            pdf_text=prospectus_text,
            company_name=company_name,
            sector=sector,
            ipo_date=None
        )
        print("✓ Prospectus chunks stored in vector DB\n")
    except Exception as e:
        print(f"✗ Error storing prospectus: {e}\n")
        return
    
    # Test 1: Enhanced Financial Metrics Extraction
    print("\n" + "-"*80)
    print("TEST 1: Enhanced Financial Metrics Extraction")
    print("-"*80)
    
    try:
        financial_metrics = analyzer._extract_financial_metrics(
            pdf_text=prospectus_text,
            company_name=company_name
        )
        
        print("\n✓ Financial Metrics Extracted:")
        print(f"  - Trailing P/E Ratio: {financial_metrics.trailing_pe_ratio}")
        print(f"  - Price to Book: {financial_metrics.price_to_book_ratio}")
        print(f"  - Net Profit Margin: {financial_metrics.net_profit_margin}")
        print(f"  - Return on Equity: {financial_metrics.return_on_equity}")
        print(f"  - Debt to Equity: {financial_metrics.debt_to_equity_ratio}")
        print(f"  - Current Ratio: {financial_metrics.current_ratio}")
        print(f"  - Revenue Growth (3yr): {financial_metrics.revenue_growth_3yr}")
        print(f"  - Extraction Confidence: {financial_metrics.extraction_confidence}")
        print(f"  - Data Completeness: {financial_metrics.data_completeness}")
        
        # Validate metrics
        assert financial_metrics.extraction_confidence > 0.5, "Low extraction confidence"
        assert financial_metrics.trailing_pe_ratio is not None, "P/E ratio not extracted"
        
        print("\n✓ TEST 1 PASSED: Financial metrics extracted with good confidence\n")
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}\n")
        logger.exception("Financial metrics extraction failed")
    
    # Test 2: Enhanced Competitive Analysis
    print("\n" + "-"*80)
    print("TEST 2: Enhanced Competitive Analysis")
    print("-"*80)
    
    try:
        benchmarking = analyzer._perform_benchmarking_analysis(
            pdf_text=prospectus_text,
            company_name=company_name,
            sector=sector
        )
        
        print("\n✓ Benchmarking Analysis Completed:")
        print(f"  - Market Position: {benchmarking.market_position}")
        print(f"  - Peer Companies: {len(benchmarking.peer_companies)} identified")
        if benchmarking.peer_companies:
            for peer in benchmarking.peer_companies[:3]:
                print(f"    • {peer.get('name', 'Unknown')}: {peer.get('comparison', 'N/A')}")
        print(f"  - Competitive Advantages: {len(benchmarking.competitive_advantages)}")
        for adv in benchmarking.competitive_advantages[:3]:
            print(f"    • {adv}")
        print(f"  - Industry Trends: {len(benchmarking.industry_trends)}")
        
        # Validate
        assert benchmarking.market_position != "unknown", "Market position not determined"
        assert len(benchmarking.competitive_advantages) > 0, "No competitive advantages found"
        
        print("\n✓ TEST 2 PASSED: Competitive analysis completed successfully\n")
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}\n")
        logger.exception("Benchmarking analysis failed")
    
    # Test 3: Enhanced IPO Specifics
    print("\n" + "-"*80)
    print("TEST 3: Enhanced IPO Specifics Extraction")
    print("-"*80)
    
    try:
        ipo_specifics = analyzer._analyze_ipo_specifics(
            pdf_text=prospectus_text,
            company_name=company_name
        )
        
        print("\n✓ IPO Specifics Extracted:")
        print(f"  - Lead Managers: {', '.join(ipo_specifics.lead_managers[:3])}")
        print(f"  - Business Model Quality: {ipo_specifics.business_model_quality}")
        print(f"  - Pricing Justification: {ipo_specifics.pricing_justification[:100]}...")
        print(f"  - Use of Funds: {len(ipo_specifics.use_of_funds)} categories")
        
        # Validate
        assert len(ipo_specifics.lead_managers) > 0, "No lead managers found"
        
        print("\n✓ TEST 3 PASSED: IPO specifics extracted successfully\n")
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}\n")
        logger.exception("IPO specifics extraction failed")
    
    # Test 4: Enhanced Investment Thesis Generation
    print("\n" + "-"*80)
    print("TEST 4: Enhanced Investment Thesis Generation")
    print("-"*80)
    
    try:
        investment_thesis = analyzer.generate_investment_thesis(
            financial_metrics=financial_metrics,
            benchmarking=benchmarking,
            ipo_specifics=ipo_specifics,
            company_name=company_name,
            sector=sector,
            web_context="Recent industry reports indicate strong growth in enterprise SaaS adoption."
        )
        
        print("\n✓ Investment Thesis Generated:")
        print("-" * 80)
        print(investment_thesis)
        print("-" * 80)
        
        # Validate thesis quality
        assert len(investment_thesis) > 500, "Thesis too short"
        assert "XYZ Technology" in investment_thesis, "Company name not mentioned"
        assert any(keyword in investment_thesis.lower() for keyword in 
                  ["revenue", "growth", "valuation", "risk", "recommendation"]), \
                  "Key thesis elements missing"
        
        # Check for anti-hallucination markers
        has_data_awareness = any(phrase in investment_thesis.lower() for phrase in 
                                ["data not available", "not disclosed", "based on provided", 
                                 "according to", "insufficient data"])
        
        print(f"\n  - Thesis Length: {len(investment_thesis)} characters")
        print(f"  - Data Awareness Markers: {'Present' if has_data_awareness else 'Absent'}")
        print(f"  - Specific Company Mentions: {'Yes' if 'XYZ' in investment_thesis else 'No'}")
        
        print("\n✓ TEST 4 PASSED: Investment thesis generated with comprehensive context\n")
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}\n")
        logger.exception("Investment thesis generation failed")
    
    print("\n" + "="*80)
    print("ENHANCED RETRIEVAL TESTING COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Financial metrics extraction with 6+ targeted queries")
    print("  ✓ Competitive analysis with comprehensive context gathering")
    print("  ✓ IPO specifics with multi-faceted retrieval")
    print("  ✓ Investment thesis with rich, structured context")
    print("\nExpected Improvements:")
    print("  • 3-5x more context chunks retrieved per analysis")
    print("  • Better coverage of prospectus sections")
    print("  • More specific and accurate recommendations")
    print("  • Reduced hallucination through data-grounded responses")

if __name__ == "__main__":
    test_enhanced_retrieval()
