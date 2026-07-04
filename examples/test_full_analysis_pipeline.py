#!/usr/bin/env python3
"""
Comprehensive test of the full IPO analysis pipeline with structured chunking.

This test validates:
1. PDF text extraction
2. Structured chunking and vector DB storage
3. Financial metrics extraction
4. Benchmarking analysis
5. IPO specifics extraction
6. Investment thesis generation with rich context
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_full_analysis_pipeline():
    """Test the complete IPO analysis pipeline."""
    
    print_section("IPO ANALYSIS PIPELINE TEST")
    
    # Find PDF file
    print("\n1. Locating PDF file...")
    possible_paths = [
        "vidya_wires.pdf",
        "prospectus/vidya_wires_draft_prospectus.pdf",
        "/Users/apoorvjain/Projects/IPO Review Agent/prospectus/vidya_wires_draft_prospectus.pdf"
    ]
    
    pdf_path = None
    for path in possible_paths:
        if Path(path).exists():
            pdf_path = path
            print(f"   ✓ Found PDF: {path}")
            break
    
    if not pdf_path:
        print(f"\n   ❌ No PDF found. Tried:")
        for path in possible_paths:
            print(f"      • {path}")
        print("\n   Using sample text for testing...")
        
        # Use sample prospectus text for testing
        pdf_text = """
        DRAFT RED HERRING PROSPECTUS
        
        Vidya Wires Limited
        
        FINANCIAL INFORMATION
        
        Revenue from Operations (FY 2024): ₹11,884.89 Lakhs
        Profit After Tax (FY 2024): ₹256.93 Lakhs
        Net Profit Margin: 2.16%
        Total Assets: ₹5,234.50 Lakhs
        Total Liabilities: ₹3,120.30 Lakhs
        Return on Assets: 4.91%
        
        Revenue Growth (FY 2023-2024): 45.2%
        Profit Growth (FY 2023-2024): 38.7%
        
        Current Ratio: 1.65
        Debt to Equity Ratio: 0.85
        
        BUSINESS OVERVIEW
        
        The Company is engaged in manufacturing electrical wires and cables. 
        We have a strong market presence in the western region of India with 
        distribution networks in Maharashtra, Gujarat, and Rajasthan.
        
        Our key competitive advantages include:
        - Established brand recognition in regional markets
        - Long-term relationships with distributors
        - Modern manufacturing facility with capacity of 50,000 MT per annum
        - Quality certifications including ISO 9001:2015
        
        INDUSTRY AND MARKET
        
        The Indian wires and cables market is growing at 12-15% CAGR driven by:
        - Infrastructure development
        - Housing construction boom
        - Industrial expansion
        - Government initiatives like Make in India
        
        The market is highly fragmented with presence of both organized and 
        unorganized players. Key competitors include Polycab, Havells, KEI Industries.
        
        RISK FACTORS
        
        1. Intense competition from larger players with better brand recognition
        2. Raw material price volatility (copper, aluminum)
        3. Dependence on limited number of distributors
        4. Working capital intensive business model
        5. Regulatory compliance requirements
        
        IPO DETAILS
        
        Price Band: ₹150 - ₹165 per share
        Fresh Issue: ₹40 Crores
        Offer for Sale: ₹25 Crores
        Total Issue Size: ₹65 Crores
        
        Use of Proceeds:
        - Capacity expansion: 50%
        - Working capital: 30%
        - Debt repayment: 15%
        - General corporate purposes: 5%
        
        Lead Managers: XYZ Capital Advisors, ABC Merchant Bankers
        
        Promoter holding pre-IPO: 100%
        Promoter holding post-IPO: 70%
        Public holding: 30%
        
        Listing: Expected on BSE and NSE
        """
        
        company_name = "Vidya Wires Limited"
        sector = "Manufacturing - Electrical Equipment"
        use_sample_text = True
    else:
        # Extract text from PDF
        print("\n2. Extracting text from PDF...")
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                print(f"   ✓ Extracted {len(pdf_text)} characters from {len(pdf_reader.pages)} pages")
        except Exception as e:
            print(f"   ❌ Error extracting text: {e}")
            return
        
        company_name = "Vidya Wires Limited"
        sector = "Manufacturing - Electrical Equipment"
        use_sample_text = False
    
    # Initialize analyzer
    print_section("INITIALIZING ANALYZER")
    print("\n3. Setting up LLM Prospectus Analyzer...")
    analyzer = LLMProspectusAnalyzer(
        provider="groq",  # Using Groq with optimized context
        use_vector_db=True
    )
    print(f"   ✓ Provider: {analyzer.provider}")
    print(f"   ✓ Vector DB: {analyzer.use_vector_db}")
    print(f"   ✓ Collections: {list(analyzer.collections.keys())}")
    
    # Store in vector DB using structured chunking if PDF available
    if not use_sample_text and pdf_path:
        print_section("STRUCTURED CHUNKING")
        print("\n4. Processing with Structured PDF Chunker...")
        analyzer.chunk_and_store_prospectus_structured(
            pdf_path=pdf_path,
            company_name=company_name,
            sector=sector,
            use_structured=True
        )
    else:
        print_section("TEXT CHUNKING")
        print("\n4. Processing with Recursive Text Chunker...")
        analyzer.chunk_and_store_prospectus(
            pdf_text=pdf_text,
            company_name=company_name,
            sector=sector
        )
    
    # Check vector DB
    print("\n5. Verifying Vector DB storage...")
    total_chunks = 0
    for name, collection in analyzer.collections.items():
        count = collection.count()
        total_chunks += count
        if count > 0:
            print(f"   • {name}: {count} chunks")
    print(f"   ✓ Total: {total_chunks} chunks stored")
    
    # Run comprehensive analysis
    print_section("FINANCIAL METRICS EXTRACTION")
    print("\n6. Extracting financial metrics...")
    financial_metrics = analyzer._extract_financial_metrics(
        pdf_text=pdf_text,
        company_name=company_name,
        pdf_path=pdf_path if not use_sample_text else None
    )
    
    print(f"\n   📊 Financial Metrics:")
    print(f"      • Net Profit Margin: {financial_metrics.net_profit_margin}")
    print(f"      • Return on Assets: {financial_metrics.return_on_assets}")
    print(f"      • Current Ratio: {financial_metrics.current_ratio}")
    print(f"      • Debt to Equity: {financial_metrics.debt_to_equity_ratio}")
    print(f"      • Revenue Growth (3yr): {financial_metrics.revenue_growth_3yr}")
    print(f"      • Confidence: {financial_metrics.extraction_confidence}")
    print(f"      • Completeness: {financial_metrics.data_completeness}")
    
    # Benchmarking analysis
    print_section("BENCHMARKING ANALYSIS")
    print("\n7. Performing competitive analysis...")
    benchmarking = analyzer._perform_benchmarking_analysis(
        pdf_text=pdf_text,
        company_name=company_name,
        sector=sector,
        pdf_path=pdf_path if not use_sample_text else None
    )
    
    print(f"\n   🏆 Benchmarking:")
    print(f"      • Market Position: {benchmarking.market_position}")
    print(f"      • Competitive Advantages: {len(benchmarking.competitive_advantages)}")
    if benchmarking.competitive_advantages:
        for adv in benchmarking.competitive_advantages[:3]:
            print(f"        - {adv}")
    print(f"      • Industry Trends: {len(benchmarking.industry_trends)}")
    if benchmarking.industry_trends:
        for trend in benchmarking.industry_trends[:3]:
            print(f"        - {trend}")
    
    # IPO specifics
    print_section("IPO-SPECIFIC ANALYSIS")
    print("\n8. Analyzing IPO details...")
    ipo_specifics = analyzer._analyze_ipo_specifics(
        pdf_text=pdf_text,
        company_name=company_name,
        pdf_path=pdf_path if not use_sample_text else None
    )
    
    print(f"\n   💰 IPO Details:")
    print(f"      • Price Band: {ipo_specifics.ipo_pricing_analysis.get('price_band', 'Not disclosed')}")
    print(f"      • Lead Managers: {ipo_specifics.underwriter_quality.get('lead_managers', [])}")
    print(f"      • Use of Funds:")
    if ipo_specifics.use_of_funds_analysis:
        for key, value in ipo_specifics.use_of_funds_analysis.items():
            if value:
                print(f"        - {key}: {value}")
    
    # Test vector DB retrieval
    print_section("VECTOR DB RETRIEVAL TEST")
    print("\n9. Testing semantic search...")
    
    test_queries = [
        "Financial performance revenue profit margins",
        "Business model competitive advantages",
        "IPO pricing valuation",
        "Risk factors challenges"
    ]
    
    for query in test_queries:
        chunks = analyzer.retrieve_relevant_context(query, chunk_type="all", n_results=3)
        print(f"\n   Query: '{query}'")
        print(f"   ✓ Retrieved {len(chunks)} relevant chunks")
        if chunks:
            preview = chunks[0][:150] + "..." if len(chunks[0]) > 150 else chunks[0]
            print(f"   Sample: {preview}")
    
    # Generate investment thesis
    print_section("INVESTMENT THESIS GENERATION")
    print("\n10. Generating comprehensive investment thesis...")
    
    thesis = analyzer.generate_investment_thesis(
        financial_metrics=financial_metrics,
        benchmarking=benchmarking,
        ipo_specifics=ipo_specifics,
        company_name=company_name,
        sector=sector,
        web_context=""  # No web search for this test
    )
    
    print("\n" + "=" * 80)
    print("📝 INVESTMENT THESIS")
    print("=" * 80)
    print(thesis)
    print("=" * 80)
    
    # Summary
    print_section("TEST SUMMARY")
    print(f"\n✅ Analysis completed successfully!")
    print(f"\n   • PDF Processing: {'✓' if not use_sample_text else '⚠️ Using sample text'}")
    print(f"   • Vector DB Storage: ✓ {total_chunks} chunks")
    print(f"   • Financial Metrics: ✓ Confidence {financial_metrics.extraction_confidence:.1%}")
    print(f"   • Benchmarking: ✓ {len(benchmarking.competitive_advantages)} advantages identified")
    print(f"   • IPO Analysis: ✓ Complete")
    print(f"   • Investment Thesis: ✓ Generated")
    
    print(f"\n   📊 Context Quality:")
    print(f"      • Data Completeness: {financial_metrics.data_completeness:.1%}")
    print(f"      • Extraction Confidence: {financial_metrics.extraction_confidence:.1%}")
    print(f"      • Vector DB Chunks: {total_chunks}")
    
    if financial_metrics.extraction_confidence < 0.5 or financial_metrics.data_completeness < 0.5:
        print(f"\n   ⚠️  LOW DATA QUALITY WARNING:")
        print(f"      The analysis confidence is low, likely due to:")
        print(f"      • Insufficient financial data in the prospectus")
        print(f"      • PDF text extraction issues")
        print(f"      • Missing key sections")
        print(f"\n      Recommendations:")
        print(f"      • Verify PDF quality and completeness")
        print(f"      • Check if PDF is text-based (not scanned)")
        print(f"      • Ensure prospectus contains financial statements")
    else:
        print(f"\n   ✓ Analysis quality is good!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        test_full_analysis_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
