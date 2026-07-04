#!/usr/bin/env python3
"""
Test script for multi-query retrieval strategy.
Validates that specialized queries retrieve actual financial tables.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from src.data_sources.prospectus_parser import ProspectusParser


def create_sample_prospectus_with_tables():
    """Create a comprehensive sample prospectus with multiple financial tables."""
    
    return """
    VIDYA WIRES LIMITED
    DRAFT RED HERRING PROSPECTUS
    
    RESTATED STATEMENT OF PROFIT AND LOSS
    (₹ in Lakhs)
    
    Particulars                           FY 2024      FY 2023      FY 2022
    Revenue from Operations              11,884.89     9,456.78     7,823.45
    Other Income                            156.23       134.56       112.34
    Total Income                         12,041.12     9,591.34     7,935.79
    
    Cost of Materials Consumed            8,234.56     6,789.45     5,678.90
    Employee Benefits Expense             1,456.78     1,234.56     1,098.76
    Other Expenses                        1,234.56     1,098.76       987.65
    Total Expenses                       10,925.90     9,122.77     7,765.31
    
    Earnings Before Interest, Tax,
    Depreciation and Amortization (EBITDA) 1,115.22      468.57       170.48
    
    Depreciation and Amortization           223.45       189.23       156.78
    Finance Costs                           145.67       123.45       98.76
    Profit Before Tax (PBT)                 746.10       155.89      -85.06
    
    Tax Expense                             210.02        25.10        -3.21
    Profit After Tax (PAT)                  536.08       130.79      -51.73
    
    RESTATED BALANCE SHEET
    (₹ in Lakhs as at March 31)
    
    Particulars                           FY 2024      FY 2023      FY 2022
    
    ASSETS
    Non-Current Assets
    Property, Plant and Equipment         3,456.78     3,123.45     2,890.12
    Capital Work-in-Progress                234.56       189.23       145.67
    Intangible Assets                        23.45        34.56        45.67
    Financial Assets - Investments          156.78       145.67       134.56
    Other Non-Current Assets                123.45       112.34       101.23
    Total Non-Current Assets              3,995.02     3,605.25     3,317.25
    
    Current Assets
    Inventories                           2,567.89     2,234.56     1,987.65
    Trade Receivables                     1,987.65     1,765.43     1,543.21
    Cash and Cash Equivalents               678.90       543.21       432.10
    Other Current Assets                    456.78       387.65       298.76
    Total Current Assets                  5,691.22     4,930.85     4,261.72
    
    TOTAL ASSETS                          9,686.24     8,536.10     7,578.97
    
    EQUITY AND LIABILITIES
    Equity Share Capital                    320.00       320.00       320.00
    Other Equity                          3,245.67     2,876.54     2,543.21
    Total Equity                          3,565.67     3,196.54     2,863.21
    
    Non-Current Liabilities
    Borrowings                            1,234.56     1,456.78     1,678.90
    Provisions                               78.90        67.89        56.78
    Total Non-Current Liabilities         1,313.46     1,524.67     1,735.68
    
    Current Liabilities
    Borrowings                            2,345.67     2,123.45     1,987.65
    Trade Payables                        1,987.65     1,432.10     1,234.56
    Other Current Liabilities               473.79       259.34       -242.13
    Total Current Liabilities             4,807.11     3,814.89     2,980.08
    
    TOTAL EQUITY AND LIABILITIES          9,686.24     8,536.10     7,578.97
    
    KEY FINANCIAL RATIOS
    
    Ratio                                 FY 2024      FY 2023      FY 2022
    
    Profitability Ratios:
    EBITDA Margin (%)                        9.39         4.95         2.18
    PAT Margin (%)                           4.51         1.38        -0.66
    Return on Equity (ROE) (%)              15.04         4.09        -1.81
    Return on Capital Employed (ROCE) (%)   12.78         3.87        -1.23
    
    Liquidity Ratios:
    Current Ratio                            1.18         1.29         1.43
    Quick Ratio                              0.65         0.71         0.76
    
    Leverage Ratios:
    Debt to Equity Ratio                     1.00         1.12         1.28
    Interest Coverage Ratio                  5.12         1.26        -0.86
    
    Efficiency Ratios:
    Asset Turnover Ratio                     1.23         1.11         1.03
    Inventory Turnover (Days)                78.5         86.3         93.1
    Trade Receivables Turnover (Days)        61.0         68.2         72.1
    Trade Payables Turnover (Days)           88.2         57.6         79.5
    
    PRODUCT-WISE REVENUE BREAKDOWN
    (₹ in Lakhs)
    
    Product Category                      FY 2024      FY 2023      FY 2022
    
    Copper Products:
    Enamelled Copper Winding Wires        2,443.33     2,434.47     2,022.98
    Paper Covered Copper Conductors       3,358.31     3,332.67     2,888.45
    Bare Copper Wire/Strips               5,160.77     3,738.10     3,707.90
    PV Ribbon (Solar Application)           224.42       206.20        87.20
    Total Copper Products                11,186.82     9,711.44     8,706.53
    
    Aluminum Products:
    Paper Covered Aluminum Strips           290.21       258.09       218.41
    Total Aluminum Products                 290.21       258.09       218.41
    
    Others (Scrap/By-products)              332.05       115.82        67.89
    
    Total Revenue from Operations        11,884.89     9,456.78     7,823.45
    
    SEGMENT-WISE PERFORMANCE
    
    Revenue Contribution (%)              FY 2024      FY 2023      FY 2022
    Copper Products                         94.32        95.90        93.52
    Aluminum Products                        2.44         2.73         2.79
    Others                                   2.79         1.22         0.87
    
    CAPACITY UTILIZATION
    
    Plant                                 FY 2024      FY 2023      FY 2022
    Anand Unit 1 (%)                        87.5         82.3         76.8
    Anand Unit 2 (%)                        92.1         88.7         83.5
    Overall Capacity Utilization (%)        89.8         85.5         80.2
    """


def test_multi_query_retrieval():
    """Test the multi-query retrieval strategy."""
    
    print("\n" + "="*70)
    print("MULTI-QUERY RETRIEVAL TEST")
    print("="*70)
    
    # Initialize analyzer
    analyzer = LLMProspectusAnalyzer(provider="gemini", use_vector_db=True)
    
    # Create sample prospectus
    print("\n📄 Creating sample prospectus with comprehensive financial tables...")
    sample_text = create_sample_prospectus_with_tables()
    print(f"✅ Sample prospectus: {len(sample_text)} characters")
    
    # Chunk and store
    print("\n🔍 Chunking and storing with table-aware detection...")
    analyzer.chunk_and_store_prospectus(
        pdf_text=sample_text,
        company_name="Vidya Wires",
        sector="Manufacturing",
        ipo_date="2025-01-22"
    )
    
    print("\n" + "="*70)
    print("TESTING MULTI-QUERY STRATEGY")
    print("="*70)
    
    # Manually test the multi-query logic
    company_name = "Vidya Wires"
    
    specialized_queries = [
        {
            "query": f"restated statement profit loss revenue EBITDA PAT net profit FY lakhs crores {company_name}",
            "description": "P&L Statement",
            "n_results": 2
        },
        {
            "query": f"key financial ratios ROE ROCE return equity debt equity current ratio {company_name}",
            "description": "Financial Ratios",
            "n_results": 2
        },
        {
            "query": f"restated balance sheet total assets liabilities equity reserves {company_name}",
            "description": "Balance Sheet",
            "n_results": 2
        }
    ]
    
    all_chunks = []
    seen_hashes = set()
    
    for query_info in specialized_queries:
        query = query_info["query"]
        description = query_info["description"]
        n_results = query_info["n_results"]
        
        print(f"\n🔎 Query {description}:")
        print(f"   Query: {query[:80]}...")
        
        # Try table chunks first
        table_chunks = analyzer.retrieve_table_chunks(
            query=query,
            chunk_type="financial",
            n_results=n_results,
            only_financial_tables=True
        )
        
        if table_chunks:
            print(f"   ✅ Found {len(table_chunks)} table chunks")
            for i, chunk_data in enumerate(table_chunks, 1):
                chunk_text = chunk_data['document']
                chunk_hash = hash(chunk_text)
                
                if chunk_hash not in seen_hashes:
                    all_chunks.append((chunk_text, description, chunk_data['metadata']))
                    seen_hashes.add(chunk_hash)
                    
                    # Show preview
                    preview = chunk_text[:150].replace('\n', ' ')
                    print(f"      Chunk {i}: {preview}...")
                else:
                    print(f"      Chunk {i}: (duplicate, skipped)")
        else:
            print(f"   ⚠️  No table chunks found for {description}")
    
    print("\n" + "="*70)
    print("RETRIEVAL SUMMARY")
    print("="*70)
    print(f"\n📊 Total unique chunks retrieved: {len(all_chunks)}")
    
    # Analyze what was retrieved
    chunk_types = {}
    for chunk_text, description, metadata in all_chunks:
        chunk_types[description] = chunk_types.get(description, 0) + 1
    
    print("\n📈 Chunks by query type:")
    for desc, count in chunk_types.items():
        print(f"   {desc}: {count} chunks")
    
    print("\n🎯 Content Analysis:")
    for i, (chunk_text, description, metadata) in enumerate(all_chunks, 1):
        print(f"\n--- Chunk {i} ({description}) ---")
        print(f"Contains Financial Table: {metadata.get('contains_financial_table', False)}")
        print(f"Multi-Year Data: {metadata.get('has_multi_year_data', False)}")
        print(f"Financial Years: {metadata.get('financial_years', 'N/A')}")
        print(f"Metric Density: {metadata.get('metric_density', 0.0):.3f}")
        print(f"Table Rows: {metadata.get('table_row_count', 0)}")
        
        # Check for specific financial data
        has_revenue = 'revenue' in chunk_text.lower()
        has_ebitda = 'ebitda' in chunk_text.lower()
        has_pat = 'pat' in chunk_text.lower() or 'profit after tax' in chunk_text.lower()
        has_roe = 'roe' in chunk_text.lower() or 'return on equity' in chunk_text.lower()
        has_assets = 'assets' in chunk_text.lower()
        has_liabilities = 'liabilities' in chunk_text.lower()
        
        print(f"Contains: ", end="")
        found_items = []
        if has_revenue: found_items.append("Revenue")
        if has_ebitda: found_items.append("EBITDA")
        if has_pat: found_items.append("PAT")
        if has_roe: found_items.append("ROE")
        if has_assets: found_items.append("Assets")
        if has_liabilities: found_items.append("Liabilities")
        print(", ".join(found_items) if found_items else "None detected")
        
        # Show snippet
        lines = chunk_text.split('\n')
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        print(f"First 3 lines: {' | '.join(non_empty_lines[:3])}")
    
    print("\n" + "="*70)
    print("✅ MULTI-QUERY RETRIEVAL TEST COMPLETE")
    print("="*70)
    
    # Check for saved file
    saved_file = f"context_chunks/Vidya_Wires/retrieved_financial_chunks_*.txt"
    print(f"\n💾 Check saved results: {saved_file}")
    
    return len(all_chunks) > 0


def main():
    """Run the test."""
    try:
        success = test_multi_query_retrieval()
        if success:
            print("\n✅ Test PASSED - Retrieved financial chunks successfully")
        else:
            print("\n❌ Test FAILED - No chunks retrieved")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
