#!/usr/bin/env python3
"""
Test script for table-aware chunking functionality.
Tests the enhanced metadata detection for financial tables.
"""

import os
import sys
from pathlib import Path
import json
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from src.data_sources.prospectus_parser import ProspectusParser

def test_table_detection():
    """Test table detection on various text samples."""
    
    analyzer = LLMProspectusAnalyzer(provider="gemini", use_vector_db=False)
    
    # Sample text with financial table
    financial_table_sample = """
    Financial Performance (₹ in Crores)
    
    Particulars                 FY2023    FY2022    FY2021
    Revenue from Operations     1,234.56   987.23    756.89
    EBITDA                       234.56   189.45    145.23
    Net Profit                   123.45    98.76     67.89
    EPS (₹)                       12.50    10.20      8.50
    """
    
    # Sample text without table
    plain_text_sample = """
    The company operates in the manufacturing sector and has shown
    consistent growth over the past three years. Revenue has increased
    from 756.89 crores to 1,234.56 crores.
    """
    
    # Sample with multi-year data but no table
    text_with_years = """
    In FY2023, the company achieved revenue of 1,234.56 crores.
    In FY2022, revenue was 987.23 crores.
    In FY2021, revenue was 756.89 crores.
    """
    
    print("\n" + "="*70)
    print("TEST 1: Financial Table Sample")
    print("="*70)
    table_info = analyzer._detect_table_structure(financial_table_sample)
    metric_density = analyzer._calculate_metric_density(financial_table_sample)
    print(f"Contains Table: {table_info['contains_table']}")
    print(f"Table Type: {table_info['table_type']}")
    print(f"Row Count: {table_info['row_count']}")
    print(f"Column Count: {table_info['column_count']}")
    print(f"Has Multi-Year Data: {table_info['has_multi_year_data']}")
    print(f"Financial Years: {table_info['financial_years']}")
    print(f"Metric Density: {metric_density:.3f}")
    
    print("\n" + "="*70)
    print("TEST 2: Plain Text Sample")
    print("="*70)
    table_info = analyzer._detect_table_structure(plain_text_sample)
    metric_density = analyzer._calculate_metric_density(plain_text_sample)
    print(f"Contains Table: {table_info['contains_table']}")
    print(f"Table Type: {table_info['table_type']}")
    print(f"Has Multi-Year Data: {table_info['has_multi_year_data']}")
    print(f"Metric Density: {metric_density:.3f}")
    
    print("\n" + "="*70)
    print("TEST 3: Text with Years (No Table)")
    print("="*70)
    table_info = analyzer._detect_table_structure(text_with_years)
    metric_density = analyzer._calculate_metric_density(text_with_years)
    print(f"Contains Table: {table_info['contains_table']}")
    print(f"Table Type: {table_info['table_type']}")
    print(f"Has Multi-Year Data: {table_info['has_multi_year_data']}")
    print(f"Financial Years: {table_info['financial_years']}")
    print(f"Metric Density: {metric_density:.3f}")


def test_chunking_with_sample_prospectus():
    """Test table-aware chunking with a sample prospectus text."""
    
    print("\n" + "="*70)
    print("TESTING TABLE-AWARE CHUNKING WITH SAMPLE TEXT")
    print("="*70)
    
    # Initialize analyzer with vector DB
    analyzer = LLMProspectusAnalyzer(provider="gemini", use_vector_db=True)
    
    # Create a sample prospectus text with financial tables
    sample_prospectus = """
    VIDYA WIRES LIMITED
    IPO PROSPECTUS
    
    Company Overview:
    Vidya Wires Limited is engaged in the business of manufacturing electrical wires and cables.
    The company has been in operation since 1985 and has established itself as a leading player
    in the wire manufacturing industry in India.
    
    FINANCIAL PERFORMANCE
    
    Statement of Profit and Loss (₹ in Lakhs)
    
    Particulars                         FY2023      FY2022      FY2021
    Revenue from Operations            11,884.89    9,456.78    7,823.45
    Other Income                          156.23      134.56      112.34
    Total Income                       12,041.12    9,591.34    7,935.79
    
    Cost of Materials                   8,234.56    6,789.45    5,678.90
    Employee Benefits                   1,456.78    1,234.56    1,098.76
    Other Expenses                      1,234.56    1,098.76      987.65
    Total Expenses                     10,925.90    9,122.77    7,765.31
    
    EBITDA                              1,115.22      468.57      170.48
    Depreciation                          234.56      198.76      156.78
    Finance Costs                          87.65       76.54       65.43
    Profit Before Tax (PBT)               793.01      193.27      -51.73
    Tax Expense                           256.93       62.48        0.00
    Profit After Tax (PAT)                536.08      130.79      -51.73
    
    BALANCE SHEET
    
    Assets (₹ in Lakhs)
    
    Particulars                         FY2023      FY2022      FY2021
    Non-Current Assets
      Property, Plant & Equipment        3,456.78    3,123.45    2,890.12
      Intangible Assets                     23.45       34.56       45.67
      Investments                          156.78      145.67      134.56
    Total Non-Current Assets             3,636.01    3,303.68    3,070.35
    
    Current Assets
      Inventories                        2,345.67    1,987.65    1,678.90
      Trade Receivables                  1,987.65    1,678.90    1,456.78
      Cash and Bank Balances               567.89      456.78      378.90
      Other Current Assets                 456.78      389.12      334.56
    Total Current Assets                 5,357.99    4,512.45    3,849.14
    
    Total Assets                         8,993.00    7,816.13    6,919.49
    
    KEY FINANCIAL RATIOS
    
    Ratio                               FY2023      FY2022      FY2021
    EBITDA Margin (%)                      9.38        4.96        2.18
    Net Profit Margin (%)                  4.51        1.38       -0.66
    Return on Equity (%)                  18.45        5.67       -2.34
    Return on Assets (%)                   5.96        1.67       -0.75
    Debt-to-Equity Ratio                   0.45        0.67        0.89
    Current Ratio                          1.78        1.65        1.52
    Interest Coverage Ratio                9.05        2.52        N/A
    
    BUSINESS MODEL AND COMPETITIVE POSITION
    
    The company operates in the organized wire and cable manufacturing segment.
    Key competitors include KEI Industries, Polycab, and Finolex Cables.
    The company has a 3.5% market share in the electrical wires segment.
    
    Our competitive advantages include:
    - Modern manufacturing facilities
    - Strong distribution network across 15 states
    - Focus on quality and customer service
    - Experienced management team
    
    IPO DETAILS
    
    Issue Size: ₹200 Crores
    Price Band: ₹800-850 per share
    Lot Size: 17 shares
    Issue Opens: January 22, 2025
    Issue Closes: January 24, 2025
    Listing Date: January 30, 2025
    
    Use of Proceeds:
    - Working Capital: ₹100 Crores (50%)
    - Debt Repayment: ₹60 Crores (30%)
    - General Corporate Purposes: ₹40 Crores (20%)
    """
    
    print(f"\n📄 Sample prospectus length: {len(sample_prospectus)} characters")
    
    # Chunk and store with table-aware metadata
    print("\n🔍 Chunking with table-aware detection...")
    analyzer.chunk_and_store_prospectus(
        pdf_text=sample_prospectus,
        company_name="Vidya Wires",
        sector="Manufacturing",
        ipo_date="2025-01-22"
    )
    
    print("\n" + "="*70)
    print("TESTING RETRIEVAL WITH TABLE PRIORITIZATION")
    print("="*70)
    
    # Test 1: Retrieve with table prioritization (default)
    print("\n1️⃣ Retrieving with table prioritization enabled...")
    query = "financial data revenue profit EBITDA ratios Vidya Wires"
    chunks_with_priority = analyzer.retrieve_relevant_context(
        query=query,
        chunk_type="financial",
        n_results=3,
        prioritize_tables=True
    )
    
    print(f"\n✅ Retrieved {len(chunks_with_priority)} chunks with table prioritization")
    for i, chunk in enumerate(chunks_with_priority[:2], 1):
        preview = chunk[:300].replace('\n', ' ')
        print(f"\n--- Chunk {i} Preview (first 300 chars) ---")
        print(preview + "...")
    
    # Test 2: Retrieve table chunks specifically
    print("\n2️⃣ Retrieving table chunks specifically...")
    table_chunks = analyzer.retrieve_table_chunks(
        query=query,
        chunk_type="financial",
        n_results=3,
        only_financial_tables=True
    )
    
    print(f"\n✅ Retrieved {len(table_chunks)} financial table chunks")
    for i, chunk_data in enumerate(table_chunks, 1):
        meta = chunk_data['metadata']
        print(f"\n--- Table Chunk {i} ---")
        print(f"  Score: {chunk_data['score']:.1f}")
        print(f"  Table Type: {meta.get('table_type', 'N/A')}")
        print(f"  Row Count: {meta.get('table_row_count', 0)}")
        print(f"  Column Count: {meta.get('table_column_count', 0)}")
        print(f"  Multi-Year: {meta.get('has_multi_year_data', False)}")
        print(f"  Years: {meta.get('financial_years', 'N/A')}")
        print(f"  Metric Density: {meta.get('metric_density', 0.0):.3f}")
        preview = chunk_data['document'][:200].replace('\n', ' ')
        print(f"  Preview: {preview}...")
    
    print("\n" + "="*70)
    print("✅ TABLE-AWARE CHUNKING TEST COMPLETE!")
    print("="*70)
    print(f"\n📂 Check context_chunks/Vidya_Wires/ for saved chunk details")
    print(f"🔍 The chunks should now prioritize financial tables with multi-year data")




def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TABLE-AWARE CHUNKING TEST SUITE")
    print("="*70)
    
    # Test 1: Basic table detection
    try:
        test_table_detection()
    except Exception as e:
        logger.error(f"Table detection test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Sample prospectus analysis with table awareness
    try:
        test_chunking_with_sample_prospectus()
    except Exception as e:
        logger.error(f"Sample prospectus test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
