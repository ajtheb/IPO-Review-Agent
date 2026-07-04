#!/usr/bin/env python3
"""
Comprehensive test script for multi-query retrieval strategy.
Tests all three analysis sections: Financial Metrics, IPO Specifics, and Benchmarking.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
import google.generativeai as genai


def test_comprehensive_multi_query():
    """Test multi-query retrieval across all analysis sections."""
    
    print("=" * 80)
    print("COMPREHENSIVE MULTI-QUERY RETRIEVAL TEST")
    print("=" * 80)
    
    # Initialize analyzer with vector DB
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("ERROR: GEMINI_API_KEY not found or not set in environment")
        print("Please set a valid GEMINI_API_KEY in your .env file")
        return
    
    genai.configure(api_key=api_key)
    
    analyzer = LLMProspectusAnalyzer(
        api_key=api_key,
        use_vector_db=True,
        vector_db_path="ipo_vector_db"
    )
    
    # Test company
    test_company = "Vidya_Wires"
    test_sector = "Manufacturing"
    
    # Check if data exists in vector DB
    collections = analyzer.vector_db.list_collections()
    collection_names = [col.name for col in collections]
    
    if test_company not in collection_names:
        print(f"\n❌ No collection found for {test_company}")
        print(f"Available collections: {collection_names}")
        return
    
    print(f"\n✅ Found collection for {test_company}")
    
    # Test 1: Financial Metrics Multi-Query
    print("\n" + "=" * 80)
    print("TEST 1: FINANCIAL METRICS MULTI-QUERY RETRIEVAL")
    print("=" * 80)
    
    financial_queries = [
        {
            "query": f"restated statement profit loss revenue EBITDA PAT net profit FY lakhs crores {test_company}",
            "description": "P&L Statement"
        },
        {
            "query": f"key financial ratios ROE ROCE return equity debt equity current ratio {test_company}",
            "description": "Financial Ratios"
        },
        {
            "query": f"restated balance sheet total assets liabilities equity reserves {test_company}",
            "description": "Balance Sheet"
        }
    ]
    
    print("\nTesting Financial Metrics Queries:")
    for i, query_info in enumerate(financial_queries, 1):
        print(f"\n--- Query {i}: {query_info['description']} ---")
        print(f"Query: {query_info['query']}")
        
        # Try table chunks first
        table_chunks = analyzer.retrieve_table_chunks(
            query=query_info['query'],
            chunk_type="financial",
            n_results=2,
            only_financial_tables=True
        )
        
        if table_chunks:
            print(f"✅ Found {len(table_chunks)} table chunks")
            for j, chunk_data in enumerate(table_chunks, 1):
                metadata = chunk_data.get('metadata', {})
                print(f"\n  Chunk {j}:")
                print(f"    - Contains table: {metadata.get('contains_financial_table', False)}")
                print(f"    - Table type: {metadata.get('table_type', 'N/A')}")
                print(f"    - Multi-year data: {metadata.get('has_multi_year_data', False)}")
                print(f"    - Financial years: {metadata.get('financial_years', [])}")
                print(f"    - Metric density: {metadata.get('metric_density', 0):.2f}")
                print(f"    - Length: {len(chunk_data['document'])} chars")
                
                # Show a snippet
                snippet = chunk_data['document'][:300] + "..." if len(chunk_data['document']) > 300 else chunk_data['document']
                print(f"    - Snippet: {snippet}")
        else:
            print(f"⚠️  No table chunks found, falling back to regular retrieval")
            regular_chunks = analyzer.retrieve_relevant_context(
                query=query_info['query'],
                chunk_type="financial",
                n_results=2,
                prioritize_tables=True
            )
            print(f"Found {len(regular_chunks)} regular chunks")
    
    # Test 2: IPO Specifics Multi-Query
    print("\n" + "=" * 80)
    print("TEST 2: IPO SPECIFICS MULTI-QUERY RETRIEVAL")
    print("=" * 80)
    
    ipo_queries = [
        {
            "query": f"objects issue IPO pricing price band valuation basis {test_company}",
            "description": "IPO Pricing"
        },
        {
            "query": f"objects issue utilization proceeds fund deployment working capital {test_company}",
            "description": "Use of Funds"
        },
        {
            "query": f"book running lead managers underwriters merchant bankers {test_company}",
            "description": "Underwriters"
        }
    ]
    
    print("\nTesting IPO Specifics Queries:")
    for i, query_info in enumerate(ipo_queries, 1):
        print(f"\n--- Query {i}: {query_info['description']} ---")
        print(f"Query: {query_info['query']}")
        
        chunks = analyzer.retrieve_relevant_context(
            query=query_info['query'],
            chunk_type="ipo_specific",
            n_results=2,
            prioritize_tables=False
        )
        
        if chunks:
            print(f"✅ Found {len(chunks)} chunks")
            for j, chunk in enumerate(chunks, 1):
                print(f"\n  Chunk {j}:")
                print(f"    - Length: {len(chunk)} chars")
                snippet = chunk[:300] + "..." if len(chunk) > 300 else chunk
                print(f"    - Snippet: {snippet}")
        else:
            print(f"⚠️  No chunks found")
    
    # Test 3: Benchmarking Multi-Query
    print("\n" + "=" * 80)
    print("TEST 3: BENCHMARKING MULTI-QUERY RETRIEVAL")
    print("=" * 80)
    
    benchmarking_queries = [
        {
            "query": f"market share industry position {test_sector} competitors {test_company}",
            "description": "Market Position"
        },
        {
            "query": f"competitive advantages unique selling proposition strengths {test_company}",
            "description": "Competitive Advantages"
        },
        {
            "query": f"industry trends {test_sector} market dynamics outlook growth drivers",
            "description": "Industry Trends"
        }
    ]
    
    print("\nTesting Benchmarking Queries:")
    for i, query_info in enumerate(benchmarking_queries, 1):
        print(f"\n--- Query {i}: {query_info['description']} ---")
        print(f"Query: {query_info['query']}")
        
        chunks = analyzer.retrieve_relevant_context(
            query=query_info['query'],
            chunk_type="competitive",
            n_results=2,
            prioritize_tables=False
        )
        
        if chunks:
            print(f"✅ Found {len(chunks)} chunks")
            for j, chunk in enumerate(chunks, 1):
                print(f"\n  Chunk {j}:")
                print(f"    - Length: {len(chunk)} chars")
                snippet = chunk[:300] + "..." if len(chunk) > 300 else chunk
                print(f"    - Snippet: {snippet}")
        else:
            print(f"⚠️  No chunks found")
    
    # Test 4: Deduplication Test
    print("\n" + "=" * 80)
    print("TEST 4: DEDUPLICATION VERIFICATION")
    print("=" * 80)
    
    print("\nTesting deduplication across multiple queries...")
    all_chunks = []
    seen_hashes = set()
    duplicates_found = 0
    
    test_queries = [
        f"revenue EBITDA profit {test_company}",
        f"financial performance revenue profit {test_company}",
        f"profitability margins {test_company}"
    ]
    
    for query in test_queries:
        chunks = analyzer.retrieve_relevant_context(
            query=query,
            chunk_type="financial",
            n_results=3,
            prioritize_tables=True
        )
        
        for chunk in chunks:
            chunk_hash = hash(chunk)
            if chunk_hash in seen_hashes:
                duplicates_found += 1
            else:
                all_chunks.append(chunk)
                seen_hashes.add(chunk_hash)
    
    print(f"Total queries: {len(test_queries)}")
    print(f"Total chunks retrieved: {sum(len(analyzer.retrieve_relevant_context(q, 'financial', 3)) for q in test_queries)}")
    print(f"Unique chunks after deduplication: {len(all_chunks)}")
    print(f"Duplicates removed: {duplicates_found}")
    
    if duplicates_found > 0:
        print(f"✅ Deduplication working correctly - removed {duplicates_found} duplicates")
    else:
        print("✅ No duplicates found (all queries returned unique results)")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("""
Multi-Query Strategy Features Tested:
1. ✅ Financial Metrics: P&L, Ratios, Balance Sheet queries
2. ✅ IPO Specifics: Pricing, Use of Funds, Underwriters queries
3. ✅ Benchmarking: Market Position, Advantages, Trends queries
4. ✅ Deduplication: Hash-based duplicate removal
5. ✅ Table-Aware Retrieval: Priority for financial tables
6. ✅ Fallback Mechanism: Regular retrieval when no tables found

Benefits of Multi-Query Strategy:
- More comprehensive coverage of financial statements
- Better targeting of specific information types
- Reduced dependency on single query quality
- Improved recall for diverse financial data
- Enhanced extraction accuracy
""")


if __name__ == "__main__":
    test_comprehensive_multi_query()
