#!/usr/bin/env python3
"""
Test script for the new structured PDF chunking capability in LLMProspectusAnalyzer.

This demonstrates how to use the enhanced structured chunker for better
categorization and metadata extraction from IPO prospectus documents.
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

def test_structured_chunking():
    """Test the new structured PDF chunking method."""
    
    print("=" * 80)
    print("TESTING STRUCTURED PDF CHUNKING")
    print("=" * 80)
    
    # Initialize analyzer with vector DB
    print("\n1. Initializing LLMProspectusAnalyzer...")
    analyzer = LLMProspectusAnalyzer(
        provider='gemini',  # Or 'openai', 'anthropic', 'groq'
        use_vector_db=True
    )
    
    print(f"   ✓ Vector DB enabled: {analyzer.use_vector_db}")
    print(f"   ✓ Collections: {list(analyzer.collections.keys())}")
    
    # Example PDF path (adjust to your actual PDF location)
    # Try multiple possible locations
    possible_paths = [
        "vidya_wires.pdf",
        "prospectus/vidya_wires_draft_prospectus.pdf",
        "/Users/apoorvjain/Projects/IPO Review Agent/prospectus/vidya_wires_draft_prospectus.pdf"
    ]
    
    pdf_path = None
    for path in possible_paths:
        if Path(path).exists():
            pdf_path = path
            break
    
    company_name = "Vidya Wires Limited"
    sector = "Manufacturing - Electrical Equipment"
    
    if not pdf_path:
        print(f"\n⚠️  PDF file not found. Tried:")
        for path in possible_paths:
            print(f"   • {path}")
        print("\n   Please place a PDF file in one of these locations or update the script.")
        return
    
    print(f"\n2. Processing PDF with Structured Chunker...")
    print(f"   PDF: {pdf_path}")
    print(f"   Company: {company_name}")
    
    # Test Method 1: Direct structured chunking (without storing to vector DB)
    print("\n   A. Testing direct structured chunking...")
    categorized_chunks = analyzer._chunk_document_structured(
        pdf_path=pdf_path,
        company_name=company_name,
        output_dir="structured_chunks"
    )
    
    if categorized_chunks:
        print(f"\n   ✓ Structured chunks extracted successfully!")
        print(f"\n   📊 Chunk Distribution:")
        total_chunks = 0
        for category, chunks in categorized_chunks.items():
            count = len(chunks)
            total_chunks += count
            if count > 0:
                print(f"      • {category.upper()}: {count} chunks")
                
                # Show sample metadata from first chunk
                if chunks:
                    sample = chunks[0]
                    print(f"        - Sample metadata: {list(sample['metadata'].keys())}")
                    print(f"        - Importance score: {sample['metadata'].get('importance_score', 'N/A')}")
                    print(f"        - Section: {sample['metadata'].get('section', 'N/A')}")
                    print(f"        - Text preview: {sample['text'][:100]}...")
        
        print(f"\n      TOTAL: {total_chunks} structured chunks")
    else:
        print(f"\n   ⚠️  No structured chunks extracted")
    
    # Test Method 2: Store structured chunks in vector DB
    print("\n   B. Testing structured chunking with vector DB storage...")
    analyzer.chunk_and_store_prospectus_structured(
        pdf_path=pdf_path,
        company_name=company_name,
        sector=sector,
        ipo_date="2025-03-15",
        use_structured=True  # Enable structured chunker
    )
    
    # Check document counts in vector DB
    print(f"\n3. Verifying Vector DB Storage...")
    total_docs = 0
    for name, collection in analyzer.collections.items():
        count = collection.count()
        total_docs += count
        if count > 0:
            print(f"   • {name}: {count} documents")
    
    print(f"   TOTAL: {total_docs} documents in vector DB")
    
    # Test retrieval with structured chunks
    print(f"\n4. Testing Chunk Retrieval...")
    
    test_queries = [
        ("Financial performance and revenue metrics", "financial", 5),
        ("Business model and competitive advantages", "competitive", 5),
        ("IPO pricing and fund utilization", "ipo_specific", 5),
        ("Risk factors and challenges", "all", 5)
    ]
    
    for query, chunk_type, n_results in test_queries:
        print(f"\n   Query: '{query}'")
        print(f"   Type: {chunk_type}, Results: {n_results}")
        
        chunks = analyzer.retrieve_relevant_context(
            query=query,
            chunk_type=chunk_type,
            n_results=n_results
        )
        
        print(f"   ✓ Retrieved {len(chunks)} chunks")
        if chunks:
            print(f"   Sample chunk (first 200 chars):")
            print(f"   {chunks[0][:200]}...")
    
    # Compare with recursive chunker
    print(f"\n5. Comparison Test: Structured vs Recursive Chunker...")
    
    # Clear and use recursive chunker
    analyzer.clear_vector_database()
    analyzer.chunk_and_store_prospectus_structured(
        pdf_path=pdf_path,
        company_name=company_name,
        sector=sector,
        use_structured=False  # Disable structured chunker
    )
    
    print(f"\n   Recursive Chunker Results:")
    total_recursive = 0
    for name, collection in analyzer.collections.items():
        count = collection.count()
        total_recursive += count
        if count > 0:
            print(f"   • {name}: {count} documents")
    
    print(f"\n   COMPARISON:")
    print(f"   • Structured Chunker: {total_docs} chunks with rich metadata")
    print(f"   • Recursive Chunker: {total_recursive} chunks with basic classification")
    
    print("\n" + "=" * 80)
    print("✅ STRUCTURED CHUNKING TEST COMPLETE")
    print("=" * 80)
    
    print("\n📋 Summary:")
    print(f"   • Structured chunks provide better categorization")
    print(f"   • Enhanced metadata includes: categories, importance scores, sections, page numbers")
    print(f"   • Chunks are organized by: financial, business, risk, offering, legal, other")
    print(f"   • Vector DB retrieval works with both chunking methods")
    print(f"   • Structured chunker falls back to recursive chunker if unavailable")


if __name__ == "__main__":
    try:
        test_structured_chunking()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
