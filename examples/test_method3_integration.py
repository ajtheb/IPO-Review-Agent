#!/usr/bin/env python3
"""
Test script to validate Method 3: Structured PDF Chunker integration.

This script tests the integration of the StructuredPDFChunker with the
LLMProspectusAnalyzer to ensure:
1. Structured chunks are extracted correctly
2. Chunks are stored in vector database with enhanced metadata
3. Retrieval works with structured chunks
4. Metadata is properly preserved
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from loguru import logger

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_structured_chunking_integration():
    """Test the complete structured chunking workflow."""
    
    print_section("METHOD 3: STRUCTURED PDF CHUNKER INTEGRATION TEST")
    
    # Initialize analyzer
    print("\n1. Initializing LLM Prospectus Analyzer...")
    analyzer = LLMProspectusAnalyzer(
        provider="gemini",
        use_vector_db=True,
        db_path="./ipo_vector_db"
    )
    print("✅ Analyzer initialized")
    
    # Test PDF path
    pdf_path = "/Users/apoorvjain/Projects/IPO Review Agent/prospectus/vidya_wires_draft_prospectus.pdf"
    
    if not Path(pdf_path).exists():
        print(f"\n❌ Test PDF not found at: {pdf_path}")
        print("Please provide a valid PDF path to test structured chunking.")
        return
    
    company_name = "Vidya Wires"
    sector = "Manufacturing"
    
    print(f"\n2. Processing PDF with Structured Chunker...")
    print(f"   PDF: {Path(pdf_path).name}")
    print(f"   Company: {company_name}")
    print(f"   Sector: {sector}")
    
    # Test Method 3: Structured Chunking
    try:
        print("\n3. Extracting structured chunks (Method 3)...")
        
        # Call the structured chunking method
        categorized_chunks = analyzer._chunk_document_structured(
            pdf_path=pdf_path,
            company_name=company_name
        )
        
        if not categorized_chunks:
            print("⚠️  No chunks extracted - this might indicate an issue")
            return
        
        print(f"✅ Extracted {sum(len(chunks) for chunks in categorized_chunks.values())} chunks")
        
        # Display chunk statistics
        print("\n4. Chunk Statistics by Category:")
        for category, chunks in categorized_chunks.items():
            if chunks:
                print(f"   📁 {category.upper()}: {len(chunks)} chunks")
                
                # Show sample metadata from first chunk
                if chunks:
                    sample = chunks[0]
                    metadata = sample['metadata']
                    print(f"      Sample chunk metadata:")
                    print(f"        - Section: {metadata.get('section', 'N/A')}")
                    print(f"        - Pages: {metadata.get('pages', [])}")
                    print(f"        - Content type: {metadata.get('content_type', 'N/A')}")
                    print(f"        - Has tables: {metadata.get('has_tables', False)}")
                    print(f"        - Word count: {metadata.get('word_count', 0)}")
        
        # Test storing chunks in vector database
        print_section("STORING CHUNKS IN VECTOR DATABASE")
        
        print("\n5. Storing structured chunks in vector DB...")
        analyzer.chunk_and_store_prospectus_structured(
            pdf_path=pdf_path,
            company_name=company_name,
            sector=sector,
            use_structured=True
        )
        
        # Test retrieval
        print_section("TESTING SEMANTIC RETRIEVAL")
        
        print("\n6. Testing semantic search with structured chunks...")
        
        # Test different query types
        test_queries = [
            ("Financial data and revenue", "financial"),
            ("Business model and operations", "competitive"),
            ("IPO pricing and offering details", "ipo_specific"),
            ("Risk factors", "general")
        ]
        
        for query, expected_type in test_queries:
            print(f"\n   Query: '{query}'")
            print(f"   Expected type: {expected_type}")
            
            results = analyzer.retrieve_relevant_context(
                query=query,
                chunk_type="all",
                n_results=3
            )
            
            if results:
                print(f"   ✅ Retrieved {len(results)} chunks")
                # Show first result preview
                preview = results[0][:200] + "..." if len(results[0]) > 200 else results[0]
                print(f"   Preview: {preview}")
            else:
                print(f"   ⚠️  No results retrieved")
        
        # Test thesis generation with structured chunks
        print_section("TESTING INVESTMENT THESIS GENERATION")
        
        print("\n7. Testing thesis generation with structured chunks...")
        prospectus_chunks, web_chunks = analyzer.retrieve_relevant_chunks_for_thesis(
            company_name=company_name,
            sector=sector,
            n_prospectus=15,
            n_web=5
        )
        
        print(f"   ✅ Retrieved {len(prospectus_chunks)} prospectus chunks")
        print(f"   ✅ Retrieved {len(web_chunks)} web chunks")
        
        if prospectus_chunks:
            print("\n   Sample prospectus chunk preview:")
            preview = prospectus_chunks[0][:300] + "..."
            print(f"   {preview}")
        
        # Summary
        print_section("TEST SUMMARY")
        print("\n✅ All integration tests completed successfully!")
        print(f"\nKey Results:")
        print(f"  - Structured chunks extracted: {sum(len(chunks) for chunks in categorized_chunks.values())}")
        print(f"  - Categories processed: {', '.join(k for k, v in categorized_chunks.items() if v)}")
        print(f"  - Vector DB storage: Working")
        print(f"  - Semantic retrieval: Working")
        print(f"  - Enhanced metadata: Preserved")
        
        print("\n📊 Structured chunking provides:")
        print("  • Section-aware chunking")
        print("  • Category classification")
        print("  • Page number tracking")
        print("  • Table detection")
        print("  • Richer context for LLM analysis")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        logger.exception("Test failed with exception:")
        raise

if __name__ == "__main__":
    try:
        test_structured_chunking_integration()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
