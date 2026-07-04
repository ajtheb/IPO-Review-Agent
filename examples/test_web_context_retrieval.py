#!/usr/bin/env python3
"""
Test script to diagnose and verify web context retrieval for investment thesis.

This script checks:
1. If web content is being stored correctly
2. If web content can be retrieved with correct metadata filters
3. The full flow from storage to thesis generation
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

def test_web_context_storage_and_retrieval():
    """Test web content storage and retrieval."""
    
    print("\n" + "="*80)
    print("WEB CONTEXT RETRIEVAL DIAGNOSTIC TEST")
    print("="*80)
    
    # Initialize analyzer with vector DB
    print("\n1. Initializing analyzer...")
    analyzer = LLMProspectusAnalyzer(provider='groq', use_vector_db=True)
    print("   ✓ Analyzer initialized")
    
    # Test data
    company_name = "Test Company Ltd"
    sector = "Technology"
    
    # Simulate web search results
    mock_search_results = [
        {
            "title": "Test Company shows strong growth in Q4 2024",
            "url": "https://example.com/test-company-growth",
            "description": "Test Company reported 25% revenue growth with expanding market share."
        },
        {
            "title": "Industry outlook: Technology sector trends",
            "url": "https://example.com/tech-sector-trends",
            "description": "The technology sector is experiencing robust growth driven by digital transformation."
        }
    ]
    
    # Step 2: Store mock web content directly (without actual scraping)
    print("\n2. Storing mock web content in vector DB...")
    
    try:
        collection = analyzer.collections['prospectus_chunks']
        
        # Create some test web chunks
        test_chunks = [
            {
                "id": f"{company_name}_web_test_0",
                "text": "Test Company reported strong financial performance with revenue growth of 25% year-over-year. The company's market position continues to strengthen with expanded client base.",
                "metadata": {
                    "company": company_name,
                    "sector": sector,
                    "chunk_type": "web_search",  # ✅ Correct field name
                    "source_url": "https://example.com/test-1",
                    "source_title": "Test Company Growth Report"
                }
            },
            {
                "id": f"{company_name}_web_test_1",
                "text": "The technology sector is experiencing rapid growth. Analysts predict continued expansion with focus on AI and cloud services. Test Company is well-positioned to capitalize on these trends.",
                "metadata": {
                    "company": company_name,
                    "sector": sector,
                    "chunk_type": "web_search",  # ✅ Correct field name
                    "source_url": "https://example.com/test-2",
                    "source_title": "Tech Sector Outlook"
                }
            },
            {
                "id": f"{company_name}_web_test_2",
                "text": "Industry experts note that Test Company's competitive advantages include strong R&D capabilities, established customer relationships, and operational efficiency. Risk factors include market competition and regulatory changes.",
                "metadata": {
                    "company": company_name,
                    "sector": sector,
                    "chunk_type": "web_search",  # ✅ Correct field name
                    "source_url": "https://example.com/test-3",
                    "source_title": "Test Company Analysis"
                }
            }
        ]
        
        # Store the test chunks
        for chunk in test_chunks:
            collection.add(
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                ids=[chunk["id"]]
            )
        
        print(f"   ✓ Stored {len(test_chunks)} test web chunks")
        
    except Exception as e:
        print(f"   ✗ Error storing test chunks: {e}")
        return
    
    # Step 3: Verify storage
    print("\n3. Verifying web content storage...")
    try:
        # Query without filter to see all content
        all_results = collection.query(
            query_texts=[f"{company_name} analysis"],
            n_results=10
        )
        
        print(f"   • Total chunks retrieved (no filter): {len(all_results['documents'][0]) if all_results['documents'] else 0}")
        
        # Query with correct filter
        web_results = collection.query(
            query_texts=[f"{company_name} financial performance"],
            n_results=5,
            where={"chunk_type": "web_search"}  # ✅ Using correct filter
        )
        
        web_count = len(web_results['documents'][0]) if web_results and web_results['documents'] else 0
        print(f"   • Web chunks retrieved (with filter): {web_count}")
        
        if web_count > 0:
            print(f"   ✓ Web content storage and retrieval WORKING")
            print(f"\n   Sample web chunk:")
            print(f"   {web_results['documents'][0][0][:150]}...")
        else:
            print(f"   ✗ Web content retrieval FAILED - no chunks found with filter")
            
            # Debug: Check metadata
            if all_results and 'metadatas' in all_results and all_results['metadatas']:
                print(f"\n   Debugging metadata from stored chunks:")
                for i, metadata_list in enumerate(all_results['metadatas'][:3]):
                    for metadata in metadata_list:
                        print(f"   Chunk {i}: {metadata}")
        
    except Exception as e:
        print(f"   ✗ Error verifying storage: {e}")
        return
    
    # Step 4: Test thesis retrieval method
    print("\n4. Testing retrieve_relevant_chunks_for_thesis method...")
    try:
        prospectus_chunks, web_chunks = analyzer.retrieve_relevant_chunks_for_thesis(
            company_name=company_name,
            sector=sector,
            n_prospectus=10,
            n_web=10
        )
        
        print(f"   • Prospectus chunks: {len(prospectus_chunks)}")
        print(f"   • Web chunks: {len(web_chunks)}")
        
        if web_chunks:
            print(f"   ✓ Web context retrieval for thesis WORKING")
            print(f"\n   Sample web chunk from thesis retrieval:")
            print(f"   {web_chunks[0][:150]}...")
        else:
            print(f"   ✗ Web context retrieval for thesis FAILED")
            print(f"   This indicates the retrieve_relevant_chunks_for_thesis method")
            print(f"   is not using the correct metadata filter")
        
    except Exception as e:
        print(f"   ✗ Error testing thesis retrieval: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if web_count > 0 and web_chunks:
        print("\n✅ SUCCESS: Web context retrieval is working correctly!")
        print("\n   The fix has resolved the metadata mismatch issue:")
        print("   • Storage uses: chunk_type='web_search'")
        print("   • Retrieval uses: where={'chunk_type': 'web_search'}")
        print("\n   Web context will now be included in investment thesis generation.")
    elif web_count > 0 and not web_chunks:
        print("\n⚠️  PARTIAL SUCCESS:")
        print("   • Direct query with filter works ✓")
        print("   • Thesis retrieval method fails ✗")
        print("\n   Check retrieve_relevant_chunks_for_thesis method for correct filter usage")
    else:
        print("\n❌ FAILURE: Web context retrieval not working")
        print("\n   Possible issues:")
        print("   • Metadata field name mismatch")
        print("   • Vector DB collection issue")
        print("   • Filter syntax error")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        test_web_context_storage_and_retrieval()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
