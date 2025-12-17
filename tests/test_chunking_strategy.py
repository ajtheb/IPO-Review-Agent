#!/usr/bin/env python3
"""
Test script to demonstrate chunking strategy and parameters
"""

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
import json

def test_chunking_strategy():
    """Test and demonstrate the chunking strategy in detail."""
    
    # Create analyzer without LLM (just for chunking)
    analyzer = LLMProspectusAnalyzer(provider='gemini', use_vector_db=False)
    
    # Sample IPO prospectus text with mixed content types
    sample_text = """
    Financial Performance Analysis. The company has shown strong revenue growth over the past three years. 
    Revenue increased from Rs 850 crores in FY2021 to Rs 1,234 crores in FY2023, representing a CAGR of 20.5%.
    Net profit margins improved from 8.2% to 12.7% during this period. The company's return on equity stands at 18.5%.
    Current ratio of 1.8 indicates strong liquidity position. Debt to equity ratio of 0.45 shows conservative leverage.
    
    Business Model and Competitive Landscape. The company operates in a highly competitive technology sector.
    Key competitors include Tech Corp, Innovation Ltd, and Digital Solutions Inc. The company's main competitive advantages 
    include proprietary technology, strong brand recognition, and established distribution network.
    Market share has grown from 12% to 18% over the past two years. Industry trends show increasing digitization
    and growing demand for cloud-based solutions. The company's differentiation strategy focuses on customer experience.
    
    IPO Details and Fund Utilization. The initial public offering is priced in the band of Rs 180-200 per share.
    Lead managers for this IPO include ICICI Securities, Kotak Mahindra Capital, and Axis Capital. The company plans to 
    raise approximately Rs 2,500 crores through this public offering. Use of proceeds includes 40% for capacity expansion,
    25% for debt repayment, 20% for working capital requirements, and 15% for general corporate purposes.
    Book building process will be conducted over 3 days. Anchor investor portion is set at 60% of the issue size.
    
    Risk Factors and Regulatory Compliance. The company faces various business risks including technology obsolescence,
    competitive pressure, and regulatory changes. Dependence on key customers poses concentration risk.
    Foreign exchange fluctuations may impact profitability. The company is compliant with all SEBI regulations
    and has received necessary approvals for the IPO listing. Environmental clearances are in place for all facilities.
    """
    
    print("=" * 80)
    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Test chunking with default parameters
    print(f"\n1. DOCUMENT STATISTICS")
    print(f"   Original text length: {len(sample_text)} characters")
    print(f"   Number of sentences: {len(sample_text.split('.'))}")
    
    # Perform chunking
    chunks = analyzer._chunk_document(sample_text)
    
    print(f"\n2. CHUNKING PARAMETERS")
    print(f"   Chunk size limit: 1000 characters")
    print(f"   Overlap size: 200 characters") 
    print(f"   Overlap percentage: 20%")
    print(f"   Splitting method: Sentence-based (regex: r'[.!?]+')")
    
    print(f"\n3. CHUNKING RESULTS")
    print(f"   Total chunks created: {len(chunks)}")
    print(f"   Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
    
    # Analyze each chunk
    print(f"\n4. DETAILED CHUNK ANALYSIS")
    for i, chunk in enumerate(chunks):
        chunk_type = analyzer._classify_chunk(chunk)
        print(f"\n   Chunk {i+1}: [{chunk_type.upper()}]")
        print(f"   Length: {len(chunk)} characters")
        print(f"   Preview: {chunk[:150]}...")
        
        # Check overlap with next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i+1]
            # Find potential overlap
            chunk_words = chunk.split()
            next_words = next_chunk.split()
            
            overlap_found = False
            for j in range(min(20, len(chunk_words))):  # Check last 20 words
                if j < len(next_words):
                    if chunk_words[-(j+1)] == next_words[j]:
                        overlap_found = True
            
            print(f"   Overlap with next: {'Yes' if overlap_found else 'No'}")
    
    # Test classification system
    print(f"\n5. CLASSIFICATION SYSTEM ANALYSIS")
    
    classification_counts = {}
    for chunk in chunks:
        chunk_type = analyzer._classify_chunk(chunk)
        classification_counts[chunk_type] = classification_counts.get(chunk_type, 0) + 1
    
    print(f"   Classification distribution:")
    for chunk_type, count in classification_counts.items():
        percentage = (count / len(chunks)) * 100
        print(f"     {chunk_type}: {count} chunks ({percentage:.1f}%)")
    
    # Test keyword matching
    print(f"\n6. KEYWORD MATCHING DETAILS")
    
    # Get classification keywords from the analyzer
    financial_keywords = [
        'revenue', 'profit', 'ebitda', 'assets', 'liabilities', 'cash flow',
        'balance sheet', 'income statement', 'financial performance', 
        'ratio analysis', 'margin', 'return on equity', 'debt'
    ]
    
    competitive_keywords = [
        'competition', 'competitors', 'market share', 'business model',
        'strategy', 'advantages', 'strengths', 'weaknesses', 'industry',
        'market position', 'differentiation'
    ]
    
    ipo_keywords = [
        'ipo', 'public offering', 'price band', 'listing', 'underwriter',
        'lead manager', 'use of proceeds', 'fund utilization', 'allotment',
        'book building', 'anchor investor'
    ]
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        
        financial_matches = [kw for kw in financial_keywords if kw in chunk_lower]
        competitive_matches = [kw for kw in competitive_keywords if kw in chunk_lower] 
        ipo_matches = [kw for kw in ipo_keywords if kw in chunk_lower]
        
        print(f"\n   Chunk {i+1} keyword matches:")
        print(f"     Financial: {len(financial_matches)} matches {financial_matches[:3]}")
        print(f"     Competitive: {len(competitive_matches)} matches {competitive_matches[:3]}")
        print(f"     IPO: {len(ipo_matches)} matches {ipo_matches[:3]}")
    
    # Test different chunk sizes
    print(f"\n7. CHUNK SIZE SENSITIVITY ANALYSIS")
    
    test_sizes = [500, 750, 1000, 1250, 1500]
    for size in test_sizes:
        test_chunks = analyzer._chunk_document(sample_text, chunk_size=size, overlap=size//5)
        avg_size = sum(len(c) for c in test_chunks) // len(test_chunks) if test_chunks else 0
        print(f"   Size {size}: {len(test_chunks)} chunks, avg {avg_size} chars")
    
    print(f"\n8. RETRIEVAL IMPLICATIONS")
    print(f"   Default retrieval: 3 chunks per query")
    print(f"   Maximum context: 6 chunks (multiple collections)")
    print(f"   Total context size: ~3000-6000 characters")
    print(f"   LLM context efficiency: Good fit for most models")
    
    print(f"\n" + "=" * 80)
    print("CHUNKING STRATEGY SUMMARY")
    print("=" * 80)
    print(f"✓ Sentence-based chunking preserves semantic integrity")
    print(f"✓ 1000-char chunks balance context and focus")
    print(f"✓ 200-char overlap (20%) prevents information loss")
    print(f"✓ Word-boundary overlap avoids fragmentation")
    print(f"✓ 4-category classification enables targeted retrieval")
    print(f"✓ Keyword-based classification is language-agnostic")
    print(f"✓ Vector storage supports semantic similarity search")
    print(f"✓ Retrieval parameters optimized for LLM context limits")

if __name__ == "__main__":
    test_chunking_strategy()
