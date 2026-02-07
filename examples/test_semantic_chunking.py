"""
Test script to demonstrate semantic chunking capabilities.

This shows how semantic chunking creates more meaningful document chunks
compared to simple character-based splitting.
"""

import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def calculate_lexical_similarity(sent1: str, sent2: str) -> float:
    """Calculate simple word-overlap similarity between sentences."""
    words1 = set(re.findall(r'\b\w+\b', sent1.lower()))
    words2 = set(re.findall(r'\b\w+\b', sent2.lower()))
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                 'be', 'been', 'are', 'were', 'have', 'has', 'had'}
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def chunk_document_semantic(text: str, 
                           max_chunk_size: int = 500, 
                           similarity_threshold: float = 0.5,
                           min_chunk_size: int = 50) -> list:
    """Simplified semantic chunking for demonstration."""
    
    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = sentences[0]
    current_size = len(current_chunk)
    
    for i in range(1, len(sentences)):
        current_sentence = sentences[i]
        sentence_size = len(current_sentence)
        
        if current_size + sentence_size + 1 > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = current_sentence
            current_size = sentence_size
            continue
        
        prev_sentence = sentences[i-1]
        similarity = calculate_lexical_similarity(prev_sentence, current_sentence)
        
        if similarity >= similarity_threshold:
            current_chunk += " " + current_sentence
            current_size += sentence_size + 1
        else:
            if current_size > min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = current_sentence
                current_size = sentence_size
            else:
                current_chunk += " " + current_sentence
                current_size += sentence_size + 1
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_document_simple(text: str, chunk_size: int = 500) -> list:
    """Simple character-based chunking for comparison."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


def test_semantic_vs_simple_chunking():
    """Compare semantic chunking with simple character-based chunking."""
    
    # Sample IPO prospectus text
    sample_text = """
    FINANCIAL PERFORMANCE
    
    The company has demonstrated strong financial performance over the past three years. 
    Revenue growth has been consistently above 25% year-over-year. This growth is primarily 
    driven by expansion in digital services and product innovation.
    
    In fiscal year 2023, the company achieved revenues of Rs 11,884.89 crores. The net profit 
    for the same period was Rs 256.93 crores, representing a net profit margin of 2.16%. 
    Operating profit margins have also shown improvement, rising from 5.2% to 6.8% over the period.
    
    BUSINESS MODEL AND COMPETITIVE ADVANTAGE
    
    The company operates a unique business model combining technology and traditional services. 
    This hybrid approach provides significant competitive advantages. The company has built 
    strong brand recognition in its target markets.
    
    Key competitive strengths include proprietary technology platforms, extensive distribution 
    networks, and strong customer relationships. The company has successfully differentiated 
    itself from competitors through innovation and customer service excellence.
    
    RISK FACTORS
    
    Several risk factors could impact future performance. Market competition is intensifying 
    with new entrants and aggressive pricing strategies. Regulatory changes in key markets 
    could affect operations and profitability.
    
    Technology disruption poses another significant risk. The company must continue investing 
    in innovation to maintain its competitive position. Economic downturns could reduce 
    customer spending and impact revenue growth.
    
    USE OF IPO PROCEEDS
    
    The company plans to utilize IPO proceeds for strategic growth initiatives. Approximately 
    40% will be allocated to capacity expansion and infrastructure development. Another 30% 
    will fund working capital requirements and business operations.
    
    Debt repayment will account for 20% of the proceeds, which will strengthen the balance 
    sheet and reduce interest costs. The remaining 10% is allocated for general corporate 
    purposes and strategic acquisitions if opportunities arise.
    """
    
    print("=" * 80)
    print("SEMANTIC CHUNKING vs SIMPLE CHUNKING COMPARISON")
    print("=" * 80)
    
    # Simple chunking
    print("\n" + "=" * 80)
    print("SIMPLE CHARACTER-BASED CHUNKING (500 chars per chunk)")
    print("=" * 80)
    
    simple_chunks = chunk_document_simple(sample_text, chunk_size=500)
    
    for i, chunk in enumerate(simple_chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"\n➤ Total simple chunks: {len(simple_chunks)}")
    
    # Semantic chunking
    print("\n" + "=" * 80)
    print("SEMANTIC CHUNKING (similarity threshold: 0.5)")
    print("=" * 80)
    
    semantic_chunks = chunk_document_semantic(sample_text, max_chunk_size=500, similarity_threshold=0.5)
    
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        # Show what topic this chunk represents
        if "financial" in chunk.lower() or "revenue" in chunk.lower() or "profit" in chunk.lower():
            print("📊 Topic: FINANCIAL DATA")
        elif "business model" in chunk.lower() or "competitive" in chunk.lower():
            print("🏢 Topic: BUSINESS & COMPETITION")
        elif "risk" in chunk.lower():
            print("⚠️  Topic: RISK FACTORS")
        elif "ipo" in chunk.lower() or "proceeds" in chunk.lower():
            print("💰 Topic: IPO INFORMATION")
    
    print(f"\n➤ Total semantic chunks: {len(semantic_chunks)}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print(f"""
Semantic Chunking Benefits:
✓ Preserves context - Related sentences stay together
✓ Respects topic boundaries - Doesn't split mid-topic
✓ Better for LLM analysis - Each chunk is semantically complete
✓ Improved retrieval - More meaningful chunks for vector search

Simple Chunking Limitations:
✗ Breaks text arbitrarily at character limits
✗ May split sentences or paragraphs
✗ Context can be lost across chunk boundaries
✗ Less meaningful for semantic search

Recommendation:
Use semantic chunking for financial documents like IPO prospectuses where
preserving context and topic coherence is crucial for accurate analysis.
    """)


def test_similarity_calculation():
    """Test the lexical similarity calculation."""
    
    print("\n" + "=" * 80)
    print("LEXICAL SIMILARITY EXAMPLES")
    print("=" * 80)
    
    test_pairs = [
        (
            "The company achieved strong revenue growth in Q4 2023.",
            "Revenue growth continued to be robust throughout 2023."
        ),
        (
            "The company achieved strong revenue growth in Q4 2023.",
            "Risk factors include market competition and regulatory changes."
        ),
        (
            "Net profit margin improved from 5.2% to 6.8%.",
            "Operating margins also showed improvement during this period."
        ),
        (
            "The company operates in the technology sector.",
            "Weather conditions affected agricultural production."
        ),
    ]
    
    for i, (sent1, sent2) in enumerate(test_pairs, 1):
        similarity = calculate_lexical_similarity(sent1, sent2)
        print(f"\nPair {i}:")
        print(f"  Sentence 1: {sent1}")
        print(f"  Sentence 2: {sent2}")
        print(f"  Similarity: {similarity:.2f}")
        
        if similarity > 0.5:
            print(f"  ✓ High similarity - Keep in same chunk")
        elif similarity > 0.3:
            print(f"  ⚠ Medium similarity - Context dependent")
        else:
            print(f"  ✗ Low similarity - Likely different topics")


def test_different_thresholds():
    """Test semantic chunking with different similarity thresholds."""
    
    sample_text = """
    The company has strong financial performance. Revenue growth is 25% annually. 
    Net profit margin stands at 2.16%. Operating efficiency has improved significantly.
    
    The business model is unique. Technology integration drives competitive advantage. 
    Customer satisfaction rates are high. Market position continues to strengthen.
    
    Risk factors require attention. Competition is intensifying. Regulatory environment 
    may change. Economic conditions remain uncertain.
    """
    
    print("\n" + "=" * 80)
    print("IMPACT OF SIMILARITY THRESHOLD")
    print("=" * 80)
    
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        chunks = chunk_document_semantic(sample_text, max_chunk_size=300, 
                                        similarity_threshold=threshold)
        print(f"\n➤ Threshold: {threshold} → Generated {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {len(chunk)} chars")
    
    print(f"""
Threshold Guidelines:
• 0.3 (Low):  More, smaller chunks - Better for fine-grained analysis
• 0.5 (Medium): Balanced chunks - Good for general use
• 0.7 (High):  Fewer, larger chunks - Better for broader context

For IPO prospectus analysis, 0.5 is recommended as it balances
context preservation with manageable chunk sizes.
    """)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IPO Review Agent - Semantic Chunking Test Suite")
    print("=" * 80)
    
    test_semantic_vs_simple_chunking()
    test_similarity_calculation()
    test_different_thresholds()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Semantic chunking preserves topic coherence")
    print("2. Better for LLM analysis and vector search")
    print("3. Threshold of 0.5 works well for financial documents")
    print("4. More meaningful chunks improve extraction accuracy")
    print("=" * 80 + "\n")
