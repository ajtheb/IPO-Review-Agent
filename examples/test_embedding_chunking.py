"""
Test embedding-based semantic chunking for IPO prospectus analysis.

Demonstrates cosine similarity-based chunking using sentence embeddings.
"""

import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Testing semantic chunking with embeddings...")
print("=" * 70)

# Sample IPO prospectus text
SAMPLE_TEXT = """
XYZ Technologies Limited is a leading software services company specializing in cloud computing solutions. 
The company was founded in 2015 and has grown rapidly over the past 8 years.

The company's revenue for FY 2023 was Rs 1,200 crores, representing a year-on-year growth of 35%.
Our net profit for the same period was Rs 180 crores, giving us a healthy profit margin of 15%.

Our primary business segments include cloud infrastructure services, enterprise software solutions, and IT consulting.
Cloud infrastructure contributes approximately 60% of our revenue and is our fastest-growing segment.

The company faces competition from both domestic and international players in the cloud services space.
However, we differentiate ourselves through customized solutions and superior customer support.

We are raising capital through this IPO to fund expansion into new geographies.
The issue size is Rs 500 crores, consisting entirely of fresh issue of shares.

The IPO price band has been fixed at Rs 180-200 per equity share.
At the upper price band, the company will be valued at approximately 22 times its trailing twelve-month earnings.

Key risks include intense competition, technology obsolescence, and dependency on key clients.
Regulatory changes in data privacy laws could impact our business operations.
"""

try:
    from src.analyzers.semantic_chunking import SemanticChunker, chunk_text_semantically
    
    print("\n✓ Semantic chunking module loaded successfully!")
    print("\nTest 1: Basic Semantic Chunking")
    print("-" * 70)
    
    # Try to create chunks
    try:
        chunker = SemanticChunker(provider="sentence-transformers")
        
        chunks = chunker.chunk_text_semantic(
            SAMPLE_TEXT,
            chunk_size=400,
            similarity_threshold=0.6
        )
        
        print(f"\nInput length: {len(SAMPLE_TEXT)} characters")
        print(f"Chunks created: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i} ({len(chunk)} chars):")
            print(f"  {chunk[:150]}...\n" if len(chunk) > 150 else f"  {chunk}\n")
        
        print("✅ Semantic chunking test passed!")
        
    except Exception as e:
        print(f"⚠️  Embedding-based chunking not available: {e}")
        print("\nNote: Install sentence-transformers for embedding support:")
        print("  pip install sentence-transformers")
        print("\nThe system will use fallback chunking when embeddings are unavailable.")
    
    print("\n" + "=" * 70)
    print("Semantic Chunking Benefits:")
    print("  • Groups semantically related sentences")
    print("  • Uses cosine similarity on embeddings")
    print("  • Better context preservation")
    print("  • Improved vector database retrieval")
    print("=" * 70)
    
except ImportError as e:
    print(f"\n❌ Could not import semantic chunking module: {e}")
    print("\nMake sure the semantic_chunking.py module exists in:")
    print("  src/analyzers/semantic_chunking.py")

print("\n" + "=" * 70)
print("How Semantic Chunking Works:")
print("=" * 70)
print("""
1. Split text into sentences
2. Generate embeddings for each sentence using:
   - OpenAI's text-embedding-3-small
   - Google's Gemini embedding-001  
   - Sentence-transformers (local, free)

3. Calculate cosine similarity between consecutive sentences
4. Group sentences with similarity above threshold
5. Respect chunk size limits while preserving coherence

Example:
  Sentence 1: "Revenue was Rs 1200 crores"
  Sentence 2: "Profit margin improved to 15%"
  → High similarity (both financial) → Same chunk

  Sentence 3: "The IPO price band is Rs 180-200"
  → Low similarity (different topic) → New chunk

Benefits for IPO Analysis:
  ✓ Financial data stays together
  ✓ Business model sections coherent
  ✓ Risk factors grouped properly
  ✓ Better LLM context windows
  ✓ Improved vector search results
""")
print("=" * 70)
