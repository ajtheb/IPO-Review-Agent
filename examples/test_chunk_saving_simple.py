#!/usr/bin/env python3
"""
Simple test to verify chunk saving is working.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

def test_chunk_saving():
    """Test that chunks are being saved to context_chunks directory."""
    
    print("=" * 80)
    print("CHUNK SAVING TEST")
    print("=" * 80)
    print()
    
    # Initialize analyzer with vector DB enabled
    analyzer = LLMProspectusAnalyzer(
        provider='groq',
        use_vector_db=True  # Enable vector DB for chunking
    )
    
    print(f"✓ Analyzer initialized")
    print(f"  Vector DB enabled: {analyzer.use_vector_db}")
    print()
    
    # Create sample prospectus text
    sample_text = """
    Financial Performance
    
    The company has shown strong revenue growth over the past three years.
    Revenue for FY2024 was Rs. 1,000 crores, up from Rs. 800 crores in FY2023.
    Net profit margin improved to 15% in FY2024 from 12% in FY2023.
    
    The company maintains a healthy balance sheet with current ratio of 2.5.
    Debt to equity ratio stands at 0.3, indicating conservative leverage.
    
    Business Model
    
    The company operates in the manufacturing sector, specializing in wires and cables.
    Key competitive advantages include vertical integration and strong distribution network.
    
    IPO Details
    
    The company is raising Rs. 320 crores through this IPO.
    Lead managers are Pantomath Capital and IDBI Capital.
    """ * 20  # Repeat to create enough text for multiple chunks
    
    company_name = "Test Company"
    
    print(f"Sample text length: {len(sample_text)} characters")
    print()
    
    # Test chunking and storage
    print("Testing chunk_and_store_prospectus()...")
    analyzer.chunk_and_store_prospectus(
        pdf_text=sample_text,
        company_name=company_name,
        sector="Manufacturing"
    )
    print()
    
    # Check if chunks were saved
    context_dir = project_root / "context_chunks" / company_name
    print(f"Checking for saved chunks in: {context_dir}")
    
    if context_dir.exists():
        chunk_files = list(context_dir.glob("*.txt"))
        print(f"✓ Context chunks directory created")
        print(f"  Files found: {len(chunk_files)}")
        print()
        
        for f in sorted(chunk_files):
            size = f.stat().st_size
            print(f"  📄 {f.name}")
            print(f"     Size: {size:,} bytes")
            
            # Show first few lines
            with open(f, 'r') as file:
                preview = file.read(200)
                print(f"     Preview: {preview[:100]}...")
            print()
    else:
        print("❌ Context chunks directory NOT created")
        print()
    
    # Test retrieval and chunk saving
    print("Testing retrieve_relevant_context() with chunk saving...")
    context_chunks = analyzer.retrieve_relevant_context(
        f"financial data revenue profit {company_name}",
        chunk_type="financial",
        n_results=3
    )
    
    print(f"  Retrieved {len(context_chunks)} chunks")
    print()
    
    # Test _save_context_chunks directly
    print("Testing _save_context_chunks() directly...")
    analyzer._save_context_chunks(
        company_name=company_name,
        context_type="test_direct_save",
        content="This is a test of direct chunk saving",
        metadata={"test": True, "method": "direct"}
    )
    print()
    
    # Final check
    if context_dir.exists():
        all_files = list(context_dir.glob("*.txt"))
        print(f"✓ TOTAL FILES IN CONTEXT DIRECTORY: {len(all_files)}")
        print()
        print("Files:")
        for f in sorted(all_files):
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print("❌ No context chunks saved")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_chunk_saving()
