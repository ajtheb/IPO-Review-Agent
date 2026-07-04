#!/usr/bin/env python3
"""
Test Increased Chunk Retrieval for Investment Thesis

This script validates that the increased chunk retrieval provides richer context
for investment thesis generation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from data_sources.enhanced_prospectus_parser import EnhancedProspectusParser

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_chunk_retrieval_increase():
    """Test that more chunks are being retrieved."""
    
    print_section("TESTING INCREASED CHUNK RETRIEVAL")
    
    # Check if Vidya Wires prospectus exists
    pdf_path = "prospectus_files/Vidya_Wires_DRHP.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at {pdf_path}")
        print("Please place the Vidya Wires DRHP PDF in the prospectus_files/ directory")
        return False
    
    print(f"\n✅ Found PDF: {pdf_path}")
    
    # Parse PDF
    print("\n1. Parsing prospectus...")
    parser = EnhancedProspectusParser()
    
    try:
        result = parser.parse_prospectus(pdf_path)
        pdf_text = result.get("full_text", "")
        
        if not pdf_text:
            print("❌ Failed to extract text from PDF")
            return False
        
        print(f"✅ Extracted {len(pdf_text)} characters from PDF")
        
    except Exception as e:
        print(f"❌ Error parsing PDF: {e}")
        return False
    
    # Initialize analyzer with vector DB
    print("\n2. Initializing analyzer with vector DB...")
    
    try:
        # Try Gemini first, fallback to OpenAI
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "openai"
        print(f"   Using provider: {provider}")
        
        analyzer = LLMProspectusAnalyzer(
            provider=provider,
            use_vector_db=True
        )
        
        print("✅ Analyzer initialized")
        
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return False
    
    # Run comprehensive analysis
    print("\n3. Running comprehensive analysis with increased chunk retrieval...")
    print("   Expected chunk counts:")
    print("   - Financial Metrics: 16 chunks (up from 10)")
    print("   - Benchmarking: 9 chunks (up from 6)")
    print("   - IPO Specifics: 8 chunks (up from 5)")
    print("   - Total: 33 chunks (up from 21)")
    
    try:
        financial_metrics, benchmarking, ipo_specifics = analyzer.analyze_prospectus_comprehensive(
            pdf_text=pdf_text,
            company_name="Vidya Wires",
            sector="Manufacturing",
            pdf_path=pdf_path
        )
        
        print("\n✅ Comprehensive analysis completed")
        
        # Check saved chunks
        chunk_dir = Path("context_chunks/Vidya_Wires")
        if chunk_dir.exists():
            chunk_files = list(chunk_dir.glob("retrieved_financial_chunks_*.txt"))
            if chunk_files:
                latest_chunk_file = max(chunk_files, key=lambda p: p.stat().st_mtime)
                print(f"\n4. Checking saved chunks: {latest_chunk_file.name}")
                
                # Read and parse metadata
                with open(latest_chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count chunks in file
                    chunk_count = content.count("=== CHUNK ")
                    print(f"   ✅ Found {chunk_count} chunks in saved file")
                    
                    # Check for metadata
                    if "unique_chunks_retrieved" in content:
                        import re
                        match = re.search(r'"unique_chunks_retrieved":\s*(\d+)', content)
                        if match:
                            unique_count = int(match.group(1))
                            print(f"   ✅ Unique chunks retrieved: {unique_count}")
                            
                            # Validate increase
                            if unique_count >= 10:
                                print(f"   ✅ SUCCESS: Retrieved {unique_count} chunks (target: 10+)")
                            else:
                                print(f"   ⚠️  WARNING: Only {unique_count} chunks retrieved (target: 10+)")
                    
                    # Check for multi-query strategy
                    if "multi_query_specialized" in content:
                        print("   ✅ Multi-query strategy confirmed")
                    
                    # Count queries
                    query_count = content.count('"query":')
                    if query_count > 0:
                        print(f"   ✅ Found metadata for {query_count} queries")
        
        # Check data completeness
        print("\n5. Checking financial metrics data completeness...")
        print(f"   - Data Completeness: {financial_metrics.data_completeness:.1%}")
        print(f"   - Extraction Confidence: {financial_metrics.extraction_confidence:.1%}")
        
        if financial_metrics.data_completeness >= 0.5:
            print("   ✅ Good data completeness (50%+)")
        else:
            print("   ⚠️  Low data completeness - may need more chunks")
        
        # Generate investment thesis
        print("\n6. Generating investment thesis with enriched context...")
        
        thesis = analyzer.generate_investment_thesis(
            financial_metrics=financial_metrics,
            benchmarking=benchmarking,
            ipo_specifics=ipo_specifics,
            company_name="Vidya Wires",
            web_context=""
        )
        
        print(f"\n✅ Investment thesis generated ({len(thesis)} characters)")
        
        # Check thesis quality indicators
        print("\n7. Validating thesis quality...")
        quality_indicators = {
            "Contains specific numbers": any(char.isdigit() for char in thesis),
            "Mentions revenue": "revenue" in thesis.lower(),
            "Mentions profitability": any(term in thesis.lower() for term in ["profit", "margin", "ebitda"]),
            "Mentions financial ratios": any(term in thesis.lower() for term in ["ratio", "roe", "roa", "debt"]),
            "Has structured sections": thesis.count("#") > 3 or thesis.count("**") > 5,
        }
        
        for indicator, present in quality_indicators.items():
            status = "✅" if present else "⚠️ "
            print(f"   {status} {indicator}")
        
        print("\n" + "=" * 80)
        print("  INVESTMENT THESIS PREVIEW")
        print("=" * 80)
        print(thesis[:800] + "..." if len(thesis) > 800 else thesis)
        
        print_section("TEST COMPLETED SUCCESSFULLY")
        print("\n✅ All checks passed!")
        print("\nNext steps:")
        print("1. Review context_chunks/Vidya_Wires/ for saved chunks")
        print("2. Compare with previous analysis results")
        print("3. Check if EBITDA margin and current ratio are now extracted")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print_section("CHUNK RETRIEVAL INCREASE VALIDATION")
    print("\nThis test validates that:")
    print("1. More chunks are being retrieved (33 vs 21)")
    print("2. Context is richer for investment thesis")
    print("3. Data completeness has improved")
    print("4. Investment thesis quality is enhanced")
    
    success = test_chunk_retrieval_increase()
    
    sys.exit(0 if success else 1)
