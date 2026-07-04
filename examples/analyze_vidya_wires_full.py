#!/usr/bin/env python3
"""
Full analysis of Vidya Wires IPO with complete chunk saving.
This script ensures ALL chunk types are saved:
1. Full prospectus text
2. All prospectus chunks (created by recursive splitter)
3. Retrieved financial chunks (from vector DB query)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusDataSource

def main():
    print("=" * 80)
    print("VIDYA WIRES IPO - FULL ANALYSIS WITH CHUNK SAVING")
    print("=" * 80)
    print()
    
    # Check API key
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️  WARNING: GROQ_API_KEY not set")
        print("   Set it with: export GROQ_API_KEY='your_key_here'")
        print("   Analysis will continue but LLM extraction will be skipped")
        print()
        # Don't return, continue with chunking to demonstrate chunk saving
    else:
        print("✓ GROQ_API_KEY found")
        print()
    
    # Initialize components
    print("1. Initializing analyzer and parser...")
    analyzer = LLMProspectusAnalyzer(
        provider='groq',
        use_vector_db=True  # Enable vector DB for chunking and retrieval
    )
    parser = EnhancedProspectusDataSource()
    print(f"   ✓ Analyzer initialized (Vector DB: {analyzer.use_vector_db})")
    print()
    
    # Search for Vidya Wires
    company_name = "Vidya Wires"
    print(f"2. Searching for '{company_name}' prospectus...")
    
    # Get enhanced IPO data which includes prospectus text
    enhanced_data = parser.get_enhanced_ipo_data(company_name, force_refresh=False)
    
    if not enhanced_data or not enhanced_data.business_description:
        print(f"   ❌ No prospectus found for '{company_name}'")
        print(f"   Try fetching it first with:")
        print(f"   parser.get_enhanced_ipo_data('{company_name}', force_refresh=True)")
        return
    
    # Extract prospectus text from business_description field (which contains the full document)
    prospectus_text = enhanced_data.business_description
    
    # Extract sector from prospectus text (simple heuristic)
    sector = "Manufacturing"  # Default
    sector_keywords = {
        "wire": "Manufacturing",
        "cable": "Manufacturing",
        "metal": "Manufacturing",
        "technology": "Technology",
        "software": "Technology",
        "pharma": "Pharmaceuticals",
        "healthcare": "Healthcare",
        "finance": "Financial Services",
        "banking": "Financial Services"
    }
    
    text_lower = prospectus_text[:5000].lower() if prospectus_text else ""
    for keyword, detected_sector in sector_keywords.items():
        if keyword in text_lower:
            sector = detected_sector
            break
    
    print(f"   ✓ Found prospectus ({len(prospectus_text):,} characters)")
    print(f"   ✓ Detected sector: {sector}")
    print()
    
    # Run full comprehensive analysis
    print("3. Running comprehensive analysis...")
    print("   This will:")
    print("   - Chunk the prospectus document")
    print("   - Store chunks in vector database")
    print("   - Save all chunks to file")
    print("   - Retrieve relevant financial chunks")
    print("   - Save retrieved chunks to file")
    print("   - Extract financial metrics using LLM")
    print()
    
    try:
        financial_metrics, benchmarking, ipo_specifics = analyzer.analyze_prospectus_comprehensive(
            pdf_text=prospectus_text,
            company_name=company_name,
            sector=sector
        )
        
        print("   ✓ Analysis completed successfully")
        print()
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check saved files
    print("4. Checking saved context chunks...")
    context_dir = project_root / "context_chunks" / "Vidya_Wires"
    
    if context_dir.exists():
        all_files = sorted(context_dir.glob("*.txt"))
        print(f"   ✓ Found {len(all_files)} chunk files")
        print()
        
        print("   Files saved:")
        for f in all_files:
            size = f.stat().st_size
            print(f"   📄 {f.name}")
            print(f"      Size: {size:,} bytes ({size/1024:.1f} KB)")
            
            # Show file type
            if "prospectus_text" in f.name:
                print(f"      Type: Full prospectus text")
            elif "all_prospectus_chunks" in f.name:
                print(f"      Type: All chunks created by recursive splitter")
            elif "retrieved_financial_chunks" in f.name:
                print(f"      Type: Retrieved chunks from vector DB")
            print()
    else:
        print(f"   ❌ Context directory not found: {context_dir}")
        return
    
    # Check vector DB stats
    print("5. Vector database statistics...")
    stats = analyzer.get_vector_db_stats()
    if stats.get('enabled'):
        print(f"   ✓ Vector DB enabled")
        print(f"   Total chunks stored: {stats.get('total_chunks', 0)}")
        print(f"   Collections:")
        for coll_name, coll_stats in stats.get('collections', {}).items():
            if isinstance(coll_stats, dict) and 'total_documents' in coll_stats:
                print(f"     - {coll_name}: {coll_stats['total_documents']} documents")
    else:
        print("   ⚠️  Vector DB not enabled")
    print()
    
    # Show metrics summary
    print("6. Financial metrics extraction summary...")
    if financial_metrics:
        print(f"   Extraction confidence: {financial_metrics.extraction_confidence:.1%}")
        print(f"   Data completeness: {financial_metrics.data_completeness:.1%}")
        
        # Count non-None metrics
        metrics_dict = {k: v for k, v in financial_metrics.__dict__.items() 
                       if v is not None and k not in ['extraction_confidence', 'data_completeness']}
        print(f"   Metrics extracted: {len(metrics_dict)}")
    else:
        print("   ⚠️  No metrics extracted")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"✓ Prospectus analyzed: {company_name}")
    print(f"✓ Context chunks saved to: {context_dir}")
    print(f"✓ Vector DB populated with chunks")
    print(f"✓ Retrieved chunks saved for review")
    print()
    print("Next steps:")
    print("1. Review saved chunk files in context_chunks/Vidya_Wires/")
    print("2. Check 'retrieved_financial_chunks' to see what LLM analyzed")
    print("3. Verify chunks prioritize financial statements and tables")
    print()


if __name__ == "__main__":
    main()
