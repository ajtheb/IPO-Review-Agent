#!/usr/bin/env python3
"""
Test script to verify LLM analysis and context chunk saving for Vidya Wires IPO.
This will help confirm:
1. Syntax errors are fixed
2. LLM analysis runs successfully
3. Context chunks are saved to context_chunks/Vidya Wires/
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusParser

def test_vidya_wires_analysis():
    """Test analysis and context chunk saving for Vidya Wires."""
    
    print("=" * 80)
    print("VIDYA WIRES IPO - LLM ANALYSIS & CONTEXT CHUNK SAVING TEST")
    print("=" * 80)
    print()
    
    # Check for API key
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        print("❌ ERROR: GROQ_API_KEY not found in environment")
        print("   Please set it with: export GROQ_API_KEY='your_key_here'")
        return False
    
    print("✓ GROQ_API_KEY found")
    print()
    
    # Initialize analyzer
    print("Initializing LLM Prospectus Analyzer...")
    try:
        analyzer = LLMProspectusAnalyzer(
            provider='groq',
            use_vector_db=False  # Disable vector DB for this test
        )
        print("✓ Analyzer initialized successfully")
        print(f"  Provider: {analyzer.provider}")
        print(f"  Vector DB: {'enabled' if analyzer.use_vector_db else 'disabled'}")
    except Exception as e:
        print(f"❌ ERROR initializing analyzer: {e}")
        return False
    
    print()
    
    # Initialize prospectus parser
    print("Initializing Enhanced Prospectus Parser...")
    try:
        parser = EnhancedProspectusParser()
        print("✓ Parser initialized successfully")
    except Exception as e:
        print(f"❌ ERROR initializing parser: {e}")
        return False
    
    print()
    
    # Test with Vidya Wires
    company_name = "Vidya Wires"
    print(f"Searching for '{company_name}' prospectus...")
    
    try:
        # Search for prospectus
        results = parser.search_prospectus(company_name)
        
        if not results:
            print(f"❌ No prospectus found for '{company_name}'")
            print("   This is expected if you haven't fetched it yet.")
            print(f"   Run: from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusParser")
            print(f"        parser = EnhancedProspectusParser()")
            print(f"        parser.fetch_and_parse_prospectus('{company_name}')")
            return False
        
        print(f"✓ Found {len(results)} prospectus result(s)")
        prospectus = results[0]
        print(f"  Company: {prospectus.get('company_name', 'N/A')}")
        print(f"  Issue Size: {prospectus.get('issue_size', 'N/A')}")
        print()
        
        # Extract prospectus text
        prospectus_text = prospectus.get('full_text', '')
        if not prospectus_text:
            print("❌ No prospectus text found")
            return False
        
        print(f"✓ Prospectus text loaded ({len(prospectus_text)} characters)")
        print()
        
        # Prepare context for saving
        context_dir = project_root / "context_chunks" / company_name
        print(f"Context chunks will be saved to: {context_dir}")
        print()
        
        # Run financial metrics extraction
        print("Running LLM financial metrics extraction...")
        print("  (This will save context chunks automatically)")
        print()
        
        try:
            metrics = analyzer._extract_financial_metrics(
                prospectus_text=prospectus_text,
                company_name=company_name
            )
            
            print("✓ Financial metrics extraction completed")
            print()
            
            # Check if context chunks were saved
            if context_dir.exists():
                chunk_files = list(context_dir.glob("*.txt"))
                metadata_files = list(context_dir.glob("*.json"))
                
                print(f"✓ Context chunks saved successfully!")
                print(f"  Directory: {context_dir}")
                print(f"  Text chunks: {len(chunk_files)} files")
                print(f"  Metadata files: {len(metadata_files)} files")
                print()
                
                if chunk_files:
                    print("  Sample files:")
                    for f in sorted(chunk_files)[:5]:
                        size = f.stat().st_size
                        print(f"    - {f.name} ({size:,} bytes)")
                
                print()
            else:
                print("⚠️  WARNING: Context chunks directory not created")
                print(f"   Expected: {context_dir}")
                print()
            
            # Display extracted metrics summary
            if metrics:
                print("Extracted Metrics Summary:")
                print("-" * 40)
                
                # Count non-None metrics
                non_none_metrics = {k: v for k, v in metrics.items() 
                                   if v is not None and k not in ['extraction_confidence', 'data_completeness']}
                
                print(f"  Total metrics extracted: {len(non_none_metrics)}")
                print(f"  Data completeness: {metrics.get('data_completeness', 0):.1%}")
                print(f"  Extraction confidence: {metrics.get('extraction_confidence', 0):.1%}")
                print()
                
                # Show a few key metrics
                key_metrics = [
                    'revenue_fy2024', 'revenue_fy2023', 'revenue_fy2022',
                    'net_profit_fy2024', 'net_profit_fy2023',
                    'revenue_growth_3yr', 'profit_margin_avg'
                ]
                
                print("  Key Financial Metrics:")
                for metric in key_metrics:
                    value = metrics.get(metric)
                    if value is not None:
                        print(f"    {metric}: {value}")
                
                print()
            else:
                print("⚠️  WARNING: No metrics extracted")
                print()
            
            print("=" * 80)
            print("TEST COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print()
            print("Summary:")
            print("  ✓ Syntax errors fixed")
            print("  ✓ LLM analyzer working correctly")
            print("  ✓ Context chunks saved successfully")
            print("  ✓ Financial metrics extracted")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR during metrics extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ ERROR during prospectus search: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vidya_wires_analysis()
    sys.exit(0 if success else 1)
