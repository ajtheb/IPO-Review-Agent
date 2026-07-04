#!/usr/bin/env python3
"""
Test EBITDA and Current Ratio Extraction

This script validates that the improved multi-query strategy and enhanced prompt
successfully extract high-priority metrics like EBITDA margin and current ratio.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusParser


def test_priority_metrics_extraction():
    """Test extraction of high-priority metrics: EBITDA margin, current ratio, etc."""
    
    print("=" * 80)
    print("TESTING HIGH-PRIORITY METRICS EXTRACTION")
    print("=" * 80)
    
    # Initialize components
    print("\n1. Initializing analyzer and parser...")
    parser = EnhancedProspectusParser()
    analyzer = LLMProspectusAnalyzer(use_vector_db=True)
    
    # Test with Vidya Wires prospectus
    test_file = "vidya_wires.pdf"
    company_name = "Vidya Wires Private Limited"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Please ensure the prospectus file exists in the current directory")
        return
    
    print(f"\n2. Parsing prospectus: {test_file}")
    enhanced_data = parser.parse_enhanced(test_file, company_name)
    
    if not enhanced_data or not enhanced_data.business_description:
        print("❌ Failed to parse prospectus or extract text content")
        return
    
    pdf_text = enhanced_data.business_description
    print(f"✅ Extracted {len(pdf_text)} characters from prospectus")
    
    # Analyze with LLM
    print("\n3. Running LLM analysis with multi-query strategy...")
    print("   Note: This will execute 5 specialized queries:")
    print("   - P&L Statement")
    print("   - Financial Ratios")
    print("   - Balance Sheet")
    print("   - EBITDA Data (specialized)")
    print("   - Liquidity Ratios (specialized)")
    
    financial_metrics, _, _ = analyzer.analyze_prospectus_comprehensive(
        pdf_text=pdf_text,
        company_name=company_name,
        sector="Manufacturing",
        pdf_path=test_file
    )
    
    # Extract financial metrics
    if not financial_metrics:
        print("❌ No financial metrics extracted")
        return
    
    print("\n" + "=" * 80)
    print("HIGH-PRIORITY METRICS EXTRACTION RESULTS")
    print("=" * 80)
    
    # Define high-priority metrics to check
    priority_metrics = {
        "current_ratio": "Current Ratio",
        "quick_ratio": "Quick Ratio", 
        "operating_profit_margin": "Operating Profit/EBITDA Margin",
        "debt_to_equity_ratio": "Debt-to-Equity Ratio",
        "return_on_equity": "Return on Equity (ROE)",
        "return_on_assets": "Return on Assets (ROA)",
        "net_profit_margin": "Net Profit Margin",
        "revenue_growth_3yr": "Revenue Growth (3-year)",
        "profit_growth_3yr": "Profit Growth (3-year)",
        "ebitda_growth_3yr": "EBITDA Growth (3-year)"
    }
    
    extracted_count = 0
    missing_count = 0
    
    print("\n📊 EXTRACTION STATUS:")
    print("-" * 80)
    
    for metric_key, metric_name in priority_metrics.items():
        value = getattr(financial_metrics, metric_key, None)
        
        if value is not None:
            extracted_count += 1
            status = "✅ EXTRACTED"
            print(f"{status:20} | {metric_name:35} | Value: {value}")
        else:
            missing_count += 1
            status = "❌ MISSING"
            print(f"{status:20} | {metric_name:35} | Value: None")
    
    # Quality scores
    print("\n" + "-" * 80)
    print("📈 EXTRACTION QUALITY SCORES:")
    print("-" * 80)
    
    confidence = getattr(financial_metrics, 'extraction_confidence', 0)
    completeness = getattr(financial_metrics, 'data_completeness', 0)
    
    print(f"Extraction Confidence: {confidence:.2%}")
    print(f"Data Completeness: {completeness:.2%}")
    print(f"Priority Metrics Extracted: {extracted_count}/{len(priority_metrics)}")
    print(f"Priority Metrics Missing: {missing_count}/{len(priority_metrics)}")
    
    # Success criteria
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    # Check if critical metrics are present
    critical_extracted = 0
    critical_metrics = ["current_ratio", "operating_profit_margin", "debt_to_equity_ratio"]
    
    for metric in critical_metrics:
        if getattr(financial_metrics, metric, None) is not None:
            critical_extracted += 1
    
    success_rate = extracted_count / len(priority_metrics)
    critical_rate = critical_extracted / len(critical_metrics)
    
    print(f"\n✅ Overall Success Rate: {success_rate:.1%} ({extracted_count}/{len(priority_metrics)})")
    print(f"✅ Critical Metrics Rate: {critical_rate:.1%} ({critical_extracted}/{len(critical_metrics)})")
    
    if critical_rate >= 0.67:  # At least 2 out of 3 critical metrics
        print("\n🎉 SUCCESS: Critical metrics extraction is working!")
    else:
        print("\n⚠️  WARNING: Critical metrics extraction needs improvement")
    
    if success_rate >= 0.7:  # At least 70% of priority metrics
        print("🎉 SUCCESS: Overall extraction is performing well!")
    else:
        print("⚠️  WARNING: Overall extraction could be improved")
    
    # Show context chunks info
    context_dir = project_root / "context_chunks" / company_name.replace(' ', '_')
    if context_dir.exists():
        print(f"\n📁 Context chunks saved to: {context_dir}")
        print("   Review these files to debug any missing metrics:")
        for f in context_dir.glob("retrieved_financial_chunks_*.txt"):
            print(f"   - {f.name}")
    
    return {
        "extracted_count": extracted_count,
        "missing_count": missing_count,
        "success_rate": success_rate,
        "critical_rate": critical_rate,
        "confidence": confidence,
        "completeness": completeness
    }


def main():
    """Main test runner."""
    print("\n" + "=" * 80)
    print("IPO REVIEW AGENT - HIGH-PRIORITY METRICS EXTRACTION TEST")
    print("=" * 80)
    print("\nThis test validates the extraction of critical financial metrics:")
    print("- Current Ratio (Liquidity)")
    print("- EBITDA Margin (Profitability)")
    print("- Debt-to-Equity (Leverage)")
    print("- ROE/ROA (Returns)")
    print("- Growth Rates")
    
    try:
        results = test_priority_metrics_extraction()
        
        if results:
            print("\n" + "=" * 80)
            print("TEST COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            # Exit with appropriate code
            if results["critical_rate"] >= 0.67 and results["success_rate"] >= 0.7:
                print("\n✅ All validation criteria met!")
                sys.exit(0)
            else:
                print("\n⚠️  Some validation criteria not met - review results above")
                sys.exit(1)
        else:
            print("\n❌ TEST FAILED - No results returned")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
