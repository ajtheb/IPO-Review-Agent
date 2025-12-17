"""
Practical example demonstrating Enhanced IPO Prospectus Integration
with sample document processing and real-world usage patterns.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

try:
    from src.data_sources.enhanced_prospectus_parser import (
        EnhancedProspectusDataSource,
        EnhancedProspectusParser,
        DataValidator,
        EnhancedFinancialData
    )
    from src.data_sources import DataSourceManager
    from config.enhanced_prospectus_config import update_config
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced prospectus integration loaded successfully")
except ImportError as e:
    print(f"❌ Enhanced prospectus integration not available: {e}")
    ENHANCED_AVAILABLE = False

class IPOProspectusDemo:
    """Demonstration class for IPO Prospectus Integration features."""
    
    def __init__(self):
        if not ENHANCED_AVAILABLE:
            raise ImportError("Enhanced prospectus integration not available")
        
        # Configure for optimal performance
        update_config(
            min_quality_threshold=0.3,
            cache_duration_hours=24,
            max_pages_to_process=50,
            enable_table_extraction=True,
            parallel_processing=True
        )
        
        self.prospectus_source = EnhancedProspectusDataSource(cache_enabled=True)
        self.data_manager = DataSourceManager(use_enhanced_prospectus=True)
        
        print("🚀 IPO Prospectus Demo initialized with enhanced features")
    
    def demo_quick_assessment(self, company_name: str) -> Dict[str, Any]:
        """Demonstrate quick prospectus data assessment."""
        print(f"\n🔍 Quick Assessment: {company_name}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Get data availability summary (fast operation)
            summary = self.prospectus_source.get_data_summary(company_name)
            assessment_time = time.time() - start_time
            
            print(f"⏱️  Assessment completed in {assessment_time:.2f} seconds")
            print(f"📁 SEBI filings found: {summary.get('sebi_filings_found', 0)}")
            print(f"💾 Data cached: {summary.get('cached', False)}")
            
            if summary.get('latest_filing_type'):
                print(f"📄 Latest filing: {summary['latest_filing_type']}")
                print(f"📅 Filing date: {summary.get('latest_filing_date', 'Unknown')}")
            
            # Provide recommendation
            filings_count = summary.get('sebi_filings_found', 0)
            if filings_count > 0:
                estimated_time = summary.get('estimated_processing_time', '2-5 minutes')
                print(f"✅ Prospectus data available - estimated processing time: {estimated_time}")
                print("💡 Recommendation: Proceed with full data extraction")
            else:
                print("❌ No SEBI filings found")
                print("💡 Recommendation: Check company name variants or use manual data entry")
            
            return {
                'assessment_time': assessment_time,
                'data_available': filings_count > 0,
                'recommendation': 'proceed' if filings_count > 0 else 'manual_entry',
                **summary
            }
            
        except Exception as e:
            print(f"❌ Assessment failed: {e}")
            return {'error': str(e), 'assessment_time': time.time() - start_time}
    
    def demo_full_extraction(self, company_name: str, force_refresh: bool = False) -> Optional[EnhancedFinancialData]:
        """Demonstrate full prospectus data extraction with detailed reporting."""
        print(f"\n📊 Full Data Extraction: {company_name}")
        print("-" * 50)
        
        if force_refresh:
            print("🔄 Force refresh enabled - bypassing cache")
        
        start_time = time.time()
        
        try:
            # Extract enhanced financial data
            enhanced_data = self.prospectus_source.get_enhanced_ipo_data(
                company_name, 
                force_refresh=force_refresh
            )
            
            extraction_time = time.time() - start_time
            print(f"⏱️  Extraction completed in {extraction_time:.2f} seconds")
            
            if enhanced_data:
                self._report_extraction_results(enhanced_data, extraction_time)
                return enhanced_data
            else:
                print("❌ No quality data could be extracted")
                print("💡 Possible reasons:")
                print("   - No SEBI filings found")
                print("   - PDF parsing failed")
                print("   - Extracted data below quality threshold")
                return None
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            print(f"⏱️  Failed after {time.time() - start_time:.2f} seconds")
            return None
    
    def _report_extraction_results(self, data: EnhancedFinancialData, extraction_time: float):
        """Generate detailed report of extraction results."""
        print("✅ Data extraction successful!")
        print(f"📈 Overall Quality Score: {data.data_quality_score:.2f}/1.00")
        print(f"🎯 Source Confidence: {data.source_confidence:.2f}/1.00")
        
        # Financial data completeness
        print("\n💰 Financial Data Extracted:")
        print(f"   Revenue data points: {len(data.revenue_data)}")
        print(f"   Profit data points: {len(data.profit_data)}")
        print(f"   EBITDA data points: {len(data.ebitda_data)}")
        print(f"   Balance sheet items: {len(data.assets_data) + len(data.liabilities_data)}")
        print(f"   Financial ratios: {len(data.key_ratios)}")
        
        # Show sample financial data
        if data.revenue_data:
            print("\n📊 Revenue Trend (₹ Crores):")
            for year, amount in sorted(data.revenue_data.items()):
                print(f"   {year}: ₹{amount:,.2f}")
        
        if data.profit_data:
            print("\n💵 Profit Trend (₹ Crores):")
            for year, amount in sorted(data.profit_data.items()):
                print(f"   {year}: ₹{amount:,.2f}")
        
        if data.key_ratios:
            print("\n📈 Key Financial Ratios:")
            for ratio, value in data.key_ratios.items():
                print(f"   {ratio.replace('_', ' ').title()}: {value:.2f}")
        
        # Qualitative data
        print(f"\n📝 Qualitative Data Extracted:")
        print(f"   Business description: {len(data.business_description)} characters")
        print(f"   Risk factors identified: {len(data.risk_factors)}")
        print(f"   Use of funds items: {len(data.use_of_funds)}")
        print(f"   Company strengths: {len(data.company_strengths)}")
        print(f"   Competitive advantages: {len(data.competitive_advantages)}")
        
        # Show sample qualitative data
        if data.business_description:
            print(f"\n🏢 Business Overview (first 200 chars):")
            print(f"   {data.business_description[:200]}...")
        
        if data.risk_factors:
            print(f"\n⚠️  Top 3 Risk Factors:")
            for i, risk in enumerate(data.risk_factors[:3], 1):
                print(f"   {i}. {risk[:80]}...")
        
        if data.use_of_funds:
            print(f"\n💼 Use of Funds:")
            for i, use in enumerate(data.use_of_funds[:3], 1):
                print(f"   {i}. {use[:80]}...")
        
        # Data quality assessment
        if data.validation_flags:
            print(f"\n⚠️  Data Validation Warnings ({len(data.validation_flags)}):")
            for flag in data.validation_flags[:3]:
                print(f"   - {flag}")
            if len(data.validation_flags) > 3:
                print(f"   ... and {len(data.validation_flags) - 3} more")
        else:
            print("\n✅ No data validation issues detected")
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"   Extraction time: {extraction_time:.2f} seconds")
        print(f"   Extraction date: {data.extraction_date}")
        
        # Quality recommendations
        self._provide_quality_recommendations(data)
    
    def _provide_quality_recommendations(self, data: EnhancedFinancialData):
        """Provide recommendations based on data quality."""
        print(f"\n💡 Quality Assessment & Recommendations:")
        
        quality = data.data_quality_score
        
        if quality >= 0.8:
            print("🟢 HIGH QUALITY DATA")
            print("   ✅ Suitable for automated analysis")
            print("   ✅ High confidence in investment recommendations")
            print("   ✅ Can be used for comparative analysis")
            
        elif quality >= 0.6:
            print("🟡 MEDIUM QUALITY DATA")
            print("   ✅ Good for preliminary analysis")
            print("   ⚠️  Recommend manual verification of key metrics")
            print("   ✅ Suitable for screening and initial assessment")
            
        elif quality >= 0.4:
            print("🟠 MODERATE QUALITY DATA")
            print("   ⚠️  Use with caution for investment decisions")
            print("   ✅ Good for basic company information")
            print("   📋 Recommend supplementing with additional research")
            
        else:
            print("🔴 LOW QUALITY DATA")
            print("   ❌ Not recommended for investment decisions")
            print("   📋 Manual data entry likely required")
            print("   🔍 Consider alternative data sources")
        
        # Specific recommendations
        if len(data.revenue_data) < 2:
            print("   📈 Recommendation: Seek additional revenue data")
        
        if len(data.risk_factors) < 3:
            print("   ⚠️  Recommendation: Manual risk assessment needed")
        
        if not data.use_of_funds:
            print("   💼 Recommendation: Research IPO fund utilization")
    
    def demo_integration_workflow(self, company_name: str, ipo_details: Dict[str, Any]):
        """Demonstrate complete integration workflow with DataSourceManager."""
        print(f"\n🔧 Complete Integration Workflow: {company_name}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Collect comprehensive IPO data
            print("📡 Step 1: Collecting comprehensive IPO data...")
            all_data = self.data_manager.collect_ipo_data(company_name, ipo_details)
            
            # Step 2: Analyze prospectus integration results
            print("📊 Step 2: Analyzing prospectus integration...")
            
            enhanced_data = all_data.get('enhanced_prospectus')
            prospectus_quality = all_data.get('prospectus_quality', {})
            
            print(f"   Extraction method: {prospectus_quality.get('extraction_method', 'unknown')}")
            
            if enhanced_data:
                print(f"   ✅ Enhanced data available (quality: {enhanced_data.data_quality_score:.2f})")
                
                # Step 3: Integrate with other data sources
                print("🔗 Step 3: Cross-referencing with market data...")
                
                market_data = {
                    'company_news': len(all_data.get('company_news', [])),
                    'market_news': len(all_data.get('market_news', [])),
                    'sector_news': len(all_data.get('sector_news', [])),
                    'recent_ipos': len(all_data.get('recent_ipo_performance', []))
                }
                
                print(f"   📰 Company news articles: {market_data['company_news']}")
                print(f"   📈 Market news articles: {market_data['market_news']}")
                print(f"   🏭 Sector news articles: {market_data['sector_news']}")
                print(f"   📋 Recent IPO references: {market_data['recent_ipos']}")
                
                # Step 4: Generate integrated analysis
                print("🧮 Step 4: Generating integrated analysis...")
                
                analysis_summary = self._generate_analysis_summary(enhanced_data, all_data)
                
                print("✅ Workflow completed successfully!")
                print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
                
                return analysis_summary
                
            else:
                error_msg = all_data.get('prospectus_error', 'Unknown error')
                print(f"   ❌ Enhanced data not available: {error_msg}")
                print("   🔄 Falling back to basic analysis...")
                
                # Use basic prospectus data if available
                basic_data = all_data.get('prospectus_financials')
                if basic_data:
                    print("   📊 Basic prospectus data available")
                    return self._generate_basic_analysis_summary(basic_data, all_data)
                else:
                    print("   📋 Using market data only")
                    return self._generate_market_only_summary(all_data)
                    
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            return {'error': str(e), 'processing_time': time.time() - start_time}
    
    def _generate_analysis_summary(self, enhanced_data: EnhancedFinancialData, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        
        # Financial analysis
        financial_summary = {}
        if enhanced_data.revenue_data:
            revenues = list(enhanced_data.revenue_data.values())
            financial_summary['latest_revenue'] = max(revenues)
            financial_summary['revenue_trend'] = 'growing' if len(revenues) > 1 and revenues[-1] > revenues[0] else 'stable'
        
        if enhanced_data.profit_data:
            profits = list(enhanced_data.profit_data.values())
            financial_summary['latest_profit'] = max(profits)
            financial_summary['profitability'] = 'positive' if profits[-1] > 0 else 'negative'
        
        # Business assessment
        business_summary = {
            'description_available': len(enhanced_data.business_description) > 100,
            'risk_factors_count': len(enhanced_data.risk_factors),
            'strengths_count': len(enhanced_data.company_strengths),
            'fund_usage_clarity': len(enhanced_data.use_of_funds) >= 3
        }
        
        # Market context
        market_summary = {
            'news_sentiment_data': len(all_data.get('company_news', [])) > 0,
            'sector_context': len(all_data.get('sector_news', [])) > 0,
            'market_conditions': all_data.get('indian_market_data', {})
        }
        
        # Overall assessment
        overall_score = (
            enhanced_data.data_quality_score * 0.4 +  # 40% weight to data quality
            (1.0 if financial_summary.get('revenue_trend') == 'growing' else 0.5) * 0.3 +  # 30% to growth
            (min(business_summary['risk_factors_count'] / 5, 1.0)) * 0.2 +  # 20% to risk assessment
            (1.0 if market_summary['news_sentiment_data'] else 0.5) * 0.1  # 10% to market data
        )
        
        print(f"\n📊 INTEGRATED ANALYSIS SUMMARY")
        print(f"Overall Assessment Score: {overall_score:.2f}/1.00")
        print(f"Data Quality: {enhanced_data.data_quality_score:.2f}")
        print(f"Financial Health: {financial_summary}")
        print(f"Business Analysis: {business_summary}")
        print(f"Market Context: Available" if market_summary['news_sentiment_data'] else "Limited")
        
        return {
            'overall_score': overall_score,
            'data_quality': enhanced_data.data_quality_score,
            'financial_summary': financial_summary,
            'business_summary': business_summary,
            'market_summary': market_summary,
            'recommendation': 'strong_buy' if overall_score >= 0.8 else 'buy' if overall_score >= 0.6 else 'hold' if overall_score >= 0.4 else 'avoid'
        }
    
    def _generate_basic_analysis_summary(self, basic_data: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary from basic prospectus data."""
        return {
            'data_source': 'basic_prospectus',
            'financial_data_available': basic_data is not None,
            'market_data_available': len(all_data.get('company_news', [])) > 0,
            'recommendation': 'manual_review_required'
        }
    
    def _generate_market_only_summary(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary from market data only."""
        return {
            'data_source': 'market_only',
            'news_articles': len(all_data.get('company_news', [])),
            'sector_articles': len(all_data.get('sector_news', [])),
            'recommendation': 'insufficient_data'
        }

def main():
    """Main demonstration function."""
    print("🎯 Enhanced IPO Prospectus Integration - Practical Demo")
    print("=" * 70)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced prospectus integration not available")
        print("📦 Install required packages: pip install tabula-py")
        return
    
    # Initialize demo
    try:
        demo = IPOProspectusDemo()
    except ImportError as e:
        print(f"❌ Demo initialization failed: {e}")
        return
    
    # Test companies (both real and hypothetical)
    test_cases = [
        {
            'company_name': 'Zomato Limited',
            'ipo_details': {
                'sector': 'Technology',
                'price_range': '72-76',
                'exchange': 'NSE',
                'issue_size': '9375 Cr'
            }
        },
        {
            'company_name': 'Paytm One97 Communications',
            'ipo_details': {
                'sector': 'Financial Services',
                'price_range': '2080-2150',
                'exchange': 'NSE/BSE',
                'issue_size': '18300 Cr'
            }
        },
        {
            'company_name': 'Life Insurance Corporation',
            'ipo_details': {
                'sector': 'Financial Services',
                'price_range': '902-949',
                'exchange': 'NSE/BSE',
                'issue_size': '21000 Cr'
            }
        }
    ]
    
    print("🚀 Running comprehensive demonstrations...")
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        company_name = test_case['company_name']
        ipo_details = test_case['ipo_details']
        
        print(f"\n{'='*70}")
        print(f"📋 TEST CASE {i}: {company_name}")
        print(f"{'='*70}")
        
        # Demo 1: Quick Assessment
        assessment = demo.demo_quick_assessment(company_name)
        
        if assessment.get('data_available'):
            # Demo 2: Full Data Extraction
            enhanced_data = demo.demo_full_extraction(company_name)
            
            # Demo 3: Complete Integration Workflow
            workflow_result = demo.demo_integration_workflow(company_name, ipo_details)
            
            results[company_name] = {
                'assessment': assessment,
                'enhanced_data': enhanced_data is not None,
                'workflow_result': workflow_result
            }
        else:
            print("⏭️  Skipping detailed demos due to no data availability")
            results[company_name] = {
                'assessment': assessment,
                'enhanced_data': False,
                'workflow_result': None
            }
        
        # Add delay between tests to be respectful to SEBI servers
        if i < len(test_cases):
            print("\n⏸️  Pausing 2 seconds before next test...")
            time.sleep(2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 DEMONSTRATION SUMMARY")
    print(f"{'='*70}")
    
    successful_assessments = sum(1 for r in results.values() if r['assessment'].get('data_available', False))
    successful_extractions = sum(1 for r in results.values() if r.get('enhanced_data', False))
    successful_workflows = sum(1 for r in results.values() if r.get('workflow_result') and r['workflow_result'].get('overall_score', 0) > 0)
    
    total_tests = len(results)
    
    print(f"📈 Results Summary:")
    print(f"   Companies tested: {total_tests}")
    print(f"   Data available: {successful_assessments}/{total_tests}")
    print(f"   Successful extractions: {successful_extractions}/{total_tests}")
    print(f"   Complete workflows: {successful_workflows}/{total_tests}")
    
    if successful_extractions > 0:
        print(f"\n✅ Enhanced prospectus integration is functional!")
        print(f"💡 Ready for production use with real IPO companies")
    else:
        print(f"\n⚠️  No data extracted in this demo")
        print(f"💡 This is normal if:")
        print(f"   - SEBI website structure has changed")
        print(f"   - Test companies don't have accessible filings")
        print(f"   - Network connectivity issues")
        print(f"\n🔧 The integration is still ready for use with:")
        print(f"   - Companies with accessible SEBI filings")
        print(f"   - Local PDF documents via direct parsing")
        print(f"   - Manual testing with your target IPO companies")
    
    print(f"\n🎯 Demo completed! Check the enhanced_prospectus_guide.md for detailed usage instructions.")

if __name__ == "__main__":
    main()
