#!/usr/bin/env python3
"""
Test Enhanced Indian Financial Pattern Detection

This script validates that the 63 enhanced patterns correctly identify
Indian financial statement structures in IPO prospectus documents.
"""

import re
from typing import Dict, List

# Import the patterns (matching the implementation)
INDIAN_FINANCIAL_PATTERNS = [
    # P&L Statement markers (13 patterns)
    r'particulars.*fy.*20\d{2}',
    r'statement.*of.*profit.*(?:and|&).*loss',
    r'revenue.*from.*operations',
    r'other.*income',
    r'total.*income',
    r'cost.*of.*(?:materials|goods|sales)',
    r'employee.*benefit.*expense',
    r'finance.*costs?',
    r'depreciation.*(?:and|&).*amortization',
    r'profit.*before.*(?:tax|interest)',
    r'profit.*after.*tax',
    r'earnings.*per.*share',
    r'ebitda',
    
    # Balance Sheet markers (17 patterns)
    r'statement.*of.*assets.*(?:and|&).*liabilities',
    r'(?:as|as\s+at).*(?:march|september|june|december).*\d{2}',
    r'assets.*liabilities',
    r'non[-\s]?current.*assets',
    r'current.*assets',
    r'property.*plant.*equipment',
    r'intangible.*assets',
    r'inventories',
    r'trade.*receivables',
    r'cash.*(?:and|&).*cash.*equivalents',
    r'current.*liabilities',
    r'non[-\s]?current.*liabilities',
    r'trade.*payables',
    r'borrowings',
    r'equity.*share.*capital',
    r'reserves.*(?:and|&).*surplus',
    r'total.*equity',
    
    # Cash Flow Statement markers (5 patterns)
    r'statement.*of.*cash.*flows?',
    r'cash.*flow.*from.*operating.*activities',
    r'cash.*flow.*from.*investing.*activities',
    r'cash.*flow.*from.*financing.*activities',
    r'net.*(?:increase|decrease).*in.*cash',
    
    # Financial Ratios markers (10 patterns)
    r'key.*financial.*(?:ratios?|metrics?|indicators?)',
    r'current.*ratio.*\d+\.\d+',
    r'debt.*(?:to|/).*equity.*ratio.*\d+\.\d+',
    r'return.*on.*equity.*\d+\.\d+',
    r'return.*on.*assets.*\d+\.\d+',
    r'return.*on.*capital.*employed',
    r'net.*profit.*margin.*\d+\.\d+',
    r'operating.*profit.*margin.*\d+\.\d+',
    r'interest.*coverage.*ratio',
    r'debt.*service.*coverage.*ratio',
    
    # Indian-specific formats (8 patterns)
    r'₹.*in.*(?:lakhs?|crores?|millions?|thousands?)',
    r'\((?:in|₹).*(?:lakhs?|crores?)\)',
    r'(?:all\s+)?amounts?.*(?:are\s+)?in.*₹?.*(?:lakhs?|crores?|millions?)',
    r'restated.*(?:statement|financials|consolidated|standalone)',
    r'(?:for|as\s+at).*(?:the\s+)?(?:year|period|six\s+months?).*ended',
    r'audited.*(?:financial|results|statements)',
    r'unaudited.*(?:financial|results)',
    
    # Multi-year data patterns (3 patterns)
    r'fy\s*20\d{2}.*fy\s*20\d{2}',
    r'march.*20\d{2}.*march.*20\d{2}',
    r'september.*20\d{2}.*september.*20\d{2}',
    
    # Summary/Key metrics tables (4 patterns)
    r'financial.*highlights?',
    r'key.*(?:performance|financial).*(?:indicators|metrics)',
    r'summary.*of.*(?:financial|restated).*(?:information|data)',
    r'selected.*financial.*(?:data|information)',
    
    # IPO-specific financial sections (4 patterns)
    r'basis.*of.*(?:issue|ipo).*price',
    r'(?:net|book).*asset.*value.*per.*share',
    r'comparison.*with.*(?:peer|listed).*companies',
    r'earning.*per.*share.*(?:basic|diluted)',
]


def test_pattern_matching(text_samples: List[Dict[str, str]]) -> Dict[str, any]:
    """Test pattern matching against various text samples."""
    
    results = {
        'total_samples': len(text_samples),
        'samples_with_matches': 0,
        'total_matches': 0,
        'pattern_coverage': {},
        'sample_results': []
    }
    
    # Initialize pattern coverage tracking
    for i, pattern in enumerate(INDIAN_FINANCIAL_PATTERNS):
        results['pattern_coverage'][f'pattern_{i}'] = 0
    
    # Test each sample
    for sample in text_samples:
        text = sample['text']
        expected_category = sample['category']
        text_lower = text.lower()
        
        matches = []
        match_count = 0
        
        for i, pattern in enumerate(INDIAN_FINANCIAL_PATTERNS):
            if re.search(pattern, text_lower):
                matches.append(i)
                match_count += 1
                results['pattern_coverage'][f'pattern_{i}'] += 1
        
        if match_count > 0:
            results['samples_with_matches'] += 1
        
        results['total_matches'] += match_count
        
        results['sample_results'].append({
            'text_preview': text[:60] + '...' if len(text) > 60 else text,
            'category': expected_category,
            'matches_found': match_count,
            'pattern_indices': matches,
            'detection_success': match_count > 0
        })
    
    return results


def main():
    """Run pattern detection tests."""
    
    print("=" * 80)
    print("ENHANCED INDIAN FINANCIAL PATTERN DETECTION TEST")
    print("=" * 80)
    print(f"\nTotal patterns implemented: {len(INDIAN_FINANCIAL_PATTERNS)}")
    
    # Test samples covering different financial statement types
    test_samples = [
        # P&L Statements
        {
            'text': 'Particulars FY 2024 FY 2023 FY 2022\nRevenue from Operations 11,884.89 10,157.18 9,169.83',
            'category': 'P&L Statement'
        },
        {
            'text': 'EBITDA was ₹2,199.87 lakhs for the year ended March 31, 2024',
            'category': 'P&L - EBITDA'
        },
        {
            'text': 'Profit After Tax (PAT) for FY2024 was ₹256.93 crores',
            'category': 'P&L - PAT'
        },
        {
            'text': 'Statement of Profit and Loss for the year ended March 31, 2024 (in ₹ Lakhs)',
            'category': 'P&L - Header'
        },
        
        # Balance Sheet
        {
            'text': 'Current Assets: Trade Receivables 1,234.56, Inventories 2,345.67, Cash and Cash Equivalents 345.78',
            'category': 'Balance Sheet - Current Assets'
        },
        {
            'text': 'Current Liabilities as at March 31, 2024: Trade Payables 567.89, Short-term Borrowings 941.41',
            'category': 'Balance Sheet - Current Liabilities'
        },
        {
            'text': 'Statement of Assets and Liabilities as at March 31, 2024 (Audited)',
            'category': 'Balance Sheet - Header'
        },
        
        # Financial Ratios
        {
            'text': 'Key Financial Ratios:\nCurrent Ratio: 2.15\nDebt to Equity Ratio: 0.87\nReturn on Equity: 20.5%',
            'category': 'Financial Ratios Table'
        },
        {
            'text': 'Current Ratio improved to 2.15 in FY 2024 from 1.89 in FY 2023',
            'category': 'Ratio Analysis'
        },
        {
            'text': 'Return on Equity (ROE) was 20.5% for FY2024',
            'category': 'ROE'
        },
        
        # Summary Tables
        {
            'text': 'Summary of Restated Financial Information (₹ in Lakhs)\nParticulars FY2024 FY2023 FY2022',
            'category': 'Summary Table'
        },
        {
            'text': 'Financial Highlights: Total Income, EBITDA, PAT, and EPS for last 3 years',
            'category': 'Financial Highlights'
        },
        
        # Multi-year data
        {
            'text': 'Revenue Growth: FY 2024: 11,884.89, FY 2023: 10,157.18, FY 2022: 9,169.83',
            'category': 'Multi-year Trend'
        },
        
        # IPO-specific
        {
            'text': 'Basis of Issue Price: NAV per share was ₹7.85 as on March 31, 2024',
            'category': 'IPO Pricing'
        },
        {
            'text': 'Comparison with Listed Peer Companies in the same industry',
            'category': 'Peer Comparison'
        },
        
        # Indian formats
        {
            'text': 'Restated Consolidated Financial Statements for the period ended September 30, 2024 (Unaudited)',
            'category': 'Indian Format'
        },
        {
            'text': 'All amounts are in ₹ Crores unless otherwise stated',
            'category': 'Currency Header'
        },
    ]
    
    print(f"\nRunning tests on {len(test_samples)} sample texts...")
    print("-" * 80)
    
    results = test_pattern_matching(test_samples)
    
    # Display results
    print(f"\n{'Category':<30} {'Preview':<35} {'Matches':<10} {'Status'}")
    print("-" * 80)
    
    for sample in results['sample_results']:
        status = "✅ PASS" if sample['detection_success'] else "❌ FAIL"
        preview = sample['text_preview'][:32] + '...' if len(sample['text_preview']) > 35 else sample['text_preview']
        print(f"{sample['category']:<30} {preview:<35} {sample['matches_found']:<10} {status}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    success_rate = (results['samples_with_matches'] / results['total_samples']) * 100
    avg_matches = results['total_matches'] / results['total_samples']
    
    print(f"\nSamples tested: {results['total_samples']}")
    print(f"Samples with matches: {results['samples_with_matches']}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total matches: {results['total_matches']}")
    print(f"Average matches per sample: {avg_matches:.2f}")
    
    # Pattern usage statistics
    patterns_used = sum(1 for v in results['pattern_coverage'].values() if v > 0)
    pattern_usage_rate = (patterns_used / len(INDIAN_FINANCIAL_PATTERNS)) * 100
    
    print(f"\nPatterns used: {patterns_used}/{len(INDIAN_FINANCIAL_PATTERNS)}")
    print(f"Pattern usage rate: {pattern_usage_rate:.1f}%")
    
    # Most frequently matched patterns
    print("\nTop 10 Most Matched Patterns:")
    print("-" * 80)
    
    sorted_patterns = sorted(
        results['pattern_coverage'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (pattern_key, match_count) in enumerate(sorted_patterns[:10]):
        if match_count > 0:
            pattern_idx = int(pattern_key.split('_')[1])
            pattern_text = INDIAN_FINANCIAL_PATTERNS[pattern_idx]
            print(f"{i+1:2d}. Pattern: {pattern_text[:60]:<60} Matches: {match_count}")
    
    # Test verdict
    print("\n" + "=" * 80)
    print("TEST VERDICT")
    print("=" * 80)
    
    if success_rate >= 95:
        print("\n🎉 EXCELLENT: Pattern detection is working exceptionally well!")
        print(f"✅ {success_rate:.1f}% of samples matched expected patterns")
    elif success_rate >= 85:
        print("\n✅ GOOD: Pattern detection is working well!")
        print(f"⚠️  {success_rate:.1f}% success rate - consider adding more patterns")
    elif success_rate >= 70:
        print("\n⚠️  ACCEPTABLE: Pattern detection needs improvement")
        print(f"❌ Only {success_rate:.1f}% success rate - review failed samples")
    else:
        print("\n❌ POOR: Pattern detection requires significant enhancement")
        print(f"❌ Only {success_rate:.1f}% success rate - patterns may be too restrictive")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if success_rate < 100:
        print("- Review failed samples to identify missing patterns")
        print("- Test against actual prospectus documents for real-world validation")
    if avg_matches < 2:
        print("- Consider if patterns are too specific (low average matches)")
    if pattern_usage_rate < 50:
        print(f"- Only {pattern_usage_rate:.1f}% of patterns used - review rarely-matched patterns")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    main()
