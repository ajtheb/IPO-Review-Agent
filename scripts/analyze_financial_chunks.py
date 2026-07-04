"""
Financial Chunks Analyzer

Analyzes structured chunks to identify and categorize financial information
from the Vidya Wires IPO prospectus.

Author: IPO Review Agent
Date: 2026-02-17
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class FinancialChunksAnalyzer:
    """Analyzes chunks to identify financial information."""
    
    # Financial keywords for detailed classification
    FINANCIAL_KEYWORDS = {
        'revenue': ['revenue', 'sales', 'turnover', 'income from operations'],
        'profit': ['profit', 'net profit', 'profit after tax', 'pat', 'profit before tax', 'pbt', 'earnings'],
        'loss': ['loss', 'net loss', 'losses'],
        'balance_sheet': ['balance sheet', 'assets', 'liabilities', 'equity', 'capital'],
        'cash_flow': ['cash flow', 'cash flows', 'cash generated', 'cash used'],
        'income_statement': ['income statement', 'profit and loss', 'statement of profit'],
        'ratios': ['ratio', 'margin', 'return on', 'debt to equity', 'current ratio', 'roe', 'roce'],
        'expenses': ['expenses', 'expenditure', 'costs', 'depreciation', 'amortization'],
        'dividends': ['dividend', 'dividend policy', 'dividend per share'],
        'capitalization': ['capitalization', 'share capital', 'equity share capital'],
        'indebtedness': ['debt', 'borrowings', 'loan', 'credit facility', 'term loan'],
        'working_capital': ['working capital', 'current assets', 'current liabilities'],
        'tax': ['tax', 'income tax', 'deferred tax', 'tax expense'],
        'reserves': ['reserves', 'retained earnings', 'surplus'],
        'ebitda': ['ebitda', 'operating profit', 'operating income'],
        'valuation': ['valuation', 'market cap', 'enterprise value', 'price to earnings', 'p/e ratio'],
    }
    
    def __init__(self, chunks_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
        """
        self.chunks_dir = Path(chunks_dir)
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
        
        self.financial_chunks = []
        self.stats = defaultdict(int)
        self.category_chunks = defaultdict(list)
        
    def load_all_chunks(self) -> List[Dict]:
        """Load all chunk files."""
        chunks = []
        chunk_files = sorted(self.chunks_dir.glob("chunk_*.json"))
        
        print(f"Loading {len(chunk_files)} chunk files...")
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
        
        return chunks
    
    def is_financial_chunk(self, chunk: Dict) -> bool:
        """
        Check if a chunk contains financial information.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            True if chunk contains financial info
        """
        metadata = chunk.get('metadata', {})
        content = chunk.get('text', '').lower()
        
        # Check if content_type is financial
        if metadata.get('content_type') == 'financial':
            return True
        
        # Check if financial is in categories
        categories = metadata.get('categories', [])
        if 'financial' in categories:
            return True
        
        # Check for financial keywords in content
        financial_keywords = [
            'revenue', 'profit', 'loss', 'balance sheet', 'cash flow',
            'income statement', 'assets', 'liabilities', 'equity',
            'financial', 'capitalization', 'indebtedness', 'dividend',
            '₹', 'million', 'crore', 'rupees', 'rs.', 'inr'
        ]
        
        return any(keyword in content for keyword in financial_keywords)
    
    def classify_financial_content(self, chunk: Dict) -> List[str]:
        """
        Classify the type of financial content in the chunk.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            List of financial content types
        """
        content = chunk.get('text', '').lower()
        classifications = []
        
        for category, keywords in self.FINANCIAL_KEYWORDS.items():
            if any(keyword in content for keyword in keywords):
                classifications.append(category)
        
        return classifications
    
    def extract_financial_numbers(self, text: str) -> List[str]:
        """
        Extract financial numbers from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of financial numbers found
        """
        import re
        
        # Pattern for Indian currency amounts
        patterns = [
            r'₹\s*[\d,]+\.?\d*\s*(?:million|crore|lakh|thousand)?',
            r'INR\s*[\d,]+\.?\d*\s*(?:million|crore|lakh|thousand)?',
            r'Rs\.?\s*[\d,]+\.?\d*\s*(?:million|crore|lakh|thousand)?',
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        return numbers[:10]  # Return first 10 matches
    
    def analyze(self):
        """Analyze all chunks for financial information."""
        print("\n" + "="*80)
        print("FINANCIAL CHUNKS ANALYSIS")
        print("="*80)
        
        chunks = self.load_all_chunks()
        print(f"\nTotal chunks loaded: {len(chunks)}")
        
        # Analyze each chunk
        for chunk in chunks:
            if self.is_financial_chunk(chunk):
                self.financial_chunks.append(chunk)
                
                # Get classifications
                classifications = self.classify_financial_content(chunk)
                
                # Store by classification
                for classification in classifications:
                    self.category_chunks[classification].append(chunk['chunk_id'])
                
                # Update stats
                self.stats['total_financial_chunks'] += 1
                metadata = chunk.get('metadata', {})
                if metadata.get('has_tables'):
                    self.stats['chunks_with_tables'] += 1
                if metadata.get('has_numbers'):
                    self.stats['chunks_with_numbers'] += 1
        
        # Print results
        self._print_results()
        
        # Save detailed results
        self._save_results()
    
    def _print_results(self):
        """Print analysis results."""
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total Financial Chunks: {self.stats['total_financial_chunks']}")
        print(f"Chunks with Tables: {self.stats['chunks_with_tables']}")
        print(f"Chunks with Numbers: {self.stats['chunks_with_numbers']}")
        
        print("\n" + "-"*80)
        print("FINANCIAL CONTENT BREAKDOWN")
        print("-"*80)
        
        sorted_categories = sorted(self.category_chunks.items(), 
                                  key=lambda x: len(x[1]), 
                                  reverse=True)
        
        for category, chunk_ids in sorted_categories:
            print(f"\n{category.upper().replace('_', ' ')}")
            print(f"  • Number of chunks: {len(chunk_ids)}")
            print(f"  • Chunk IDs (first 10): {chunk_ids[:10]}")
        
        print("\n" + "-"*80)
        print("SAMPLE FINANCIAL CHUNKS")
        print("-"*80)
        
        # Show samples from different categories
        samples = {}
        for category, chunk_ids in sorted_categories[:5]:  # Top 5 categories
            if chunk_ids:
                chunk_id = chunk_ids[0]
                chunk = next((c for c in self.financial_chunks if c['chunk_id'] == chunk_id), None)
                if chunk:
                    samples[category] = chunk
        
        for category, chunk in samples.items():
            print(f"\n{category.upper().replace('_', ' ')} - Chunk #{chunk['chunk_id']}")
            print(f"Section: {chunk.get('metadata', {}).get('section', 'N/A')}")
            print(f"Pages: {chunk.get('metadata', {}).get('pages', [])}")
            text = chunk.get('text', '')
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"Preview: {preview}")
            
            # Extract and show financial numbers
            numbers = self.extract_financial_numbers(text)
            if numbers:
                print(f"Financial figures found: {', '.join(numbers[:5])}")
    
    def _save_results(self):
        """Save analysis results to file."""
        output_dir = self.chunks_dir.parent / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary = {
            'total_chunks_analyzed': self.stats.get('total_financial_chunks', 0) + 
                                    len(self.load_all_chunks()) - self.stats.get('total_financial_chunks', 0),
            'financial_chunks': self.stats['total_financial_chunks'],
            'chunks_with_tables': self.stats['chunks_with_tables'],
            'chunks_with_numbers': self.stats['chunks_with_numbers'],
            'content_breakdown': {
                category: {
                    'count': len(chunk_ids),
                    'chunk_ids': chunk_ids
                }
                for category, chunk_ids in self.category_chunks.items()
            }
        }
        
        summary_file = output_dir / "financial_chunks_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Summary saved to: {summary_file}")
        
        # Save detailed financial chunks
        financial_chunks_file = output_dir / "financial_chunks_detailed.json"
        with open(financial_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.financial_chunks, f, indent=2, ensure_ascii=False)
        print(f"✅ Detailed financial chunks saved to: {financial_chunks_file}")
        
        # Create a report
        self._create_report(output_dir)
    
    def _create_report(self, output_dir: Path):
        """Create a human-readable report."""
        report = f"""
{'='*80}
FINANCIAL CHUNKS ANALYSIS REPORT
{'='*80}

Document: Vidya Wires IPO Prospectus
Analysis Date: 2026-02-17

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Total Financial Chunks Identified: {self.stats['total_financial_chunks']}
Chunks with Financial Tables: {self.stats['chunks_with_tables']}
Chunks with Numerical Data: {self.stats['chunks_with_numbers']}

{'='*80}
FINANCIAL CONTENT CATEGORIES
{'='*80}
"""
        
        sorted_categories = sorted(self.category_chunks.items(), 
                                  key=lambda x: len(x[1]), 
                                  reverse=True)
        
        for category, chunk_ids in sorted_categories:
            percentage = (len(chunk_ids) / self.stats['total_financial_chunks'] * 100) if self.stats['total_financial_chunks'] > 0 else 0
            report += f"""
{category.upper().replace('_', ' ')}:
  • Chunks: {len(chunk_ids)}
  • Percentage: {percentage:.1f}%
  • Representative Chunk IDs: {chunk_ids[:20]}
"""
        
        report += f"""
{'='*80}
KEY FINANCIAL SECTIONS DETECTED
{'='*80}
"""
        
        # Get unique sections from financial chunks
        sections = set()
        for chunk in self.financial_chunks[:100]:  # First 100 financial chunks
            section = chunk.get('metadata', {}).get('section', '')
            if section and len(section) > 5:
                sections.add(section)
        
        for section in sorted(sections)[:30]:  # Show top 30
            report += f"  • {section}\n"
        
        report += f"""
{'='*80}
RECOMMENDATIONS
{'='*80}

Based on this analysis:

1. COMPREHENSIVE COVERAGE: {self.stats['total_financial_chunks']} chunks contain 
   financial information, indicating extensive financial data in the prospectus.

2. DATA-RICH CONTENT: {self.stats['chunks_with_tables']} chunks include financial 
   tables, providing structured quantitative data.

3. FOCUS AREAS: 
   - Revenue and profit analysis chunks
   - Balance sheet and asset information
   - Cash flow statements
   - Debt and capitalization details
   - Financial ratios and metrics

4. SUITABLE FOR ANALYSIS: The chunks are well-structured for:
   - Automated financial statement extraction
   - Ratio calculation and analysis
   - Trend analysis across fiscal years
   - Risk assessment based on financial metrics
   - Investment recommendation generation

{'='*80}
"""
        
        report_file = output_dir / "financial_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Analysis report saved to: {report_file}")
        
        print("\n" + report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze structured chunks for financial information'
    )
    parser.add_argument(
        '--chunks-dir',
        default='structured_chunks/Vidya Wires/chunks',
        help='Directory containing chunk JSON files'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = FinancialChunksAnalyzer(args.chunks_dir)
        analyzer.analyze()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
