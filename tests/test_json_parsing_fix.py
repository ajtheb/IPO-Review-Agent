#!/usr/bin/env python3
"""
Test script to demonstrate improved JSON parsing for LLM responses
"""

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
import json

def test_json_parsing_improvements():
    """Test the improved JSON parsing with various malformed responses."""
    
    analyzer = LLMProspectusAnalyzer(provider='gemini', use_vector_db=False)
    
    # Test cases with different types of malformed JSON
    test_cases = [
        {
            "name": "Truncated JSON (original error case)",
            "response": '''```json
{
  "trailing_pe_ratio": 28.0,
  "forward_pe_ratio": null,
  "price_to_book_ratio": 2.8,
  "price_to_sales_ratio": 1.9,
  "ev_to_ebitda_ratio": 24.0...'''
        },
        {
            "name": "JSON with markdown and missing closing braces",
            "response": '''```json
{
  "trailing_pe_ratio": 28.0,
  "price_to_book_ratio": 2.8,
  "return_on_equity": 18.5,
  "current_ratio": 1.8,
  "debt_to_equity_ratio": 0.45,
  "extraction_confidence": 0.8,
  "data_completeness": 0.7
```'''
        },
        {
            "name": "JSON with trailing comma and incomplete structure",
            "response": '''{
  "trailing_pe_ratio": 28.0,
  "price_to_book_ratio": 2.8,
  "return_on_equity": 18.5,
  "current_ratio": 1.8,
  "debt_to_equity_ratio": 0.45,
  "extraction_confidence": 0.8,'''
        },
        {
            "name": "JSON with extra quotes around null values",
            "response": '''{
  "trailing_pe_ratio": 28.0,
  "forward_pe_ratio": "null",
  "price_to_book_ratio": 2.8,
  "return_on_equity": 18.5,
  "extraction_confidence": 0.8,
  "data_completeness": 0.7
}'''
        }
    ]
    
    print("Testing JSON parsing improvements...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 40)
        
        # Show original response
        print(f"Original response: {test_case['response'][:100]}...")
        
        try:
            # Apply the JSON fixes
            fixed_json = analyzer._fix_json_issues(test_case['response'])
            print(f"Fixed JSON: {fixed_json[:200]}...")
            
            # Try to parse the fixed JSON
            parsed_data = json.loads(fixed_json)
            print(f"✅ Successfully parsed! Found {len(parsed_data)} fields")
            
            # Show some extracted values
            if 'trailing_pe_ratio' in parsed_data:
                print(f"   P/E Ratio: {parsed_data['trailing_pe_ratio']}")
            if 'return_on_equity' in parsed_data:
                print(f"   ROE: {parsed_data['return_on_equity']}")
            if 'extraction_confidence' in parsed_data:
                print(f"   Confidence: {parsed_data['extraction_confidence']}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing still failed: {e}")
            
            # Test partial extraction as fallback
            print("   Trying partial extraction...")
            partial_data = analyzer._extract_partial_financial_data(test_case['response'])
            if partial_data:
                print(f"   ✅ Partial extraction successful! Found {sum(1 for v in partial_data.values() if v is not None)} metrics")
                if partial_data.get('trailing_pe_ratio'):
                    print(f"      P/E Ratio: {partial_data['trailing_pe_ratio']}")
                if partial_data.get('return_on_equity'):
                    print(f"      ROE: {partial_data['return_on_equity']}")
            else:
                print("   ❌ Partial extraction also failed")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("JSON parsing improvement tests completed!")

if __name__ == "__main__":
    test_json_parsing_improvements()
