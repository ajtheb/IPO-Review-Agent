"""
Test script to demonstrate arithmetic expression evaluation in financial metrics.

This shows how the LLM can return expressions like "(256.93 / 11884.89) * 100"
which will be automatically evaluated to numerical values.
"""

import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analyzers.llm_prospectus_analyzer import (
    safe_eval, 
    evaluate_expressions_in_dict,
    LLMFinancialMetrics
)
import json


def test_safe_eval():
    """Test the safe_eval function with various expressions."""
    print("=" * 60)
    print("Testing safe_eval function")
    print("=" * 60)
    
    test_cases = [
        ("(256.93 / 11884.89) * 100", "Net profit margin calculation"),
        ("2414.09 / 1255.38", "Debt to equity ratio"),
        ("1429.74 / 2462.09", "Current ratio"),
        ("(256.93 / 1255.38) * 100", "Return on equity percentage"),
        ("100 - 45.5", "Simple subtraction"),
        ("(1500 + 2500) / 4", "Average calculation"),
    ]
    
    for expression, description in test_cases:
        try:
            result = safe_eval(expression)
            print(f"\n✓ {description}")
            print(f"  Expression: {expression}")
            print(f"  Result: {round(result, 2)}")
        except Exception as e:
            print(f"\n✗ {description}")
            print(f"  Expression: {expression}")
            print(f"  Error: {e}")


def test_evaluate_expressions_in_dict():
    """Test evaluating expressions in a dictionary structure."""
    print("\n" + "=" * 60)
    print("Testing evaluate_expressions_in_dict function")
    print("=" * 60)
    
    # Simulate an LLM response with arithmetic expressions
    sample_response = {
        "net_profit_margin": "(256.93 / 11884.89) * 100",
        "return_on_equity": "(256.93 / 1255.38) * 100",
        "return_on_assets": "(256.93 / 12929.74) * 100",
        "current_ratio": "1429.74 / 2462.09",
        "quick_ratio": "(1429.74 - 0) / 2462.09",
        "debt_to_equity_ratio": "2414.09 / 1255.38",
        "debt_to_assets_ratio": "2414.09 / 12929.74",
        "trailing_pe_ratio": 25.5,  # Direct number
        "price_to_book_ratio": None,  # Null value
        "extraction_confidence": 0.8,
        "data_completeness": 0.9
    }
    
    print("\nOriginal data (with expressions):")
    print(json.dumps(sample_response, indent=2))
    
    # Evaluate expressions
    evaluated_data = evaluate_expressions_in_dict(sample_response.copy())
    
    print("\nEvaluated data (expressions converted to numbers):")
    print(json.dumps(evaluated_data, indent=2))
    
    # Show the conversion
    print("\nConversion details:")
    for key in sample_response.keys():
        original = sample_response[key]
        evaluated = evaluated_data[key]
        if original != evaluated and isinstance(original, str):
            print(f"  {key}:")
            print(f"    Before: {original}")
            print(f"    After: {evaluated}")


def test_with_llm_financial_metrics():
    """Test creating LLMFinancialMetrics from evaluated data."""
    print("\n" + "=" * 60)
    print("Testing LLMFinancialMetrics with expression evaluation")
    print("=" * 60)
    
    # Simulate LLM response
    llm_response_text = '''
    {
        "net_profit_margin": "(256.93 / 11884.89) * 100",
        "return_on_equity": "(256.93 / 1255.38) * 100",
        "return_on_assets": "(256.93 / 12929.74) * 100",
        "current_ratio": "1429.74 / 2462.09",
        "quick_ratio": "1429.74 / 2462.09",
        "debt_to_equity_ratio": "2414.09 / 1255.38",
        "debt_to_assets_ratio": "2414.09 / 12929.74",
        "gross_profit_margin": "(3500 / 11884.89) * 100",
        "operating_profit_margin": "(800 / 11884.89) * 100",
        "trailing_pe_ratio": 25.5,
        "extraction_confidence": 0.85,
        "data_completeness": 0.75
    }
    '''
    
    # Parse JSON and evaluate expressions
    data = json.loads(llm_response_text)
    evaluated_data = evaluate_expressions_in_dict(data)
    
    # Create LLMFinancialMetrics object
    metrics = LLMFinancialMetrics(**evaluated_data)
    
    print("\nCreated LLMFinancialMetrics object:")
    print(f"  Net Profit Margin: {metrics.net_profit_margin}%")
    print(f"  Return on Equity: {metrics.return_on_equity}%")
    print(f"  Return on Assets: {metrics.return_on_assets}%")
    print(f"  Current Ratio: {metrics.current_ratio}")
    print(f"  Debt to Equity Ratio: {metrics.debt_to_equity_ratio}")
    print(f"  Trailing P/E Ratio: {metrics.trailing_pe_ratio}")
    print(f"  Extraction Confidence: {metrics.extraction_confidence}")
    print(f"  Data Completeness: {metrics.data_completeness}")


def test_nested_dict_evaluation():
    """Test evaluation with nested dictionaries."""
    print("\n" + "=" * 60)
    print("Testing nested dictionary evaluation")
    print("=" * 60)
    
    nested_data = {
        "ipo_pricing_analysis": {
            "price_band": "Rs 100-120 per share",
            "valuation_multiple": "(5000 / 250) * 1.2",  # Expression in nested dict
            "market_cap": "5000 * 120"
        },
        "use_of_funds_analysis": {
            "capex_percentage": "(500 / 2000) * 100",
            "debt_repayment_percentage": "(800 / 2000) * 100",
            "working_capital_percentage": "(700 / 2000) * 100"
        },
        "company_name": "Example Corp"
    }
    
    print("\nOriginal nested data:")
    print(json.dumps(nested_data, indent=2))
    
    evaluated = evaluate_expressions_in_dict(nested_data.copy())
    
    print("\nEvaluated nested data:")
    print(json.dumps(evaluated, indent=2))


def test_unsafe_expressions():
    """Test that unsafe expressions are rejected."""
    print("\n" + "=" * 60)
    print("Testing unsafe expression rejection")
    print("=" * 60)
    
    unsafe_cases = [
        "__import__('os').system('ls')",  # Code injection attempt
        "eval('1+1')",  # Nested eval
        "open('/etc/passwd')",  # File access
        "100 + __builtins__",  # Access to builtins
    ]
    
    for unsafe_expr in unsafe_cases:
        try:
            result = safe_eval(unsafe_expr)
            print(f"✗ SECURITY ISSUE: Expression '{unsafe_expr}' was evaluated: {result}")
        except ValueError as e:
            print(f"✓ Correctly rejected: '{unsafe_expr[:30]}...'")
        except Exception as e:
            print(f"✓ Blocked with error: '{unsafe_expr[:30]}...' - {type(e).__name__}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IPO Review Agent - Expression Evaluation Test Suite")
    print("=" * 60)
    
    test_safe_eval()
    test_evaluate_expressions_in_dict()
    test_with_llm_financial_metrics()
    test_nested_dict_evaluation()
    test_unsafe_expressions()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
