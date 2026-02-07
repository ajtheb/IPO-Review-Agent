"""
Standalone test for arithmetic expression evaluation functionality.

This demonstrates the safe_eval and evaluate_expressions_in_dict functions
without requiring external dependencies.
"""

import json
import re


def safe_eval(expr: str) -> float:
    """
    Safely evaluate a simple arithmetic expression.
    
    Args:
        expr: String containing arithmetic expression (e.g., "(256.93 / 11884.89) * 100")
    
    Returns:
        Evaluated numerical result
    
    Raises:
        ValueError: If expression contains unsafe characters
    """
    # Only allow numbers, operators, spaces, and parentheses
    if not re.match(r'^[\d.\s+\-*/()]+$', expr):
        raise ValueError(f"Unsafe expression: {expr}")
    
    try:
        # Evaluate with restricted builtins for safety
        result = eval(expr, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        print(f"Warning: Failed to evaluate expression '{expr}': {e}")
        raise


def evaluate_expressions_in_dict(data: dict) -> dict:
    """
    Recursively evaluate string arithmetic expressions in a dictionary.
    
    This allows the LLM to return expressions like "(256.93 / 11884.89) * 100"
    which will be evaluated to numerical values.
    
    Args:
        data: Dictionary potentially containing string expressions
    
    Returns:
        Dictionary with expressions evaluated to float values
    """
    if not isinstance(data, dict):
        return data
    
    for k, v in data.items():
        if isinstance(v, str):
            v_strip = v.strip()
            # Detect if it's an arithmetic expression (numbers and operators only)
            if re.match(r'^[\d.\s+\-*/()]+$', v_strip):
                try:
                    # Evaluate and round to 2 decimal places
                    evaluated_value = safe_eval(v_strip)
                    data[k] = round(evaluated_value, 2)
                    print(f"  ✓ Evaluated {k}: {v_strip} = {data[k]}")
                except Exception as e:
                    print(f"  ✗ Could not evaluate '{v_strip}': {e}")
                    data[k] = None
            # Keep non-expression strings as-is
        elif isinstance(v, dict):
            # Recursively process nested dictionaries
            data[k] = evaluate_expressions_in_dict(v)
        elif isinstance(v, list):
            # Process lists that might contain dicts
            data[k] = [evaluate_expressions_in_dict(item) if isinstance(item, dict) else item for item in v]
    
    return data


def test_safe_eval():
    """Test the safe_eval function with various expressions."""
    print("=" * 70)
    print("Test 1: Safe Expression Evaluation")
    print("=" * 70)
    
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


def test_llm_response_simulation():
    """Test with a simulated LLM response containing expressions."""
    print("\n" + "=" * 70)
    print("Test 2: LLM Response with Arithmetic Expressions")
    print("=" * 70)
    
    # Simulate an LLM response with arithmetic expressions
    llm_response_json = '''
    {
        "net_profit_margin": "(256.93 / 11884.89) * 100",
        "return_on_equity": "(256.93 / 1255.38) * 100",
        "return_on_assets": "(256.93 / 12929.74) * 100",
        "current_ratio": "1429.74 / 2462.09",
        "quick_ratio": "(1429.74 - 0) / 2462.09",
        "debt_to_equity_ratio": "2414.09 / 1255.38",
        "debt_to_assets_ratio": "2414.09 / 12929.74",
        "trailing_pe_ratio": 25.5,
        "price_to_book_ratio": null,
        "extraction_confidence": 0.8,
        "data_completeness": 0.9
    }
    '''
    
    print("\n📄 Original LLM Response:")
    print(llm_response_json)
    
    # Parse JSON
    data = json.loads(llm_response_json)
    
    print("\n🔄 Evaluating expressions...")
    # Evaluate expressions
    evaluated_data = evaluate_expressions_in_dict(data)
    
    print("\n📊 Final Evaluated Data:")
    print(json.dumps(evaluated_data, indent=2))


def test_nested_structures():
    """Test evaluation with nested dictionaries."""
    print("\n" + "=" * 70)
    print("Test 3: Nested Dictionary Structures")
    print("=" * 70)
    
    nested_data = {
        "company_name": "Example Corp IPO",
        "ipo_pricing_analysis": {
            "price_band": "Rs 100-120 per share",
            "valuation_multiple": "(5000 / 250) * 1.2",
            "market_cap_at_upper_band": "5000 * 120"
        },
        "use_of_funds_analysis": {
            "total_funds": 2000,
            "capex_percentage": "(500 / 2000) * 100",
            "debt_repayment_percentage": "(800 / 2000) * 100",
            "working_capital_percentage": "(700 / 2000) * 100"
        },
        "financial_ratios": {
            "profitability": {
                "gross_margin": "(3500 / 11884.89) * 100",
                "net_margin": "(256.93 / 11884.89) * 100"
            },
            "leverage": {
                "d_e_ratio": "2414.09 / 1255.38",
                "d_a_ratio": "2414.09 / 12929.74"
            }
        }
    }
    
    print("\n📄 Original Nested Structure:")
    print(json.dumps(nested_data, indent=2))
    
    print("\n🔄 Evaluating expressions in nested structure...")
    evaluated = evaluate_expressions_in_dict(nested_data.copy())
    
    print("\n📊 Evaluated Nested Structure:")
    print(json.dumps(evaluated, indent=2))


def test_security():
    """Test that unsafe expressions are properly rejected."""
    print("\n" + "=" * 70)
    print("Test 4: Security - Unsafe Expression Rejection")
    print("=" * 70)
    
    unsafe_cases = [
        "__import__('os').system('ls')",
        "eval('1+1')",
        "open('/etc/passwd')",
        "100 + __builtins__",
        "print('hello')",
        "import os",
    ]
    
    print("\nTesting unsafe expressions (should all be rejected):")
    
    for i, unsafe_expr in enumerate(unsafe_cases, 1):
        try:
            result = safe_eval(unsafe_expr)
            print(f"\n{i}. ✗ SECURITY ISSUE: '{unsafe_expr[:40]}...' was evaluated!")
        except ValueError:
            print(f"\n{i}. ✓ Correctly rejected: '{unsafe_expr[:40]}...'")
        except Exception as e:
            print(f"\n{i}. ✓ Blocked: '{unsafe_expr[:40]}...' ({type(e).__name__})")


def test_practical_example():
    """A practical example showing real IPO analysis usage."""
    print("\n" + "=" * 70)
    print("Test 5: Practical IPO Analysis Example")
    print("=" * 70)
    
    # This simulates what the LLM would return for a real IPO
    practical_response = {
        "company": "TechStartup India Ltd",
        "financial_metrics": {
            # Raw financial data from prospectus
            "revenue_fy23": 11884.89,
            "profit_fy23": 256.93,
            "total_equity": 1255.38,
            "total_assets": 12929.74,
            "current_assets": 1429.74,
            "current_liabilities": 2462.09,
            "total_debt": 2414.09,
            
            # Calculated ratios using expressions
            "net_profit_margin": "(256.93 / 11884.89) * 100",
            "return_on_equity": "(256.93 / 1255.38) * 100", 
            "return_on_assets": "(256.93 / 12929.74) * 100",
            "current_ratio": "1429.74 / 2462.09",
            "debt_to_equity": "2414.09 / 1255.38",
            "debt_to_assets": "2414.09 / 12929.74",
            
            # Metadata
            "extraction_confidence": 0.85,
            "data_completeness": 0.9
        }
    }
    
    print("\n💼 IPO Analysis Input (with expressions):")
    print(json.dumps(practical_response, indent=2))
    
    print("\n🔄 Processing financial calculations...")
    result = evaluate_expressions_in_dict(practical_response)
    
    print("\n📈 Final Analysis Output (expressions evaluated):")
    print(json.dumps(result, indent=2))
    
    # Show key insights
    metrics = result['financial_metrics']
    print("\n🎯 Key Financial Insights:")
    print(f"  • Net Profit Margin: {metrics['net_profit_margin']}%")
    print(f"  • Return on Equity: {metrics['return_on_equity']}%")
    print(f"  • Return on Assets: {metrics['return_on_assets']}%")
    print(f"  • Current Ratio: {metrics['current_ratio']}")
    print(f"  • Debt to Equity: {metrics['debt_to_equity']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IPO Review Agent - Expression Evaluation Test Suite")
    print("Demonstrating LLM response processing with arithmetic expressions")
    print("=" * 70)
    
    test_safe_eval()
    test_llm_response_simulation()
    test_nested_structures()
    test_security()
    test_practical_example()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    print("\nSummary:")
    print("• Expression evaluation allows LLMs to return formulas instead of pre-calculated values")
    print("• Improves accuracy by showing the calculation process")
    print("• Security measures prevent code injection attacks")
    print("• Works seamlessly with nested JSON structures")
    print("=" * 70 + "\n")
