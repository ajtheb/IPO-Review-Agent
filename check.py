import ast
import re

def safe_eval(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression."""
    # Only allow numbers, operators, and parentheses
    if not re.match(r'^[\d\.\s\+\-\*\/\(\)]+$', expr):
        raise ValueError(f"Unsafe expression: {expr}")
    return eval(expr, {"__builtins__": {}})

def evaluate_expressions_in_dict(data: dict) -> dict:
    """Recursively evaluate string expressions in a dict."""
    for k, v in data.items():
        if isinstance(v, str):
            v_strip = v.strip()
            # Detect if it's an arithmetic expression
            if re.match(r'^[\d\.\s\+\-\*\/\(\)]+$', v_strip):
                try:
                    data[k] = round(safe_eval(v_strip), 2)
                except Exception:
                    data[k] = None
        elif isinstance(v, dict):
            data[k] = evaluate_expressions_in_dict(v)
    return data

# Example usage:
import json

response = '''
{
    "net_profit_margin": "(256.93 / 11884.89) * 100",
    "return_on_equity": "(256.93 / 1255.38) * 100",
    "return_on_assets": "(256.93 / 12929.74) * 100",
    "current_ratio": "(1429.74 / 2462.09)",
    "quick_ratio": "(1429.74 / 2462.09)",
    "debt_to_equity_ratio": "(2414.09 / 1255.38)",
    "debt_to_assets_ratio": "(2414.09 / 12929.74)",
    "extraction_confidence": 0.8,
    "data_completeness": 0.9
}
'''

data = json.loads(response)
clean_data = evaluate_expressions_in_dict(data)
print(clean_data)