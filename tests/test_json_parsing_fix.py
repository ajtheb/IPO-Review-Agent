"""
Test suite for LLMProspectusAnalyzer's JSON extraction/parsing helpers.

These are pure string -> dict helpers (no LLM/network calls), and have been
the subject of several past bug fixes (truncated JSON, missing commas,
quoted null/boolean values). use_vector_db=False keeps instantiation free
of any ChromaDB disk writes.
"""

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer


class TestExtractJsonFromResponse(unittest.TestCase):
    """Tests for LLMProspectusAnalyzer._extract_json_from_response."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = LLMProspectusAnalyzer(provider="groq", use_vector_db=False)

    def test_clean_json(self):
        result = self.analyzer._extract_json_from_response('{"a": 1, "b": "text"}')
        self.assertEqual(result, '{"a": 1, "b": "text"}')

    def test_json_in_markdown_json_fence(self):
        response = '```json\n{"a": 1}\n```'
        result = self.analyzer._extract_json_from_response(response)
        self.assertEqual(result, '{"a": 1}')

    def test_json_in_plain_markdown_fence(self):
        response = '```\n{"a": 1}\n```'
        result = self.analyzer._extract_json_from_response(response)
        self.assertEqual(result, '{"a": 1}')

    def test_ignores_trailing_text_after_json(self):
        response = '{"a": 1} some trailing commentary from the LLM'
        result = self.analyzer._extract_json_from_response(response)
        self.assertEqual(result, '{"a": 1}')

    def test_no_json_returns_none(self):
        result = self.analyzer._extract_json_from_response('no json here at all')
        self.assertIsNone(result)


class TestParseJsonWithFallbacks(unittest.TestCase):
    """Tests for LLMProspectusAnalyzer._parse_json_with_fallbacks."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = LLMProspectusAnalyzer(provider="groq", use_vector_db=False)

    def test_clean_json(self):
        data = self.analyzer._parse_json_with_fallbacks('{"trailing_pe_ratio": 28.0}')
        self.assertEqual(data['trailing_pe_ratio'], 28.0)

    def test_truncated_json_with_markdown_fence(self):
        # Real bug case: LLM response cut off mid-value, wrapped in a ```json fence.
        response = '''```json
{
  "trailing_pe_ratio": 28.0,
  "forward_pe_ratio": null,
  "price_to_book_ratio": 2.8,
  "price_to_sales_ratio": 1.9,
  "ev_to_ebitda_ratio": 24.0...'''
        data = self.analyzer._parse_json_with_fallbacks(response, description="test")
        self.assertIsNotNone(data)
        self.assertEqual(data.get('trailing_pe_ratio'), 28.0)
        self.assertEqual(data.get('price_to_book_ratio'), 2.8)

    def test_missing_closing_brace(self):
        response = '''```json
{
  "trailing_pe_ratio": 28.0,
  "price_to_book_ratio": 2.8,
  "return_on_equity": 18.5,
  "current_ratio": 1.8,
  "debt_to_equity_ratio": 0.45,
  "extraction_confidence": 0.8,
  "data_completeness": 0.7
```'''
        data = self.analyzer._parse_json_with_fallbacks(response, description="test")
        self.assertIsNotNone(data)
        self.assertEqual(data.get('return_on_equity'), 18.5)

    def test_trailing_comma_and_incomplete_structure(self):
        response = '''{
  "trailing_pe_ratio": 28.0,
  "price_to_book_ratio": 2.8,
  "return_on_equity": 18.5,
  "current_ratio": 1.8,
  "debt_to_equity_ratio": 0.45,
  "extraction_confidence": 0.8,'''
        data = self.analyzer._parse_json_with_fallbacks(response, description="test")
        self.assertIsNotNone(data)
        self.assertEqual(data.get('debt_to_equity_ratio'), 0.45)

    def test_quoted_null_value_stays_a_string_when_json_is_otherwise_valid(self):
        # _fix_json_issues normalizes "null"/"true"/"false" strings to real JSON
        # literals, but only runs when direct parsing fails first. This input is
        # syntactically valid JSON on its own, so "Direct JSON extraction" (attempt 1)
        # succeeds and short-circuits before the fix-up tier ever sees it.
        response = '''{
  "trailing_pe_ratio": 28.0,
  "forward_pe_ratio": "null",
  "extraction_confidence": 0.8
}'''
        data = self.analyzer._parse_json_with_fallbacks(response, description="test")
        self.assertIsNotNone(data)
        self.assertEqual(data.get('forward_pe_ratio'), "null")

    def test_quoted_null_value_normalized_when_direct_parse_fails(self):
        # Same quoted "null" value, but now paired with a trailing comma that makes
        # direct parsing fail, forcing the "Fixed JSON" tier (_fix_json_issues) to run
        # and normalize "null" -> real None.
        response = '''{
  "trailing_pe_ratio": 28.0,
  "forward_pe_ratio": "null",
  "extraction_confidence": 0.8,
}'''
        data = self.analyzer._parse_json_with_fallbacks(response, description="test")
        self.assertIsNotNone(data)
        self.assertIsNone(data.get('forward_pe_ratio'))

    def test_evaluates_arithmetic_expressions(self):
        response = '{"gross_profit_margin": "(256.93 / 11884.89) * 100"}'
        data = self.analyzer._parse_json_with_fallbacks(response, description="test")
        self.assertIsNotNone(data)
        self.assertAlmostEqual(data['gross_profit_margin'], (256.93 / 11884.89) * 100, places=2)

    def test_garbage_input_returns_none(self):
        data = self.analyzer._parse_json_with_fallbacks('not json, not fixable, not anything useful')
        self.assertIsNone(data)


if __name__ == '__main__':
    unittest.main()
