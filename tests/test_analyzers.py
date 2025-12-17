"""
Test suite for IPO Review Agent.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models import FinancialMetrics, RiskLevel
from src.analyzers import FinancialAnalyzer, RiskAnalyzer


class TestFinancialAnalyzer(unittest.TestCase):
    """Test cases for FinancialAnalyzer."""
    
    def setUp(self):
        self.analyzer = FinancialAnalyzer()
    
    def test_growth_rate_calculation(self):
        """Test revenue growth rate calculation."""
        # Test positive growth
        data = {'2021': 1000, '2022': 1200, '2023': 1440}
        growth_rate = self.analyzer._calculate_growth_rate(data)
        self.assertAlmostEqual(growth_rate, 0.2, places=2)  # 20% CAGR
        
        # Test negative growth
        data = {'2021': 1000, '2022': 900, '2023': 800}
        growth_rate = self.analyzer._calculate_growth_rate(data)
        self.assertLess(growth_rate, 0)
    
    def test_profit_margin_calculation(self):
        """Test profit margin calculation."""
        revenue = {'2021': 1000, '2022': 1200}
        profit = {'2021': 100, '2022': 150}
        
        margin = self.analyzer._calculate_profit_margin(revenue, profit)
        expected_margin = (100/1000 + 150/1200) / 2
        self.assertAlmostEqual(margin, expected_margin, places=3)


class TestRiskAnalyzer(unittest.TestCase):
    """Test cases for RiskAnalyzer."""
    
    def setUp(self):
        self.analyzer = RiskAnalyzer()
    
    def test_financial_risk_assessment(self):
        """Test financial risk assessment."""
        # Low risk scenario
        metrics = FinancialMetrics(
            revenue={}, profit={}, assets={}, liabilities={},
            revenue_growth_rate=0.20,
            profit_margin=0.15
        )
        
        risk_factors = []
        risk_level = self.analyzer._assess_financial_risk(metrics, risk_factors)
        self.assertEqual(risk_level, RiskLevel.LOW)
        
        # High risk scenario
        metrics_high_risk = FinancialMetrics(
            revenue={}, profit={}, assets={}, liabilities={},
            revenue_growth_rate=-0.10,
            profit_margin=-0.05
        )
        
        risk_factors_high = []
        risk_level_high = self.analyzer._assess_financial_risk(metrics_high_risk, risk_factors_high)
        self.assertIn(risk_level_high, [RiskLevel.HIGH, RiskLevel.MODERATE])


if __name__ == '__main__':
    unittest.main()
