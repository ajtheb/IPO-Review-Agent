"""
Test suite for IPOReviewAgent's scoring/recommendation logic (src/agent.py).

These methods are pure functions over the dataclasses in src/models - they
don't touch self.data_manager/self.financial_analyzer/etc - so tests
construct the agent via __new__() to skip __init__ (which builds
DataSourceManager and the analyzers, requiring network/API access).
"""

import unittest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.agent import IPOReviewAgent
from src.models import (
    FinancialMetrics, NewsAnalysis, RiskAssessment, StrengthsAndWeaknesses,
    RiskLevel, InvestmentRecommendation
)


def make_metrics(revenue_growth_rate=None, profit_margin=None):
    return FinancialMetrics(
        revenue={}, profit={}, assets={}, liabilities={},
        revenue_growth_rate=revenue_growth_rate,
        profit_margin=profit_margin,
    )


def make_news(sentiment_score=0.0):
    return NewsAnalysis(sentiment_score=sentiment_score, key_themes=[])


def make_risk(overall=RiskLevel.MODERATE):
    return RiskAssessment(
        overall_risk=overall,
        financial_risk=RiskLevel.MODERATE,
        market_risk=RiskLevel.MODERATE,
        operational_risk=RiskLevel.MODERATE,
        risk_factors=[],
        risk_mitigation=[],
    )


def make_strengths_weaknesses(strengths=None, weaknesses=None):
    return StrengthsAndWeaknesses(
        strengths=strengths or [],
        weaknesses=weaknesses or [],
        opportunities=[],
        threats=[],
        competitive_advantages=[],
    )


class TestPredictListingGains(unittest.TestCase):
    """Tests for IPOReviewAgent._predict_listing_gains."""

    @classmethod
    def setUpClass(cls):
        cls.agent = IPOReviewAgent.__new__(IPOReviewAgent)

    def test_high_growth_bonus(self):
        gain = self.agent._predict_listing_gains(
            make_metrics(revenue_growth_rate=0.25), make_news(), make_risk()
        )
        # base 10 + 15 high-growth bonus, neutral sentiment/moderate risk = no further adjustment
        self.assertAlmostEqual(gain, 25.0, places=2)

    def test_negative_growth_penalty(self):
        gain = self.agent._predict_listing_gains(
            make_metrics(revenue_growth_rate=-0.1), make_news(), make_risk()
        )
        self.assertAlmostEqual(gain, -10.0, places=2)

    def test_high_risk_dampens_gain(self):
        metrics = make_metrics(revenue_growth_rate=0.25)
        gain_moderate = self.agent._predict_listing_gains(metrics, make_news(), make_risk(RiskLevel.MODERATE))
        gain_high = self.agent._predict_listing_gains(metrics, make_news(), make_risk(RiskLevel.HIGH))
        self.assertLess(gain_high, gain_moderate)

    def test_low_risk_boosts_gain(self):
        metrics = make_metrics(revenue_growth_rate=0.25)
        gain_moderate = self.agent._predict_listing_gains(metrics, make_news(), make_risk(RiskLevel.MODERATE))
        gain_low = self.agent._predict_listing_gains(metrics, make_news(), make_risk(RiskLevel.LOW))
        self.assertGreater(gain_low, gain_moderate)

    def test_clamped_to_upper_bound(self):
        gain = self.agent._predict_listing_gains(
            make_metrics(revenue_growth_rate=0.9), make_news(sentiment_score=20.0), make_risk(RiskLevel.LOW)
        )
        self.assertEqual(gain, 100)

    def test_stays_within_bounds(self):
        gain = self.agent._predict_listing_gains(
            make_metrics(revenue_growth_rate=0.25), make_news(sentiment_score=0.5), make_risk(RiskLevel.LOW)
        )
        self.assertLessEqual(gain, 100)
        self.assertGreaterEqual(gain, -50)


class TestCalculateLongTermScore(unittest.TestCase):
    """Tests for IPOReviewAgent._calculate_long_term_score."""

    @classmethod
    def setUpClass(cls):
        cls.agent = IPOReviewAgent.__new__(IPOReviewAgent)

    def test_high_profit_margin_bonus(self):
        score = self.agent._calculate_long_term_score(
            make_metrics(profit_margin=0.15), make_risk(), make_strengths_weaknesses()
        )
        self.assertGreater(score, 5.0)

    def test_negative_profit_margin_penalty(self):
        score = self.agent._calculate_long_term_score(
            make_metrics(profit_margin=-0.05), make_risk(), make_strengths_weaknesses()
        )
        self.assertLess(score, 5.0)

    def test_clamped_to_upper_bound(self):
        score = self.agent._calculate_long_term_score(
            make_metrics(profit_margin=0.5, revenue_growth_rate=0.5),
            make_risk(RiskLevel.LOW),
            make_strengths_weaknesses(strengths=["a"] * 20),
        )
        self.assertLessEqual(score, 10)

    def test_stays_within_bounds(self):
        score = self.agent._calculate_long_term_score(
            make_metrics(profit_margin=-0.9, revenue_growth_rate=-0.9),
            make_risk(RiskLevel.VERY_HIGH),
            make_strengths_weaknesses(weaknesses=["a"] * 20),
        )
        self.assertGreaterEqual(score, 0)

    def test_strengths_score_higher_than_weaknesses(self):
        metrics = make_metrics()
        score_with_strengths = self.agent._calculate_long_term_score(
            metrics, make_risk(), make_strengths_weaknesses(strengths=["a", "b"])
        )
        score_with_weaknesses = self.agent._calculate_long_term_score(
            metrics, make_risk(), make_strengths_weaknesses(weaknesses=["a", "b"])
        )
        self.assertGreater(score_with_strengths, score_with_weaknesses)


class TestGenerateRecommendation(unittest.TestCase):
    """Tests for IPOReviewAgent._generate_recommendation."""

    @classmethod
    def setUpClass(cls):
        cls.agent = IPOReviewAgent.__new__(IPOReviewAgent)

    def test_strong_buy(self):
        rec = self.agent._generate_recommendation(25, 8.5, make_risk(RiskLevel.LOW))
        self.assertEqual(rec, InvestmentRecommendation.STRONG_BUY)

    def test_buy(self):
        rec = self.agent._generate_recommendation(15, 7.0, make_risk(RiskLevel.MODERATE))
        self.assertEqual(rec, InvestmentRecommendation.BUY)

    def test_hold(self):
        rec = self.agent._generate_recommendation(5, 5.0, make_risk(RiskLevel.MODERATE))
        self.assertEqual(rec, InvestmentRecommendation.HOLD)

    def test_avoid_on_very_high_risk_overrides_decent_score(self):
        # long_term_score alone would qualify for HOLD, but Very High risk should override it
        rec = self.agent._generate_recommendation(5, 5.0, make_risk(RiskLevel.VERY_HIGH))
        self.assertEqual(rec, InvestmentRecommendation.AVOID)

    def test_avoid_on_low_score(self):
        rec = self.agent._generate_recommendation(-10, 2.0, make_risk(RiskLevel.HIGH))
        self.assertEqual(rec, InvestmentRecommendation.AVOID)


class TestEstimateMarketCap(unittest.TestCase):
    """Tests for IPOReviewAgent._estimate_market_cap."""

    @classmethod
    def setUpClass(cls):
        cls.agent = IPOReviewAgent.__new__(IPOReviewAgent)

    def test_tuple_price_range(self):
        cap = self.agent._estimate_market_cap({'price_range': (100, 120)})
        self.assertAlmostEqual(cap, 110 * 10_000_000)

    def test_string_price_range(self):
        cap = self.agent._estimate_market_cap({'price_range': '100-120'})
        self.assertAlmostEqual(cap, 110 * 10_000_000)

    def test_single_string_price(self):
        cap = self.agent._estimate_market_cap({'price_range': '150'})
        self.assertAlmostEqual(cap, 150 * 10_000_000)

    def test_invalid_string_falls_back_to_default(self):
        cap = self.agent._estimate_market_cap({'price_range': 'not-a-price'})
        self.assertAlmostEqual(cap, 110 * 10_000_000)  # default (100, 120)

    def test_missing_key_uses_default(self):
        cap = self.agent._estimate_market_cap({})
        self.assertAlmostEqual(cap, 110 * 10_000_000)


if __name__ == '__main__':
    unittest.main()
