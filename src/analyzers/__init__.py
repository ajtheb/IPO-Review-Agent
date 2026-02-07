"""
Core analysis engines for IPO evaluation.
Enhanced with LLM-powered prospectus analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import statistics
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..models import (
    FinancialMetrics, RiskLevel, RiskAssessment, 
    StrengthsAndWeaknesses, NewsAnalysis, InvestmentRecommendation
)

# Lazy import LLM analyzer to avoid circular dependency
LLM_ANALYZER_AVAILABLE = False
LLMProspectusAnalyzer = None
LLMFinancialMetrics = None
BenchmarkingAnalysis = None
IPOSpecificMetrics = None
integrate_llm_analysis = None

def _load_llm_analyzer():
    """Lazy load LLM analyzer modules to avoid circular imports."""
    global LLM_ANALYZER_AVAILABLE, LLMProspectusAnalyzer, LLMFinancialMetrics
    global BenchmarkingAnalysis, IPOSpecificMetrics, integrate_llm_analysis
    
    if LLM_ANALYZER_AVAILABLE:
        return True
    
    try:
        from .llm_prospectus_analyzer import (
            LLMProspectusAnalyzer as _LLMProspectusAnalyzer,
            LLMFinancialMetrics as _LLMFinancialMetrics,
            BenchmarkingAnalysis as _BenchmarkingAnalysis,
            IPOSpecificMetrics as _IPOSpecificMetrics,
            integrate_llm_analysis as _integrate_llm_analysis
        )
        
        LLMProspectusAnalyzer = _LLMProspectusAnalyzer
        LLMFinancialMetrics = _LLMFinancialMetrics
        BenchmarkingAnalysis = _BenchmarkingAnalysis
        IPOSpecificMetrics = _IPOSpecificMetrics
        integrate_llm_analysis = _integrate_llm_analysis
        LLM_ANALYZER_AVAILABLE = True
        
        logger.info("LLM prospectus analyzer loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"LLM analyzer not available: {e}")
        return False


class EnhancedFinancialAnalyzer:
    """Enhanced financial analyzer with LLM-powered prospectus analysis."""
    
    def __init__(self, llm_provider: str = "openai"):
        """Initialize enhanced analyzer with LLM integration."""
        self.llm_provider = llm_provider
        self.llm_analyzer = None
        
        # Lazy load LLM analyzer
        if _load_llm_analyzer():
            try:
                self.llm_analyzer = LLMProspectusAnalyzer(provider=llm_provider)
                logger.info(f"LLM analyzer initialized with {llm_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM analyzer: {e}")
        
        # Traditional benchmarks (enhanced)
        self.industry_benchmarks = {
            'Technology': {
                'profit_margin': 0.15, 'revenue_growth': 0.20, 'pe_ratio': 25.0,
                'debt_to_equity': 0.3, 'current_ratio': 2.0, 'roe': 0.18
            },
            'Healthcare': {
                'profit_margin': 0.12, 'revenue_growth': 0.15, 'pe_ratio': 22.0,
                'debt_to_equity': 0.4, 'current_ratio': 1.8, 'roe': 0.15
            },
            'Financial Services': {
                'profit_margin': 0.25, 'revenue_growth': 0.10, 'pe_ratio': 12.0,
                'debt_to_equity': 4.0, 'current_ratio': 1.0, 'roe': 0.12
            },
            'Manufacturing': {
                'profit_margin': 0.08, 'revenue_growth': 0.12, 'pe_ratio': 15.0,
                'debt_to_equity': 0.6, 'current_ratio': 1.5, 'roe': 0.12
            },
            'Consumer Goods': {
                'profit_margin': 0.10, 'revenue_growth': 0.08, 'pe_ratio': 18.0,
                'debt_to_equity': 0.4, 'current_ratio': 1.6, 'roe': 0.14
            },
            'Default': {
                'profit_margin': 0.10, 'revenue_growth': 0.12, 'pe_ratio': 18.0,
                'debt_to_equity': 0.5, 'current_ratio': 1.5, 'roe': 0.12
            }
        }
    
    def analyze_comprehensive(self, 
                            financial_data: Dict[str, Any], 
                            company_name: str,
                            sector: str = "") -> Dict[str, Any]:
        """
        Comprehensive financial analysis using both traditional methods and LLM.
        """
        results = {}
        
        # Traditional financial analysis
        traditional_metrics = self.calculate_financial_metrics(financial_data)
        results['traditional_metrics'] = traditional_metrics
        
        # LLM-powered analysis if prospectus text is available
        prospectus_text = financial_data.get('prospectus_text', '')
        
        # Debug logging
        logger.info(f"Prospectus text available: {bool(prospectus_text)}, Length: {len(prospectus_text)}")
        logger.info(f"LLM analyzer available: {bool(self.llm_analyzer)}")
        
        if prospectus_text and self.llm_analyzer:
            logger.info(f"Performing LLM analysis for {company_name}")
            
            try:
                # First, chunk and store the prospectus in vector database
                ipo_date = financial_data.get('ipo_date')
                self.llm_analyzer.chunk_and_store_prospectus(
                    prospectus_text, 
                    company_name, 
                    sector, 
                    ipo_date
                )
                logger.info(f"Successfully chunked and stored prospectus for {company_name}")
                
                # Get comprehensive LLM analysis
                llm_analysis = integrate_llm_analysis(
                    company_name, prospectus_text, sector, self.llm_provider
                )
                results['llm_analysis'] = llm_analysis
                
                # Merge LLM financial metrics with traditional metrics
                enhanced_metrics = self._merge_financial_metrics(
                    traditional_metrics, 
                    llm_analysis['llm_financial_metrics']
                )
                results['enhanced_metrics'] = enhanced_metrics
                
                # Advanced valuation analysis
                valuation_analysis = self._perform_valuation_analysis(
                    llm_analysis['llm_financial_metrics'], 
                    sector
                )
                results['valuation_analysis'] = valuation_analysis
                
                # Peer benchmarking
                peer_analysis = self._analyze_peer_comparison(
                    llm_analysis['llm_benchmarking'],
                    llm_analysis['llm_financial_metrics'],
                    sector
                )
                results['peer_analysis'] = peer_analysis
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                results['llm_error'] = str(e)
        elif self.llm_analyzer and not prospectus_text:
            # Generate a basic investment thesis even without prospectus text
            logger.warning(f"No prospectus text available for {company_name}, generating basic LLM analysis")
            try:
                # Create a simple context from available data
                basic_context = self._create_basic_context(financial_data, company_name, sector)
                llm_analysis = integrate_llm_analysis(
                    company_name, basic_context, sector, self.llm_provider
                )
                results['llm_analysis'] = llm_analysis
                logger.info(f"Generated basic LLM analysis from available data")
            except Exception as e:
                logger.error(f"Basic LLM analysis generation failed: {e}")
                results['llm_error'] = str(e)
        else:
            logger.warning(f"LLM analysis skipped - prospectus_text: {bool(prospectus_text)}, llm_analyzer: {bool(self.llm_analyzer)}")
        
        # Quality assessment
        results['analysis_quality'] = self._assess_analysis_quality(results)
        
        return results
    
    def _create_basic_context(self, financial_data: Dict[str, Any], company_name: str, sector: str) -> str:
        """Create a basic context string from available data when no prospectus is available."""
        context_parts = [f"Company: {company_name}", f"Sector: {sector}"]
        
        # Add IPO details if available
        ipo_details = financial_data.get('ipo_details', {})
        if ipo_details:
            context_parts.append(f"\nIPO Details:")
            for key, value in ipo_details.items():
                context_parts.append(f"  {key}: {value}")
        
        # Add any available description
        if 'description' in financial_data:
            context_parts.append(f"\nBusiness Description:\n{financial_data['description']}")
        
        # Add prospectus summary if available
        if 'prospectus_summary' in financial_data:
            summary = financial_data['prospectus_summary']
            if isinstance(summary, dict):
                context_parts.append(f"\nProspectus Summary:")
                for key, value in summary.items():
                    if isinstance(value, (str, int, float)):
                        context_parts.append(f"  {key}: {value}")
            elif isinstance(summary, str):
                context_parts.append(f"\nProspectus Summary:\n{summary}")
        
        # Add news headlines if available
        company_news = financial_data.get('company_news', [])
        if company_news:
            context_parts.append(f"\nRecent News ({len(company_news)} articles):")
            for article in company_news[:5]:  # Top 5 articles
                title = article.get('title', '')
                if title:
                    context_parts.append(f"  - {title}")
        
        return "\n".join(context_parts)
    
    def _merge_financial_metrics(self, 
                               traditional: FinancialMetrics, 
                               llm_metrics: Any) -> FinancialMetrics:  # Using Any to avoid circular import
        """Merge traditional and LLM-extracted financial metrics."""
        
        # Start with traditional metrics
        merged = FinancialMetrics(
            revenue=traditional.revenue,
            profit=traditional.profit,
            assets=traditional.assets,
            liabilities=traditional.liabilities
        )
        
        # Override with LLM data where available and reliable
        if llm_metrics.extraction_confidence and llm_metrics.extraction_confidence > 0.7:
            
            # Profitability ratios
            if llm_metrics.gross_profit_margin:
                merged.gross_profit_margin = llm_metrics.gross_profit_margin
            
            if llm_metrics.net_profit_margin:
                merged.profit_margin = llm_metrics.net_profit_margin
            
            if llm_metrics.return_on_equity:
                merged.return_on_equity = llm_metrics.return_on_equity
                
            # Liquidity ratios
            if llm_metrics.current_ratio:
                merged.current_ratio = llm_metrics.current_ratio
                
            # Leverage ratios  
            if llm_metrics.debt_to_equity_ratio:
                merged.debt_to_equity = llm_metrics.debt_to_equity_ratio
                
            # Growth metrics
            if llm_metrics.revenue_growth_3yr:
                merged.revenue_growth_rate = llm_metrics.revenue_growth_3yr
        
        return merged
    
    def _perform_valuation_analysis(self, 
                                  llm_metrics: Any,  # Using Any to avoid circular import
                                  sector: str) -> Dict[str, Any]:
        """Perform advanced valuation analysis using LLM-extracted metrics."""
        
        valuation = {
            'pe_analysis': {},
            'book_value_analysis': {},
            'sales_multiple_analysis': {},
            'dcf_inputs': {},
            'relative_valuation': {}
        }
        
        # P/E Analysis
        if llm_metrics.trailing_pe_ratio or llm_metrics.forward_pe_ratio:
            sector_pe = self.industry_benchmarks.get(sector, {}).get('pe_ratio', 18.0)
            
            current_pe = llm_metrics.trailing_pe_ratio or llm_metrics.forward_pe_ratio
            
            valuation['pe_analysis'] = {
                'current_pe': current_pe,
                'sector_average_pe': sector_pe,
                'relative_premium': (current_pe - sector_pe) / sector_pe if current_pe else None,
                'assessment': self._assess_pe_valuation(current_pe, sector_pe)
            }
        
        # Book Value Analysis
        if llm_metrics.price_to_book_ratio:
            valuation['book_value_analysis'] = {
                'pb_ratio': llm_metrics.price_to_book_ratio,
                'assessment': 'Undervalued' if llm_metrics.price_to_book_ratio < 1.5 else 'Fairly Valued' if llm_metrics.price_to_book_ratio < 3.0 else 'Overvalued'
            }
        
        # Sales Multiple Analysis
        if llm_metrics.price_to_sales_ratio:
            valuation['sales_multiple_analysis'] = {
                'ps_ratio': llm_metrics.price_to_sales_ratio,
                'assessment': 'Attractive' if llm_metrics.price_to_sales_ratio < 5.0 else 'Fair' if llm_metrics.price_to_sales_ratio < 10.0 else 'Expensive'
            }
        
        # DCF Inputs preparation
        valuation['dcf_inputs'] = {
            'revenue_growth': llm_metrics.revenue_growth_3yr,
            'profit_margin': llm_metrics.net_profit_margin,
            'roe': llm_metrics.return_on_equity,
            'debt_equity': llm_metrics.debt_to_equity_ratio,
            'note': 'Inputs for DCF model - requires risk-free rate and market premium'
        }
        
        return valuation
    
    def _analyze_peer_comparison(self,
                               benchmarking: Any,  # Using Any to avoid circular import
                               llm_metrics: Any,  # Using Any to avoid circular import
                               sector: str) -> Dict[str, Any]:
        """Analyze peer comparison using LLM benchmarking data."""
        
        peer_analysis = {
            'market_position': benchmarking.market_position,
            'competitive_assessment': {},
            'relative_performance': {},
            'investment_case': {}
        }
        
        # Competitive Assessment
        peer_analysis['competitive_assessment'] = {
            'advantages_count': len(benchmarking.competitive_advantages),
            'disadvantages_count': len(benchmarking.competitive_disadvantages),
            'net_competitive_position': len(benchmarking.competitive_advantages) - len(benchmarking.competitive_disadvantages),
            'key_advantages': benchmarking.competitive_advantages[:3],
            'key_concerns': benchmarking.competitive_disadvantages[:3]
        }
        
        # Relative Performance vs Sector
        sector_benchmarks = self.industry_benchmarks.get(sector, {})
        
        if llm_metrics.net_profit_margin and 'profit_margin' in sector_benchmarks:
            margin_vs_sector = llm_metrics.net_profit_margin - sector_benchmarks['profit_margin']
            peer_analysis['relative_performance']['profit_margin'] = {
                'company': llm_metrics.net_profit_margin,
                'sector': sector_benchmarks['profit_margin'],
                'difference': margin_vs_sector,
                'assessment': 'Above Sector' if margin_vs_sector > 0.02 else 'Below Sector' if margin_vs_sector < -0.02 else 'In-line'
            }
        
        if llm_metrics.return_on_equity and 'roe' in sector_benchmarks:
            roe_vs_sector = llm_metrics.return_on_equity - sector_benchmarks['roe']
            peer_analysis['relative_performance']['roe'] = {
                'company': llm_metrics.return_on_equity,
                'sector': sector_benchmarks['roe'],
                'difference': roe_vs_sector,
                'assessment': 'Superior' if roe_vs_sector > 0.03 else 'Inferior' if roe_vs_sector < -0.03 else 'Average'
            }
        
        # Investment Case Summary
        strengths = benchmarking.competitive_advantages
        concerns = benchmarking.competitive_disadvantages
        
        # Calculate competitive score first
        competitive_score = self._calculate_competitive_score(benchmarking, llm_metrics)
        
        peer_analysis['investment_case'] = {
            'investment_strengths': strengths[:5],
            'investment_concerns': concerns[:5],
            'overall_competitive_score': competitive_score,
        }
        
        # Add recommendation bias after setting competitive score
        peer_analysis['investment_case']['recommendation_bias'] = self._determine_recommendation_bias(peer_analysis)
        
        return peer_analysis
    
    def _assess_pe_valuation(self, current_pe: float, sector_pe: float) -> str:
        """Assess P/E valuation relative to sector."""
        if not current_pe or not sector_pe:
            return "Insufficient Data"
        
        premium = (current_pe - sector_pe) / sector_pe
        
        if premium > 0.3:
            return "Significantly Overvalued"
        elif premium > 0.15:
            return "Moderately Overvalued"
        elif premium > -0.15:
            return "Fairly Valued"
        elif premium > -0.3:
            return "Moderately Undervalued"
        else:
            return "Significantly Undervalued"
    
    def _calculate_competitive_score(self, 
                                   benchmarking: Any,  # Using Any to avoid circular import
                                   metrics: Any) -> float:  # Using Any to avoid circular import
        """Calculate overall competitive score (0-10)."""
        score = 5.0  # Base score
        
        try:
            # Competitive position (with safe attribute access)
            market_position = getattr(benchmarking, 'market_position', None)
            if market_position == "leader":
                score += 2.0
            elif market_position == "challenger":
                score += 1.0
            elif market_position == "follower":
                score -= 1.0
            
            # Advantages vs disadvantages (with safe attribute access)
            competitive_advantages = getattr(benchmarking, 'competitive_advantages', [])
            competitive_disadvantages = getattr(benchmarking, 'competitive_disadvantages', [])
            
            if competitive_advantages and competitive_disadvantages:
                net_advantages = len(competitive_advantages) - len(competitive_disadvantages)
                score += min(net_advantages * 0.3, 2.0)  # Cap at +2.0
            
            # Financial performance indicators (with safe attribute access)
            roe = getattr(metrics, 'return_on_equity', None)
            if roe and roe > 0.15:
                score += 0.5
            
            net_margin = getattr(metrics, 'net_profit_margin', None)
            if net_margin and net_margin > 0.12:
                score += 0.5
            
            debt_equity = getattr(metrics, 'debt_to_equity_ratio', None)
            if debt_equity and debt_equity < 0.5:
                score += 0.5
        
        except Exception as e:
            logger.warning(f"Error calculating competitive score: {e}")
            score = 5.0  # Return default score on error
        
        return max(0, min(10, score))
    
    def _determine_recommendation_bias(self, peer_analysis: Dict[str, Any]) -> str:
        """Determine recommendation bias based on peer analysis."""
        competitive_score = peer_analysis['investment_case']['overall_competitive_score']
        
        if competitive_score >= 7.5:
            return "Positive Bias"
        elif competitive_score >= 6.0:
            return "Neutral to Positive"
        elif competitive_score >= 4.0:
            return "Neutral"
        else:
            return "Negative Bias"
    
    def _assess_analysis_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of the analysis."""
        quality = {
            'data_sources': [],
            'confidence_level': 0.0,
            'completeness_score': 0.0,
            'reliability_assessment': 'Low'
        }
        
        # Check available data sources
        if 'traditional_metrics' in results:
            quality['data_sources'].append('Traditional Financial Analysis')
        
        if 'llm_analysis' in results:
            quality['data_sources'].append('LLM Prospectus Analysis')
            
            # Use LLM confidence if available
            llm_metrics = results['llm_analysis'].get('llm_financial_metrics')
            if llm_metrics and hasattr(llm_metrics, 'extraction_confidence'):
                quality['confidence_level'] = llm_metrics.extraction_confidence
        
        # Calculate completeness score
        completeness = 0.0
        if results.get('enhanced_metrics'):
            metrics = results['enhanced_metrics']
            available_metrics = sum([
                1 if metrics.profit_margin else 0,
                1 if metrics.revenue_growth_rate else 0,
                1 if metrics.return_on_equity else 0,
                1 if metrics.current_ratio else 0,
                1 if metrics.debt_to_equity else 0
            ])
            completeness = available_metrics / 5.0
        
        quality['completeness_score'] = completeness
        
        # Overall reliability assessment
        if quality['confidence_level'] > 0.8 and completeness > 0.8:
            quality['reliability_assessment'] = 'High'
        elif quality['confidence_level'] > 0.6 and completeness > 0.6:
            quality['reliability_assessment'] = 'Medium'
        else:
            quality['reliability_assessment'] = 'Low'
        
        return quality
    
    # Maintain backward compatibility with existing FinancialAnalyzer
    def calculate_financial_metrics(self, financial_data: Dict[str, Any]) -> FinancialMetrics:
        """Calculate comprehensive financial metrics from financial statements or prospectus data."""
        try:
            # First, try to use prospectus data if available
            prospectus_financials = financial_data.get('prospectus_financials')
            if prospectus_financials:
                logger.info("Using prospectus financial data for analysis")
                return self._process_prospectus_financials(prospectus_financials)
            
            # Fall back to traditional financial statements
            income_stmt = financial_data.get('income_statement', pd.DataFrame())
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            
            if income_stmt.empty:
                logger.warning("No financial data available - using estimation methods")
                return self._estimate_financial_metrics(financial_data)
            
            # Extract revenue data (last 3 years)
            revenue_data = {}
            profit_data = {}
            
            # Get revenue from income statement
            revenue_rows = ['Total Revenue', 'Revenue', 'Net Revenue']
            profit_rows = ['Net Income', 'Net Profit', 'Earnings']
            
            for col in income_stmt.columns[:3]:  # Last 3 years
                year = str(col.year) if hasattr(col, 'year') else str(col)
                
                # Find revenue
                for row_name in revenue_rows:
                    if row_name in income_stmt.index:
                        revenue_data[year] = float(income_stmt.loc[row_name, col])
                        break
                
                # Find profit
                for row_name in profit_rows:
                    if row_name in income_stmt.index:
                        profit_data[year] = float(income_stmt.loc[row_name, col])
                        break
            
            # Calculate growth rates and margins
            revenue_growth = self._calculate_growth_rate(revenue_data)
            profit_margin = self._calculate_profit_margin(revenue_data, profit_data)
            
            return FinancialMetrics(
                revenue=revenue_data,
                profit=profit_data,
                assets={},  # Would extract from balance sheet
                liabilities={},  # Would extract from balance sheet
                revenue_growth_rate=revenue_growth,
                profit_margin=profit_margin
            )
        
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return FinancialMetrics(revenue={}, profit={}, assets={}, liabilities={})
    
    def _process_prospectus_financials(self, prospectus_financials) -> FinancialMetrics:
        """Process financial data extracted from IPO prospectus."""
        try:
            revenue_data = prospectus_financials.revenue_data
            profit_data = prospectus_financials.profit_data
            assets_data = prospectus_financials.assets_data
            liabilities_data = prospectus_financials.liabilities_data
            
            # Calculate derived metrics
            revenue_growth = self._calculate_growth_rate(revenue_data)
            profit_margin = self._calculate_profit_margin(revenue_data, profit_data)
            
            # Calculate additional ratios if data is available
            debt_to_equity = None
            return_on_equity = None
            current_ratio = None
            
            if assets_data and liabilities_data:
                # Calculate debt-to-equity ratio
                latest_year = max(assets_data.keys()) if assets_data else None
                if latest_year and latest_year in liabilities_data:
                    total_assets = assets_data[latest_year]
                    total_liabilities = liabilities_data[latest_year]
                    equity = total_assets - total_liabilities
                    if equity > 0:
                        debt_to_equity = total_liabilities / equity
            
            # Use key ratios if available from prospectus
            key_ratios = prospectus_financials.key_ratios
            if key_ratios:
                debt_to_equity = key_ratios.get('debt_to_equity', debt_to_equity)
                return_on_equity = key_ratios.get('roe', return_on_equity)
                current_ratio = key_ratios.get('current_ratio', current_ratio)
            
            logger.info(f"Successfully processed prospectus financials with {len(revenue_data)} years of data")
            
            return FinancialMetrics(
                revenue=revenue_data,
                profit=profit_data,
                assets=assets_data,
                liabilities=liabilities_data,
                revenue_growth_rate=revenue_growth,
                profit_margin=profit_margin,
                debt_to_equity=debt_to_equity,
                return_on_equity=return_on_equity,
                current_ratio=current_ratio
            )
        
        except Exception as e:
            logger.error(f"Error processing prospectus financials: {e}")
            return FinancialMetrics(revenue={}, profit={}, assets={}, liabilities={})
    
    def _estimate_financial_metrics(self, financial_data: Dict[str, Any]) -> FinancialMetrics:
        """Estimate financial metrics when no concrete data is available."""
        # Use IPO details and industry benchmarks for estimation
        ipo_details = financial_data.get('ipo_details', {})
        sector = ipo_details.get('sector', 'Default')
        
        # Get industry benchmarks
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from config import config
        benchmarks = config.industry_benchmarks.get(sector, config.industry_benchmarks['Default'])
        
        # Create estimated financial metrics based on IPO valuation and industry averages
        estimated_revenue_growth = benchmarks.get('revenue_growth', 0.12)
        estimated_profit_margin = benchmarks.get('profit_margin', 0.10)
        
        logger.info(f"Using estimated metrics for {sector} sector: growth={estimated_revenue_growth:.1%}, margin={estimated_profit_margin:.1%}")
        
        return FinancialMetrics(
            revenue={},  # No historical data available
            profit={},   # No historical data available
            assets={},
            liabilities={},
            revenue_growth_rate=estimated_revenue_growth,
            profit_margin=estimated_profit_margin
        )
    
    def _calculate_growth_rate(self, data: Dict[str, float]) -> Optional[float]:
        """Calculate compound annual growth rate."""
        if len(data) < 2:
            return None
        
        values = list(data.values())
        if len(values) < 2 or values[0] <= 0:
            return None
        
        years = len(values) - 1
        cagr = (values[-1] / values[0]) ** (1/years) - 1
        return cagr
    
    def _calculate_profit_margin(self, revenue: Dict[str, float], profit: Dict[str, float]) -> Optional[float]:
        """Calculate average profit margin."""
        margins = []
        for year in revenue.keys():
            if year in profit and revenue[year] > 0:
                margin = profit[year] / revenue[year]
                margins.append(margin)
        
        return statistics.mean(margins) if margins else None
    
    def benchmark_analysis(self, metrics: FinancialMetrics, industry: str) -> Dict[str, str]:
        """Compare company metrics against industry benchmarks."""
        benchmarks = self.industry_benchmarks.get(industry, self.industry_benchmarks['Default'])
        analysis = {}
        
        if metrics.profit_margin is not None:
            if metrics.profit_margin > benchmarks['profit_margin']:
                analysis['profit_margin'] = "Above Industry Average"
            else:
                analysis['profit_margin'] = "Below Industry Average"
        
        if metrics.revenue_growth_rate is not None:
            if metrics.revenue_growth_rate > benchmarks['revenue_growth']:
                analysis['revenue_growth'] = "Above Industry Average"
            else:
                analysis['revenue_growth'] = "Below Industry Average"
        
        return analysis


class SentimentAnalyzer:
    """Analyzes news sentiment and market intelligence."""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_news_sentiment(self, news_articles: List[Dict[str, Any]]) -> NewsAnalysis:
        """Analyze sentiment of news articles."""
        if not news_articles:
            return NewsAnalysis(
                sentiment_score=0.0,
                news_articles=[],
                key_themes=[],
                market_trends=[],
                analyst_opinions=[]
            )
        
        sentiment_scores = []
        key_themes = []
        
        for article in news_articles[:20]:  # Limit to 20 most relevant articles
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            
            # VADER sentiment analysis
            vader_score = self.vader_analyzer.polarity_scores(text)
            sentiment_scores.append(vader_score['compound'])
            
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            sentiment_scores.append(blob.sentiment.polarity)
            
            # Extract key themes (simplified keyword extraction)
            themes = self._extract_themes(text)
            key_themes.extend(themes)
        
        # Calculate overall sentiment
        overall_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Get most common themes
        theme_counts = {}
        for theme in key_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_themes = [theme for theme, count in top_themes]
        
        return NewsAnalysis(
            sentiment_score=overall_sentiment,
            news_articles=news_articles[:10],  # Store top 10 articles
            key_themes=top_themes,
            market_trends=self._identify_market_trends(news_articles),
            analyst_opinions=[]
        )
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract key themes from text."""
        # Financial keywords to look for
        financial_keywords = [
            'growth', 'profit', 'revenue', 'loss', 'debt', 'investment',
            'market share', 'competition', 'innovation', 'risk', 'opportunity',
            'valuation', 'ipo', 'listing', 'stock', 'shares'
        ]
        
        text_lower = text.lower()
        found_themes = [keyword for keyword in financial_keywords if keyword in text_lower]
        return found_themes
    
    def _identify_market_trends(self, news_articles: List[Dict[str, Any]]) -> List[str]:
        """Identify market trends from news articles."""
        trends = []
        
        # Look for trend indicators in headlines
        trend_keywords = {
            'bullish': ['surge', 'soar', 'rally', 'boom', 'uptick'],
            'bearish': ['plunge', 'crash', 'decline', 'fall', 'downturn'],
            'volatility': ['volatile', 'uncertainty', 'fluctuation'],
            'sector_rotation': ['rotation', 'shift', 'movement']
        }
        
        for article in news_articles:
            title = article.get('title', '').lower()
            for trend_type, keywords in trend_keywords.items():
                if any(keyword in title for keyword in keywords):
                    trends.append(trend_type)
        
        return list(set(trends))  # Remove duplicates


class RiskAnalyzer:
    """Analyzes various risk factors for IPO investment."""
    
    def assess_risks(self, 
                    financial_metrics: FinancialMetrics,
                    market_data: Dict[str, Any],
                    news_analysis: NewsAnalysis,
                    company_info: Dict[str, Any]) -> RiskAssessment:
        """Comprehensive risk assessment."""
        
        risk_factors = []
        financial_risk = self._assess_financial_risk(financial_metrics, risk_factors)
        market_risk = self._assess_market_risk(market_data, news_analysis, risk_factors)
        operational_risk = self._assess_operational_risk(company_info, risk_factors)
        
        # Determine overall risk
        risk_levels = [financial_risk, market_risk, operational_risk]
        risk_scores = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
        avg_score = sum(risk_scores[risk.value] for risk in risk_levels) / len(risk_levels)
        
        if avg_score <= 1.5:
            overall_risk = RiskLevel.LOW
        elif avg_score <= 2.5:
            overall_risk = RiskLevel.MODERATE
        elif avg_score <= 3.5:
            overall_risk = RiskLevel.HIGH
        else:
            overall_risk = RiskLevel.VERY_HIGH
        
        return RiskAssessment(
            overall_risk=overall_risk,
            financial_risk=financial_risk,
            market_risk=market_risk,
            operational_risk=operational_risk,
            risk_factors=risk_factors,
            risk_mitigation=self._suggest_risk_mitigation(risk_factors)
        )
    
    def _assess_financial_risk(self, metrics: FinancialMetrics, risk_factors: List[str]) -> RiskLevel:
        """Assess financial risks."""
        risk_score = 0
        
        # Revenue growth risk
        if metrics.revenue_growth_rate is not None:
            if metrics.revenue_growth_rate < 0:
                risk_score += 2
                risk_factors.append("Declining revenue growth")
            elif metrics.revenue_growth_rate < 0.05:
                risk_score += 1
                risk_factors.append("Low revenue growth")
        
        # Profitability risk
        if metrics.profit_margin is not None:
            if metrics.profit_margin < 0:
                risk_score += 2
                risk_factors.append("Company is not profitable")
            elif metrics.profit_margin < 0.05:
                risk_score += 1
                risk_factors.append("Low profit margins")
        
        if risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_market_risk(self, market_data: Dict[str, Any], news_analysis: NewsAnalysis, risk_factors: List[str]) -> RiskLevel:
        """Assess market-related risks."""
        risk_score = 0
        
        # Sentiment risk
        if news_analysis.sentiment_score < -0.3:
            risk_score += 2
            risk_factors.append("Negative market sentiment")
        elif news_analysis.sentiment_score < 0:
            risk_score += 1
            risk_factors.append("Mixed market sentiment")
        
        # Market trend risks
        if 'bearish' in news_analysis.market_trends:
            risk_score += 1
            risk_factors.append("Bearish market conditions")
        
        if 'volatility' in news_analysis.market_trends:
            risk_score += 1
            risk_factors.append("High market volatility")
        
        if risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_operational_risk(self, company_info: Dict[str, Any], risk_factors: List[str]) -> RiskLevel:
        """Assess operational risks."""
        risk_score = 0
        
        # Industry-specific risks
        sector = company_info.get('sector', '')
        high_risk_sectors = ['Technology', 'Biotech', 'Cryptocurrency']
        
        if sector in high_risk_sectors:
            risk_score += 1
            risk_factors.append(f"High-risk sector: {sector}")
        
        # Size risk (smaller companies generally riskier)
        market_cap = company_info.get('market_cap', 0)
        if market_cap and market_cap < 1e9:  # Less than $1B
            risk_score += 1
            risk_factors.append("Small market capitalization")
        
        if risk_score >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _suggest_risk_mitigation(self, risk_factors: List[str]) -> List[str]:
        """Suggest risk mitigation strategies."""
        mitigations = []
        
        risk_mitigation_map = {
            "declining revenue": "Monitor quarterly results closely",
            "negative profit": "Wait for profitability before investing", 
            "negative sentiment": "Consider dollar-cost averaging",
            "high volatility": "Use limit orders and position sizing",
            "small market cap": "Limit position size and diversify",
            "high risk sector": "Increase due diligence and research",
            "negative margins": "Focus on cash flow analysis",
            "stagnant revenue": "Look for turnaround catalysts"
        }
        
        for factor in risk_factors:
            factor_lower = factor.lower()
            for risk_key, mitigation in risk_mitigation_map.items():
                if risk_key in factor_lower:
                    mitigations.append(mitigation)
                    break
        
        # Add generic mitigations if none found
        if not mitigations:
            mitigations = [
                "Diversify investment portfolio",
                "Monitor company performance regularly",
                "Set stop-loss limits"
            ]
        
        return list(set(mitigations))  # Remove duplicates


class BusinessAnalyzer:
    """Analyzes business strengths, weaknesses, and competitive position."""
    
    def analyze_business_fundamentals(self,
                                   company_info: Dict[str, Any],
                                   financial_metrics: FinancialMetrics,
                                   market_data: Dict[str, Any]) -> StrengthsAndWeaknesses:
        """Analyze business strengths and weaknesses."""
        
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []
        competitive_advantages = []
        
        # First, try to use prospectus data if available
        prospectus_financials = market_data.get('prospectus_financials')
        if prospectus_financials:
            logger.info("Using prospectus business analysis data")
            
            # Use strengths from prospectus
            if prospectus_financials.company_strengths:
                for strength in prospectus_financials.company_strengths[:5]:
                    strengths.append(strength[:100])  # Limit length
                    competitive_advantages.append(strength[:100])
            
            # Use risk factors as threats/weaknesses
            if prospectus_financials.risk_factors:
                for risk in prospectus_financials.risk_factors[:5]:
                    if any(keyword in risk.lower() for keyword in ['competition', 'market', 'regulatory']):
                        threats.append(risk[:100])
                    else:
                        weaknesses.append(risk[:100])
        
        # Analyze financial strengths/weaknesses
        if financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate > 0.15:
            strengths.append("Strong revenue growth trajectory")
        elif financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate < 0:
            weaknesses.append("Declining revenue trends")
        
        if financial_metrics.profit_margin and financial_metrics.profit_margin > 0.15:
            strengths.append("High profit margins indicating operational efficiency")
            competitive_advantages.append("Efficient cost management")
        elif financial_metrics.profit_margin and financial_metrics.profit_margin < 0:
            weaknesses.append("Unprofitable operations")
        
        # Debt analysis from prospectus data
        if financial_metrics.debt_to_equity:
            if financial_metrics.debt_to_equity < 0.3:
                strengths.append("Low debt burden")
            elif financial_metrics.debt_to_equity > 0.8:
                weaknesses.append("High debt levels")
        
        # Sector-specific analysis for Indian market
        sector = company_info.get('sector', '')
        if sector == 'Technology':
            opportunities.extend([
                "Digital India initiatives driving growth",
                "Growing internet penetration in rural areas"
            ])
            threats.extend([
                "Intense competition from global tech giants",
                "Rapid technological changes"
            ])
        elif sector == 'Financial Services':
            opportunities.extend([
                "Financial inclusion initiatives",
                "Growing middle class in India"
            ])
            threats.extend([
                "Regulatory changes in financial sector",
                "RBI policy changes"
            ])
        elif sector == 'Consumer Goods':
            opportunities.extend([
                "Rising disposable income",
                "E-commerce growth in India"
            ])
            threats.extend([
                "Economic slowdown impact on consumption",
                "Supply chain disruptions"
            ])
        
        # Market position analysis (adjusted for Indian market in Crores)
        market_cap = company_info.get('market_cap', 0)
        if market_cap:
            market_cap_cr = market_cap / 10000000  # Convert to crores
            if market_cap_cr > 1000:  # > ₹1000 Cr
                strengths.append("Large market capitalization")
            elif market_cap_cr < 100:  # < ₹100 Cr
                opportunities.append("High growth potential as smaller company")
                weaknesses.append("Limited financial resources")
        
        # IPO-specific considerations
        ipo_details = market_data.get('ipo_details', {})
        if ipo_details:
            price_range = ipo_details.get('price_range', (0, 0))
            if price_range[1] > 1000:  # High-priced IPO
                threats.append("High IPO pricing may limit retail participation")
            
            if ipo_details.get('exchange') == 'Both NSE & BSE':
                strengths.append("Dual exchange listing provides better liquidity")
        
        # Remove duplicates and limit lengths
        strengths = list(dict.fromkeys(strengths[:7]))  # Remove duplicates, max 7
        weaknesses = list(dict.fromkeys(weaknesses[:5]))
        opportunities = list(dict.fromkeys(opportunities[:5]))
        threats = list(dict.fromkeys(threats[:5]))
        competitive_advantages = list(dict.fromkeys(competitive_advantages[:5]))
        
        return StrengthsAndWeaknesses(
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            competitive_advantages=competitive_advantages
        )


# Backward compatibility aliases and simplified analyzer classes
class FinancialAnalyzer:
    """Traditional financial analyzer for backward compatibility."""
    
    def __init__(self):
        self.enhanced_analyzer = EnhancedFinancialAnalyzer()
    
    def calculate_financial_metrics(self, financial_data: Dict[str, Any]) -> FinancialMetrics:
        """Calculate financial metrics using enhanced analyzer."""
        return self.enhanced_analyzer.calculate_financial_metrics(financial_data)


class SentimentAnalyzer:
    """News sentiment analyzer."""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_news_sentiment(self, news_articles: List[Dict]) -> NewsAnalysis:
        """Analyze sentiment of news articles."""
        if not news_articles:
            return NewsAnalysis(
                sentiment_score=0.0,
                key_themes=[],
                news_volume=0,
                positive_mentions=0,
                negative_mentions=0
            )
        
        sentiments = []
        themes = []
        positive_count = 0
        negative_count = 0
        
        for article in news_articles[:50]:  # Limit to 50 articles
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Vader sentiment
            vader_score = self.vader_analyzer.polarity_scores(text)
            compound_score = vader_score['compound']
            sentiments.append(compound_score)
            
            if compound_score > 0.1:
                positive_count += 1
            elif compound_score < -0.1:
                negative_count += 1
            
            # Extract themes (simple keyword extraction)
            blob = TextBlob(text.lower())
            words = [word for word in blob.words if len(word) > 4]
            themes.extend(words[:5])  # Top 5 words per article
        
        # Calculate overall sentiment
        avg_sentiment = statistics.mean(sentiments) if sentiments else 0.0
        
        # Get most common themes
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        key_themes = [theme[0] for theme in top_themes[:10]]
        
        return NewsAnalysis(
            sentiment_score=avg_sentiment,
            key_themes=key_themes,
            news_volume=len(news_articles),
            positive_mentions=positive_count,
            negative_mentions=negative_count
        )


class RiskAnalyzer:
    """Risk assessment analyzer."""
    
    def assess_risks(self, 
                    financial_metrics: FinancialMetrics,
                    raw_data: Dict[str, Any],
                    news_analysis: NewsAnalysis,
                    company_info: Dict[str, Any]) -> RiskAssessment:
        """Assess various risk factors for the IPO."""
        
        risk_factors = []
        financial_risk = RiskLevel.LOW
        market_risk = RiskLevel.LOW
        operational_risk = RiskLevel.LOW
        
        # Financial Risk Assessment
        if financial_metrics.profit_margin is not None:
            if financial_metrics.profit_margin < 0:
                financial_risk = RiskLevel.HIGH
                risk_factors.append("Negative profit margins indicate financial distress")
            elif financial_metrics.profit_margin < 0.05:
                financial_risk = RiskLevel.MODERATE
                risk_factors.append("Low profit margins may impact sustainability")
        
        if financial_metrics.revenue_growth_rate is not None:
            if financial_metrics.revenue_growth_rate < -0.1:
                financial_risk = RiskLevel.HIGH
                risk_factors.append("Declining revenue trend")
            elif financial_metrics.revenue_growth_rate < 0:
                financial_risk = RiskLevel.MODERATE
                risk_factors.append("Stagnant revenue growth")
        
        # Market Risk Assessment  
        if news_analysis.sentiment_score < -0.2:
            market_risk = RiskLevel.HIGH
            risk_factors.append("Negative market sentiment")
        elif news_analysis.sentiment_score < 0:
            market_risk = RiskLevel.MODERATE
            risk_factors.append("Neutral to negative market sentiment")
        
        # Operational Risk Assessment
        sector = company_info.get('sector', '').lower()
        high_risk_sectors = ['cryptocurrency', 'biotech', 'pharmaceutical', 'mining']
        if any(risk_sector in sector for risk_sector in high_risk_sectors):
            operational_risk = RiskLevel.HIGH
            risk_factors.append(f"High-risk sector: {sector}")
        
        # Overall Risk Calculation
        risk_scores = {
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2, 
            RiskLevel.HIGH: 3,
            RiskLevel.VERY_HIGH: 4
        }
        
        avg_risk_score = (
            risk_scores[financial_risk] + 
            risk_scores[market_risk] + 
            risk_scores[operational_risk]
        ) / 3
        
        if avg_risk_score >= 3.5:
            overall_risk = RiskLevel.VERY_HIGH
        elif avg_risk_score >= 2.5:
            overall_risk = RiskLevel.HIGH
        elif avg_risk_score >= 1.5:
            overall_risk = RiskLevel.MODERATE
        else:
            overall_risk = RiskLevel.LOW
        
        return RiskAssessment(
            overall_risk=overall_risk,
            financial_risk=financial_risk,
            market_risk=market_risk,
            operational_risk=operational_risk,
            risk_factors=risk_factors,
            risk_mitigation=self._suggest_risk_mitigation(risk_factors)
        )
    
    def _suggest_risk_mitigation(self, risk_factors: List[str]) -> List[str]:
        """Suggest risk mitigation strategies."""
        mitigations = []
        
        risk_mitigation_map = {
            "declining revenue": "Monitor quarterly results closely",
            "negative profit": "Wait for profitability before investing", 
            "negative sentiment": "Consider dollar-cost averaging",
            "high volatility": "Use limit orders and position sizing",
            "small market cap": "Limit position size and diversify",
            "high risk sector": "Increase due diligence and research",
            "negative margins": "Focus on cash flow analysis",
            "stagnant revenue": "Look for turnaround catalysts"
        }
        
        for factor in risk_factors:
            factor_lower = factor.lower()
            for risk_key, mitigation in risk_mitigation_map.items():
                if risk_key in factor_lower:
                    mitigations.append(mitigation)
                    break
        
        # Add generic mitigations if none found
        if not mitigations:
            mitigations = [
                "Diversify investment portfolio",
                "Monitor company performance regularly", 
                "Set stop-loss limits"
            ]
        
        return list(set(mitigations))  # Remove duplicates


class BusinessAnalyzer:
    """Business fundamentals analyzer."""
    
    def analyze_business_fundamentals(self,
                                    company_info: Dict[str, Any],
                                    financial_metrics: FinancialMetrics, 
                                    raw_data: Dict[str, Any]) -> StrengthsAndWeaknesses:
        """Analyze business strengths, weaknesses, opportunities, and threats."""
        
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []
        competitive_advantages = []
        
        # Financial Strengths/Weaknesses
        if financial_metrics.profit_margin and financial_metrics.profit_margin > 0.15:
            strengths.append("Strong profit margins indicate efficient operations")
        elif financial_metrics.profit_margin and financial_metrics.profit_margin < 0:
            weaknesses.append("Negative profit margins")
        
        if financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate > 0.2:
            strengths.append("High revenue growth rate")
            competitive_advantages.append("Strong market demand for products/services")
        elif financial_metrics.revenue_growth_rate and financial_metrics.revenue_growth_rate < 0:
            weaknesses.append("Declining revenue trend")
        
        # Sector-specific analysis
        sector = company_info.get('sector', '').lower()
        
        if 'technology' in sector:
            opportunities.append("Growing digital transformation market")
            strengths.append("Technology sector with scalability potential")
        elif 'healthcare' in sector:
            opportunities.append("Aging population driving healthcare demand")
            strengths.append("Essential healthcare services")
        elif 'financial' in sector:
            opportunities.append("Financial inclusion and digitization trends")
        
        # Market cap analysis
        market_cap = company_info.get('market_cap', 0)
        if market_cap > 50000000000:  # 50B+
            strengths.append("Large market capitalization indicates market confidence")
        elif market_cap < 1000000000:  # <1B
            opportunities.append("Small cap with high growth potential")
            threats.append("Limited resources compared to larger competitors")
        
        # Generic IPO considerations
        opportunities.extend([
            "Access to public capital markets",
            "Enhanced brand visibility post-IPO",
            "Potential for strategic acquisitions"
        ])
        
        threats.extend([
            "Market volatility affecting stock performance",
            "Increased regulatory compliance requirements",
            "Public scrutiny and reporting obligations"
        ])
        
        return StrengthsAndWeaknesses(
            strengths=strengths[:7],
            weaknesses=weaknesses[:5], 
            opportunities=opportunities[:5],
            threats=threats[:5],
            competitive_advantages=competitive_advantages[:3]
        )
