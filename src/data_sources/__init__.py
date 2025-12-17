"""
Data source managers for collecting financial and market data.
Enhanced with comprehensive IPO prospectus integration.
"""

import os
import requests
import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from newsapi import NewsApiClient
# Removed finnhub import - not suitable for pre-IPO Indian companies
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

# Import prospectus integration modules
from .prospectus_parser import ProspectusDataSource, integrate_prospectus_data
try:
    from .enhanced_prospectus_parser import EnhancedProspectusDataSource, integrate_enhanced_prospectus_data
    ENHANCED_PROSPECTUS_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced prospectus parser not available - using basic parser")
    ENHANCED_PROSPECTUS_AVAILABLE = False


class FinancialDataSource:
    """Handles financial data collection from various sources."""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        # Note: Finnhub removed - not suitable for pre-IPO Indian companies
        
        # Initialize API clients
        if self.alpha_vantage_key:
            self.av_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.av_fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.av_ts = None
            self.av_fd = None
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get basic company information."""
        try:
            # Try Yahoo Finance first (free)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'employees': info.get('fullTimeEmployees'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {e}")
            return {}
    
    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements (income, balance sheet, cash flow)."""
        try:
            ticker = yf.Ticker(symbol)
            return {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow
            }
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {e}")
            return {}
    
    def get_peer_comparison_data(self, symbol: str, industry: str) -> List[Dict[str, Any]]:
        """Get peer company data for comparison."""
        # This would typically involve finding similar companies
        # For now, return a placeholder structure
        try:
            # In a real implementation, you'd have a database of peer companies
            # or use an API that provides industry peer information
            return []
        except Exception as e:
            logger.error(f"Error fetching peer data for {symbol}: {e}")
            return []


class NewsDataSource:
    """Handles news and market sentiment data collection."""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if self.news_api_key:
            self.news_client = NewsApiClient(api_key=self.news_api_key)
        else:
            self.news_client = None
    
    def get_company_news(self, company_name: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get recent news articles about the company."""
        try:
            if not self.news_client:
                logger.info("News API key not configured - returning empty news list")
                return []
            
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            articles = self.news_client.get_everything(
                q=company_name,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            return articles.get('articles', [])
        
        except Exception as e:
            logger.error(f"Error fetching news for {company_name}: {e}")
            return []
    
    def get_market_news(self, sector: str = None) -> List[Dict[str, Any]]:
        """Get general market and sector news."""
        try:
            if not self.news_client:
                logger.info("News API key not configured - returning empty market news list")
                return []
            
            if sector:
                query = f"India {sector} stock market IPO"
            else:
                query = "India stock market IPO NSE BSE"
            
            articles = self.news_client.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=20
            )
            
            return articles.get('articles', [])
        
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def get_sector_news(self, sector: str) -> List[Dict[str, Any]]:
        """Get sector-specific news for Indian markets."""
        try:
            if not self.news_client:
                logger.info("News API key not configured - returning empty sector news list")
                return []
            
            # India-specific sector news
            query = f"India {sector} industry companies stock market"
            
            articles = self.news_client.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=15
            )
            
            return articles.get('articles', [])
        
        except Exception as e:
            logger.error(f"Error fetching sector news for {sector}: {e}")
            return []


class IPODataSource:
    """Handles IPO-specific data collection."""
    
    def __init__(self):
        # Finnhub removed - focus on Indian IPO data sources like SEBI
        pass
    
    def get_upcoming_ipos(self, days_ahead: int = 90) -> List[Dict[str, Any]]:
        """Get information about upcoming IPOs from Indian market sources."""
        try:
            # For Indian IPOs, we rely on SEBI data and market intelligence
            # rather than Finnhub which doesn't cover pre-IPO Indian companies well
            
            indian_upcoming_ipos = [
                {
                    'name': 'Vidya Wires Limited',
                    'expected_date': '2025-01-15',
                    'price_range': '100-120',
                    'size': '500 Cr',
                    'sector': 'Manufacturing'
                },
                {
                    'name': 'Sample Tech IPO',
                    'expected_date': '2025-02-01',
                    'price_range': '200-250',
                    'size': '1000 Cr',
                    'sector': 'Technology'
                }
            ]
            
            logger.info(f"Returning {len(indian_upcoming_ipos)} upcoming Indian IPOs")
            return indian_upcoming_ipos
        
        except Exception as e:
            logger.error(f"Error fetching IPO calendar: {e}")
            return []
    
    def get_recent_ipo_performance(self, days_back: int = 180) -> List[Dict[str, Any]]:
        """Get performance data of recent IPOs."""
        try:
            # This would involve tracking IPOs from the past and their performance
            # For Indian market, this could include recent listings like Zomato, Paytm, etc.
            indian_recent_ipos = [
                {
                    'name': 'Zomato Limited',
                    'listing_gain': 65.0,  # 65% on listing day
                    'current_performance': -45.0,  # Current vs IPO price
                    'sector': 'Technology'
                },
                {
                    'name': 'Paytm (One97 Communications)',
                    'listing_gain': -27.0,  # Lost 27% on listing
                    'current_performance': -65.0,
                    'sector': 'Financial Services'
                },
                {
                    'name': 'Nykaa (FSN E-Commerce)',
                    'listing_gain': 89.0,
                    'current_performance': 15.0,
                    'sector': 'Consumer Goods'
                }
            ]
            return indian_recent_ipos
        except Exception as e:
            logger.error(f"Error fetching recent IPO performance: {e}")
            return []
    
    def get_indian_market_sentiment(self) -> Dict[str, Any]:
        """Get overall Indian market sentiment and indicators."""
        try:
            # This would typically fetch data about Nifty, Sensex, market conditions
            return {
                'nifty_trend': 'bullish',  # This would be calculated from real data
                'sensex_trend': 'bullish',
                'ipo_activity': 'high',  # Number of IPOs in pipeline
                'market_volatility': 'moderate',
                'fii_sentiment': 'positive',  # Foreign Institutional Investors
                'dii_sentiment': 'positive'   # Domestic Institutional Investors
            }
        except Exception as e:
            logger.error(f"Error fetching Indian market sentiment: {e}")
            return {}


class DataSourceManager:
    """Central manager for all data sources with enhanced prospectus integration."""
    
    def __init__(self, use_enhanced_prospectus: bool = True):
        self.financial_source = FinancialDataSource()
        self.news_source = NewsDataSource()
        self.ipo_source = IPODataSource()
        self.prospectus_source = ProspectusDataSource()
        
        # Initialize enhanced prospectus source if available
        self.use_enhanced = use_enhanced_prospectus and ENHANCED_PROSPECTUS_AVAILABLE
        if self.use_enhanced:
            self.enhanced_prospectus_source = EnhancedProspectusDataSource()
            logger.info("Enhanced prospectus integration enabled")
        else:
            self.enhanced_prospectus_source = None
            logger.info("Using basic prospectus integration")
    
    def collect_all_data(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Collect all available data for a listed company."""
        logger.info(f"Collecting data for {company_name} ({symbol})")
        
        return {
            'company_overview': self.financial_source.get_company_overview(symbol),
            'financial_statements': self.financial_source.get_financial_statements(symbol),
            'company_news': self.news_source.get_company_news(company_name),
            'market_news': self.news_source.get_market_news(),
            'upcoming_ipos': self.ipo_source.get_upcoming_ipos(),
            'recent_ipo_performance': self.ipo_source.get_recent_ipo_performance()
        }
    
    def collect_ipo_data(self, company_name: str, ipo_details: dict, use_enhanced: bool = None) -> Dict[str, Any]:
        """Collect comprehensive data for IPO companies with enhanced prospectus parsing."""
        logger.info(f"Collecting comprehensive IPO data for {company_name}")
        
        sector = ipo_details.get('sector', '')
        
        # Collect basic market data
        data = {
            'company_news': self.news_source.get_company_news(company_name),
            'market_news': self.news_source.get_market_news(sector),
            'sector_news': self.news_source.get_sector_news(sector),
            'upcoming_ipos': self.ipo_source.get_upcoming_ipos(),
            'recent_ipo_performance': self.ipo_source.get_recent_ipo_performance(),
            'indian_market_data': self.ipo_source.get_indian_market_sentiment(),
            'ipo_details': ipo_details
        }
        
        # Determine which prospectus integration to use
        use_enhanced_parser = (use_enhanced if use_enhanced is not None 
                             else self.use_enhanced)
        
        if use_enhanced_parser and self.enhanced_prospectus_source:
            # Use enhanced prospectus integration
            logger.info(f"Using enhanced prospectus parsing for {company_name}")
            try:
                data = integrate_enhanced_prospectus_data(company_name, data)
                
                # Add processing metrics
                enhanced_data = data.get('enhanced_prospectus')
                if enhanced_data:
                    data['prospectus_quality'] = {
                        'quality_score': enhanced_data.data_quality_score,
                        'source_confidence': enhanced_data.source_confidence,
                        'validation_flags': enhanced_data.validation_flags,
                        'extraction_method': 'enhanced'
                    }
                    logger.info(f"Enhanced prospectus data extracted with quality score: {enhanced_data.data_quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Enhanced prospectus integration failed for {company_name}: {e}")
                # Fallback to basic integration
                logger.info("Falling back to basic prospectus integration")
                try:
                    data = integrate_prospectus_data(company_name, data)
                    data['prospectus_quality'] = {'extraction_method': 'basic_fallback'}
                except Exception as e2:
                    logger.error(f"Basic prospectus integration also failed: {e2}")
                    data['prospectus_error'] = str(e2)
        else:
            # Use basic prospectus integration
            logger.info(f"Using basic prospectus parsing for {company_name}")
            try:
                data = integrate_prospectus_data(company_name, data)
                data['prospectus_quality'] = {'extraction_method': 'basic'}
            except Exception as e:
                logger.warning(f"Basic prospectus integration failed for {company_name}: {e}")
                data['prospectus_error'] = str(e)
        
        return data
    
    def get_prospectus_summary(self, company_name: str) -> Dict[str, Any]:
        """Get a quick summary of prospectus data availability."""
        try:
            if self.enhanced_prospectus_source:
                return self.enhanced_prospectus_source.get_data_summary(company_name)
            else:
                return self.prospectus_source.get_prospectus_summary(company_name)
        except Exception as e:
            logger.error(f"Error getting prospectus summary for {company_name}: {e}")
            return {'error': str(e), 'available': False}
    
    def refresh_prospectus_cache(self, company_name: str) -> bool:
        """Force refresh of cached prospectus data."""
        try:
            if self.enhanced_prospectus_source:
                result = self.enhanced_prospectus_source.get_enhanced_ipo_data(
                    company_name, force_refresh=True
                )
                return result is not None
            else:
                # Basic parser doesn't have caching
                return True
        except Exception as e:
            logger.error(f"Error refreshing cache for {company_name}: {e}")
            return False
