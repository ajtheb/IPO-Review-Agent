"""
Grey Market Premium (GMP) Data Fetcher for Indian IPOs.

This module fetches Grey Market Premium data for IPOs using multiple sources:
1. Web scraping from popular IPO tracking websites
2. Fallback to web search APIs
3. Caching mechanism to avoid excessive requests

GMP indicates the premium at which IPO shares are trading in the unofficial grey market
before listing, providing insights into expected listing gains.
"""

import os
import re
import requests
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from loguru import logger
import json
from urllib.parse import quote_plus
import time


class GMPFetcher:
    """Fetches Grey Market Premium data for Indian IPOs."""
    
    # Popular Indian IPO tracking websites
    GMP_SOURCES = {
        'investorgain': 'https://www.investorgain.com/report/live-ipo-gmp/331/',
        'chittorgarh': 'https://www.chittorgarh.com/ipo',
        'ipowatch': 'https://www.ipowatch.in'
    }
    
    def __init__(self, cache_duration_hours: int = 6, use_llm_fallback: bool = True):
        """
        Initialize the GMP fetcher.
        
        Args:
            cache_duration_hours: How long to cache GMP data (default: 6 hours)
            use_llm_fallback: Whether to use LLM extraction when scraping fails (default: True)
        """
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.use_llm_fallback = use_llm_fallback
        self.llm_extractor = None
        
        # Initialize LLM extractor if enabled
        if use_llm_fallback:
            try:
                from src.data_sources.llm_gmp_extractor import LLMGMPExtractor
                self.llm_extractor = LLMGMPExtractor(provider="groq")
                logger.info("LLM fallback enabled for GMP extraction with Groq")
            except Exception as e:
                logger.warning(f"Could not initialize LLM extractor: {e}")
                self.use_llm_fallback = False
        
    def get_gmp(self, company_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch Grey Market Premium data for a company.
        
        Args:
            company_name: Name of the company (e.g., "Vidya Wires")
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing GMP data:
            {
                'company_name': str,
                'gmp_price': float or None,
                'gmp_percentage': float or None,
                'issue_price': float or None,
                'expected_listing_price': float or None,
                'estimated_listing_gain': float or None,
                'last_updated': datetime,
                'source': str,
                'status': str  # 'active', 'not_found', 'error'
            }
        """
        # Check cache
        if use_cache and company_name in self.cache:
            cached_data = self.cache[company_name]
            if datetime.now() - cached_data['last_updated'] < self.cache_duration:
                logger.info(f"Using cached GMP data for {company_name}")
                return cached_data
        
        logger.info(f"Fetching GMP data for {company_name}")
        
        # Try multiple sources
        gmp_data = None
        
        # Try InvestorGain first (most reliable structure)
        try:
            gmp_data = self._fetch_from_investorgain(company_name)
            if gmp_data and gmp_data['status'] == 'active':
                self.cache[company_name] = gmp_data
                return gmp_data
        except Exception as e:
            logger.warning(f"InvestorGain fetch failed: {e}")
        
        # Try Chittorgarh as fallback
        try:
            gmp_data = self._fetch_from_chittorgarh(company_name)
            if gmp_data and gmp_data['status'] == 'active':
                self.cache[company_name] = gmp_data
                return gmp_data
        except Exception as e:
            logger.warning(f"Chittorgarh fetch failed: {e}")
        
        # Try IPOWatch as second fallback
        try:
            gmp_data = self._fetch_from_ipowatch(company_name)
            if gmp_data and gmp_data['status'] == 'active':
                self.cache[company_name] = gmp_data
                return gmp_data
        except Exception as e:
            logger.warning(f"IPOWatch fetch failed: {e}")
        
        # If all scraping fails, try LLM-based extraction as a last resort
        if self.use_llm_fallback:
            try:
                logger.info("Attempting LLM-based extraction for GMP data")
                gmp_data = self._fetch_with_llm_extraction(company_name)
                if gmp_data and gmp_data['status'] == 'active':
                    self.cache[company_name] = gmp_data
                    return gmp_data
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
        
        # Return not found status
        result = {
            'company_name': company_name,
            'gmp_price': None,
            'gmp_percentage': None,
            'issue_price': None,
            'expected_listing_price': None,
            'estimated_listing_gain': None,
            'last_updated': datetime.now(),
            'source': 'none',
            'status': 'not_found',
            'message': 'GMP data not available for this company'
        }
        
        self.cache[company_name] = result
        return result
    
    def _fetch_from_investorgain(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Fetch GMP from InvestorGain website."""
        try:
            url = self.GMP_SOURCES['investorgain']
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if page uses dynamic loading (spinner present)
            spinner = soup.find('div', class_='spinner-border')
            if spinner:
                logger.warning(f"InvestorGain uses dynamic content loading - static HTML parsing won't work")
                logger.info(f"Consider using Selenium or Playwright for dynamic content")
                
                # Try to find any mentions in the static content
                page_text = soup.get_text()
                if company_name.lower() in page_text.lower():
                    logger.info(f"{company_name} is mentioned on the page but GMP data not in static HTML")
                    return {
                        'company_name': company_name,
                        'gmp_price': None,
                        'gmp_percentage': None,
                        'issue_price': None,
                        'expected_listing_price': None,
                        'estimated_listing_gain': None,
                        'last_updated': datetime.now(),
                        'source': 'investorgain',
                        'status': 'not_available',
                        'message': 'IPO mentioned but GMP data requires JavaScript to load'
                    }
                return None
            
            # Find the table containing GMP data
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    # Check if this row contains our company
                    row_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                    if self._fuzzy_match(company_name, row_text):
                        logger.info(f"Match found in InvestorGain for {company_name}")
                        return self._parse_investorgain_row(cells, company_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from InvestorGain: {e}")
            return None
    
    def _parse_investorgain_row(self, cells: List, company_name: str) -> Dict[str, Any]:
        """Parse GMP data from InvestorGain table row."""
        try:
            # Typical structure: Company | Price Band | GMP | Estimated Listing | ...
            # This may vary, so we'll use flexible parsing
            
            data = {
                'company_name': company_name,
                'gmp_price': None,
                'gmp_percentage': None,
                'issue_price': None,
                'expected_listing_price': None,
                'estimated_listing_gain': None,
                'last_updated': datetime.now(),
                'source': 'investorgain',
                'status': 'active'
            }
            
            # Extract numeric values from cells
            for i, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                
                # Extract GMP price (usually in format ₹XX or XX)
                if 'gmp' in text.lower() or (i > 0 and i < len(cells) - 1):
                    gmp_match = re.search(r'[₹\$]?\s*(\d+(?:\.\d+)?)', text)
                    if gmp_match and data['gmp_price'] is None:
                        try:
                            data['gmp_price'] = float(gmp_match.group(1))
                        except:
                            pass
                
                # Extract issue price
                if 'price' in text.lower() or 'band' in text.lower():
                    price_match = re.search(r'[₹\$]?\s*(\d+(?:\.\d+)?)', text)
                    if price_match:
                        try:
                            data['issue_price'] = float(price_match.group(1))
                        except:
                            pass
            
            # Calculate derived metrics
            if data['gmp_price'] and data['issue_price']:
                data['expected_listing_price'] = data['issue_price'] + data['gmp_price']
                data['gmp_percentage'] = (data['gmp_price'] / data['issue_price']) * 100
                data['estimated_listing_gain'] = data['gmp_percentage']
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing InvestorGain row: {e}")
            return None
    
    def _fetch_from_chittorgarh(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Fetch GMP from Chittorgarh website."""
        try:
            url = self.GMP_SOURCES['chittorgarh']
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Chittorgarh uses a different table structure
            tables = soup.find_all('table', class_='table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    row_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                    if self._fuzzy_match(company_name, row_text):
                        return self._parse_chittorgarh_row(cells, company_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from Chittorgarh: {e}")
            return None
    
    def _parse_chittorgarh_row(self, cells: List, company_name: str) -> Dict[str, Any]:
        """Parse GMP data from Chittorgarh table row."""
        try:
            data = {
                'company_name': company_name,
                'gmp_price': None,
                'gmp_percentage': None,
                'issue_price': None,
                'expected_listing_price': None,
                'estimated_listing_gain': None,
                'last_updated': datetime.now(),
                'source': 'chittorgarh',
                'status': 'active'
            }
            
            # Extract values from cells
            for cell in cells:
                text = cell.get_text(strip=True)
                
                # Look for GMP value
                if '₹' in text or 'Rs' in text:
                    numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
                    if numbers:
                        try:
                            val = float(numbers[0])
                            if val > 0 and val < 10000:  # Reasonable GMP range
                                if data['gmp_price'] is None:
                                    data['gmp_price'] = val
                                elif data['issue_price'] is None:
                                    data['issue_price'] = val
                        except:
                            pass
            
            # Calculate derived metrics
            if data['gmp_price'] and data['issue_price']:
                data['expected_listing_price'] = data['issue_price'] + data['gmp_price']
                data['gmp_percentage'] = (data['gmp_price'] / data['issue_price']) * 100
                data['estimated_listing_gain'] = data['gmp_percentage']
            
            return data if data['gmp_price'] else None
            
        except Exception as e:
            logger.error(f"Error parsing Chittorgarh row: {e}")
            return None
    
    def _fetch_from_ipowatch(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Fetch GMP from IPOWatch website."""
        try:
            url = self.GMP_SOURCES['ipowatch']
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # IPOWatch uses divs and spans, different structure
            content = soup.get_text()
            
            # Look for company name and nearby GMP data
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if self._fuzzy_match(company_name, line):
                    # Look in nearby lines for GMP data
                    context = '\n'.join(lines[max(0, i-3):min(len(lines), i+3)])
                    return self._parse_ipowatch_context(context, company_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from IPOWatch: {e}")
            return None
    
    def _parse_ipowatch_context(self, context: str, company_name: str) -> Dict[str, Any]:
        """Parse GMP data from IPOWatch context."""
        try:
            data = {
                'company_name': company_name,
                'gmp_price': None,
                'gmp_percentage': None,
                'issue_price': None,
                'expected_listing_price': None,
                'estimated_listing_gain': None,
                'last_updated': datetime.now(),
                'source': 'ipowatch',
                'status': 'active'
            }
            
            # Extract GMP mentions
            gmp_match = re.search(r'GMP[:\s]+[₹\$]?\s*(\d+(?:\.\d+)?)', context, re.IGNORECASE)
            if gmp_match:
                data['gmp_price'] = float(gmp_match.group(1))
            
            # Extract price band
            price_match = re.search(r'Price[:\s]+[₹\$]?\s*(\d+(?:\.\d+)?)', context, re.IGNORECASE)
            if price_match:
                data['issue_price'] = float(price_match.group(1))
            
            # Calculate derived metrics
            if data['gmp_price'] and data['issue_price']:
                data['expected_listing_price'] = data['issue_price'] + data['gmp_price']
                data['gmp_percentage'] = (data['gmp_price'] / data['issue_price']) * 100
                data['estimated_listing_gain'] = data['gmp_percentage']
            
            return data if data['gmp_price'] else None
            
        except Exception as e:
            logger.error(f"Error parsing IPOWatch context: {e}")
            return None
    
    def _fuzzy_match(self, query: str, text: str, threshold: float = 0.6) -> bool:
        """
        Fuzzy match company name in text.
        
        Args:
            query: Company name to search for
            text: Text to search in
            threshold: Minimum similarity ratio (0-1)
            
        Returns:
            True if match found
        """
        query = query.lower().strip()
        text = text.lower()
        
        # Exact match
        if query in text:
            return True
        
        # Word-by-word match
        query_words = set(query.split())
        text_words = set(text.split())
        
        # Calculate Jaccard similarity
        if query_words:
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                return True
        
        # Partial match (all significant words present)
        significant_words = [w for w in query_words if len(w) > 3]
        if significant_words:
            matches = sum(1 for word in significant_words if word in text)
            if matches / len(significant_words) >= threshold:
                return True
        
        return False
    
    def get_multiple_gmp(self, company_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch GMP data for multiple companies.
        
        Args:
            company_names: List of company names
            
        Returns:
            Dictionary mapping company names to their GMP data
        """
        results = {}
        
        for company_name in company_names:
            try:
                results[company_name] = self.get_gmp(company_name)
                # Rate limiting to avoid overwhelming servers
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching GMP for {company_name}: {e}")
                results[company_name] = {
                    'company_name': company_name,
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def clear_cache(self, company_name: Optional[str] = None):
        """
        Clear cached GMP data.
        
        Args:
            company_name: Specific company to clear, or None to clear all
        """
        if company_name:
            self.cache.pop(company_name, None)
            logger.info(f"Cleared cache for {company_name}")
        else:
            self.cache.clear()
            logger.info("Cleared all GMP cache")
    
    def format_gmp_report(self, gmp_data: Dict[str, Any]) -> str:
        """
        Format GMP data into a readable report.
        
        Args:
            gmp_data: GMP data dictionary
            
        Returns:
            Formatted string report
        """
        if gmp_data['status'] == 'not_found':
            return f"❌ Grey Market Premium data not available for {gmp_data['company_name']}"
        
        if gmp_data['status'] == 'error':
            return f"⚠️ Error fetching GMP data for {gmp_data['company_name']}"
        
        report = []
        report.append(f"📊 Grey Market Premium Report for {gmp_data['company_name']}")
        report.append("=" * 60)
        
        if gmp_data['issue_price']:
            report.append(f"💰 Issue Price: ₹{gmp_data['issue_price']:.2f}")
        
        if gmp_data['gmp_price']:
            report.append(f"📈 GMP: ₹{gmp_data['gmp_price']:.2f}")
        
        if gmp_data['gmp_percentage']:
            emoji = "🔥" if gmp_data['gmp_percentage'] > 50 else "✅" if gmp_data['gmp_percentage'] > 20 else "📊"
            report.append(f"{emoji} GMP Percentage: {gmp_data['gmp_percentage']:.2f}%")
        
        if gmp_data['expected_listing_price']:
            report.append(f"🎯 Expected Listing Price: ₹{gmp_data['expected_listing_price']:.2f}")
        
        if gmp_data['estimated_listing_gain']:
            report.append(f"💹 Estimated Listing Gain: {gmp_data['estimated_listing_gain']:.2f}%")
        
        report.append(f"\n📅 Last Updated: {gmp_data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🔗 Source: {gmp_data['source']}")
        
        # Add interpretation
        report.append("\n💡 Interpretation:")
        if gmp_data['gmp_percentage']:
            if gmp_data['gmp_percentage'] > 50:
                report.append("   🔥 Very Strong Grey Market Premium - High demand expected")
            elif gmp_data['gmp_percentage'] > 30:
                report.append("   ✅ Strong Grey Market Premium - Good listing gains expected")
            elif gmp_data['gmp_percentage'] > 10:
                report.append("   📊 Moderate Grey Market Premium - Positive sentiment")
            elif gmp_data['gmp_percentage'] > 0:
                report.append("   ⚠️  Low Grey Market Premium - Mixed sentiment")
            else:
                report.append("   ❌ Negative/Zero GMP - Caution advised")
        
        report.append("\n⚠️  Note: GMP is unofficial and subject to change. It's an indicator, not a guarantee.")
        
        return '\n'.join(report)
    
    def _fetch_with_llm_extraction(self, company_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch GMP using LLM-based extraction as a fallback.
        
        This method scrapes the page content and uses LLM to extract GMP data
        from the text, even when JavaScript rendering is required.
        """
        if not self.llm_extractor:
            logger.warning("LLM extractor not initialized")
            return None
        
        logger.info(f"Attempting LLM-based extraction for {company_name}")
        print(f"\n{'='*80}")
        print(f"🤖 LLM EXTRACTION MODE for: {company_name}")
        print(f"{'='*80}\n")
        
        # Try each source and extract with LLM
        for source_name, url in self.GMP_SOURCES.items():
            try:
                logger.info(f"Fetching content from {source_name} for LLM extraction")
                print(f"📡 Fetching from: {source_name}")
                print(f"🔗 URL: {url}")
                
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                print(f"✅ Content fetched successfully")
                print(f"📄 Content size: {len(response.text):,} characters\n")
                
                # Use LLM to extract data from the raw HTML
                # Enable chunk printing to show what's being sent to the LLM
                result = self.llm_extractor.extract_gmp_from_scraped_content(
                    company_name=company_name,
                    html_content=response.text,
                    print_chunks=True,  # Enable chunk printing
                    save_chunks=True,   # Enable chunk saving
                    chunks_folder="gmp_chunks",  # Save to gmp_chunks folder
                    source=source_name  # Pass source name for better identification
                )
                
                if result and result.get('status') in ['success', 'active']:
                    logger.info(f"LLM successfully extracted GMP data from {source_name}")
                    print(f"\n✅ Successfully extracted GMP data from {source_name}")
                    print(f"{'='*80}\n")
                    # Normalize status
                    result['status'] = 'active'
                    result['source'] = f"{source_name}_llm"
                    return result
                else:
                    logger.debug(f"LLM could not extract from {source_name}: {result.get('message', 'unknown')}")
                    print(f"⚠️  No GMP data found in {source_name}, trying next source...\n")
                    
            except Exception as e:
                logger.warning(f"Error during LLM extraction from {source_name}: {e}")
                print(f"❌ Error with {source_name}: {e}\n")
                continue
        
        logger.warning(f"LLM extraction failed for all sources for {company_name}")
        print(f"❌ LLM extraction failed for all sources")
        print(f"{'='*80}\n")
        return None
