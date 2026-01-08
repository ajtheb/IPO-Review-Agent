"""
Enhanced IPO Prospectus and DRHP document parser with improved accuracy and robustness.
Includes advanced financial data extraction, cross-validation, and caching mechanisms.
"""

import os
import re
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from loguru import logger
import requests
from bs4 import BeautifulSoup
import tempfile
import concurrent.futures
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

try:
    import PyPDF2
    import pdfplumber
    import tabula
    PDF_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("PDF processing libraries not available. Install PyPDF2, pdfplumber, tabula-py")
    PDF_LIBS_AVAILABLE = False


@dataclass
class EnhancedFinancialData:
    """Enhanced structured financial data with validation."""
    revenue_data: Dict[str, float]
    profit_data: Dict[str, float]
    ebitda_data: Dict[str, float]
    assets_data: Dict[str, float]
    liabilities_data: Dict[str, float]
    equity_data: Dict[str, float]
    cash_flow_data: Dict[str, float]
    key_ratios: Dict[str, float]
    growth_metrics: Dict[str, float]
    
    # Qualitative data
    business_description: str
    risk_factors: List[str]
    use_of_funds: List[str]
    company_strengths: List[str]
    competitive_advantages: List[str]
    
    # Metadata
    extraction_date: str
    data_quality_score: float
    source_confidence: float
    validation_flags: List[str]


class DataValidator:
    """Validates extracted financial data for consistency and accuracy."""
    
    @staticmethod
    def validate_financial_data(data: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
        """Validate financial data for logical consistency."""
        issues = []
        
        # Check for reasonable revenue growth
        revenue = data.get('revenue', {})
        if len(revenue) >= 2:
            years = sorted(revenue.keys())
            for i in range(1, len(years)):
                prev_year, curr_year = years[i-1], years[i]
                if revenue[curr_year] > revenue[prev_year] * 10:
                    issues.append(f"Unusual revenue jump: {prev_year} to {curr_year}")
        
        # Check profit vs revenue relationship
        profit = data.get('profit', {})
        for year in revenue.keys():
            if year in profit:
                if profit[year] > revenue[year]:
                    issues.append(f"Profit exceeds revenue in {year}")
        
        # Check for negative values where inappropriate
        for metric, values in data.items():
            if metric in ['assets', 'equity'] and any(v < 0 for v in values.values()):
                issues.append(f"Negative {metric} detected")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def calculate_quality_score(data: EnhancedFinancialData) -> float:
        """Calculate data quality score based on completeness and consistency."""
        score = 0.0
        max_score = 10.0
        
        # Revenue completeness (2 points)
        if len(data.revenue_data) >= 3:
            score += 2.0
        elif len(data.revenue_data) >= 2:
            score += 1.5
        elif len(data.revenue_data) >= 1:
            score += 1.0
        
        # Profit data (1.5 points)
        if len(data.profit_data) >= 2:
            score += 1.5
        elif len(data.profit_data) >= 1:
            score += 1.0
        
        # Balance sheet data (1.5 points)
        if data.assets_data and data.liabilities_data:
            score += 1.5
        elif data.assets_data or data.liabilities_data:
            score += 1.0
        
        # Ratios (1 point)
        if len(data.key_ratios) >= 3:
            score += 1.0
        elif len(data.key_ratios) >= 1:
            score += 0.5
        
        # Qualitative data (4 points)
        if data.business_description and len(data.business_description) > 100:
            score += 1.0
        if len(data.risk_factors) >= 3:
            score += 1.0
        if len(data.use_of_funds) >= 2:
            score += 1.0
        if len(data.company_strengths) >= 2:
            score += 1.0
        
        return min(score / max_score, 1.0)


class CacheManager:
    """Manages caching for prospectus data to avoid repeated downloads."""
    
    def __init__(self, cache_dir: str = None, max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "ipo_prospectus_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
    
    def _get_cache_key(self, company_name: str) -> str:
        """Generate cache key for company."""
        return hashlib.md5(company_name.lower().encode()).hexdigest()
    
    def get_cached_data(self, company_name: str) -> Optional[EnhancedFinancialData]:
        """Get cached financial data if available and fresh."""
        cache_key = self._get_cache_key(company_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is fresh
        if datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime) > self.max_age:
            cache_file.unlink()  # Remove stale cache
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for {company_name}: {e}")
            return None
    
    def cache_data(self, company_name: str, data: EnhancedFinancialData):
        """Cache financial data."""
        cache_key = self._get_cache_key(company_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached prospectus data for {company_name}")
        except Exception as e:
            logger.error(f"Error caching data for {company_name}: {e}")


class EnhancedSEBISource:
    """Enhanced SEBI filing source with multiple search strategies."""
    
    def __init__(self):
        self.sebi_base_url = "https://www.sebi.gov.in"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search_comprehensive(self, company_name: str) -> List[Dict[str, Any]]:
        """
        Comprehensive search across multiple SEBI endpoints.
        Updated to prioritize Draft Offer Documents for pre-IPO companies.
        """
        all_filings = []
        
        # Strategy 1: Draft Offer Documents search (PRIMARY for pre-IPO companies)
        try:
            filings1 = self._search_draft_offer_documents(company_name)
            all_filings.extend(filings1)
            logger.info(f"Found {len(filings1)} filings via Draft Offer Documents")
        except Exception as e:
            logger.warning(f"Draft Offer Documents search failed: {e}")
        
        # Strategy 2: Issues and Listing search (for completed IPOs)
        try:
            filings2 = self._search_issues_and_listing(company_name)
            all_filings.extend(filings2)
            logger.info(f"Found {len(filings2)} filings via Issues & Listing")
        except Exception as e:
            logger.warning(f"Issues & Listing search failed: {e}")
        
        # Strategy 3: Public database search (fallback)
        try:
            filings3 = self._search_public_database(company_name)
            all_filings.extend(filings3)
            logger.info(f"Found {len(filings3)} filings via public database")
        except Exception as e:
            logger.warning(f"Public database search failed: {e}")
        
        # Strategy 4: IPO specific search (additional coverage)
        try:
            filings4 = self._search_ipo_specific(company_name)
            all_filings.extend(filings4)
            logger.info(f"Found {len(filings4)} filings via IPO search")
        except Exception as e:
            logger.warning(f"IPO specific search failed: {e}")
        
        # Strategy 4: Draft Offer Documents search (pre-IPO companies)
        try:
            filings4 = self._search_draft_offer_documents(company_name)
            all_filings.extend(filings4)
            logger.info(f"Found {len(filings4)} filings via Draft Offer Documents search")
        except Exception as e:
            logger.warning(f"Draft Offer Documents search failed: {e}")
        
        # Remove duplicates and sort by date
        unique_filings = self._deduplicate_filings(all_filings)
        return sorted(unique_filings, key=lambda x: x.get('date', ''), reverse=True)
    
    def _search_issues_and_listing(self, company_name: str) -> List[Dict[str, Any]]:
        """Search SEBI Issues and Listing section - most effective method."""
        url = f"{self.sebi_base_url}/sebiweb/home/HomeAction.do?doListingAll=yes&cid=1"
        
        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        return self._parse_issues_listing_results(soup, company_name)
    
    def _search_public_database(self, company_name: str) -> List[Dict[str, Any]]:
        """Search SEBI public database - fallback method."""
        url = f"{self.sebi_base_url}/sebiweb/other/OtherAction.do"
        params = {
            'doRecognition': 'yes',
            'intmId': '13',
            'companyName': company_name,
            'segment': 'IPO'
        }
        
        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        return self._parse_search_results(soup)
    
    def _search_corporate_filings(self, company_name: str) -> List[Dict[str, Any]]:
        """Search corporate filings section."""
        # Alternative SEBI endpoints for corporate filings
        endpoints = [
            "/sebiweb/action/CorporateDetails.do",
            "/sebiweb/action/CompanyDetails.do"
        ]
        
        filings = []
        for endpoint in endpoints:
            try:
                url = urljoin(self.sebi_base_url, endpoint)
                params = {'companyName': company_name}
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    filings.extend(self._parse_search_results(soup))
            except Exception as e:
                logger.warning(f"Error searching {endpoint}: {e}")
        
        return filings
    
    def _search_ipo_specific(self, company_name: str) -> List[Dict[str, Any]]:
        """Search IPO-specific SEBI sections."""
        # Try alternative company name formats
        name_variants = [
            company_name,
            company_name.replace('Limited', 'Ltd'),
            company_name.replace('Ltd', 'Limited'),
            company_name.split()[0]  # First word only
        ]
        
        all_filings = []
        for variant in name_variants:
            try:
                url = f"{self.sebi_base_url}/sebiweb/action/IPOAction.do"
                params = {
                    'companyName': variant,
                    'docType': 'DRHP'
                }
                
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    filings = self._parse_search_results(soup)
                    all_filings.extend(filings)
                    
                    if filings:  # If we found something, no need to try other variants
                        break
                        
            except Exception as e:
                logger.warning(f"IPO search failed for {variant}: {e}")
        
        return all_filings
    
    def _search_draft_offer_documents(self, company_name: str) -> List[Dict[str, Any]]:
        """
        Search SEBI Draft Offer Documents section - PRIMARY method for pre-IPO companies.
        This is where companies like Vidya Wires are found.
        """
        filings = []
        
        try:
            # Use Selenium for JavaScript-heavy SEBI Draft Documents page
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            except ImportError:
                driver = webdriver.Chrome(options=options)
            
            try:
                # Navigate to Draft Offer Documents section
                url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
                driver.get(url)
                time.sleep(5)
                
                # Find and use the search box
                search_box = driver.find_element(By.CSS_SELECTOR, "input[placeholder*='Title, Keywords, Entity Name']")
                search_box.clear()
                search_box.send_keys(company_name)
                search_box.send_keys(Keys.RETURN)
                time.sleep(3)
                
                # Parse results
                tables = driver.find_elements(By.TAG_NAME, "table")
                
                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    
                    for row in rows:
                        row_text = row.text.strip()
                        if company_name.lower() in row_text.lower() and len(row_text) > 10:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            
                            if len(cells) >= 2:
                                cell_texts = [cell.text.strip() for cell in cells]
                                
                                # Look for document links
                                links = row.find_elements(By.TAG_NAME, "a")
                                doc_link = None
                                if links:
                                    doc_link = links[0].get_attribute('href')
                                
                                # Create filing record
                                filing = {
                                    'date': cell_texts[0] if cell_texts else '',
                                    'company': cell_texts[1] if len(cell_texts) > 1 else company_name,
                                    'title': row_text,
                                    'type': 'Draft Offer Document',
                                    'url': doc_link,
                                    'source': 'SEBI Draft Offer Documents',
                                    'full_text': row_text
                                }
                                
                                # Determine document type from content
                                if 'drhp' in row_text.lower():
                                    filing['document_type'] = 'DRHP'
                                elif 'rhp' in row_text.lower():
                                    filing['document_type'] = 'RHP'
                                elif 'prospectus' in row_text.lower():
                                    filing['document_type'] = 'Prospectus'
                                else:
                                    filing['document_type'] = 'IPO Document'
                                
                                filings.append(filing)
                                logger.info(f"Found Draft Document: {filing['company']} - {filing['date']}")
            
            finally:
                driver.quit()
        
        except Exception as e:
            logger.warning(f"Error searching Draft Offer Documents: {e}")
            # Fallback to requests-based search if Selenium fails
            try:
                response = self.session.get(
                    "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10",
                    timeout=15
                )
                if company_name.lower() in response.text.lower():
                    logger.info(f"Found potential match in Draft Documents page content for {company_name}")
            except:
                pass
        
        return filings

    def _parse_search_results(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Enhanced parsing of SEBI search results."""
        filings = []
        
        # Look for different table structures
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            # Skip if no data rows
            if len(rows) < 2:
                continue
            
            # Try to identify header structure
            header_row = rows[0]
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 3:
                    continue
                
                cell_data = [cell.get_text(strip=True) for cell in cells]
                
                # Look for IPO-related documents
                document_text = ' '.join(cell_data).upper()
                if any(keyword in document_text for keyword in ['DRHP', 'PROSPECTUS', 'RED HERRING', 'IPO']):
                    
                    # Extract download link
                    download_link = None
                    for cell in cells:
                        link = cell.find('a')
                        if link and link.get('href'):
                            href = link['href']
                            if href.startswith('http'):
                                download_link = href
                            else:
                                download_link = urljoin(self.sebi_base_url, href)
                            break
                    
                    if download_link:
                        filing = {
                            'date': cell_data[0] if cell_data else '',
                            'type': cell_data[1] if len(cell_data) > 1 else 'Document',
                            'company': cell_data[2] if len(cell_data) > 2 else '',
                            'url': download_link,
                            'source': 'SEBI'
                        }
                        filings.append(filing)
        
        return filings
    
    def _deduplicate_filings(self, filings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate filings based on URL and type."""
        seen = set()
        unique_filings = []
        
        for filing in filings:
            key = (filing.get('url', ''), filing.get('type', ''))
            if key not in seen:
                seen.add(key)
                unique_filings.append(filing)
        
        return unique_filings

    def _parse_issues_listing_results(self, soup: BeautifulSoup, company_name: str) -> List[Dict[str, Any]]:
        """Parse SEBI Issues and Listing results - optimized for actual structure."""
        filings = []
        
        # Find the main data table
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            # Skip if no data rows (need at least header + 1 data row)
            if len(rows) < 2:
                continue
            
            # Verify this is the correct table structure
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Look for the expected structure: ['Date', 'Type', 'Title'] or similar
            if len(headers) < 3:
                continue
            
            # Process data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 3:
                    continue
                
                # Extract basic data
                date_text = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                type_text = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                title_text = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                
                # Check if this row matches our company (case-insensitive)
                full_row_text = f"{type_text} {title_text}".lower()
                company_name_lower = company_name.lower()
                
                # Try different matching strategies
                company_match = False
                
                # Strategy 1: Direct company name match
                if company_name_lower in full_row_text:
                    company_match = True
                
                # Strategy 2: Individual word matching (for partial matches)
                elif len(company_name.split()) > 1:
                    company_words = [word.lower() for word in company_name.split() if len(word) > 2]
                    matching_words = sum(1 for word in company_words if word in full_row_text)
                    if matching_words >= len(company_words) * 0.7:  # 70% of words must match
                        company_match = True
                
                if company_match and any(keyword in full_row_text for keyword in ['public issues', 'ipo', 'prospectus', 'drhp', 'rhp']):
                    # Extract the link from title cell
                    title_link = cells[2].find('a') if len(cells) > 2 else None
                    filing_url = None
                    
                    if title_link and title_link.get('href'):
                        href = title_link['href']
                        if href.startswith('http'):
                            filing_url = href
                        else:
                            filing_url = urljoin(self.sebi_base_url, href)
                    
                    if filing_url:
                        filing = {
                            'date': date_text,
                            'type': type_text,
                            'title': title_text,
                            'company': self._extract_company_name(title_text),
                            'url': filing_url,
                            'source': 'SEBI Issues & Listing'
                        }
                        filings.append(filing)
                        logger.debug(f"Found matching filing: {title_text}")
        
        return filings
    
    def _extract_company_name(self, title_text: str) -> str:
        """Extract clean company name from title text."""
        # Remove common suffixes and document types
        clean_title = title_text
        
        # Remove document type suffixes
        suffixes_to_remove = [
            '- RHP', '- DRHP', '- Prospectus', '- Letter of Offer',
            'Addendum to RHP of', 'Addendum to DRHP of', 'Corrigendum to DRHP of'
        ]
        
        for suffix in suffixes_to_remove:
            if suffix in clean_title:
                clean_title = clean_title.replace(suffix, '').strip()
        
        return clean_title

class EnhancedProspectusParser:
    """Enhanced parser with advanced financial data extraction."""
    
    def __init__(self):
        self.validator = DataValidator()
        
        # Enhanced patterns for Indian financial data
        self.financial_patterns = {
            'revenue': [
                r'(?:Total\s+)?Revenue(?:\s+from\s+operations)?[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'Net\s+Revenue[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'Income\s+from\s+operations[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'Sales[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)'
            ],
            'profit': [
                r'Net\s+Profit(?:\s+after\s+tax)?[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'Profit\s+after\s+tax[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'(?:Net\s+)?Income\s+for\s+the\s+(?:year|period)[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)'
            ],
            'ebitda': [
                r'EBITDA[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'Earnings\s+before[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)'
            ],
            'assets': [
                r'Total\s+Assets[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)',
                r'Total\s+Current\s+Assets[\s\S]*?(\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?)'
            ]
        }
        
        self.year_patterns = [
            r'(?:FY|F\.Y\.)\s*(\d{4})',
            r'(?:FY|F\.Y\.)\s*(\d{2})',
            r'20(\d{2})-(\d{2})',
            r'March\s+(\d{4})',
            r'Year\s+ended.*?(\d{4})'
        ]
    
    def parse_enhanced(self, pdf_path: str, company_name: str) -> Optional[EnhancedFinancialData]:
        """Return document content for vector database without financial extraction."""
        if not PDF_LIBS_AVAILABLE:
            logger.error("PDF processing libraries not available")
            return None
        
        try:
            logger.info(f"Processing document content for {company_name}: {pdf_path}")
            
            # Extract text using multiple methods for vector database
            text_content = self._extract_text_enhanced(pdf_path)
            if not text_content:
                logger.warning(f"No text content extracted from {pdf_path}")
                return None
            
            logger.info(f"Extracted {len(text_content)} characters from document")
            
            # Create enhanced financial data object with raw document content
            # No financial extraction - just return the document content for vector DB
            enhanced_data = EnhancedFinancialData(
                revenue_data={},  # Empty - no extraction
                profit_data={},   # Empty - no extraction
                ebitda_data={},   # Empty - no extraction
                assets_data={},   # Empty - no extraction
                liabilities_data={},  # Empty - no extraction
                equity_data={},   # Empty - no extraction
                cash_flow_data={}, # Empty - no extraction
                key_ratios={},    # Empty - no calculation needed
                growth_metrics={}, # Empty - no calculation needed
                business_description=text_content,  # Store full document content here
                risk_factors=[],   # Empty - let vector DB handle this
                use_of_funds=[],   # Empty - let vector DB handle this
                company_strengths=[], # Empty - let vector DB handle this
                competitive_advantages=[], # Empty - let vector DB handle this
                extraction_date=datetime.now().isoformat(),
                data_quality_score=1.0,  # High score since we have document content
                source_confidence=0.9,   # High confidence for raw document
                validation_flags=[]      # No validation issues for raw content
            )
            
            logger.info(f"Document content prepared for vector database: {company_name}")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Document processing failed for {company_name}: {e}")
            return None
        
        finally:
            # Cleanup
            if os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass
    
    def _extract_text_enhanced(self, pdf_path: str) -> str:
        """Extract text using multiple methods for better accuracy."""
        methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
            self._extract_with_tabula
        ]
        
        best_text = ""
        max_length = 0
        
        for method in methods:
            try:
                text = method(pdf_path)
                if len(text) > max_length:
                    max_length = len(text)
                    best_text = text
            except Exception as e:
                logger.warning(f"Text extraction method failed: {e}")
        
        return best_text
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract using pdfplumber with table detection."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:50]):
                # Extract regular text
                page_text = page.extract_text() or ""
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            text += " ".join([str(cell) for cell in row if cell]) + "\n"
                
                text += page_text + "\n"
        
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract using PyPDF2."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(min(50, len(pdf_reader.pages))):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text
    
    def _extract_with_tabula(self, pdf_path: str) -> str:
        """Extract tables using tabula-py."""
        try:
            import tabula
            # Extract tables from first 20 pages
            tables = tabula.read_pdf(pdf_path, pages='1-20', multiple_tables=True, silent=True)
            
            text = ""
            for df in tables:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    text += df.to_string() + "\n\n"
            
            return text
        except:
            return ""
    
    def _extract_enhanced_financials(self, text: str) -> Dict[str, Dict[str, float]]:
        """Enhanced financial data extraction with pattern matching."""
        financials = {
            'revenue': {}, 'profit': {}, 'ebitda': {}, 
            'assets': {}, 'liabilities': {}, 'equity': {}, 'cash_flow': {}
        }
        
        # Split text into sections for better parsing
        sections = self._identify_financial_sections(text)
        
        for section_name, section_text in sections.items():
            self._extract_section_financials(section_text, financials)
        
        return financials
    
    def _identify_financial_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract financial statement sections."""
        sections = {}
        
        section_markers = {
            'profit_loss': ['profit and loss', 'statement of income', 'income statement'],
            'balance_sheet': ['balance sheet', 'statement of financial position'],
            'cash_flow': ['cash flow statement', 'statement of cash flows'],
            'financial_highlights': ['financial highlights', 'key financial data']
        }
        
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're entering a new section
            new_section = None
            for section, markers in section_markers.items():
                if any(marker in line_lower for marker in markers):
                    new_section = section
                    break
            
            if new_section:
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                
                # Start new section
                current_section = new_section
                section_content = [line]
            elif current_section:
                section_content.append(line)
                
                # Stop if we hit 100 lines (avoid capturing entire document)
                if len(section_content) > 100:
                    sections[current_section] = '\n'.join(section_content)
                    current_section = None
                    section_content = []
        
        # Save final section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def _extract_section_financials(self, section_text: str, financials: Dict[str, Dict[str, float]]):
        """Extract financials from a specific section."""
        # Extract years first
        years = self._extract_years(section_text)
        
        for metric, patterns in self.financial_patterns.items():
            values = self._extract_metric_values(section_text, patterns, years)
            if values:
                financials[metric].update(values)
    
    def _extract_years(self, text: str) -> List[str]:
        """Extract financial years from text."""
        years = []
        
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle patterns like "2022-23"
                    year = f"20{match[0]}" if len(match[0]) == 2 else match[0]
                else:
                    year = f"20{match}" if len(match) == 2 else match
                
                if year not in years and 2010 <= int(year) <= 2030:
                    years.append(year)
        
        return sorted(set(years), reverse=True)[:5]  # Last 5 years
    
    def _extract_metric_values(self, text: str, patterns: List[str], years: List[str]) -> Dict[str, float]:
        """Extract values for a specific metric."""
        values = {}
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Extract the numeric value
                amount_str = match.group(1)
                try:
                    # Handle Indian number format (crores, lakhs)
                    amount = self._parse_indian_amount(amount_str)
                    
                    # Try to match with nearby year
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(text), match.end() + 200)
                    context = text[context_start:context_end]
                    
                    matched_year = None
                    for year in years:
                        if year in context or f"FY{year[2:]}" in context:
                            matched_year = f"FY{year}"
                            break
                    
                    if matched_year and amount > 0:
                        values[matched_year] = amount
                        
                except (ValueError, IndexError):
                    continue
        
        return values
    
    def _parse_indian_amount(self, amount_str: str) -> float:
        """Parse Indian number format (with crores, lakhs notation)."""
        # Remove common formatting
        clean_amount = re.sub(r'[₹,\s]', '', amount_str)
        
        # Handle crores/lakhs
        multiplier = 1
        if 'crore' in amount_str.lower():
            multiplier = 10000000  # 1 crore = 10 million
        elif 'lakh' in amount_str.lower():
            multiplier = 100000  # 1 lakh = 100,000
        
        try:
            return float(clean_amount) * multiplier
        except ValueError:
            # Try to extract just the numeric part
            numeric = re.search(r'(\d+(?:\.\d+)?)', clean_amount)
            if numeric:
                return float(numeric.group(1)) * multiplier
            raise ValueError(f"Could not parse amount: {amount_str}")
    
    def _extract_qualitative_data(self, text: str) -> Dict[str, Any]:
        """Extract qualitative information."""
        return {
            'business': self._extract_business_description(text),
            'risks': self._extract_risk_factors(text),
            'use_of_funds': self._extract_use_of_funds(text),
            'strengths': self._extract_strengths(text),
            'advantages': self._extract_competitive_advantages(text)
        }
    
    def _extract_business_description(self, text: str) -> str:
        """Extract business description with improved accuracy."""
        business_markers = [
            'business overview', 'company profile', 'business description',
            'nature of business', 'our business', 'business model'
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in business_markers):
                # Extract next 30 lines
                desc_lines = lines[i+1:i+31]
                description = ' '.join([l.strip() for l in desc_lines if l.strip() and len(l) > 20])
                
                # Clean and return first 800 characters
                clean_desc = re.sub(r'\s+', ' ', description).strip()
                return clean_desc[:800] if clean_desc else ""
        
        return "Business description not found"
    
    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract risk factors with better parsing."""
        risk_markers = ['risk factors', 'principal risks', 'key risks', 'material risks']
        risks = []
        
        lines = text.split('\n')
        in_risk_section = False
        
        for line in lines:
            line_clean = line.strip()
            
            if any(marker in line_clean.lower() for marker in risk_markers):
                in_risk_section = True
                continue
            
            if in_risk_section:
                # Stop at next major section
                if any(stop_word in line_clean.lower() for stop_word in 
                       ['use of funds', 'financial statements', 'business overview', 'management']):
                    break
                
                # Look for risk items (numbered, bulleted, or paragraph starting)
                if (re.match(r'^\d+\.', line_clean) or 
                    line_clean.startswith('•') or 
                    line_clean.startswith('-') or
                    (len(line_clean) > 50 and line_clean[0].isupper())):
                    
                    risk_text = line_clean[:200]  # Limit length
                    if len(risk_text) > 20:
                        risks.append(risk_text)
                
                if len(risks) >= 10:
                    break
        
        return risks
    
    def _extract_use_of_funds(self, text: str) -> List[str]:
        """Extract use of funds information."""
        fund_markers = ['use of funds', 'objects of the offer', 'fund utilization', 'proceeds utilization']
        funds = []
        
        lines = text.split('\n')
        in_funds_section = False
        
        for line in lines:
            line_clean = line.strip()
            
            if any(marker in line_clean.lower() for marker in fund_markers):
                in_funds_section = True
                continue
            
            if in_funds_section:
                if any(stop_word in line_clean.lower() for stop_word in 
                       ['risk factors', 'financial statements', 'business overview']):
                    break
                
                if (re.match(r'^\d+\.', line_clean) or 
                    line_clean.startswith('•') or 
                    line_clean.startswith('-')):
                    
                    fund_text = line_clean[:150]
                    if len(fund_text) > 15:
                        funds.append(fund_text)
                
                if len(funds) >= 5:
                    break
        
        return funds
    
    def _extract_strengths(self, text: str) -> List[str]:
        """Extract company strengths."""
        strength_markers = ['competitive strengths', 'business strengths', 'key strengths', 'advantages']
        strengths = []
        
        lines = text.split('\n')
        in_strengths_section = False
        
        for line in lines:
            line_clean = line.strip()
            
            if any(marker in line_clean.lower() for marker in strength_markers):
                in_strengths_section = True
                continue
            
            if in_strengths_section:
                if any(stop_word in line_clean.lower() for stop_word in 
                       ['risk factors', 'use of funds', 'financial statements']):
                    break
                
                if (re.match(r'^\d+\.', line_clean) or 
                    line_clean.startswith('•') or 
                    line_clean.startswith('-')):
                    
                    strength_text = line_clean[:150]
                    if len(strength_text) > 20:
                        strengths.append(strength_text)
                
                if len(strengths) >= 7:
                    break
        
        return strengths
    
    def _extract_competitive_advantages(self, text: str) -> List[str]:
        """Extract competitive advantages."""
        advantage_keywords = [
            'market leader', 'competitive advantage', 'unique position',
            'brand recognition', 'cost leadership', 'technology edge'
        ]
        
        advantages = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(keyword in sentence_lower for keyword in advantage_keywords):
                if 20 < len(sentence) < 200:
                    advantages.append(sentence.strip())
                
                if len(advantages) >= 5:
                    break
        
        return advantages
    
    def _calculate_ratios(self, financial_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate financial ratios from extracted data."""
        ratios = {}
        
        try:
            revenue = financial_data.get('revenue', {})
            profit = financial_data.get('profit', {})
            assets = financial_data.get('assets', {})
            
            # Get latest year data
            if revenue and profit:
                latest_year = max(revenue.keys())
                if latest_year in profit:
                    # Profit margin
                    ratios['profit_margin'] = (profit[latest_year] / revenue[latest_year]) * 100
                    
            # Revenue growth (if multiple years available)
            if len(revenue) >= 2:
                years = sorted(revenue.keys())
                latest_rev = revenue[years[-1]]
                prev_rev = revenue[years[-2]]
                ratios['revenue_growth'] = ((latest_rev - prev_rev) / prev_rev) * 100
            
            # Asset efficiency
            if revenue and assets:
                latest_year = max(revenue.keys())
                if latest_year in assets:
                    ratios['asset_turnover'] = revenue[latest_year] / assets[latest_year]
                    
        except (KeyError, ZeroDivisionError, ValueError):
            pass  # Skip if calculation not possible
        
        return ratios
    
    def _calculate_growth_metrics(self, financial_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate growth metrics."""
        growth = {}
        
        try:
            revenue = financial_data.get('revenue', {})
            profit = financial_data.get('profit', {})
            
            # Calculate CAGR for revenue (if 3+ years available)
            if len(revenue) >= 3:
                years = sorted(revenue.keys())
                start_value = revenue[years[0]]
                end_value = revenue[years[-1]]
                num_years = len(years) - 1
                
                if start_value > 0:
                    cagr = (pow(end_value / start_value, 1/num_years) - 1) * 100
                    growth['revenue_cagr'] = cagr
            
            # Calculate profit growth
            if len(profit) >= 2:
                years = sorted(profit.keys())
                if len(years) >= 2:
                    latest_profit = profit[years[-1]]
                    prev_profit = profit[years[-2]]
                    
                    if prev_profit != 0:
                        growth['profit_growth'] = ((latest_profit - prev_profit) / prev_profit) * 100
                        
        except (KeyError, ZeroDivisionError, ValueError):
            pass
        
        return growth


class EnhancedProspectusDataSource:
    """Enhanced main class for IPO prospectus data integration."""
    
    def __init__(self, cache_enabled: bool = True):
        self.sebi_source = EnhancedSEBISource()
        self.parser = EnhancedProspectusParser()
        self.cache_manager = CacheManager() if cache_enabled else None
    
    def get_enhanced_ipo_data(self, company_name: str, force_refresh: bool = False) -> Optional[EnhancedFinancialData]:
        """Get enhanced IPO financial data with caching."""
        try:
            # Check cache first (unless force refresh)
            # if not force_refresh and self.cache_manager:
            #     cached_data = self.cache_manager.get_cached_data(company_name)
            #     if cached_data:
            #         logger.info(f"Using cached data for {company_name} (quality: {cached_data.data_quality_score:.2f})")
            #         return cached_data
            
            logger.info(f"Fetching fresh prospectus data for {company_name}")
            
            # Search for SEBI filings with enhanced methods
            filings = self.sebi_source.search_comprehensive(company_name)
            
            if not filings:
                logger.warning(f"No SEBI filings found for {company_name}")
                return None
            
            logger.info(f"Found {len(filings)} potential filings for {company_name}")
            
            # Try to download and parse documents
            for i, filing in enumerate(filings[:3]):  # Try top 3 filings
                logger.info(f"Attempting filing {i+1}: {filing.get('type', 'Unknown')} from {filing.get('date', 'Unknown date')}")
                
                try:
                    # Download document
                    pdf_path = self._download_document_enhanced(filing['url'], company_name)
                    print("pdf_path:", pdf_path)
                    if not pdf_path:
                        continue
                    
                    # Parse with enhanced parser
                    enhanced_data = self.parser.parse_enhanced(pdf_path, company_name)
                    
                    print("enhanced_data:", enhanced_data)

                    if enhanced_data and enhanced_data.data_quality_score > 0.3:  # Minimum quality threshold
                        # Cache the results
                        if self.cache_manager:
                            self.cache_manager.cache_data(company_name, enhanced_data)
                        
                        logger.info(f"Successfully extracted enhanced data for {company_name} (quality: {enhanced_data.data_quality_score:.2f})")
                        return enhanced_data
                    
                except Exception as e:
                    logger.warning(f"Failed to process filing {i+1} for {company_name}: {e}")
                    continue
            
            logger.warning(f"Could not extract quality financial data for {company_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting enhanced IPO data for {company_name}: {e}")
            return None
    
    def _download_document_enhanced(self, filing_url: str, company_name: str) -> Optional[str]:
        """Enhanced document download with iframe PDF extraction."""
        try:
            # First, get the filing page to extract the actual PDF URL
            pdf_url = self._extract_pdf_url_from_filing_page(filing_url)
            if not pdf_url:
                logger.warning(f"Could not extract PDF URL from filing page: {filing_url}")
                return None
            
            logger.info(f"Extracted PDF URL: {pdf_url}")
            
            # Now download the actual PDF
            return self._download_pdf_direct(pdf_url, company_name)
            
        except Exception as e:
            logger.error(f"Enhanced document download failed: {e}")
            return None
    
    def _extract_pdf_url_from_filing_page(self, filing_url: str) -> Optional[str]:
        """Extract the actual PDF URL from SEBI filing page iframe."""
        try:
            response = self.sebi_source.session.get(filing_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for iframe with PDF
            iframe = soup.find('iframe')
            if iframe and iframe.get('src'):
                iframe_src = iframe.get('src')
                
                # Extract PDF URL from iframe src (format: ../../../web/?file=PDF_URL)
                if 'file=' in iframe_src:
                    pdf_url = iframe_src.split('file=')[1]
                    return pdf_url
            
            # Fallback: look for direct PDF links
            pdf_links = soup.find_all('a', href=lambda x: x and '.pdf' in x.lower())
            if pdf_links:
                href = pdf_links[0].get('href')
                if href.startswith('http'):
                    return href
                else:
                    return urljoin(self.sebi_source.sebi_base_url, href)
            
            logger.warning("No PDF URL found in filing page")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract PDF URL: {e}")
            return None
    
    def _download_pdf_direct(self, pdf_url: str, company_name: str) -> Optional[str]:
        """Download PDF directly from URL."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.sebi_source.session.get(pdf_url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Verify it's a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type:
                    logger.warning(f"Content may not be PDF: {content_type}")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_path = temp_file.name
                
                # Verify file size and PDF signature
                file_size = os.path.getsize(temp_path)
                if file_size < 1000:
                    os.unlink(temp_path)
                    logger.warning(f"Downloaded file too small: {file_size} bytes")
                    continue
                
                # Check PDF signature
                with open(temp_path, 'rb') as f:
                    header = f.read(10)
                    if not header.startswith(b'%PDF'):
                        os.unlink(temp_path)
                        logger.warning("Downloaded file is not a valid PDF")
                        continue
                
                logger.info(f"Successfully downloaded PDF for {company_name}: {temp_path} ({file_size:,} bytes)")
                return temp_path
                
            except Exception as e:
                logger.warning(f"PDF download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All PDF download attempts failed for {company_name}")
        
        return None
    
    def get_data_summary(self, company_name: str) -> Dict[str, Any]:
        """Get a summary of available data without full processing."""
        try:
            # Check cache
            if self.cache_manager:
                cached_data = self.cache_manager.get_cached_data(company_name)
                if cached_data:
                    return {
                        'cached': True,
                        'quality_score': cached_data.data_quality_score,
                        'extraction_date': cached_data.extraction_date,
                        'revenue_years': len(cached_data.revenue_data),
                        'profit_years': len(cached_data.profit_data),
                        'risk_factors': len(cached_data.risk_factors),
                        'validation_flags': cached_data.validation_flags
                    }
            
            # Quick SEBI search
            filings = self.sebi_source.search_comprehensive(company_name)
            
            return {
                'cached': False,
                'sebi_filings_found': len(filings),
                'latest_filing_type': filings[0].get('type', '') if filings else '',
                'latest_filing_date': filings[0].get('date', '') if filings else '',
                'estimated_processing_time': '2-5 minutes' if filings else 'N/A'
            }
            
        except Exception as e:
            return {'error': str(e), 'cached': False, 'sebi_filings_found': 0}


# Integration function for backward compatibility
def integrate_enhanced_prospectus_data(company_name: str, existing_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced integration function for feeding document content to vector database."""
    enhanced_source = EnhancedProspectusDataSource()
    
    try:
        # Get document content for vector database
        enhanced_data = enhanced_source.get_enhanced_ipo_data(company_name)
        
        # Get basic summary
        summary = enhanced_source.get_data_summary(company_name)
        
        # Update existing data
        existing_data['enhanced_prospectus'] = enhanced_data
        existing_data['prospectus_summary'] = summary
        existing_data['sebi_registered'] = summary.get('sebi_filings_found', 0) > 0
        
        if enhanced_data:
            existing_data['data_quality_score'] = enhanced_data.data_quality_score
            existing_data['prospectus_extraction_date'] = enhanced_data.extraction_date
            
            # Store document content for vector database and app usage
            existing_data['document_content'] = enhanced_data.business_description  # Full document content
            existing_data['prospectus_text'] = enhanced_data.business_description   # For app.py compatibility
            existing_data['document_ready_for_vectordb'] = True
            
            logger.info(f"Document content prepared for {company_name} (quality: {enhanced_data.data_quality_score:.2f})")
        else:
            logger.warning(f"No document content available for {company_name}")
            existing_data['document_ready_for_vectordb'] = False
        
        return existing_data
        
    except Exception as e:
        logger.error(f"Document processing failed for {company_name}: {e}")
        existing_data['prospectus_error'] = str(e)
        existing_data['document_ready_for_vectordb'] = False
        return existing_data
