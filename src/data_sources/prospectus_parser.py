"""
IPO Prospectus and DRHP document parser for extracting financial data.
Integrates with SEBI filings and company prospectus documents.
"""

import os
import re
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from loguru import logger
import PyPDF2
import pdfplumber
from bs4 import BeautifulSoup
import tempfile
from dataclasses import dataclass


@dataclass
class ProspectusFinancials:
    """Structured financial data extracted from prospectus."""
    revenue_data: Dict[str, float]  # Year -> Revenue in Crores
    profit_data: Dict[str, float]   # Year -> Profit in Crores
    assets_data: Dict[str, float]   # Year -> Total Assets
    liabilities_data: Dict[str, float]  # Year -> Total Liabilities
    key_ratios: Dict[str, float]    # Financial ratios
    business_description: str
    risk_factors: List[str]
    use_of_funds: List[str]
    company_strengths: List[str]


class SEBIFilingSource:
    """Fetches IPO documents from SEBI and company websites."""
    
    def __init__(self):
        self.sebi_base_url = "https://www.sebi.gov.in"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_sebi_filings(self, company_name: str) -> List[Dict[str, Any]]:
        """Search for IPO filings on SEBI website."""
        try:
            # SEBI's public database search
            search_url = f"{self.sebi_base_url}/sebiweb/other/OtherAction.do?doRecognition=yes&intmId=13"
            
            # Search parameters for IPO filings
            params = {
                'companyName': company_name,
                'segment': 'IPO',
                'fromDate': '01-01-2020',  # Last 5 years
                'toDate': datetime.now().strftime('%d-%m-%Y')
            }
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                filings = self._parse_sebi_search_results(soup, company_name)
                return filings
            else:
                logger.warning(f"SEBI search failed with status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching SEBI filings for {company_name}: {e}")
            return []
    
    def _parse_sebi_search_results(self, soup: BeautifulSoup, company_name: str) -> List[Dict[str, Any]]:
        """Parse SEBI search results to extract document links."""
        filings = []
        
        try:
            # Look for table rows containing filing information
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        # Extract filing information
                        filing_date = cells[0].get_text(strip=True)
                        document_type = cells[1].get_text(strip=True)
                        company = cells[2].get_text(strip=True)
                        
                        # Look for DRHP or Prospectus documents
                        if any(keyword in document_type.upper() for keyword in ['DRHP', 'PROSPECTUS', 'RED HERRING']):
                            if company_name.lower() in company.lower():
                                # Find download link
                                link_element = cells[-1].find('a')
                                if link_element and link_element.get('href'):
                                    filing = {
                                        'date': filing_date,
                                        'type': document_type,
                                        'company': company,
                                        'url': self.sebi_base_url + link_element['href'],
                                        'source': 'SEBI'
                                    }
                                    filings.append(filing)
            
            logger.info(f"Found {len(filings)} SEBI filings for {company_name}")
            return filings
            
        except Exception as e:
            logger.error(f"Error parsing SEBI results: {e}")
            return []
    
    def download_document(self, url: str, company_name: str) -> Optional[str]:
        """Download IPO document and return local file path."""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                logger.info(f"Downloaded prospectus for {company_name}: {temp_path}")
                return temp_path
            else:
                logger.error(f"Failed to download document: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            return None


class ProspectusParser:
    """Parses IPO prospectus documents to extract financial data."""
    
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
            'cash flow', 'ebitda', 'net worth', 'equity', 'debt'
        ]
        
        self.section_keywords = {
            'financials': ['financial statements', 'audited financial', 'profit and loss', 'balance sheet'],
            'business': ['business overview', 'company profile', 'business description'],
            'risks': ['risk factors', 'principal risks', 'key risks'],
            'use_of_funds': ['use of funds', 'objects of the offer', 'fund utilization'],
            'strengths': ['competitive strengths', 'business strengths', 'key strengths']
        }
    
    def parse_prospectus(self, pdf_path: str, company_name: str) -> Optional[ProspectusFinancials]:
        """Parse IPO prospectus PDF to extract structured financial data."""
        try:
            logger.info(f"Parsing prospectus for {company_name}: {pdf_path}")
            
            # Extract text from PDF
            text_content = self._extract_pdf_text(pdf_path)
            if not text_content:
                return None
            
            # Parse different sections
            financials = self._extract_financial_data(text_content)
            business_desc = self._extract_business_description(text_content)
            risk_factors = self._extract_risk_factors(text_content)
            use_of_funds = self._extract_use_of_funds(text_content)
            strengths = self._extract_company_strengths(text_content)
            
            return ProspectusFinancials(
                revenue_data=financials.get('revenue', {}),
                profit_data=financials.get('profit', {}),
                assets_data=financials.get('assets', {}),
                liabilities_data=financials.get('liabilities', {}),
                key_ratios=financials.get('ratios', {}),
                business_description=business_desc,
                risk_factors=risk_factors,
                use_of_funds=use_of_funds,
                company_strengths=strengths
            )
            
        except Exception as e:
            logger.error(f"Error parsing prospectus for {company_name}: {e}")
            return None
        
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text content from PDF using multiple methods."""
        text_content = ""
        
        try:
            # Method 1: Use pdfplumber (better for tables)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:50]:  # Limit to first 50 pages for performance
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            if text_content.strip():
                return text_content
                
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        try:
            # Method 2: Use PyPDF2 as fallback
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(min(50, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
            
            return text_content
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return ""
    
    def _extract_financial_data(self, text: str) -> Dict[str, Dict[str, float]]:
        """Extract financial statements data from prospectus text."""
        financials = {'revenue': {}, 'profit': {}, 'assets': {}, 'liabilities': {}, 'ratios': {}}
        
        # Look for financial tables and statements
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for financial statement sections
            if any(keyword in line_lower for keyword in self.section_keywords['financials']):
                # Extract financial data from next 50 lines
                financial_section = '\n'.join(lines[i:i+50])
                self._parse_financial_section(financial_section, financials)
        
        return financials
    
    def _parse_financial_section(self, section: str, financials: Dict[str, Dict[str, float]]):
        """Parse a financial statement section to extract numbers."""
        lines = section.split('\n')
        
        # Regex patterns for financial data (Indian format)
        amount_pattern = r'[\d,]+\.?\d*'  # Matches numbers with commas
        year_pattern = r'20\d{2}|FY\s*\d{2}'  # Matches years like 2023 or FY23
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Look for revenue patterns
            if any(keyword in line_clean.lower() for keyword in ['total revenue', 'net revenue', 'income from operations']):
                amounts = re.findall(amount_pattern, line_clean)
                years = re.findall(year_pattern, line_clean)
                
                # Match amounts to years
                for i, amount in enumerate(amounts[:3]):  # Last 3 years
                    try:
                        # Convert to float (handle Indian number format)
                        amount_float = float(amount.replace(',', ''))
                        year = f"FY{2021+i}" if i < len(years) else f"FY{2023-i}"
                        financials['revenue'][year] = amount_float
                    except ValueError:
                        continue
            
            # Look for profit patterns
            elif any(keyword in line_clean.lower() for keyword in ['net profit', 'profit after tax', 'net income']):
                amounts = re.findall(amount_pattern, line_clean)
                for i, amount in enumerate(amounts[:3]):
                    try:
                        amount_float = float(amount.replace(',', ''))
                        year = f"FY{2021+i}" if i < 3 else f"FY{2023-i}"
                        financials['profit'][year] = amount_float
                    except ValueError:
                        continue
    
    def _extract_business_description(self, text: str) -> str:
        """Extract business description from prospectus."""
        lines = text.split('\n')
        description = ""
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in self.section_keywords['business']):
                # Extract next 20 lines as business description
                desc_lines = lines[i+1:i+21]
                description = ' '.join([l.strip() for l in desc_lines if l.strip()])
                break
        
        return description[:1000] if description else "Business description not found in prospectus"
    
    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract risk factors from prospectus."""
        risk_factors = []
        lines = text.split('\n')
        
        in_risk_section = False
        for line in lines:
            line_clean = line.strip()
            
            if any(keyword in line_clean.lower() for keyword in self.section_keywords['risks']):
                in_risk_section = True
                continue
            
            if in_risk_section:
                # Stop if we hit another major section
                if any(keyword in line_clean.lower() for keyword in ['use of funds', 'financial statements', 'business overview']):
                    break
                
                # Look for bullet points or numbered risks
                if re.match(r'^\d+\.|\•|\-', line_clean) and len(line_clean) > 20:
                    risk_factors.append(line_clean[:200])  # Limit length
                    
                if len(risk_factors) >= 10:  # Limit to top 10 risks
                    break
        
        return risk_factors
    
    def _extract_use_of_funds(self, text: str) -> List[str]:
        """Extract use of funds information."""
        use_of_funds = []
        lines = text.split('\n')
        
        in_funds_section = False
        for line in lines:
            line_clean = line.strip()
            
            if any(keyword in line_clean.lower() for keyword in self.section_keywords['use_of_funds']):
                in_funds_section = True
                continue
            
            if in_funds_section:
                if any(keyword in line_clean.lower() for keyword in ['risk factors', 'financial statements']):
                    break
                
                if re.match(r'^\d+\.|\•|\-', line_clean) and len(line_clean) > 15:
                    use_of_funds.append(line_clean[:150])
                    
                if len(use_of_funds) >= 5:
                    break
        
        return use_of_funds
    
    def _extract_company_strengths(self, text: str) -> List[str]:
        """Extract company strengths from prospectus."""
        strengths = []
        lines = text.split('\n')
        
        in_strengths_section = False
        for line in lines:
            line_clean = line.strip()
            
            if any(keyword in line_clean.lower() for keyword in self.section_keywords['strengths']):
                in_strengths_section = True
                continue
            
            if in_strengths_section:
                if any(keyword in line_clean.lower() for keyword in ['risk factors', 'use of funds']):
                    break
                
                if re.match(r'^\d+\.|\•|\-', line_clean) and len(line_clean) > 20:
                    strengths.append(line_clean[:150])
                    
                if len(strengths) >= 5:
                    break
        
        return strengths


class ProspectusDataSource:
    """Main class for IPO prospectus data integration."""
    
    def __init__(self):
        self.sebi_source = SEBIFilingSource()
        self.parser = ProspectusParser()
        self._cache = {}  # Simple caching to avoid re-downloading
    
    def get_ipo_financials(self, company_name: str) -> Optional[ProspectusFinancials]:
        """Get financial data from IPO prospectus documents."""
        try:
            # Check cache first
            cache_key = company_name.lower().replace(' ', '_')
            if cache_key in self._cache:
                logger.info(f"Using cached prospectus data for {company_name}")
                return self._cache[cache_key]
            
            logger.info(f"Searching for IPO prospectus: {company_name}")
            
            # Search for SEBI filings
            filings = self.sebi_source.search_sebi_filings(company_name)
            
            if not filings:
                logger.warning(f"No SEBI filings found for {company_name}")
                return None
            
            # Try to download and parse the most recent DRHP
            for filing in filings[:3]:  # Try up to 3 most recent filings
                logger.info(f"Attempting to download: {filing['type']} from {filing['date']}")
                
                pdf_path = self.sebi_source.download_document(filing['url'], company_name)
                if pdf_path:
                    financials = self.parser.parse_prospectus(pdf_path, company_name)
                    if financials:
                        # Cache the results
                        self._cache[cache_key] = financials
                        logger.info(f"Successfully extracted prospectus data for {company_name}")
                        return financials
            
            logger.warning(f"Could not extract financial data from prospectus for {company_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting IPO financials for {company_name}: {e}")
            return None
    
    def get_prospectus_summary(self, company_name: str) -> Dict[str, Any]:
        """Get a summary of prospectus information without full parsing."""
        try:
            filings = self.sebi_source.search_sebi_filings(company_name)
            
            summary = {
                'filings_found': len(filings),
                'latest_filing': None,
                'document_types': [],
                'sebi_registered': len(filings) > 0
            }
            
            if filings:
                latest = filings[0]
                summary['latest_filing'] = {
                    'date': latest['date'],
                    'type': latest['type'],
                    'url': latest['url']
                }
                summary['document_types'] = [f['type'] for f in filings]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting prospectus summary for {company_name}: {e}")
            return {'error': str(e), 'sebi_registered': False}


# Integration function for the main data source manager
def integrate_prospectus_data(company_name: str, existing_data: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate prospectus data with existing IPO data."""
    prospectus_source = ProspectusDataSource()
    
    # Get prospectus financial data
    prospectus_financials = prospectus_source.get_ipo_financials(company_name)
    
    # Get basic prospectus summary
    prospectus_summary = prospectus_source.get_prospectus_summary(company_name)
    
    # Add to existing data
    existing_data['prospectus_financials'] = prospectus_financials
    existing_data['prospectus_summary'] = prospectus_summary
    existing_data['sebi_registered'] = prospectus_summary.get('sebi_registered', False)
    
    if prospectus_financials:
        logger.info(f"Enhanced {company_name} data with prospectus financials")
    else:
        logger.warning(f"No prospectus financial data available for {company_name}")
    
    return existing_data
