#!/usr/bin/env python3
"""
Robust SEBI IPO Document Extractor with Session Recovery and Retry Logic

This script implements robust error handling and session recovery for large-scale
extraction of IPO filings from SEBI website, with the ability to resume interrupted
sessions and handle browser crashes gracefully.

Author: IPO Review Agent
Date: 2024
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    WebDriverException, TimeoutException, NoSuchElementException,
    ElementClickInterceptedException, StaleElementReferenceException
)
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Try to use webdriver-manager for automatic ChromeDriver setup
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sebi_extractor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustSEBIExtractor:
    """Robust SEBI IPO document extractor with session recovery."""
    
    def __init__(self, headless=True, max_retries=3, session_file='sebi_session.json'):
        self.headless = headless
        self.max_retries = max_retries
        self.session_file = session_file
        self.driver = None
        self.wait = None
        self.session_data = self.load_session()
        
        # Extraction statistics
        self.stats = {
            'total_extracted': 0,
            'pages_processed': 0,
            'errors_encountered': 0,
            'retries_attempted': 0,
            'session_recoveries': 0
        }
    
    def load_session(self):
        """Load previous session data if available."""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"📁 Loaded session from {self.session_file}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
        
        return {
            'last_page': 0,
            'extracted_data': [],
            'last_update': None,
            'total_records_expected': 0
        }
    
    def save_session(self, current_page=None, data=None):
        """Save current session state."""
        try:
            if current_page is not None:
                self.session_data['last_page'] = current_page
            if data is not None:
                self.session_data['extracted_data'] = data
            
            self.session_data['last_update'] = datetime.now().isoformat()
            self.session_data['stats'] = self.stats
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            logger.info(f"💾 Session saved to {self.session_file}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def setup_driver(self):
        """Initialize Chrome driver with robust options."""
        options = Options()
        
        if self.headless:
            options.add_argument("--headless=new")
        
        # Enhanced stability options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")  # Speed up loading
        options.add_argument("--disable-javascript")  # Basic functionality only
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        # Memory management
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=4096")
        
        # Initialize driver with retries
        for attempt in range(self.max_retries):
            try:
                if WEBDRIVER_MANAGER_AVAILABLE:
                    service = Service(ChromeDriverManager().install())
                    self.driver = webdriver.Chrome(service=service, options=options)
                else:
                    self.driver = webdriver.Chrome(options=options)
                
                self.wait = WebDriverWait(self.driver, 15)
                logger.info("✅ Chrome driver initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5)
                    self.stats['retries_attempted'] += 1
                else:
                    logger.error("❌ Failed to initialize Chrome driver after all retries")
                    return False
        
        return False
    
    def recover_session(self):
        """Recover from browser session crash."""
        logger.warning("🔄 Attempting session recovery...")
        
        try:
            if self.driver:
                self.driver.quit()
        except:
            pass
        
        self.driver = None
        self.wait = None
        
        # Wait before retry
        time.sleep(10)
        
        if self.setup_driver():
            self.stats['session_recoveries'] += 1
            logger.info("✅ Session recovered successfully")
            return True
        else:
            logger.error("❌ Session recovery failed")
            return False
    
    def safe_execute(self, func, *args, **kwargs):
        """Execute function with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (WebDriverException, TimeoutException) as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                self.stats['errors_encountered'] += 1
                
                if attempt < self.max_retries - 1:
                    if "chrome not reachable" in str(e).lower() or "session deleted" in str(e).lower():
                        logger.warning("Browser session lost, attempting recovery...")
                        if self.recover_session():
                            continue
                        else:
                            break
                    else:
                        time.sleep(5)
                        self.stats['retries_attempted'] += 1
                else:
                    logger.error(f"Function failed after {self.max_retries} attempts")
                    raise
        
        return None
    
    def navigate_to_page(self, page_num):
        """Navigate to specific page number."""
        try:
            logger.info(f"📄 Navigating to page {page_num}")
            
            # Look for page number link
            page_links = self.driver.find_elements(By.XPATH, f"//a[text()='{page_num}']")
            
            if page_links:
                self.driver.execute_script("arguments[0].click();", page_links[0])
                time.sleep(3)
                return True
            else:
                logger.warning(f"Page {page_num} link not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to navigate to page {page_num}: {e}")
            return False
    
    def extract_page_data(self):
        """Extract IPO data from current page."""
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            tables = soup.find_all('table')
            
            if not tables:
                logger.warning("No tables found on page")
                return []
            
            main_table = tables[0]
            rows = main_table.find_all('tr')
            
            page_data = []
            for i, row in enumerate(rows[1:], 1):  # Skip header
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 3:
                    date = cols[0].get_text(strip=True) if len(cols) > 0 else ''
                    issue_type = cols[1].get_text(strip=True) if len(cols) > 1 else ''
                    title = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                    
                    # Extract link
                    title_link = cols[2].find('a') if len(cols) > 2 else None
                    doc_link = title_link.get('href') if title_link else None
                    
                    # Only include Public Issues (IPO-related)
                    if 'public issues' in issue_type.lower():
                        company_name = self.clean_company_name(title)
                        
                        page_data.append({
                            'Date': date,
                            'Type': issue_type,
                            'Company': company_name,
                            'Title': title,
                            'Doc_Link': doc_link,
                            'Extracted_At': datetime.now().isoformat()
                        })
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error extracting page data: {e}")
            return []
    
    def clean_company_name(self, title):
        """Clean company name from title."""
        company_name = title.replace('- RHP', '').replace('- DRHP', '').replace('- Prospectus', '').strip()
        company_name = company_name.replace('Addendum to RHP of', '').replace('Addendum to DRHP of', '').strip()
        company_name = company_name.replace('Corrigendum to DRHP of', '').strip()
        return company_name
    
    def extract_all_pages(self, start_page=None, max_pages=200):
        """Extract IPO data from all pages with robust error handling."""
        if not self.setup_driver():
            logger.error("Failed to initialize driver")
            return []
        
        try:
            # Navigate to SEBI page
            url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&cid=1"
            logger.info(f"Loading SEBI Issues & Listing page: {url}")
            
            self.safe_execute(self.driver.get, url)
            time.sleep(5)
            
            # Wait for table to load
            table = self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            logger.info("✅ Page loaded successfully")
            
            # Get pagination info
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            pagination_info = soup.find('div', string=lambda text: text and 'records' in text and 'to' in text)
            
            if pagination_info:
                info_text = pagination_info.get_text(strip=True)
                logger.info(f"Pagination info: {info_text}")
                
                import re
                total_match = re.search(r'of (\d+) records', info_text)
                total_records = int(total_match.group(1)) if total_match else 0
                self.session_data['total_records_expected'] = total_records
                logger.info(f"Total records available: {total_records:,}")
            
            # Determine starting page
            current_page = start_page or self.session_data.get('last_page', 1)
            all_data = self.session_data.get('extracted_data', [])
            
            if current_page > 1:
                logger.info(f"🔄 Resuming from page {current_page} with {len(all_data)} existing records")
            
            # Extract pages
            pages_processed = 0
            consecutive_errors = 0
            
            while current_page <= max_pages and consecutive_errors < 5:
                try:
                    logger.info(f"\n--- Processing Page {current_page} ---")
                    
                    # Navigate to specific page if needed
                    if current_page > 1:
                        nav_success = self.safe_execute(self.navigate_to_page, current_page)
                        if not nav_success:
                            logger.warning(f"Failed to navigate to page {current_page}")
                            consecutive_errors += 1
                            if consecutive_errors >= 3:
                                logger.error("Too many navigation failures, stopping extraction")
                                break
                            current_page += 1
                            continue
                    
                    # Extract data from current page
                    page_data = self.safe_execute(self.extract_page_data)
                    
                    if page_data:
                        all_data.extend(page_data)
                        self.stats['total_extracted'] += len(page_data)
                        logger.info(f"✅ Extracted {len(page_data)} records from page {current_page}")
                        consecutive_errors = 0
                    else:
                        logger.warning(f"No data extracted from page {current_page}")
                        consecutive_errors += 1
                    
                    # Save progress every 10 pages
                    if current_page % 10 == 0 or consecutive_errors > 0:
                        self.save_session(current_page, all_data)
                        logger.info(f"💾 Progress saved: {len(all_data)} total records")
                    
                    # Check for next page
                    try:
                        next_links = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'Next') or contains(text(), '>')]")
                        
                        if not next_links:
                            next_page_num = current_page + 1
                            next_page_links = self.driver.find_elements(By.XPATH, f"//a[text()='{next_page_num}']")
                            if next_page_links:
                                next_links = next_page_links
                        
                        if next_links and next_links[0].is_enabled():
                            logger.info(f"Moving to page {current_page + 1}")
                            self.driver.execute_script("arguments[0].click();", next_links[0])
                            time.sleep(3)
                            current_page += 1
                            pages_processed += 1
                        else:
                            logger.info("No more pages available")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error navigating to next page: {e}")
                        consecutive_errors += 1
                        current_page += 1
                
                except KeyboardInterrupt:
                    logger.info("\n⏹️  Extraction interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error processing page {current_page}: {e}")
                    consecutive_errors += 1
                    current_page += 1
            
            # Final save
            self.save_session(current_page, all_data)
            self.stats['pages_processed'] = pages_processed
            
            logger.info(f"\n📊 Extraction completed:")
            logger.info(f"  Total records: {len(all_data)}")
            logger.info(f"  Pages processed: {pages_processed}")
            logger.info(f"  Errors encountered: {self.stats['errors_encountered']}")
            logger.info(f"  Retries attempted: {self.stats['retries_attempted']}")
            logger.info(f"  Session recoveries: {self.stats['session_recoveries']}")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Critical error in extraction: {e}")
            return self.session_data.get('extracted_data', [])
        
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
    
    def export_data(self, data, filename='robust_sebi_ipo_documents.xlsx'):
        """Export extracted data to Excel."""
        try:
            if data:
                df = pd.DataFrame(data)
                df.to_excel(filename, index=False)
                logger.info(f"✅ Data exported to {filename}")
                return True
            else:
                logger.warning("No data to export")
                return False
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False
    
    def resume_extraction(self):
        """Resume extraction from last saved state."""
        logger.info("🔄 Resuming extraction from last saved state...")
        
        if self.session_data.get('last_page', 0) > 0:
            start_page = self.session_data['last_page']
            existing_data = len(self.session_data.get('extracted_data', []))
            logger.info(f"Resuming from page {start_page} with {existing_data} existing records")
            
            return self.extract_all_pages(start_page=start_page)
        else:
            logger.info("No previous session found, starting fresh extraction")
            return self.extract_all_pages()


def main():
    """Main function with interactive options."""
    print("🔍 Robust SEBI IPO Document Extractor")
    print("=====================================")
    
    extractor = RobustSEBIExtractor(headless=True)
    
    # Check for existing session
    if extractor.session_data.get('last_page', 0) > 0:
        existing_records = len(extractor.session_data.get('extracted_data', []))
        last_page = extractor.session_data['last_page']
        
        print(f"\n📁 Found existing session:")
        print(f"  Last page processed: {last_page}")
        print(f"  Records extracted: {existing_records}")
        print(f"  Last update: {extractor.session_data.get('last_update', 'Unknown')}")
        
        choice = input("\nResume extraction? (y/n/fresh): ").strip().lower()
        
        if choice == 'y':
            data = extractor.resume_extraction()
        elif choice == 'fresh':
            # Clear session and start fresh
            extractor.session_data = {'last_page': 0, 'extracted_data': []}
            data = extractor.extract_all_pages()
        else:
            print("Extraction cancelled")
            return
    else:
        data = extractor.extract_all_pages()
    
    # Export results
    if data:
        filename = f"sebi_ipo_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        extractor.export_data(data, filename)
        
        print(f"\n📊 Final Statistics:")
        print(f"  Total records extracted: {len(data)}")
        print(f"  File saved: {filename}")
        print(f"  Session file: {extractor.session_file}")
    else:
        print("\n❌ No data extracted")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Extractor stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error occurred. Check sebi_extractor.log for details")
