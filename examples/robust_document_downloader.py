#!/usr/bin/env python3
"""
Robust IPO Document Downloader with Batch Processing and Resume Capability

This script downloads IPO prospectus documents from SEBI with robust error handling,
batch processing limits, and the ability to resume interrupted downloads.

Author: IPO Review Agent
Date: 2024
"""

import pandas as pd
import requests
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

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
        logging.FileHandler('document_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustDocumentDownloader:
    """Robust IPO document downloader with batch processing."""
    
    def __init__(self, download_dir='downloaded_documents', max_retries=3, 
                 batch_size=5, delay_between_downloads=2):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.delay_between_downloads = delay_between_downloads
        
        self.driver = None
        self.session_file = 'download_session.json'
        self.download_log = self.load_download_log()
        
        # Statistics
        self.stats = {
            'total_attempted': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_existing': 0,
            'bytes_downloaded': 0
        }
    
    def load_download_log(self):
        """Load previous download session."""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load download log: {e}")
        
        return {
            'downloaded_files': [],
            'failed_downloads': [],
            'last_update': None
        }
    
    def save_download_log(self):
        """Save download session state."""
        try:
            self.download_log['last_update'] = datetime.now().isoformat()
            self.download_log['stats'] = self.stats
            
            with open(self.session_file, 'w') as f:
                json.dump(self.download_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save download log: {e}")
    
    def setup_driver(self):
        """Setup Chrome driver for document access."""
        try:
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)
            
            logger.info("✅ Chrome driver initialized for document access")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            return False
    
    def cleanup_driver(self):
        """Clean up Chrome driver."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
    
    def generate_safe_filename(self, company_name, date, doc_type="IPO"):
        """Generate safe filename for document."""
        # Clean company name
        safe_company = ''.join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_company = safe_company.replace(' ', '_')[:50]  # Limit length
        
        # Clean date
        safe_date = date.replace(' ', '_').replace(',', '').replace('/', '_')
        
        # Generate filename
        filename = f"{safe_company}_{safe_date}_{doc_type}.pdf"
        return filename
    
    def is_already_downloaded(self, filename):
        """Check if file was already downloaded."""
        file_path = self.download_dir / filename
        
        # Check if file exists and in download log
        if file_path.exists() and filename in self.download_log.get('downloaded_files', []):
            return True
        
        return False
    
    def extract_pdf_url_from_filing_page(self, filing_url):
        """Extract PDF URL from SEBI filing page."""
        try:
            if not self.driver:
                if not self.setup_driver():
                    return None
            
            # Ensure full URL
            if not filing_url.startswith('http'):
                filing_url = 'https://www.sebi.gov.in' + filing_url
            
            logger.info(f"Loading filing page: {filing_url}")
            self.driver.get(filing_url)
            time.sleep(3)
            
            # Parse page to find PDF iframe
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            iframe = soup.find('iframe')
            
            if iframe and iframe.get('src'):
                iframe_src = iframe.get('src')
                if 'file=' in iframe_src:
                    pdf_url = iframe_src.split('file=')[1]
                    return pdf_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting PDF URL: {e}")
            return None
    
    def download_single_document(self, document_info):
        """Download a single IPO document with retries."""
        company = document_info.get('Company', 'Unknown')
        date = document_info.get('Date', 'Unknown')
        title = document_info.get('Title', 'Unknown')
        doc_link = document_info.get('Doc_Link', '')
        
        # Generate filename
        filename = self.generate_safe_filename(company, date)
        file_path = self.download_dir / filename
        
        # Check if already downloaded
        if self.is_already_downloaded(filename):
            logger.info(f"⏭️  Skipping {company} - already downloaded")
            self.stats['skipped_existing'] += 1
            return True, f"Already downloaded: {filename}"
        
        logger.info(f"📄 Downloading: {company}")
        logger.info(f"   Date: {date}")
        logger.info(f"   Title: {title[:80]}...")
        
        # Attempt download with retries
        for attempt in range(self.max_retries):
            try:
                # Extract PDF URL from filing page
                pdf_url = self.extract_pdf_url_from_filing_page(doc_link)
                
                if not pdf_url:
                    logger.warning(f"Could not extract PDF URL for {company}")
                    continue
                
                # Download PDF
                logger.info(f"Downloading PDF from: {pdf_url}")
                
                response = requests.get(pdf_url, timeout=60, stream=True)
                response.raise_for_status()
                
                # Check if response is actually a PDF
                content_type = response.headers.get('content-type', '')
                if 'pdf' not in content_type.lower():
                    logger.warning(f"Response may not be PDF (Content-Type: {content_type})")
                
                # Save file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = file_path.stat().st_size
                
                # Validate file size (should be reasonably large for PDF)
                if file_size < 1024:  # Less than 1KB is suspicious
                    logger.warning(f"Downloaded file is very small ({file_size} bytes)")
                    file_path.unlink()  # Delete suspicious file
                    continue
                
                # Success
                self.stats['successful_downloads'] += 1
                self.stats['bytes_downloaded'] += file_size
                
                # Update download log
                self.download_log['downloaded_files'].append(filename)
                self.save_download_log()
                
                logger.info(f"✅ Downloaded: {filename} ({file_size:,} bytes)")
                return True, f"Downloaded: {filename} ({file_size:,} bytes)"
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed for {company}: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    # Final failure
                    self.stats['failed_downloads'] += 1
                    
                    # Log failed download
                    failure_info = {
                        'company': company,
                        'date': date,
                        'title': title,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.download_log['failed_downloads'].append(failure_info)
                    self.save_download_log()
                    
                    return False, f"Failed after {self.max_retries} attempts: {e}"
        
        return False, "Download failed"
    
    def batch_download_documents(self, documents, max_downloads=None):
        """Download documents in batches with limits."""
        if max_downloads:
            documents = documents[:max_downloads]
        
        total_docs = len(documents)
        logger.info(f"📦 Starting batch download of {total_docs} documents")
        logger.info(f"   Download directory: {self.download_dir}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Delay between downloads: {self.delay_between_downloads}s")
        
        try:
            for i, doc in enumerate(documents, 1):
                logger.info(f"\n--- Download {i}/{total_docs} ---")
                self.stats['total_attempted'] += 1
                
                success, message = self.download_single_document(doc)
                
                if success:
                    logger.info(f"✅ {message}")
                else:
                    logger.error(f"❌ {message}")
                
                # Progress update
                if i % 5 == 0:
                    self.print_progress_summary(i, total_docs)
                
                # Delay between downloads (except for last one)
                if i < total_docs and success:
                    time.sleep(self.delay_between_downloads)
                
                # Batch checkpoint save
                if i % self.batch_size == 0:
                    self.save_download_log()
                    logger.info(f"💾 Progress saved at document {i}")
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Download interrupted by user")
        
        finally:
            self.cleanup_driver()
            self.save_download_log()
            self.print_final_summary()
    
    def print_progress_summary(self, current, total):
        """Print progress summary."""
        progress = (current / total) * 100
        logger.info(f"📊 Progress: {current}/{total} ({progress:.1f}%)")
        logger.info(f"   ✅ Successful: {self.stats['successful_downloads']}")
        logger.info(f"   ❌ Failed: {self.stats['failed_downloads']}")
        logger.info(f"   ⏭️  Skipped: {self.stats['skipped_existing']}")
    
    def print_final_summary(self):
        """Print final download summary."""
        logger.info(f"\n📊 Download Summary:")
        logger.info(f"   Total attempted: {self.stats['total_attempted']}")
        logger.info(f"   Successful downloads: {self.stats['successful_downloads']}")
        logger.info(f"   Failed downloads: {self.stats['failed_downloads']}")
        logger.info(f"   Skipped existing: {self.stats['skipped_existing']}")
        logger.info(f"   Total bytes downloaded: {self.stats['bytes_downloaded']:,}")
        logger.info(f"   Download directory: {self.download_dir}")
    
    def filter_downloadable_documents(self, df):
        """Filter DataFrame for downloadable IPO documents."""
        # Filter for documents with links and IPO-related content
        downloadable = df[
            (df['Doc_Link'].notna()) & 
            (df['Doc_Link'] != '') &
            (df['Type'].str.contains('public issues', case=False, na=False)) &
            (df['Title'].str.contains('RHP|DRHP|Prospectus', case=False, na=False))
        ].copy()
        
        # Remove duplicates based on company and date
        downloadable = downloadable.drop_duplicates(subset=['Company', 'Date'])
        
        logger.info(f"📋 Found {len(downloadable)} downloadable documents")
        return downloadable
    
    def interactive_download_menu(self, df):
        """Interactive menu for document selection and download."""
        downloadable_docs = self.filter_downloadable_documents(df)
        
        if downloadable_docs.empty:
            logger.warning("No downloadable documents found")
            return
        
        print(f"\n📋 Interactive Document Download Menu")
        print(f"Found {len(downloadable_docs)} downloadable IPO documents")
        print("-" * 60)
        
        # Show company summary
        companies = downloadable_docs['Company'].value_counts()
        print(f"\nTop companies with documents:")
        for company, count in companies.head(10).items():
            print(f"  • {company}: {count} documents")
        
        print(f"\n📥 Download Options:")
        print(f"1. Download all documents ({len(downloadable_docs)})")
        print(f"2. Download limited batch (specify number)")
        print(f"3. Download by date range")
        print(f"4. Download specific company documents")
        print(f"5. Show detailed document list")
        print(f"6. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    confirm = input(f"Download ALL {len(downloadable_docs)} documents? (y/n): ")
                    if confirm.lower() == 'y':
                        docs_to_download = downloadable_docs.to_dict('records')
                        self.batch_download_documents(docs_to_download)
                    break
                
                elif choice == '2':
                    try:
                        limit = int(input("Enter number of documents to download: "))
                        limit = min(limit, len(downloadable_docs))
                        docs_to_download = downloadable_docs.head(limit).to_dict('records')
                        self.batch_download_documents(docs_to_download)
                    except ValueError:
                        print("❌ Invalid number")
                        continue
                    break
                
                elif choice == '3':
                    print("Date range download not implemented yet")
                    continue
                
                elif choice == '4':
                    company_name = input("Enter company name (partial match): ").strip()
                    company_docs = downloadable_docs[
                        downloadable_docs['Company'].str.contains(company_name, case=False, na=False)
                    ]
                    
                    if company_docs.empty:
                        print(f"❌ No documents found for '{company_name}'")
                        continue
                    
                    print(f"Found {len(company_docs)} documents for companies matching '{company_name}'")
                    docs_to_download = company_docs.to_dict('records')
                    self.batch_download_documents(docs_to_download)
                    break
                
                elif choice == '5':
                    self.show_document_details(downloadable_docs)
                    continue
                
                elif choice == '6':
                    print("Download cancelled")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-6")
            
            except KeyboardInterrupt:
                print("\n⏹️  Selection cancelled")
                break
    
    def show_document_details(self, df, max_show=20):
        """Show detailed document list."""
        print(f"\n📋 Document Details (showing first {max_show}):")
        print("-" * 100)
        
        for i, row in df.head(max_show).iterrows():
            print(f"{i+1:2d}. {row['Company']}")
            print(f"     Date: {row['Date']}")
            print(f"     Type: {row['Type']}")
            print(f"     Title: {row['Title'][:70]}...")
            print()


def main():
    """Main function for document downloading."""
    print("📥 Robust IPO Document Downloader")
    print("=================================")
    
    # Load extracted data
    excel_files = list(Path('.').glob('*sebi_ipo_documents*.xlsx'))
    
    if not excel_files:
        print("❌ No SEBI IPO data files found!")
        print("   Please run the extractor first to generate data files")
        return
    
    # Use most recent file
    latest_file = max(excel_files, key=os.path.getctime)
    print(f"📁 Loading data from: {latest_file}")
    
    try:
        df = pd.read_excel(latest_file)
        print(f"✅ Loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize downloader
    downloader = RobustDocumentDownloader(
        download_dir='ipo_documents',
        batch_size=3,  # Conservative batch size
        delay_between_downloads=3  # 3 second delay
    )
    
    # Start interactive menu
    downloader.interactive_download_menu(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Downloader stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error occurred. Check document_downloader.log for details")
