from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests
import sys
from pathlib import Path

# Try to use webdriver-manager for automatic ChromeDriver setup
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False
    print("webdriver-manager not installed. Install with: pip install webdriver-manager")

# Setup Chrome options
options = Options()
options.add_argument("--headless=new")  # Remove for visible browser
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-web-security")
options.add_argument("--allow-running-insecure-content")

# Initialize Chrome driver with automatic detection
try:
    if WEBDRIVER_MANAGER_AVAILABLE:
        # Use webdriver-manager for automatic ChromeDriver setup
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Try to use Chrome without specifying driver path (modern Selenium auto-detection)
        driver = webdriver.Chrome(options=options)
except Exception as e:
    print(f"Failed to initialize Chrome driver: {e}")
    print("\nTrying alternative methods...")
    
    # Try common ChromeDriver locations on macOS
    common_paths = [
        "/usr/local/bin/chromedriver",
        "/opt/homebrew/bin/chromedriver",
        str(Path.home() / "Downloads/chromedriver"),
        "/Applications/chromedriver"
    ]
    
    driver = None
    for path in common_paths:
        try:
            if Path(path).exists():
                service = Service(path)
                driver = webdriver.Chrome(service=service, options=options)
                print(f"Successfully using ChromeDriver at: {path}")
                break
        except Exception:
            continue
    
    if driver is None:
        print("❌ Could not initialize Chrome driver!")
        print("📝 Please install ChromeDriver:")
        print("   Option 1: pip install webdriver-manager")
        print("   Option 2: brew install chromedriver")
        print("   Option 3: Download from https://chromedriver.chromium.org/")
        sys.exit(1)

# Use the correct SEBI Issues & Listing endpoint we discovered
url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&cid=1"
print(f"Loading SEBI Issues & Listing page: {url}")
driver.get(url)
wait = WebDriverWait(driver, 15)

try:
    # Wait for the page to load completely
    print("Waiting for page to load...")
    time.sleep(5)
    
    # Wait for table to be present (SEBI doesn't typically have CAPTCHAs on this page)
    table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    print("✅ Table found, extracting IPO data...")
    
    # Get page source and parse with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Find the main data table (should be first table)
    tables = soup.find_all('table')
    if not tables:
        raise Exception("No tables found on page")
    
    main_table = tables[0]
    rows = main_table.find_all('tr')
    print(f"Found table with {len(rows)} rows")
    
    # Extract header
    if rows:
        header_row = rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        print(f"Table headers: {headers}")
    
    # Check pagination info
    pagination_info = soup.find('div', string=lambda text: text and 'records' in text and 'to' in text)
    if pagination_info:
        info_text = pagination_info.get_text(strip=True)
        print(f"Pagination info: {info_text}")
        
        # Extract total records count
        import re
        total_match = re.search(r'of (\d+) records', info_text)
        total_records = int(total_match.group(1)) if total_match else 0
        print(f"Total records available: {total_records:,}")
        
        if total_records > 25:
            print("⚠️  Multiple pages detected! Will extract all pages...")
    
    # Extract IPO data from all pages
    ipo_data = []
    current_page = 1
    max_pages = 100  # Safety limit to prevent infinite loops
    
    while current_page <= max_pages:
        print(f"\n--- Processing Page {current_page} ---")
        
        # Parse current page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tables = soup.find_all('table')
        if not tables:
            break
            
        main_table = tables[0]
        rows = main_table.find_all('tr')
        
        page_data_count = 0
        for i, row in enumerate(rows[1:], 1):  # Skip header row
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 3:  # Ensure we have at least Date, Type, Title
                date = cols[0].get_text(strip=True) if len(cols) > 0 else ''
                issue_type = cols[1].get_text(strip=True) if len(cols) > 1 else ''
                title = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                
                # Extract link from title cell
                title_link = cols[2].find('a') if len(cols) > 2 else None
                doc_link = title_link.get('href') if title_link else None
                
                # Only include Public Issues (IPO-related)
                if 'public issues' in issue_type.lower():
                    company_name = title.replace('- RHP', '').replace('- DRHP', '').replace('- Prospectus', '').strip()
                    company_name = company_name.replace('Addendum to RHP of', '').replace('Addendum to DRHP of', '').replace('Corrigendum to DRHP of', '').strip()
                    
                    ipo_data.append({
                        'Page': current_page,
                        'Date': date,
                        'Type': issue_type,
                        'Company': company_name,
                        'Title': title,
                        'Doc_Link': doc_link
                    })
                    page_data_count += 1
        
        print(f"  Extracted {page_data_count} IPO records from page {current_page}")
        
        # Try to find and click "Next" button
        try:
            # Look for next page link/button
            next_links = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next') or contains(text(), '>') or contains(@class, 'next')]")
            
            if not next_links:
                # Try pagination numbers - look for next page number
                next_page_num = current_page + 1
                next_page_links = driver.find_elements(By.XPATH, f"//a[text()='{next_page_num}']")
                if next_page_links:
                    next_links = next_page_links
            
            if next_links and next_links[0].is_enabled():
                print(f"  Clicking to page {current_page + 1}...")
                driver.execute_script("arguments[0].click();", next_links[0])
                time.sleep(3)  # Wait for page to load
                current_page += 1
            else:
                print("  No more pages found")
                break
                
        except Exception as e:
            print(f"  Error navigating to next page: {e}")
            break
    
    print(f"\n📊 Found {len(ipo_data)} total IPO-related filings across {current_page} pages")
    
    # Save to Excel
    if ipo_data:
        df = pd.DataFrame(ipo_data)
        excel_file = 'sebi_ipo_documents.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"✅ IPO details saved to {excel_file}")
        
        # Try to download a sample PDF document
        sample_entry = None
        for entry in ipo_data:
            if entry['Doc_Link'] and ('rhp' in entry['Title'].lower() or 'prospectus' in entry['Title'].lower()):
                sample_entry = entry
                break
        
        if sample_entry and sample_entry['Doc_Link']:
            print(f"\n📄 Attempting to download sample document: {sample_entry['Title']}")
            
            # Navigate to the filing page to extract PDF URL
            filing_url = 'Vidya '
            if not filing_url.startswith('http'):
                filing_url = 'https://www.sebi.gov.in' + filing_url
            
            print(f"Loading filing page: {filing_url}")
            driver.get(filing_url)
            time.sleep(3)
            
            # Look for iframe with PDF
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            iframe = page_soup.find('iframe')
            
            if iframe and iframe.get('src'):
                iframe_src = iframe.get('src')
                if 'file=' in iframe_src:
                    pdf_url = iframe_src.split('file=')[1]
                    print(f"Found PDF URL: {pdf_url}")
                    
                    # Download the PDF
                    pdf_response = requests.get(pdf_url, timeout=30)
                    if pdf_response.status_code == 200:
                        pdf_filename = f"sample_{sample_entry['Company'].replace(' ', '_')}_DRHP.pdf"
                        with open(pdf_filename, 'wb') as f:
                            f.write(pdf_response.content)
                        print(f"✅ Sample DRHP downloaded: {pdf_filename} ({len(pdf_response.content):,} bytes)")
                    else:
                        print(f"❌ Failed to download PDF: {pdf_response.status_code}")
                else:
                    print("❌ No PDF file parameter found in iframe")
            else:
                print("❌ No iframe found on filing page")
    else:
        print("❌ No IPO data found")
        
except Exception as e:
    print(f"Error: {e}. Check selectors/page changes [web:12]")
finally:
    driver.quit()

def download_ipo_document(driver, entry):
    """Download a specific IPO document."""
    try:
        print(f"\n📄 Downloading: {entry['Title']}")
        print(f"Company: {entry['Company']}")
        print(f"Date: {entry['Date']}")
        
        # Navigate to the filing page to extract PDF URL
        filing_url = entry['Doc_Link']
        if not filing_url.startswith('http'):
            filing_url = 'https://www.sebi.gov.in' + filing_url
        
        print(f"Loading filing page: {filing_url}")
        driver.get(filing_url)
        time.sleep(3)
        
        # Look for iframe with PDF
        page_soup = BeautifulSoup(driver.page_source, 'html.parser')
        iframe = page_soup.find('iframe')
        
        if iframe and iframe.get('src'):
            iframe_src = iframe.get('src')
            if 'file=' in iframe_src:
                pdf_url = iframe_src.split('file=')[1]
                print(f"Found PDF URL: {pdf_url}")
                
                # Download the PDF
                pdf_response = requests.get(pdf_url, timeout=30)
                if pdf_response.status_code == 200:
                    # Create safe filename
                    safe_company = entry['Company'].replace(' ', '_').replace('/', '_').replace('\\', '_')
                    safe_company = ''.join(c for c in safe_company if c.isalnum() or c in ('_', '-'))[:50]
                    pdf_filename = f"{safe_company}_{entry['Date'].replace(' ', '_').replace(',', '')}_IPO.pdf"
                    
                    with open(pdf_filename, 'wb') as f:
                        f.write(pdf_response.content)
                    print(f"✅ Downloaded: {pdf_filename} ({len(pdf_response.content):,} bytes)")
                    return True
                else:
                    print(f"❌ Failed to download PDF: {pdf_response.status_code}")
                    return False
            else:
                print("❌ No PDF file parameter found in iframe")
                return False
        else:
            print("❌ No iframe found on filing page")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading document: {e}")
        return False


def show_document_selection_menu(driver, downloadable_docs):
    """Show interactive menu for document selection."""
    try:
        print(f"\n📋 Available IPO Documents ({len(downloadable_docs)} total):")
        print("-" * 80)
        
        # Group by company for better display
        companies = {}
        for i, doc in enumerate(downloadable_docs):
            company = doc['Company']
            if company not in companies:
                companies[company] = []
            companies[company].append((i, doc))
        
        # Display grouped documents
        doc_index = 1
        display_map = {}
        
        for company, docs in sorted(companies.items()):
            print(f"\n🏢 {company}:")
            for orig_idx, doc in docs:
                doc_type = "RHP" if "rhp" in doc['Title'].lower() else "Prospectus" if "prospectus" in doc['Title'].lower() else "Document"
                print(f"  {doc_index:2d}. [{doc_type}] {doc['Date']} - {doc['Title'][:60]}")
                display_map[doc_index] = orig_idx
                doc_index += 1
        
        print(f"\n📥 Download Options:")
        print(f"Enter document number (1-{len(downloadable_docs)})")
        print(f"Enter 'q' to quit")
        print(f"Enter 'all' to download all documents")
        
        while True:
            choice = input(f"\nSelect document to download: ").strip().lower()
            
            if choice == 'q':
                print("⏭️  Selection cancelled")
                break
            elif choice == 'all':
                print(f"📦 Downloading all {len(downloadable_docs)} documents...")
                batch_download_documents(driver, downloadable_docs, max_downloads=10)
                break
            else:
                try:
                    doc_num = int(choice)
                    if 1 <= doc_num <= len(downloadable_docs):
                        orig_idx = display_map[doc_num]
                        selected_doc = downloadable_docs[orig_idx]
                        success = download_ipo_document(driver, selected_doc)
                        
                        if success:
                            print(f"\n✅ Download completed!")
                            cont = input("Download another document? (y/n): ").strip().lower()
                            if cont != 'y':
                                break
                        else:
                            print(f"\n❌ Download failed. Try another document? (y/n): ")
                            cont = input().strip().lower()
                            if cont != 'y':
                                break
                    else:
                        print(f"❌ Invalid number. Please enter 1-{len(downloadable_docs)}")
                except ValueError:
                    print("❌ Invalid input. Please enter a number or 'q' to quit")
                    
    except KeyboardInterrupt:
        print("\n⏭️  Selection cancelled by user")
    except Exception as e:
        print(f"❌ Error in document selection: {e}")


def batch_download_documents(driver, downloadable_docs, max_downloads=5):
    """Download multiple documents in batch with limits."""
    try:
        total_docs = len(downloadable_docs)
        max_downloads = min(max_downloads, total_docs)
        
        print(f"📦 Batch downloading up to {max_downloads} documents...")
        print(f"⚠️  Note: Limited to {max_downloads} downloads to prevent overload")
        
        downloaded = 0
        failed = 0
        
        for i, doc in enumerate(downloadable_docs[:max_downloads]):
            print(f"\n--- Download {i+1}/{max_downloads} ---")
            success = download_ipo_document(driver, doc)
            
            if success:
                downloaded += 1
            else:
                failed += 1
            
            # Small delay between downloads
            if i < max_downloads - 1:
                print("⏳ Waiting 2 seconds before next download...")
                time.sleep(2)
        
        print(f"\n📊 Batch Download Summary:")
        print(f"✅ Successfully downloaded: {downloaded}")
        print(f"❌ Failed downloads: {failed}")
        print(f"📁 Check current directory for downloaded files")
        
    except Exception as e:
        print(f"❌ Error in batch download: {e}")
