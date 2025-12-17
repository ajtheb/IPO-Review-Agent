# Robust SEBI IPO Document Extraction - Usage Guide

## Overview

This guide covers the enhanced, robust SEBI IPO document extraction system that can handle large-scale extraction with session recovery, retry logic, and graceful error handling.

## New Features

### 1. Session Recovery and Resume Capability
- **Automatic session saving**: Progress is saved every 10 pages
- **Resume interrupted extractions**: Can restart from last processed page
- **Browser crash recovery**: Automatically recovers from browser disconnections
- **Progress persistence**: All data is preserved across sessions

### 2. Robust Error Handling
- **Retry logic**: Multiple attempts for failed operations
- **Graceful degradation**: Continues extraction even with some failures
- **Error logging**: Detailed logging to file and console
- **Statistics tracking**: Comprehensive extraction statistics

### 3. Batch Document Download
- **Rate limiting**: Controlled download speed to prevent server overload
- **Resume downloads**: Skip already downloaded files
- **Batch size management**: Configurable batch sizes
- **Download verification**: File size and content validation

## File Structure

```
examples/
├── robust_sebi_extractor.py     # Main extraction script with session recovery
├── robust_document_downloader.py # Batch document downloader
├── session_manager.py           # Session management utilities
├── test.py                      # Original extraction script (legacy)
└── logs/
    ├── sebi_extractor.log       # Extraction logs
    └── document_downloader.log  # Download logs
```

## Usage Instructions

### 1. Large-Scale Data Extraction

#### First-Time Extraction
```bash
# Run the robust extractor
python examples/robust_sebi_extractor.py

# The script will:
# - Extract all IPO filings from SEBI
# - Handle pagination automatically
# - Save progress every 10 pages
# - Recover from browser crashes
# - Generate comprehensive logs
```

#### Resume Interrupted Extraction
```bash
# If extraction was interrupted, run the same command
python examples/robust_sebi_extractor.py

# The script will automatically detect the previous session and ask:
# "Resume extraction? (y/n/fresh): "
# - y: Resume from last page
# - n: Cancel
# - fresh: Start over (clears session)
```

### 2. Document Download

#### Interactive Download
```bash
# Run the document downloader
python examples/robust_document_downloader.py

# Interactive menu options:
# 1. Download all documents
# 2. Download limited batch (specify number)
# 3. Download by date range
# 4. Download specific company documents
# 5. Show detailed document list
# 6. Exit
```

#### Configuration Options
The downloader can be configured by editing the script:
```python
downloader = RobustDocumentDownloader(
    download_dir='ipo_documents',    # Download directory
    batch_size=3,                    # Documents per batch
    delay_between_downloads=3        # Seconds between downloads
)
```

### 3. Session Management

#### Check Status
```bash
# View current session status
python examples/session_manager.py status
```

#### Generate Report
```bash
# Generate comprehensive progress report
python examples/session_manager.py report
```

#### Other Commands
```bash
# Analyze extraction data
python examples/session_manager.py analyze

# Merge session data to Excel
python examples/session_manager.py merge

# Export failed downloads for retry
python examples/session_manager.py failed

# Clean session files (creates backups)
python examples/session_manager.py clean
```

## Configuration and Customization

### Extraction Configuration

#### Basic Settings
```python
extractor = RobustSEBIExtractor(
    headless=True,           # Run browser in background
    max_retries=3,          # Retry attempts per operation
    session_file='sebi_session.json'  # Session persistence file
)
```

#### Advanced Settings
```python
# In setup_driver() method, modify options:
options.add_argument("--headless=new")      # Visible: remove this line
options.add_argument("--disable-images")    # Speed up: keep this
options.add_argument("--disable-javascript") # Basic only: keep this
```

### Download Configuration

#### Rate Limiting
```python
downloader = RobustDocumentDownloader(
    batch_size=5,                    # Increase for faster downloads
    delay_between_downloads=2,       # Decrease for speed (careful!)
    max_retries=3                    # Retry attempts per file
)
```

#### File Organization
```python
# Custom filename generation
def generate_safe_filename(self, company_name, date, doc_type="IPO"):
    # Customize filename format here
    safe_company = company_name.replace(' ', '_')[:50]
    return f"{safe_company}_{date}_{doc_type}.pdf"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Browser Crashes at High Page Numbers
**Symptoms**: "chrome not reachable" or "session deleted" errors
**Solution**: The robust extractor automatically handles this
```bash
# Check logs for recovery attempts
tail -f sebi_extractor.log

# If persistent issues, reduce batch processing:
# Edit robust_sebi_extractor.py and modify:
# - Increase delay_between_downloads
# - Add more memory management options
```

#### 2. Download Failures
**Symptoms**: PDF downloads fail or produce small files
**Solution**: 
```bash
# Check failed downloads
python examples/session_manager.py failed

# Retry with modified settings
# Edit robust_document_downloader.py:
# - Increase timeout values
# - Add more retry attempts
# - Reduce batch size
```

#### 3. Memory Issues
**Symptoms**: Browser becomes slow or unresponsive
**Solution**: Enhanced memory management is already implemented
```python
# Additional options in robust_sebi_extractor.py:
options.add_argument("--memory-pressure-off")
options.add_argument("--max_old_space_size=4096")
options.add_argument("--disable-images")
options.add_argument("--disable-plugins")
```

### Performance Optimization

#### 1. Speed vs. Reliability Trade-offs
```python
# For maximum reliability (recommended):
delay_between_downloads = 3
batch_size = 3
max_retries = 3

# For faster extraction (higher risk):
delay_between_downloads = 1
batch_size = 10
max_retries = 2
```

#### 2. Resource Management
```bash
# Monitor system resources during extraction
htop  # or Activity Monitor on macOS

# If memory usage is high:
# - Reduce batch_size
# - Increase delays
# - Use headless mode (default)
```

## File Outputs

### Generated Files

#### Data Files
- `sebi_ipo_documents_YYYYMMDD_HHMMSS.xlsx` - Extracted IPO data
- `robust_sebi_ipo_documents.xlsx` - Latest extraction results

#### Session Files
- `sebi_session.json` - Extraction session state
- `download_session.json` - Download session state

#### Log Files
- `sebi_extractor.log` - Extraction operation logs
- `document_downloader.log` - Download operation logs

#### Downloaded Documents
- `ipo_documents/` - Directory containing downloaded PDFs
- Naming format: `Company_Name_Date_IPO.pdf`

### Data Structure

#### Excel File Columns
- `Date` - Filing date
- `Type` - Document type (Public Issues, etc.)
- `Company` - Cleaned company name
- `Title` - Full document title
- `Doc_Link` - Link to SEBI filing page
- `Extracted_At` - Timestamp of extraction

#### Session File Structure
```json
{
  "last_page": 96,
  "extracted_data": [...],
  "total_records_expected": 1500,
  "last_update": "2024-01-15T10:30:00",
  "stats": {
    "pages_processed": 95,
    "errors_encountered": 5,
    "retries_attempted": 12,
    "session_recoveries": 2
  }
}
```

## Best Practices

### 1. Regular Monitoring
```bash
# Check progress during long extractions
python examples/session_manager.py status

# Monitor logs in real-time
tail -f sebi_extractor.log
```

### 2. Backup Management
```bash
# Sessions are automatically backed up
# Manual backup before major operations:
cp sebi_session.json sebi_session_backup.json
cp download_session.json download_session_backup.json
```

### 3. Incremental Extraction
```bash
# For regular updates, run extraction periodically
# The system will automatically continue from where it left off
# New filings will be added to existing data
```

### 4. Resource Planning
- **Disk Space**: Budget ~50-100MB per 1000 documents
- **Time**: Expect 1-2 minutes per page with full extraction
- **Bandwidth**: Consider SEBI server limits, use reasonable delays

## Integration with IPO Analysis

### Using Extracted Data
```python
# Load extracted data for analysis
import pandas as pd
df = pd.read_excel('sebi_ipo_documents_latest.xlsx')

# Filter for specific companies or date ranges
recent_ipos = df[df['Date'].str.contains('2024')]

# Use with IPO Review Agent
from src.analyzers.business_analyzer import BusinessAnalyzer
analyzer = BusinessAnalyzer()

# Analyze using prospectus data
for _, row in recent_ipos.iterrows():
    company = row['Company']
    # Load corresponding PDF and analyze
```

### Prospectus Integration
```python
# Use with enhanced prospectus parser
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusParser

parser = EnhancedProspectusParser()
for pdf_file in Path('ipo_documents').glob('*.pdf'):
    prospectus_data = parser.parse_pdf_prospectus(str(pdf_file))
    # Integrate with IPO analysis workflow
```

This robust system ensures reliable large-scale extraction of SEBI IPO documents while maintaining data integrity and providing comprehensive error recovery mechanisms.
