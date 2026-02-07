# Brave Search + Website Scraping Integration

## Overview

The IPO Review Agent now **scrapes actual website content** from Brave Search results to extract Grey Market Premium (GMP) data. This provides more accurate and comprehensive data than relying solely on search result descriptions.

## How It Works

### 1. **Brave Search API**
- Searches for: `"{company_name} IPO GMP grey market premium today"`
- Returns top 5 search results with titles, URLs, and descriptions
- Focuses on Indian results (`country: IN`) with fresh data (`freshness: pd` = past day)

### 2. **Website Scraping**
- **Scrapes top 3 URLs** from search results
- Uses `requests` library with proper user agent headers
- Extracts clean text from HTML using `BeautifulSoup4`
- Removes scripts, styles, navigation, headers, and footers
- Saves both raw HTML and extracted text for debugging

### 3. **LLM Extraction**
- Combines search snippets + scraped website content
- Uses Groq API (llama-3.3-70b-versatile) to extract structured GMP data
- Truncates content to avoid token limits (8000 chars per website)
- Returns JSON with GMP price, percentage, dates, and confidence

## Saved Data for Debugging

All data is saved to `gmp_chunks/` folder:

### Search Results
```
{company}_brave_search_{timestamp}.txt
```
Contains JSON with search results, URLs, and descriptions.

### Raw HTML
```
{company}_{domain}_raw_html_{timestamp}.html
```
Complete HTML source from scraped websites with metadata comments.

### Extracted Text
```
{company}_{domain}_extracted_text_{timestamp}.txt
```
Clean text content extracted from HTML with source URL and timestamp.

## Code Example

```python
from src.data_sources.llm_gmp_extractor import LLMGMPExtractor

# Initialize with Groq + Brave Search
extractor = LLMGMPExtractor(
    provider="groq",
    use_brave_search=True
)

# Extract GMP with website scraping
result = extractor.extract_gmp(
    company_name="Biopol Chemicals",
    use_brave=True,
    save_chunks=True,  # Save all scraped content
    print_chunks=True  # Print search results
)

# Result includes:
# - gmp_price, gmp_percentage, issue_price
# - scraped_urls: list of URLs that were scraped
# - search_results_count: number of Brave results
```

## Key Methods

### `scrape_url_content(url, timeout=10)`
- Scrapes HTML from a URL
- Uses proper user agent to avoid blocking
- Returns HTML string or None on error

### `extract_text_from_html(html_content)`
- Parses HTML with BeautifulSoup
- Removes scripts, styles, nav, header, footer
- Returns cleaned text content

### `save_scraped_content(company_name, url, html_content, text_content, folder)`
- Saves raw HTML and extracted text to files
- Creates organized filenames with timestamp and domain
- Adds metadata comments with source URL and timestamp

### `extract_gmp_from_brave_results(company_name, search_results, scrape_websites=True, save_scraped=True)`
- **NEW:** Now scrapes websites if `scrape_websites=True`
- Combines search snippets + scraped content
- Uses LLM to extract GMP data
- Returns structured result with metadata

## Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key
BRAVE_API_KEY=your_brave_api_key
```

### Parameters
- `scrape_websites=True`: Enable website scraping (default: True)
- `save_scraped=True`: Save HTML and text files (default: True)
- `chunks_folder="gmp_chunks"`: Where to save files
- `timeout=10`: Request timeout in seconds
- `max_length=8000`: Max characters per website for LLM

## Advantages

### Before (Search Snippets Only)
- ❌ Limited to ~200 character descriptions
- ❌ Often missing actual GMP values
- ❌ No access to full page content

### After (With Website Scraping)
- ✅ Full access to website content
- ✅ Captures GMP tables, lists, and formatted data
- ✅ More context for accurate extraction
- ✅ Saved HTML/text for debugging and verification

## Common GMP Websites

The system automatically finds and scrapes these popular Indian GMP sources:

1. **ChanakayaNiPothi** (`chanakyanipothi.com`)
   - Comprehensive GMP tables
   - Daily updates
   - Historical data

2. **IPO Watch** (`ipowatch.in`)
   - Individual IPO pages
   - GMP with commentary
   - Market sentiment

3. **IPO Central** (`ipocentral.in`)
   - Real-time GMP tracking
   - Performance analysis

4. **Invest or Gain** (`investorgain.com`)
   - GMP lists
   - Subscription data

## Testing

### Run Complete Test
```bash
python examples/test_brave_scraping.py
```

This will:
1. Search Brave for multiple companies
2. Scrape top 3 websites for each
3. Extract GMP using Groq LLM
4. Save all data to `gmp_chunks/`
5. Print results with metadata

### Test Single Company
```python
from src.data_sources.llm_gmp_extractor import LLMGMPExtractor

extractor = LLMGMPExtractor(provider="groq", use_brave_search=True)

result = extractor.extract_gmp(
    company_name="Fractal Analytics",
    use_brave=True,
    save_chunks=True,
    print_chunks=True
)

print(f"GMP: ₹{result['gmp_price']} ({result['gmp_percentage']}%)")
print(f"Scraped {len(result['scraped_urls'])} websites")
```

## Error Handling

### Scraping Failures
- Continues if a URL fails to scrape
- Logs error but doesn't fail entire extraction
- Falls back to remaining URLs

### Parsing Failures
- Falls back to simple HTML cleaning if BeautifulSoup fails
- Logs warning but continues

### Rate Limiting
- Uses 10-second timeout per request
- Scrapes only top 3 results to avoid overwhelming servers
- Proper user agent to appear as legitimate browser

## Future Improvements

1. **JavaScript Rendering**
   - Use Selenium/Playwright for JS-heavy sites
   - Handle dynamic content loading

2. **Parallel Scraping**
   - Speed up by scraping URLs concurrently
   - Use ThreadPoolExecutor

3. **Caching**
   - Cache scraped content for 1 hour
   - Avoid re-scraping same URLs

4. **Smart URL Selection**
   - Prioritize known reliable GMP sources
   - Skip news/promotional sites

5. **Extraction Validation**
   - Cross-verify GMP from multiple sources
   - Flag inconsistencies

## Troubleshooting

### No HTML Saved
**Issue:** `gmp_chunks/` folder is empty

**Solution:**
- Ensure `save_chunks=True` in `extract_gmp()`
- Check file permissions on `gmp_chunks/` folder
- Verify scraping didn't fail (check logs)

### Empty Extracted Text
**Issue:** Text file exists but is empty

**Solution:**
- Website might be JavaScript-rendered
- Check raw HTML file for actual content
- Consider using Selenium for that site

### GMP Not Found Despite Scraping
**Issue:** Websites scraped but GMP not extracted

**Solution:**
- Check extracted text files manually
- GMP might be in unexpected format
- Improve LLM prompt for that specific format
- Use `print_chunks=True` to debug

## Performance

- **Search:** ~500ms (Brave API)
- **Scraping:** ~1-3s per URL (depends on website)
- **LLM Extraction:** ~2-5s (Groq API)
- **Total:** ~10-15s for 3 URLs

## Conclusion

Website scraping dramatically improves GMP extraction accuracy by providing full page content to the LLM instead of short search snippets. All data is saved for transparency and debugging, making the system reliable and auditable.
