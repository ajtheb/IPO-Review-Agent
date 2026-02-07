"""
LLM-based GMP Extractor

This module uses LLMs to extract GMP data from scraped web content by:
1. Chunking the HTML/text content
2. Finding chunks that mention the company name
3. Using LLM to extract structured GMP data from the relevant chunks

This approach works around JavaScript rendering limitations by using
the LLM's ability to understand unstructured text.
"""

import os
import re
import json
from typing import Dict, Optional, List, Any
from datetime import datetime
from loguru import logger

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Install with: pip install openai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed. Install with: pip install groq")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Install with: pip install requests")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")


class LLMGMPExtractor:
    """Extract GMP data from scraped content using LLMs and Brave Search API."""
    
    def __init__(self, provider: str = "groq", model: str = None, use_brave_search: bool = True):
        """
        Initialize LLM GMP extractor.
        
        Args:
            provider: 'groq', 'gemini', or 'openai'
            model: Model name (defaults to llama-3.3-70b-versatile, gemini-2.0-flash-exp, or gpt-4o-mini)
            use_brave_search: Whether to use Brave Search API for finding GMP data
        """
        self.provider = provider.lower()
        self.use_brave_search = use_brave_search
        
        if self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq not installed. Install with: pip install groq")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self.model = model or "llama-3.3-70b-versatile"
            self.client = Groq(api_key=api_key)
            
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.model = model or "gemini-2.0-flash-exp"
            self.client = genai.GenerativeModel(self.model)
            
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.model = model or "gpt-4o-mini"
            self.client = OpenAI(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'groq', 'gemini', or 'openai'")
        
        # Initialize Brave Search if enabled
        self.brave_api_key = None
        if self.use_brave_search:
            self.brave_api_key = os.getenv("BRAVE_API_KEY")
            if self.brave_api_key:
                logger.info("Brave Search API enabled for GMP search")
            else:
                logger.warning("BRAVE_API_KEY not set, Brave Search disabled")
                self.use_brave_search = False
        
        logger.info(f"Initialized LLM GMP Extractor with {self.provider} ({self.model})")
    
    def search_gmp_with_brave(self, company_name: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for GMP information using Brave Search API.
        
        Args:
            company_name: Name of the company
            max_results: Maximum number of search results to return
            
        Returns:
            List of search results with title, url, and description
        """
        if not self.use_brave_search or not self.brave_api_key:
            logger.warning("Brave Search not available")
            return []
        
        try:
            # Construct search query
            query = f"{company_name} IPO GMP grey market premium today"
            
            # Brave Search API endpoint
            url = "https://api.search.brave.com/res/v1/web/search"
            
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.brave_api_key
            }
            
            params = {
                "q": query,
                "count": max_results,
                "country": "IN",  # Focus on Indian results
                "search_lang": "en",
                "freshness": "pd"  # Past day for latest GMP data
            }
            
            logger.info(f"Searching Brave for: {query}")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract web results
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "age": result.get("age", "")
                    })
            
            logger.info(f"Found {len(results)} search results from Brave")
            return results
            
        except Exception as e:
            logger.error(f"Error searching with Brave API: {e}")
            return []
    
    def scrape_url_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Scrape the HTML content from a URL.
        
        Args:
            url: URL to scrape
            timeout: Request timeout in seconds
            
        Returns:
            HTML content as string, or None if failed
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available for scraping")
            return None
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"Scraping URL: {url}")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            logger.info(f"Successfully scraped {len(response.text)} characters from {url}")
            return response.text
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content using BeautifulSoup.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available, returning raw HTML")
            return self._clean_html(html_content)
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'header', 'footer']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error parsing HTML with BeautifulSoup: {e}")
            return self._clean_html(html_content)
    
    def save_scraped_content(
        self,
        company_name: str,
        url: str,
        html_content: str,
        text_content: str,
        folder: str = "gmp_chunks"
    ) -> None:
        """
        Save scraped HTML and extracted text to files for debugging.
        
        Args:
            company_name: Name of the company
            url: Source URL
            html_content: Raw HTML content
            text_content: Extracted text content
            folder: Folder to save files
        """
        try:
            os.makedirs(folder, exist_ok=True)
            
            # Create safe filename
            safe_name = re.sub(r'[^\w\s-]', '_', company_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = url.split('/')[2] if len(url.split('/')) > 2 else "unknown"
            
            # Save HTML
            html_filename = f"{safe_name}_{domain}_raw_html_{timestamp}.html"
            html_path = os.path.join(folder, html_filename)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Source URL: {url} -->\n")
                f.write(f"<!-- Scraped at: {datetime.now().isoformat()} -->\n\n")
                f.write(html_content)
            
            # Save extracted text
            text_filename = f"{safe_name}_{domain}_extracted_text_{timestamp}.txt"
            text_path = os.path.join(folder, text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Source URL: {url}\n")
                f.write(f"Scraped at: {datetime.now().isoformat()}\n")
                f.write(f"Company: {company_name}\n")
                f.write("="*80 + "\n\n")
                f.write(text_content)
            
            logger.info(f"Saved scraped content to {html_filename} and {text_filename}")
            
        except Exception as e:
            logger.error(f"Error saving scraped content: {e}")

    def extract_gmp_from_brave_results(
        self,
        company_name: str,
        search_results: List[Dict[str, str]],
        scrape_websites: bool = True,
        save_scraped: bool = True,
        chunks_folder: str = "gmp_chunks"
    ) -> Dict[str, Any]:
        """
        Extract GMP data from Brave search results using LLM.
        
        Args:
            company_name: Name of the company
            search_results: List of search results from Brave
            scrape_websites: Whether to scrape website content for more data
            save_scraped: Whether to save scraped HTML and text content
            chunks_folder: Folder path to save chunks (default: "gmp_chunks")
            
        Returns:
            Dictionary containing GMP data or error status
        """
        if not search_results:
            return {
                'company_name': company_name,
                'status': 'not_found',
                'message': 'No search results found'
            }
        
        # Combine search results into context
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):
            context_parts.append(f"Result {i}:")
            context_parts.append(f"Title: {result['title']}")
            context_parts.append(f"URL: {result['url']}")
            context_parts.append(f"Description: {result['description']}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Scrape website content if enabled
        scraped_contexts = []
        if scrape_websites:
            for i, result in enumerate(search_results[:3], 1):  # Scrape top 3 results
                url = result.get("url")
                if url:
                    logger.info(f"Scraping content from: {url}")
                    html_content = self.scrape_url_content(url)
                    if html_content:
                        # Extract text from HTML
                        text_content = self.extract_text_from_html(html_content)
                        
                        # Save scraped content for debugging
                        if save_scraped:
                            self.save_scraped_content(
                                company_name,
                                url,
                                html_content,
                                text_content,
                                chunks_folder
                            )
                        
                        # Add to context with truncation for LLM processing
                        max_length = 8000  # Limit per website to avoid token limits
                        if len(text_content) > max_length:
                            text_content = text_content[:max_length] + "..."
                        
                        scraped_contexts.append(f"\n--- Content from {url} ---\n{text_content}\n")
                        logger.info(f"Added {len(text_content)} characters from {url}")
        
        # Combine search snippets and scraped content
        if scraped_contexts:
            context = context + "\n\n" + "\n".join(scraped_contexts)
            logger.info(f"Total context length with scraped content: {len(context)} characters")
        
        # Use LLM to extract GMP data from combined context
        result = self._extract_with_llm(company_name, context)
        
        # Add metadata about sources
        if result.get('status') != 'error':
            result['scraped_urls'] = [r['url'] for r in search_results[:3]] if scrape_websites else []
            result['search_results_count'] = len(search_results)
        
        return result
    
    def extract_gmp_from_scraped_content(
        self, 
        company_name: str, 
        html_content: str,
        chunk_size: int = 8000,  # Increased from 3000 to capture more context
        print_chunks: bool = False,
        save_chunks: bool = False,
        chunks_folder: str = "gmp_chunks",
        source: str = ""
    ) -> Dict[str, Any]:
        """
        Extract GMP data from scraped HTML content.
        
        Args:
            company_name: Name of the company to search for
            html_content: Raw HTML or text content from the webpage
            chunk_size: Size of text chunks for processing
            print_chunks: Whether to print relevant chunks for debugging
            save_chunks: Whether to save chunks to a folder
            chunks_folder: Folder path to save chunks (default: "gmp_chunks")
            source: Source name for better identification (e.g., "investorgain")
            
        Returns:
            Dictionary containing GMP data or error status
        """
        logger.info(f"Extracting GMP for {company_name} from scraped content")
        
        # Step 1: Clean and chunk the content
        cleaned_text = self._clean_html(html_content)
        chunks = self._chunk_text(cleaned_text, chunk_size)
        
        logger.info(f"Created {len(chunks)} chunks from content")
        
        # Step 2: Find relevant chunks mentioning the company
        relevant_chunks = self._find_relevant_chunks(company_name, chunks)
        
        if not relevant_chunks:
            logger.warning(f"No chunks found mentioning {company_name}")
            return {
                'company_name': company_name,
                'gmp_price': None,
                'gmp_percentage': None,
                'issue_price': None,
                'expected_listing_price': None,
                'estimated_listing_gain': None,
                'last_updated': datetime.now(),
                'source': 'llm_extraction',
                'status': 'not_found',
                'message': f'No content found mentioning {company_name}'
            }
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Save chunks to folder if enabled
        if save_chunks:
            self._save_chunks_to_folder(company_name, relevant_chunks, chunks_folder, source)
        
        # Print relevant chunks for debugging if enabled
        if print_chunks:
            print("\n" + "="*80)
            print(f"📄 RELEVANT CHUNKS FOR: {company_name}")
            print(f"   Found {len(relevant_chunks)} relevant chunks, showing top 5")
            print("="*80)
            for i, chunk in enumerate(relevant_chunks[:5], 1):
                print(f"\n--- Chunk {i} (length: {len(chunk)} chars) ---")
                # Print first 500 chars of each chunk
                preview = chunk[:500] + "..." if len(chunk) > 500 else chunk
                print(preview)
            print("\n" + "="*80 + "\n")
        
        # Step 3: Use LLM to extract GMP data from relevant chunks
        combined_context = "\n\n".join(relevant_chunks[:5])  # Use top 5 chunks
        
        try:
            gmp_data = self._extract_with_llm(company_name, combined_context)
            
            if gmp_data and gmp_data.get('status') == 'success':
                logger.info(f"Successfully extracted GMP data for {company_name}")
                return gmp_data
            else:
                logger.warning(f"LLM could not extract GMP data for {company_name}")
                return {
                    'company_name': company_name,
                    'gmp_price': None,
                    'gmp_percentage': None,
                    'issue_price': None,
                    'expected_listing_price': None,
                    'estimated_listing_gain': None,
                    'last_updated': datetime.now(),
                    'source': 'llm_extraction',
                    'status': 'not_available',
                    'message': 'GMP data could not be extracted from content'
                }
                
        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}")
            return {
                'company_name': company_name,
                'status': 'error',
                'message': f'Error during extraction: {str(e)}',
                'source': 'llm_extraction'
            }
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content to extract readable text."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style tags
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _find_relevant_chunks(self, company_name: str, chunks: List[str]) -> List[str]:
        """Find chunks that mention the company name with expanded context."""
        relevant = []
        
        # Normalize company name for matching
        company_lower = company_name.lower()
        company_keywords = company_lower.split()
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            
            # Check for exact match
            if company_lower in chunk_lower:
                # Add current chunk and potentially adjacent chunks for more context
                if i > 0 and chunks[i-1] not in relevant:
                    relevant.append(chunks[i-1])  # Previous chunk
                relevant.append(chunk)  # Current chunk
                if i < len(chunks) - 1 and chunks[i+1] not in relevant:
                    relevant.append(chunks[i+1])  # Next chunk
                continue
            
            # Check for keyword match (at least 2 keywords for multi-word names)
            if len(company_keywords) > 1:
                matches = sum(1 for kw in company_keywords if kw in chunk_lower)
                if matches >= min(2, len(company_keywords)):
                    # Add context chunks
                    if i > 0 and chunks[i-1] not in relevant:
                        relevant.append(chunks[i-1])
                    relevant.append(chunk)
                    if i < len(chunks) - 1 and chunks[i+1] not in relevant:
                        relevant.append(chunks[i+1])
            elif company_keywords[0] in chunk_lower:
                relevant.append(chunk)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_relevant = []
        for chunk in relevant:
            if chunk not in seen:
                seen.add(chunk)
                unique_relevant.append(chunk)
        
        return unique_relevant
    
    def _extract_with_llm(self, company_name: str, context: str) -> Dict[str, Any]:
        """Use LLM to extract GMP data from context."""
        
        prompt = f"""You are a financial data extraction expert. Extract Grey Market Premium (GMP) information for "{company_name}" from the following text.

Text content:
{context[:8000]}

Extract the following information if available:
1. GMP Price (in ₹ or numeric value)
2. GMP Percentage (%)
3. Issue Price (₹)
4. Expected Listing Price (₹)
5. IPO Status (Open, Upcoming, Closed, Listed)
6. Opening Date
7. Closing Date
8. Listing Date

Return the data in JSON format with these exact keys:
{{
    "company_name": "{company_name}",
    "gmp_price": <number or null>,
    "gmp_percentage": <number or null>,
    "issue_price": <number or null>,
    "expected_listing_price": <number or null>,
    "status": "<success|not_found>",
    "ipo_status": "<Open|Upcoming|Closed|Listed|null>",
    "open_date": "<date or null>",
    "close_date": "<date or null>",
    "listing_date": "<date or null>",
    "confidence": "<high|medium|low>",
    "notes": "<any relevant observations>"
}}

Important:
- Return ONLY the JSON object, no markdown formatting
- Use null for missing values, not "Not found" or "N/A"
- Extract numeric values only (remove ₹, %, commas)
- If GMP data is not found, set status to "not_found"
- If data is partially available, mark what you found
"""

        try:
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                response_text = response.text
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                response_text = response.choices[0].message.content
            
            # Clean response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Add metadata
            data['last_updated'] = datetime.now()
            data['source'] = f'llm_extraction_{self.provider}'
            data['extraction_method'] = 'llm_chunked_analysis'
            
            # Calculate derived fields if possible
            if data.get('gmp_price') and data.get('issue_price'):
                if not data.get('expected_listing_price'):
                    data['expected_listing_price'] = data['gmp_price'] + data['issue_price']
                if not data.get('gmp_percentage'):
                    data['gmp_percentage'] = (data['gmp_price'] / data['issue_price']) * 100
            
            data['estimated_listing_gain'] = data.get('gmp_percentage')
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return {'status': 'error', 'message': 'Failed to parse LLM response'}
        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _save_chunks_to_folder(self, company_name: str, chunks: List[str], folder: str, source: str = ""):
        """Save relevant chunks to a folder as text files."""
        import os
        from datetime import datetime
        
        os.makedirs(folder, exist_ok=True)
        
        # Create timestamp for unique identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize company name for filename
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_company_name = safe_company_name.replace(' ', '_')
        
        for i, chunk in enumerate(chunks, 1):
            # Create descriptive file name
            if source:
                file_name = f"{safe_company_name}_{source}_chunk_{i}_{timestamp}.txt"
            else:
                file_name = f"{safe_company_name}_chunk_{i}_{timestamp}.txt"
            
            file_path = os.path.join(folder, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Company: {company_name}\n")
                f.write(f"Source: {source or 'Unknown'}\n")
                f.write(f"Chunk: {i} of {len(chunks)}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Length: {len(chunk)} characters\n")
                f.write("="*80 + "\n\n")
                f.write(chunk)
            
            logger.info(f"Saved chunk {i} to {file_path}")
        
        print(f"💾 Saved {len(chunks)} chunk(s) to: {folder}/")
    
    def extract_gmp(
        self,
        company_name: str,
        use_brave: bool = True,
        html_content: str = None,
        save_chunks: bool = False,
        print_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        Extract GMP data for a company using multiple methods.
        
        Args:
            company_name: Name of the company
            use_brave: Whether to try Brave Search first
            html_content: Optional HTML content to extract from (fallback)
            save_chunks: Whether to save extracted chunks
            print_chunks: Whether to print chunks for debugging
            
        Returns:
            Dictionary containing GMP data or error status
        """
        logger.info(f"Extracting GMP for {company_name}")
        
        # Try Brave Search first if enabled
        if use_brave and self.use_brave_search:
            print(f"\n🔍 Searching Brave for: {company_name} IPO GMP")
            search_results = self.search_gmp_with_brave(company_name)
            
            if search_results:
                print(f"✅ Found {len(search_results)} search results")
                
                # Print search results if requested
                if print_chunks:
                    print("\n" + "="*80)
                    print(f"🔎 BRAVE SEARCH RESULTS FOR: {company_name}")
                    print("="*80)
                    for i, result in enumerate(search_results[:5], 1):
                        print(f"\n--- Result {i} ---")
                        print(f"Title: {result['title']}")
                        print(f"URL: {result['url']}")
                        print(f"Description: {result['description'][:200]}...")
                    print("\n" + "="*80 + "\n")
                
                # Save search results if requested
                if save_chunks:
                    self._save_search_results(company_name, search_results, "gmp_chunks")
                
                # Extract GMP from search results (now with website scraping)
                print(f"\n🌐 Scraping websites for detailed GMP data...")
                result = self.extract_gmp_from_brave_results(
                    company_name=company_name,
                    search_results=search_results,
                    scrape_websites=True,
                    save_scraped=save_chunks,
                    chunks_folder="gmp_chunks"
                )
                if result.get('status') == 'success':
                    print(f"✅ Successfully extracted GMP from scraped websites")
                    return result
                else:
                    print(f"⚠️  Could not extract GMP from search results or scraped websites")
        
        # Fallback to HTML content extraction if provided
        if html_content:
            print(f"\n📄 Falling back to HTML content extraction")
            return self.extract_gmp_from_scraped_content(
                company_name=company_name,
                html_content=html_content,
                print_chunks=print_chunks,
                save_chunks=save_chunks,
                source="fallback"
            )
        
        # No data found
        return {
            'company_name': company_name,
            'gmp_price': None,
            'gmp_percentage': None,
            'issue_price': None,
            'expected_listing_price': None,
            'estimated_listing_gain': None,
            'last_updated': datetime.now(),
            'source': 'none',
            'status': 'not_found',
            'message': 'GMP data not available'
        }
    
    def _save_search_results(self, company_name: str, results: List[Dict[str, str]], folder: str):
        """Save Brave search results to a file."""
        import os
        from datetime import datetime
        
        os.makedirs(folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_company_name = safe_company_name.replace(' ', '_')
        
        file_name = f"{safe_company_name}_brave_search_{timestamp}.txt"
        file_path = os.path.join(folder, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Company: {company_name}\n")
            f.write(f"Source: Brave Search API\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Results Count: {len(results)}\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Result {i}:\n")
                f.write(f"  Title: {result['title']}\n")
                f.write(f"  URL: {result['url']}\n")
                f.write(f"  Description: {result['description']}\n")
                if result.get('age'):
                    f.write(f"  Age: {result['age']}\n")
                f.write("\n")
        
        print(f"💾 Saved search results to: {file_path}")
        logger.info(f"Saved search results to {file_path}")


def test_llm_gmp_extraction():
    """Test LLM-based GMP extraction with sample content."""
    
    # Sample scraped content (simulating what's in gmp.log)
    sample_content = """
    <div class="container">
        <h1>Live IPO GMP</h1>
        <p>IPO GMP is trending for Kanishk Aluminium India, Msafe Equipments, 
        Accretion Nutraveda, CKK Retail Mart, Hannah Joseph Hospital, 
        Kasturi Metal Composite, Shayona Engineering, NFP Sampoorna Foods, 
        Biopol Chemicals IPOs.</p>
        
        <div class="table">
            <p>Biopol Chemicals NSE SME IPO details:</p>
            <p>Issue Price: ₹85</p>
            <p>GMP: ₹25 (29.41%)</p>
            <p>Expected Listing: ₹110</p>
            <p>Status: Open</p>
        </div>
    </div>
    """
    
    try:
        extractor = LLMGMPExtractor(provider="groq")
        result = extractor.extract_gmp_from_scraped_content(
            company_name="Biopol Chemicals",
            html_content=sample_content
        )
        
        print("\n" + "="*80)
        print("LLM GMP EXTRACTION TEST")
        print("="*80)
        print(f"\nCompany: {result['company_name']}")
        print(f"Status: {result['status']}")
        print(f"GMP Price: ₹{result.get('gmp_price', 'N/A')}")
        print(f"GMP %: {result.get('gmp_percentage', 'N/A')}%")
        print(f"Issue Price: ₹{result.get('issue_price', 'N/A')}")
        print(f"Expected Listing: ₹{result.get('expected_listing_price', 'N/A')}")
        print(f"Source: {result.get('source', 'N/A')}")
        
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']}")
        if 'notes' in result:
            print(f"Notes: {result['notes']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_llm_gmp_extraction()
