"""
LLM-Powered Prospectus Analyzer for Advanced IPO Analysis

This module uses Large Language Models to extract and analyze complex financial data,
P/E ratios, benchmarking information, and other critical details from IPO prospectus documents.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import openai
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime

# For Brave Search API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - Brave Search disabled")

# Alternative LLM providers
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def safe_eval(expr: str) -> float:
    """
    Safely evaluate a simple arithmetic expression.
    
    Args:
        expr: String containing arithmetic expression (e.g., "(256.93 / 11884.89) * 100")
    
    Returns:
        Evaluated numerical result
    
    Raises:
        ValueError: If expression contains unsafe characters
    """
    # Only allow numbers, operators, spaces, and parentheses
    if not re.match(r'^[\d.\s+\-*/()]+$', expr):
        raise ValueError(f"Unsafe expression: {expr}")
    
    try:
        # Evaluate with restricted builtins for safety
        result = eval(expr, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        logger.warning(f"Failed to evaluate expression '{expr}': {e}")
        raise


def evaluate_expressions_in_dict(data: dict) -> dict:
    """
    Recursively evaluate string arithmetic expressions in a dictionary.
    
    This allows the LLM to return expressions like "(256.93 / 11884.89) * 100"
    which will be evaluated to numerical values.
    
    Args:
        data: Dictionary potentially containing string expressions
    
    Returns:
        Dictionary with expressions evaluated to float values
    """
    if not isinstance(data, dict):
        return data
    
    for k, v in data.items():
        if isinstance(v, str):
            v_strip = v.strip()
            # Detect if it's an arithmetic expression (numbers and operators only)
            if re.match(r'^[\d.\s+\-*/()]+$', v_strip):
                try:
                    # Evaluate and round to 2 decimal places
                    evaluated_value = safe_eval(v_strip)
                    data[k] = round(evaluated_value, 2)
                    logger.debug(f"Evaluated expression '{v_strip}' = {data[k]}")
                except Exception as e:
                    logger.warning(f"Could not evaluate expression '{v_strip}': {e}")
                    data[k] = None
            # Keep non-expression strings as-is
        elif isinstance(v, dict):
            # Recursively process nested dictionaries
            data[k] = evaluate_expressions_in_dict(v)
        elif isinstance(v, list):
            # Process lists that might contain dicts
            data[k] = [evaluate_expressions_in_dict(item) if isinstance(item, dict) else item for item in v]
    
    return data


@dataclass
class LLMFinancialMetrics:
    """Advanced financial metrics extracted by LLM."""
    # Valuation Metrics
    trailing_pe_ratio: Optional[float] = None
    forward_pe_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    price_to_sales_ratio: Optional[float] = None
    ev_to_ebitda_ratio: Optional[float] = None
    price_to_earnings_growth_ratio: Optional[float] = None
    
    # Profitability Ratios
    gross_profit_margin: Optional[float] = None
    operating_profit_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    
    # Liquidity Ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    
    # Leverage Ratios
    debt_to_equity_ratio: Optional[float] = None
    debt_to_assets_ratio: Optional[float] = None
    interest_coverage_ratio: Optional[float] = None
    
    # Efficiency Ratios
    asset_turnover_ratio: Optional[float] = None
    inventory_turnover_ratio: Optional[float] = None
    receivables_turnover_ratio: Optional[float] = None
    
    # Growth Metrics
    revenue_growth_3yr: Optional[float] = None
    profit_growth_3yr: Optional[float] = None
    ebitda_growth_3yr: Optional[float] = None
    
    # Cash Flow Metrics
    operating_cash_flow: Optional[Dict[str, float]] = None
    free_cash_flow: Optional[Dict[str, float]] = None
    cash_flow_to_debt_ratio: Optional[float] = None
    
    # Quality Scores
    extraction_confidence: Optional[float] = None
    data_completeness: Optional[float] = None


@dataclass
class BenchmarkingAnalysis:
    """Benchmarking analysis results."""
    sector_comparison: Dict[str, Any] = field(default_factory=dict)
    peer_companies: List[Dict[str, Any]] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)
    competitive_disadvantages: List[str] = field(default_factory=list)
    industry_trends: List[str] = field(default_factory=list)
    market_position: str = "unknown"
    market_share_analysis: Optional[str] = None
    overall_competitive_score: float = 0.0


@dataclass
class IPOSpecificMetrics:
    """IPO-specific metrics and analysis."""
    ipo_pricing_analysis: Dict[str, Any]
    underwriter_quality: Dict[str, Any]
    use_of_funds_analysis: Dict[str, Any]
    lock_in_analysis: Dict[str, Any]
    promoter_background: Dict[str, Any]
    business_model_assessment: Dict[str, Any]
    growth_strategy_analysis: Dict[str, Any]
    regulatory_compliance: Dict[str, Any]


class LLMProspectusAnalyzer:
    """Advanced LLM-powered prospectus analyzer with vector DB support."""
    
    def __init__(self, provider: str = "openai", use_vector_db: bool = True, db_path: str = "./ipo_vector_db"):
        """Initialize with preferred LLM provider and optional vector DB."""
        self.provider = provider
        self.client = None
        self.use_vector_db = use_vector_db and CHROMA_AVAILABLE
        self.vector_client = None
        self.collections = {}
        
        # Initialize vector database if available and requested
        if self.use_vector_db:
            try:
                self.vector_client = chromadb.PersistentClient(path=db_path)
                self._setup_vector_collections()
                logger.info("Vector database initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize vector database: {e}")
                self.use_vector_db = False
        
        # Initialize the appropriate client
        if provider == "openai":
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                openai.api_key = openai_key
                self.client = openai
            else:
                logger.warning("OpenAI API key not found")
                
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                self.client = anthropic.Anthropic(api_key=anthropic_key)
            else:
                logger.warning("Anthropic API key not found")
                
        elif provider == "groq" and GROQ_AVAILABLE:
            groq_key = os.getenv('GROQ_API_KEY')
            if groq_key:
                self.client = Groq(api_key=groq_key)
                # Store available models for fallback
                self.groq_models = [
                    "llama-3.1-8b-instant",      # Primary choice
                    "llama3-8b-8192",       # Alternative option
                ]
                logger.info("Groq client initialized with model fallback support")
            else:
                logger.warning("Groq API key not found")
                
        elif provider == "gemini" and GEMINI_AVAILABLE:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                genai.configure(api_key=gemini_key)
                # Use the correct model name for Gemini
                try:
                    # Try the latest models first
                    self.client = genai.GenerativeModel('gemini-2.5-flash')
                    logger.info("Gemini 2.5 Flash model initialized successfully")
                except Exception:
                    try:
                        # Fallback to Gemini Pro
                        self.client = genai.GenerativeModel('gemini-2.5-pro')
                        logger.info("Gemini Pro model initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize Gemini model: {e}")
                        self.client = None
            else:
                logger.warning("Gemini API key not found")
        
        if not self.client:
            logger.error(f"Failed to initialize {provider} client")
    
    def search_brave_for_ipo_context(self, company_name: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for IPO information using Brave Search API.
        
        Args:
            company_name: Name of the company
            max_results: Maximum number of search results to return
            
        Returns:
            List of search results with title, url, and description
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available - Brave Search disabled")
            return []
        
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            logger.warning("BRAVE_API_KEY not set - Brave Search disabled")
            return []
        
        try:
            # Construct search query for general IPO analysis
            query = f"{company_name} IPO analysis business model financials competitors market position"
            
            # Brave Search API endpoint
            url = "https://api.search.brave.com/res/v1/web/search"
            
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": brave_api_key
            }
            
            params = {
                "q": query,
                "count": max_results,
                "country": "IN",  # Focus on Indian results
                "search_lang": "en",
                "freshness": "pw"  # Past week for recent analysis
            }
            
            logger.info(f"Searching Brave for IPO context: {query}")
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
            
            logger.info(f"Found {len(results)} search results from Brave for IPO context")
            return results
            
        except Exception as e:
            logger.error(f"Error searching with Brave API: {e}")
            return []
    
    def _format_brave_context(self, search_results: List[Dict[str, str]]) -> str:
        """
        Format Brave search results into a readable context string.
        
        Args:
            search_results: List of search results from Brave API
            
        Returns:
            Formatted context string for LLM
        """
        if not search_results:
            return ""
        
        context_parts = [
            "=== WEB SEARCH CONTEXT (from Brave Search API) ===\n",
            f"Found {len(search_results)} relevant articles/sources:\n"
        ]
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"\n--- Source {i} ---")
            context_parts.append(f"Title: {result.get('title', 'N/A')}")
            context_parts.append(f"URL: {result.get('url', 'N/A')}")
            context_parts.append(f"Description: {result.get('description', 'N/A')}")
            if result.get('age'):
                context_parts.append(f"Published: {result.get('age')}")
        
        context_parts.append("\n=== END WEB SEARCH CONTEXT ===\n")
        return "\n".join(context_parts)

    def _setup_vector_collections(self):
        """Set up vector database collections for different types of content."""
        try:
            # Collection for prospectus document chunks
            self.collections['prospectus_chunks'] = self.vector_client.get_or_create_collection(
                name="prospectus_chunks",
                metadata={"description": "IPO prospectus document chunks for context retrieval"}
            )
            
            # Collection for financial sections specifically
            self.collections['financial_sections'] = self.vector_client.get_or_create_collection(
                name="financial_sections", 
                metadata={"description": "Financial data sections from IPO prospectus"}
            )
            
            # Collection for competitive analysis sections
            self.collections['competitive_sections'] = self.vector_client.get_or_create_collection(
                name="competitive_sections",
                metadata={"description": "Business model and competitive analysis sections"}
            )
            
            # Collection for IPO-specific sections
            self.collections['ipo_sections'] = self.vector_client.get_or_create_collection(
                name="ipo_sections",
                metadata={"description": "IPO pricing, underwriters, and fund usage sections"}
            )
            
            logger.info(f"Set up {len(self.collections)} vector collections")
            
        except Exception as e:
            logger.error(f"Failed to setup vector collections: {e}")
            self.use_vector_db = False
    
    def chunk_and_store_prospectus(self, pdf_text: str, company_name: str, sector: str = "", ipo_date: str = None):
        """
        Chunk the prospectus document and store in vector database for retrieval.
        Clears the vector database before storing new PDF to start from empty state.
        """
        if not self.use_vector_db:
            logger.warning("Vector database not available, skipping document storage")
            return
        
        try:
            logger.info(f"Chunking and storing prospectus for {company_name}")
            
            # Clear vector database before storing new PDF
            self.clear_vector_database()
            
            # Split document into chunks
            # chunks = self._chunk_document(pdf_text)
            # logger.info(f"Created {len(chunks)} chunks from prospectus")

            # Split document into chunks using recursive splitter
            chunks = self._chunk_document_recursive(pdf_text)
            logger.info(f"Created {len(chunks)} chunks from prospectus using recursive splitter")

            # Classify and store chunks in appropriate collections
            for i, chunk in enumerate(chunks):
                chunk_type = self._classify_chunk(chunk)
                chunk_id = f"{company_name}_{chunk_type}_{i}"
                
                metadata = {
                    "company": company_name,
                    "sector": sector,
                    "chunk_type": chunk_type,
                    "chunk_index": i,
                    "ipo_date": ipo_date or "unknown",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store in appropriate collection based on chunk type
                if chunk_type == "financial":
                    collection = self.collections['financial_sections']
                elif chunk_type == "competitive":
                    collection = self.collections['competitive_sections'] 
                elif chunk_type == "ipo_specific":
                    collection = self.collections['ipo_sections']
                else:
                    collection = self.collections['prospectus_chunks']
                
                # Add chunk to collection
                collection.add(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
            
            print(f"Successfully stored {len(chunks)} chunks for {company_name}")
            logger.info(f"Successfully stored {len(chunks)} chunks for {company_name}")
            
        except Exception as e:
            print(f"Error chunking and storing prospectus: {e}")
            logger.error(f"Error chunking and storing prospectus: {e}")
    
    def _chunk_document(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """
        Split document into overlapping chunks for better context retrieval.
        """
        chunks = []
        
        # Split by sentences first for better semantic boundaries
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence
                current_length += sentence_length + 1
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_document_recursive(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """
        Split document into overlapping chunks using a recursive character text splitter.
        Tries to split by paragraphs, then sentences, then characters for optimal chunking.
        """
        import re

        def recursive_split(text: str, chunk_size: int, overlap: int, level: int = 0) -> List[str]:
            # Level 0: Paragraphs, Level 1: Sentences, Level 2: Characters
            if len(text) <= chunk_size:
                return [text.strip()]
            separators = ["\n\n", r'[.!?]+', '']  # Paragraph, sentence, char
            sep = separators[level]
            if sep:
                if level == 0:
                    splits = text.split(sep)
                else:
                    splits = re.split(sep, text)
            else:
                splits = list(text)
            chunks = []
            current = ""
            for i, part in enumerate(splits):
                part = part.strip()
                if not part:
                    continue
                if current:
                    candidate = current + (sep if sep and not current.endswith(sep) else "") + part
                else:
                    candidate = part
                if len(candidate) > chunk_size:
                    if current:
                        # If current is not empty, finalize it and start new
                        chunks.append(current.strip())
                        # Overlap
                        overlap_text = current[-overlap:] if overlap > 0 else ""
                        current = overlap_text + (sep if sep else "") + part
                    else:
                        # If a single part is too big, go deeper
                        if level < 2:
                            subchunks = recursive_split(part, chunk_size, overlap, level + 1)
                            chunks.extend(subchunks)
                            current = ""
                        else:
                            # At char level, just cut
                            chunks.append(part[:chunk_size])
                            current = part[chunk_size - overlap:] if overlap > 0 else ""
                else:
                    current = candidate
            if current.strip():
                chunks.append(current.strip())
            return chunks

        return recursive_split(text, chunk_size, overlap, 0)
    
    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """Get the last overlap_chars characters for chunk overlap."""
        if len(text) <= overlap_chars:
            return text
        
        # Try to break at word boundary
        overlap_text = text[-overlap_chars:]
        last_space = overlap_text.find(' ')
        
        if last_space != -1:
            return overlap_text[last_space:].strip()
        else:
            return overlap_text
    
    def _classify_chunk(self, chunk: str) -> str:
        """
        Classify document chunk type based on content keywords.
        """
        chunk_lower = chunk.lower()
        
        # Financial keywords
        financial_keywords = [
            'revenue', 'profit', 'ebitda', 'assets', 'liabilities', 'cash flow',
            'balance sheet', 'income statement', 'financial performance', 
            'ratio analysis', 'margin', 'return on equity', 'debt'
        ]
        
        # Competitive/business keywords  
        competitive_keywords = [
            'competition', 'competitors', 'market share', 'business model',
            'strategy', 'advantages', 'strengths', 'weaknesses', 'industry',
            'market position', 'differentiation'
        ]
        
        # IPO-specific keywords
        ipo_keywords = [
            'ipo', 'public offering', 'price band', 'listing', 'underwriter',
            'lead manager', 'use of proceeds', 'fund utilization', 'allotment',
            'book building', 'anchor investor'
        ]
        
        # Count keyword matches
        financial_score = sum(1 for keyword in financial_keywords if keyword in chunk_lower)
        competitive_score = sum(1 for keyword in competitive_keywords if keyword in chunk_lower)
        ipo_score = sum(1 for keyword in ipo_keywords if keyword in chunk_lower)
        
        # Classify based on highest score
        if financial_score >= competitive_score and financial_score >= ipo_score:
            return "financial"
        elif competitive_score >= ipo_score:
            return "competitive" 
        elif ipo_score > 0:
            return "ipo_specific"
        else:
            return "general"
    
    def retrieve_relevant_context(self, query: str, chunk_type: str = "all", n_results: int = 3) -> List[str]:
        """
        Retrieve relevant document chunks for enhancing LLM analysis.
        """
        if not self.use_vector_db:
            return []
        
        try:
            contexts = []
            
            if chunk_type == "all":
                # Search all collections
                collections_to_search = self.collections.values()
            else:
                # Search specific collection
                collection_map = {
                    "financial": self.collections.get('financial_sections'),
                    "competitive": self.collections.get('competitive_sections'),
                    "ipo_specific": self.collections.get('ipo_sections'),
                    "general": self.collections.get('prospectus_chunks')
                }
                collection = collection_map.get(chunk_type)
                collections_to_search = [collection] if collection else []
            
            for collection in collections_to_search:
                try:
                    results = collection.query(
                        query_texts=[query],
                        n_results=n_results
                    )
                    
                    if results and results['documents']:
                        contexts.extend(results['documents'][0])  # Flatten the nested list
                        
                except Exception as e:
                    logger.warning(f"Error querying collection: {e}")
                    continue
            
            print(f"Retrieved {len(contexts)} context chunks for query: {query[:50]}...")
            logger.info(f"Retrieved {len(contexts)} context chunks for query")
            return contexts[:n_results * 2]  # Limit total results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def analyze_prospectus_comprehensive(self, 
                                       pdf_text: str, 
                                       company_name: str,
                                       sector: str = "",
                                       pdf_path: str = None) -> Tuple[LLMFinancialMetrics, BenchmarkingAnalysis, IPOSpecificMetrics]:
        """
        Comprehensive LLM-powered analysis of IPO prospectus with vector DB enhancement.
        """
        logger.info(f"Starting comprehensive LLM analysis for {company_name}")
        logger.info(f"Using pdf text length: {len(pdf_text)} characters")
        
        # Store document chunks in vector DB for future retrieval
        if self.use_vector_db and len(pdf_text) > 500:
            self.chunk_and_store_prospectus(pdf_text, company_name, sector)
        
        # Split analysis into focused chunks for better accuracy
        financial_metrics = self._extract_financial_metrics(pdf_text, company_name, pdf_path)
        print("---Financial metrics---")
        print(financial_metrics)
        benchmarking = self._perform_benchmarking_analysis(pdf_text, company_name, sector, pdf_path)
        ipo_specifics = self._analyze_ipo_specifics(pdf_text, company_name, pdf_path)
        
        return financial_metrics, benchmarking, ipo_specifics
    
    def _extract_financial_metrics(self, pdf_text: str, company_name: str, pdf_path: str = None) -> LLMFinancialMetrics:
        """Extract detailed financial metrics using LLM with vector context enhancement."""
        
        # Get relevant financial context from vector DB
        financial_context = ""
        if self.use_vector_db:
            print(f"Vector DB enabled, searching for context...")
            context_chunks = self.retrieve_relevant_context(
                f"financial data revenue profit EBITDA ratios {company_name}", 
                chunk_type="financial", 
                n_results=3
            )
            print(f"Retrieved {len(context_chunks)} context chunks from vector DB")
            if context_chunks:
                financial_context = f"\nRelevant financial context:\n" + "\n---\n".join(context_chunks)
                print(f"Financial context length: {len(financial_context)}")
            else:
                print("No context chunks found - this is expected for first run")
        else:
            print("Vector DB not enabled")
        print("financial_context: ", financial_context)
        # Create a more concise prompt to avoid truncation issues
        prompt = f"""Extract financial metrics for {company_name} from the document text ONLY.

{financial_context}

⚠️ CRITICAL ANTI-HALLUCINATION RULES:
1. Extract ONLY metrics explicitly stated in the document - DO NOT calculate or infer values
2. If a metric is not clearly stated, return null (not 0, not estimated)
3. DO NOT use industry averages, benchmarks, or external knowledge
4. DO NOT make assumptions about missing data
5. If you see a range (e.g., "10-15%"), use the midpoint or note it in comments
6. Set extraction_confidence based on data clarity (0.0-1.0)
7. Set data_completeness based on how many fields have actual values (0.0-1.0)

CRITICAL: Respond with ONLY complete, valid JSON. No markdown, no explanations, no truncation.
The JSON MUST end with both "extraction_confidence" and "data_completeness" fields and a closing brace.

You can provide values as:
1. Direct numbers from document: 25.5
2. Arithmetic expressions if you see calculations: "(256.93 / 11884.89) * 100"
3. null if data not found in document (NEVER guess or use external knowledge)

Required JSON structure (use exact field names):
{{
    "trailing_pe_ratio": "float, expression, or null - ONLY if explicitly stated",
    "price_to_book_ratio": "float, expression, or null - ONLY if explicitly stated",
    "price_to_sales_ratio": "float, expression, or null - ONLY if explicitly stated",
    "gross_profit_margin": "float, expression, or null (as percentage) - ONLY from document",
    "operating_profit_margin": "float, expression, or null (as percentage) - ONLY from document",
    "net_profit_margin": "float, expression, or null (as percentage) - ONLY from document",
    "return_on_equity": "float, expression, or null (as percentage) - ONLY from document",
    "return_on_assets": "float, expression, or null (as percentage) - ONLY from document",
    "current_ratio": "float, expression, or null - ONLY from document",
    "quick_ratio": "float, expression, or null - ONLY from document",
    "debt_to_equity_ratio": "float, expression, or null - ONLY from document",
    "debt_to_assets_ratio": "float, expression, or null - ONLY from document",
    "interest_coverage_ratio": "float, expression, or null - ONLY from document",
    "revenue_growth_3yr": "float, expression, or null (as percentage) - ONLY from document",
    "profit_growth_3yr": "float, expression, or null (as percentage) - ONLY from document",
    "ebitda_growth_3yr": "float, expression, or null (as percentage) - ONLY from document",
    "extraction_confidence": "0.0-1.0 based on data clarity and completeness",
    "data_completeness": "0.0-1.0 based on percentage of non-null fields"
}}

Example: "net_profit_margin": "(256.93 / 11884.89) * 100" will be evaluated to 2.16
If data is unclear or not found, use null - NEVER make assumptions."""
        print("financial prompt: ", prompt)
        
        # Try multiple approaches to get a complete response
        for attempt in range(3):  # Try up to 3 times
            try:
                # Increase max tokens based on attempt
                max_tokens = 2000 + (attempt * 1000)  # Start with 2000, then 3000, then 4000
                
                print(f"Attempt {attempt + 1}: Calling LLM with max_tokens={max_tokens}")
                response = self._call_llm(prompt, max_tokens=max_tokens, temperature=0.1)
                
                print(f"Response length: {len(response) if response else 0}")
                if response:
                    # Show more of the response for debugging
                    if len(response) > 1000:
                        print("Response start:", response[:300])
                        print("Response end:", response[-300:])
                    else:
                        print("Response:", response)
                
                if response:
                    # Parse JSON response with improved error handling
                    try:
                        # Use robust JSON extraction method
                        json_text = self._extract_json_from_response(response)
                        
                        if not json_text:
                            # Check if response looks truncated and we should retry
                            is_truncated = (
                                'extraction_confidence' not in response or
                                'data_completeness' not in response or
                                response.count('{') != response.count('}')
                            )
                            
                        # Use robust JSON parsing with fallbacks
                        metrics_data = self._parse_json_with_fallbacks(response, "financial metrics")
                        allowed_fields = set(LLMFinancialMetrics.__dataclass_fields__.keys())
                        filtered_metrics_data = {k: v for k, v in metrics_data.items() if k in allowed_fields}
                        if filtered_metrics_data:
                            # Validate that we have the required fields
                            if 'extraction_confidence' not in filtered_metrics_data:
                                filtered_metrics_data['extraction_confidence'] = 0.5
                            if 'data_completeness' not in filtered_metrics_data:
                                filtered_metrics_data['data_completeness'] = 0.5

                            print(f"Successfully parsed JSON with confidence: {filtered_metrics_data.get('extraction_confidence', 'unknown')}")
                            return LLMFinancialMetrics(**filtered_metrics_data)
                        else:
                            # If JSON parsing fails completely, continue to next attempt
                            if attempt < 2:
                                print(f"JSON parsing failed completely, trying again with more tokens...")
                                continue
                            else:
                                raise json.JSONDecodeError("Could not extract valid JSON after all attempts", response, 0)
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error on attempt {attempt + 1}: {e}")
                        print(f"Error position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
                        print(f"Problematic text around error: {json_text[max(0, e.pos-50):e.pos+50] if hasattr(e, 'pos') and e.pos > 0 else 'N/A'}")
                        
                        # Only try partial extraction on the last attempt
                        if attempt == 2:
                            logger.warning(f"Failed to parse JSON on all attempts, trying partial extraction")
                            partial_data = self._extract_partial_financial_data(response)
                            if partial_data:
                                logger.info(f"Successfully extracted partial financial data")
                                return LLMFinancialMetrics(**partial_data)
                        else:
                            logger.warning(f"JSON parsing failed on attempt {attempt + 1}, retrying...")
                            continue
                    
                    except Exception as json_error:
                        logger.error(f"Unexpected error parsing JSON on attempt {attempt + 1}: {json_error}")
                        if attempt == 2:
                            # Try partial extraction as last resort
                            partial_data = self._extract_partial_financial_data(response)
                            if partial_data:
                                return LLMFinancialMetrics(**partial_data)
                        continue
                else:
                    print(f"Empty response on attempt {attempt + 1}")
                    if attempt < 2:
                        continue
                        
            except Exception as e:
                logger.error(f"Error in financial metrics extraction attempt {attempt + 1}: {e}")
                if attempt == 2:  # Last attempt
                    return LLMFinancialMetrics(extraction_confidence=0.0)
                continue
        
        # If all attempts failed
        logger.error("All attempts to extract financial metrics failed")
        return LLMFinancialMetrics(extraction_confidence=0.0)
    
    def _perform_benchmarking_analysis(self, pdf_text: str, company_name: str, sector: str, pdf_path: str = None) -> BenchmarkingAnalysis:
        """Perform benchmarking analysis using LLM with competitive context enhancement."""
        
        # Get relevant competitive context from vector DB
        competitive_context = ""
        if self.use_vector_db:
            context_chunks = self.retrieve_relevant_context(
                f"competitive analysis market position {sector} competition {company_name}",
                chunk_type="competitive",
                n_results=5
            )
            if context_chunks:
                competitive_context = f"\nRelevant competitive context:\n" + "\n---\n".join(context_chunks)
        
        prompt = f"""
        Analyze competitive position of {company_name} in {sector} sector using ONLY the provided document.

        {competitive_context}

        ⚠️ CRITICAL ANTI-HALLUCINATION RULES:
        1. Extract ONLY information explicitly stated in the document
        2. DO NOT add competitor names unless they are mentioned in the document
        3. DO NOT assume market position - only state what the document claims
        4. DO NOT use external knowledge about the sector or competitors
        5. If sector comparison data is not in document, mark as "unknown" or leave arrays empty
        6. Be explicit when company makes claims vs. when data is verified
        7. If no competitive information is found, return minimal/empty structures

        Find ONLY from the provided text:
        - Market position (ONLY if explicitly claimed or implied in document)
        - Main competitors (ONLY those mentioned by name in document)
        - Key advantages (ONLY those stated in document)
        - Industry trends (ONLY those mentioned in document)

        Return JSON format only:
        {{
            "sector_comparison": {{
                "revenue_growth_vs_sector": "above/below/inline/unknown - based ONLY on document",
                "profitability_vs_sector": "above/below/inline/unknown - based ONLY on document",
                "debt_levels_vs_sector": "above/below/inline/unknown - based ONLY on document",
                "key_metrics_comparison": ["ONLY metrics explicitly compared in document - use empty array if none"]
            }},
            "peer_companies": [
                {{"name": "ONLY companies mentioned in document", "similarity": "high/medium/low", "comparison": "ONLY comparisons stated in document"}}
            ],
            "market_position": "leader/challenger/follower/niche/unknown - based ONLY on document claims",
            "competitive_advantages": ["ONLY advantages explicitly stated - empty array if none found"],
            "competitive_disadvantages": ["ONLY if explicitly mentioned - empty array if none found"],
            "industry_trends": ["ONLY trends mentioned in document - empty array if none found"],
            "market_share_analysis": "ONLY if market share data is provided - otherwise 'Not disclosed in document'"
        }}

        REMEMBER: Empty arrays or "unknown" values are better than invented information.
        """
        print("benchmarking prompt: ", prompt)
        try:
            response = self._call_llm(prompt, max_tokens=1500, temperature=0.2)
            
            if response:
                try:
                    # Use robust JSON parsing with fallbacks
                    benchmark_data = self._parse_json_with_fallbacks(response, "benchmarking")
                    
                    if benchmark_data:
                        print("benchmarking data:")
                        print(benchmark_data)
                        return BenchmarkingAnalysis(**benchmark_data)
                    else:
                        logger.error("Could not extract valid JSON from benchmarking response")
                        # Try to extract partial data
                        partial_data = self._extract_partial_benchmarking(response)
                        if partial_data:
                            return BenchmarkingAnalysis(**partial_data)
                        
                except Exception as parse_e:
                    logger.error(f"Error parsing benchmarking data structure: {parse_e}")
                    logger.error(f"Response was: {response[:500]}...")
                    
                    # Try to extract partial data as final fallback
                    try:
                        partial_data = self._extract_partial_benchmarking(response)
                        if partial_data:
                            return BenchmarkingAnalysis(**partial_data)
                    except Exception as partial_e:
                        logger.error(f"Partial data extraction also failed: {partial_e}")
            
        except Exception as e:
            logger.error(f"Error in benchmarking analysis: {e}")
        
        # Return default benchmarking analysis
        return BenchmarkingAnalysis(
            sector_comparison={},
            peer_companies=[],
            market_position="unknown",
            competitive_advantages=[],
            competitive_disadvantages=[],
            industry_trends=[]
        )
    
    def _analyze_ipo_specifics(self, pdf_text: str, company_name: str, pdf_path: str = None) -> IPOSpecificMetrics:
        """Analyze IPO-specific factors using LLM with IPO context enhancement."""
        
        # Get relevant IPO context from vector DB
        ipo_context = ""
        if self.use_vector_db:
            context_chunks = self.retrieve_relevant_context(
                f"IPO pricing underwriters fund utilization {company_name}",
                chunk_type="ipo_specific", 
                n_results=3
            )
            if context_chunks:
                ipo_context = f"\nRelevant IPO context:\n" + "\n---\n".join(context_chunks)
        
        prompt = f"""
        Extract IPO information for {company_name} from the provided document ONLY.

        {ipo_context}

        ⚠️ CRITICAL ANTI-HALLUCINATION RULES:
        1. Extract ONLY information explicitly stated in the document
        2. DO NOT invent lead manager names - only list those mentioned
        3. DO NOT assess business model quality without explicit data - use "unknown" if not clear
        4. DO NOT make up pricing justifications - only quote what document states
        5. DO NOT estimate use of funds percentages - use null if not provided
        6. DO NOT rate promoter governance without explicit information
        7. For subjective assessments (sustainability, scalability), explicitly note these are document claims

        Find ONLY from the provided text:
        - IPO price range (exact values from document)
        - Use of funds breakdown (exact percentages if stated)
        - Lead managers (names as mentioned)
        - Business model details (as described in document)
        - Key risks (as listed in document)

        Return JSON only:
        {{
            "ipo_pricing_analysis": {{
                "price_band": "Rs X-Y per share - ONLY if stated, otherwise 'Not disclosed'",
                "valuation_method": "ONLY if methodology is mentioned, otherwise 'Not disclosed'",
                "peer_comparison": "ONLY comparisons stated in document, otherwise 'Not provided'",
                "pricing_justification": "ONLY reasons given in document, otherwise 'Not disclosed'"
            }},
            "underwriter_quality": {{
                "lead_managers": ["ONLY names mentioned - empty array if none"],
                "reputation_score": "ONLY if reputation is discussed - otherwise 'unknown'",
                "track_record": "ONLY if track record data provided - otherwise 'Not disclosed'"
            }},
            "use_of_funds_analysis": {{
                "capex_percentage": "float if stated, null if not",
                "debt_repayment_percentage": "float if stated, null if not",
                "working_capital_percentage": "float if stated, null if not",
                "other_purposes": "ONLY purposes mentioned in document"
            }},
            "lock_in_analysis": {{
                "promoter_lock_in": "ONLY if lock-in details are stated, otherwise 'Not disclosed'",
                "investor_lock_in": "ONLY if mentioned, otherwise 'Not disclosed'",
                "impact_assessment": "ONLY if discussed in document, otherwise 'Not assessed'"
            }},
            "promoter_background": {{
                "experience_years": "int if stated, null if not",
                "industry_expertise": "ONLY expertise mentioned in document",
                "track_record": "ONLY track record stated in document",
                "governance_score": "ONLY if governance is assessed in document - otherwise 'unknown'"
            }},
            "business_model_assessment": {{
                "sustainability": "high/medium/low/unknown - mark as 'unknown' if not clearly stated",
                "scalability": "high/medium/low/unknown - mark as 'unknown' if not clearly stated",
                "competitive_moat": "ONLY moats described in document, 'None identified' if unclear",
                "technology_edge": "ONLY if technology advantages are mentioned, 'None mentioned' if not"
            }},
            "growth_strategy_analysis": {{
                "expansion_timeline": "ONLY timelines stated in document, 'Not specified' if absent",
                "market_opportunity": "ONLY opportunity described in document", 
                "execution_capability": "ONLY capabilities mentioned, 'Not assessed' if absent",
                "capital_requirements": "ONLY capital needs stated, 'Not specified' if absent"
            }},
            "regulatory_compliance": {{
                "compliance_status": "compliant/issues/unknown - ONLY if compliance is discussed",
                "pending_approvals": "ONLY approvals mentioned, 'None mentioned' if absent",
                "regulatory_risks": ["ONLY risks listed in document - empty array if none"]
            }}
        }}

        REMEMBER: "Not disclosed", "unknown", null, and empty arrays are acceptable and preferred over speculation.
        """
        print("ipo specifics prompt: ", prompt)      
        try:
            response = self._call_llm(prompt, max_tokens=2500, temperature=0.15)
            
            if response:
                try:
                    # Use robust JSON parsing with fallbacks
                    ipo_data = self._parse_json_with_fallbacks(response, "IPO specifics")
                    
                    if ipo_data:
                        print("IPO specifics data:")
                        print(ipo_data)
                        return IPOSpecificMetrics(**ipo_data)
                    else:
                        logger.error("Could not extract valid JSON from response")
                        # Try to extract partial data using fallback method
                        try:
                            partial_data = self._extract_partial_ipo_data(response)
                            if partial_data:
                                return IPOSpecificMetrics(**partial_data)
                        except Exception as fallback_e:
                            logger.error(f"Fallback parsing also failed: {fallback_e}")
                        
                except Exception as parse_e:
                    logger.error(f"Error parsing IPO data structure: {parse_e}")
                    logger.error(f"Response was: {response[:500]}...")
                    # Try to extract partial data as final fallback
                    try:
                        partial_data = self._extract_partial_ipo_data(response)
                        if partial_data:
                            return IPOSpecificMetrics(**partial_data)
                    except Exception as final_fallback_e:
                        logger.error(f"Final fallback parsing also failed: {final_fallback_e}")
            
        except Exception as e:
            logger.error(f"Error in IPO specifics analysis: {e}")
        
        # Return default IPO metrics
        return IPOSpecificMetrics(
            ipo_pricing_analysis={},
            underwriter_quality={},
            use_of_funds_analysis={},
            lock_in_analysis={},
            promoter_background={},
            business_model_assessment={},
            growth_strategy_analysis={},
            regulatory_compliance={}
        )
    
    def _call_llm(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.1) -> Optional[str]:
        """Call the configured LLM with the given prompt."""
        
        try:
            if self.provider == "openai" and self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4",  # or "gpt-3.5-turbo" for cost efficiency
                    messages=[
                        {"role": "system", "content": "You are a financial analyst specializing in IPO analysis and prospectus evaluation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic" and self.client:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == "groq" and self.client:
                # Try models with fallback support
                models_to_try = getattr(self, 'groq_models', ["llama3-70b-8192", "llama3-8b-8192"])
                
                for model_name in models_to_try:
                    try:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a financial analyst specializing in IPO analysis."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        if "decommissioned" in str(e).lower() or "not found" in str(e).lower():
                            logger.warning(f"Groq model {model_name} is not available, trying next...")
                            continue
                        else:
                            # For other errors, log and continue to next model
                            logger.warning(f"Groq model {model_name} failed: {e}")
                            continue
                
                # If all models failed
                logger.error("All Groq models failed")
                return None
                
            elif self.provider == "gemini" and self.client:
                # Gemini uses different generation config
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40
                )
                
                # Configure safety settings to be more permissive for financial analysis
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
                
                # Simplify prompt for Gemini
                simple_prompt = f"""Extract financial data from this business document.

{prompt}"""
                
                try:
                    response = self.client.generate_content(
                        simple_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Handle different response formats
                    if hasattr(response, 'text') and response.text:
                        return response.text
                    elif hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                return candidate.content.parts[0].text
                    
                    # Log finish reason for debugging
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                        logger.warning(f"Gemini response empty, finish_reason: {finish_reason}")
                        
                        # Try with an even simpler prompt if blocked
                        if finish_reason == 2:  # SAFETY
                            logger.info("Retrying with simplified prompt due to safety filter")
                            minimal_prompt = "Analyze this financial document and return key metrics as JSON."
                            retry_response = self.client.generate_content(
                                minimal_prompt,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                            )
                            if hasattr(retry_response, 'text') and retry_response.text:
                                return retry_response.text
                    
                    return None
                    
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
        
        return None
    
    def generate_investment_thesis(self, 
                                 financial_metrics: LLMFinancialMetrics,
                                 benchmarking: BenchmarkingAnalysis,
                                 ipo_specifics: IPOSpecificMetrics,
                                 company_name: str,
                                 web_context: str = "") -> str:
        """Generate comprehensive investment thesis using LLM analysis and web search context."""
        
        # Convert dataclasses to dict for JSON serialization
        metrics_dict = asdict(financial_metrics)
        benchmark_dict = asdict(benchmarking)
        ipo_dict = asdict(ipo_specifics)
        
        # Add web context section if available
        web_context_section = ""
        if web_context:
            web_context_section = f"\n\n{web_context}\n"
        
        prompt = f"""
        Generate a detailed investment thesis for {company_name} based ONLY on the data provided below.
        
        ⚠️ CRITICAL ANTI-HALLUCINATION RULES:
        1. Use ONLY the data provided in this prompt - DO NOT make up or assume any information
        2. If data is missing (null, N/A, or not provided), explicitly state "Data not available" or "Not disclosed"
        3. DO NOT invent financial metrics, company details, or market information
        4. DO NOT make claims about competitors, market share, or industry unless explicitly stated in the data
        5. If you cannot make an assessment due to missing data, clearly state the limitation
        6. Quote specific numbers from the provided data when making claims

        === INTERNAL ANALYSIS (From Prospectus) ===
        Financial Metrics: {json.dumps(metrics_dict, indent=2)}
        Benchmarking: {json.dumps(benchmark_dict, indent=2)}
        IPO Specifics: {json.dumps(ipo_dict, indent=2)}
        {web_context_section}
        
        Generate a structured investment thesis covering:

        1. EXECUTIVE SUMMARY (2-3 sentences using ONLY provided data)
        2. KEY STRENGTHS (ONLY strengths explicitly mentioned or derivable from provided data - if limited data, acknowledge this)
        3. KEY CONCERNS (ONLY concerns from provided data - clearly mark speculative risks as such)
        4. VALUATION ASSESSMENT (ONLY if sufficient financial metrics are provided - otherwise state "Cannot assess due to insufficient data")
        5. INVESTMENT RECOMMENDATION (Based ONLY on available data - if data is limited, recommend "Hold/Avoid pending more information")
        6. RISK-REWARD ASSESSMENT (Based ONLY on provided data - explicitly note data gaps)
        7. TARGET PRICE ESTIMATE (ONLY if valuation metrics are available - otherwise state "Cannot estimate")
        8. MARKET CONTEXT (ONLY if web search context is provided above - summarize what was found, don't add external knowledge)
        9. DATA QUALITY ASSESSMENT (Rate the completeness of available data: High/Medium/Low and list key missing metrics)

        CRITICAL: 
        - Every claim must be traceable to the provided data
        - Prefer "Data not available" over speculation
        - Be explicit when making reasonable inferences vs. stating facts
        - If web context contradicts or enhances prospectus data, note the discrepancy
        """
        
        try:
            thesis = self._call_llm(prompt, max_tokens=1200, temperature=0.2)
            return thesis if thesis else "Investment thesis generation failed."
            
        except Exception as e:
            logger.error(f"Failed to generate investment thesis: {e}")
            return "Unable to generate investment thesis due to analysis error."

    def _fix_json_issues(self, json_text: str) -> str:
        """Fix common JSON formatting issues in LLM responses."""
        try:
            # Remove common problematic characters and fix formatting
            json_text = json_text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            
            json_text = json_text.strip()
            
            # Remove trailing ellipsis that might cause truncation issues
            if json_text.endswith('...'):
                json_text = json_text[:-3]
            
            # Fix common formatting issues
            json_text = json_text.replace('\n', ' ')  # Remove newlines but preserve structure
            json_text = json_text.replace('\r', '')   # Remove carriage returns
            json_text = json_text.replace('\t', ' ')  # Replace tabs with spaces
            
            # Fix multiple spaces
            json_text = re.sub(r'\s+', ' ', json_text)
            
            # Fix common JSON syntax issues
            json_text = json_text.replace('\\', '\\\\')  # Escape backslashes
            json_text = json_text.replace('"null"', 'null')  # Fix quoted null values
            json_text = json_text.replace('"true"', 'true')  # Fix quoted boolean values
            json_text = json_text.replace('"false"', 'false')  # Fix quoted boolean values
            
            # Fix missing commas between JSON properties (common LLM error)
            # Look for patterns like `"key": "value" "nextkey":` and add comma
            json_text = re.sub(r'("\s*:\s*"[^"]*")\s+(")', r'\1, \2', json_text)
            json_text = re.sub(r'("\s*:\s*[^",}\]]+)\s+(")', r'\1, \2', json_text)
            json_text = re.sub(r'(\])\s+(")', r'\1, \2', json_text)  # After arrays
            json_text = re.sub(r'(\})\s+(")', r'\1, \2', json_text)  # After objects
            
            # Fix missing commas in arrays
            json_text = re.sub(r'("\s*)\s+(")', r'\1, \2', json_text)
            
            # Fix trailing commas (which are invalid in JSON)
            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
            
            # Handle incomplete JSON objects (more robust approach)
            if json_text.startswith('{') and not json_text.endswith('}'):
                # Try to find where the JSON got truncated
                print(f"Detected incomplete JSON, attempting to fix...")
                
                # Find the last complete key-value pair
                last_complete_pair = -1
                brace_count = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(json_text):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                        elif char == ',' and brace_count == 1:
                            # This is a top-level comma, mark as potential cut point
                            last_complete_pair = i
                
                # Truncate at the last complete pair and close the JSON
                if last_complete_pair > 0:
                    json_text = json_text[:last_complete_pair] + '}'
                    print(f"Truncated JSON at position {last_complete_pair}")
                else:
                    # If no complete pairs found, just add missing closing brace
                    json_text = json_text.rstrip()
                    if json_text.endswith(','):
                        json_text = json_text[:-1]  # Remove trailing comma
                    json_text += '}'
                    print(f"Added closing brace to JSON")
            
            # Fix unterminated strings (common issue)
            if json_text.count('"') % 2 != 0:
                json_text += '"'
            
            # Fix missing closing braces/brackets
            open_braces = json_text.count('{')
            close_braces = json_text.count('}')
            if open_braces > close_braces:
                json_text += '}' * (open_braces - close_braces)
            
            open_brackets = json_text.count('[')
            close_brackets = json_text.count(']')
            if open_brackets > close_brackets:
                json_text += ']' * (open_brackets - close_brackets)
            
            return json_text.strip()
            
        except Exception as e:
            logger.error(f"Error fixing JSON issues: {e}")
            return json_text

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling extra data after JSON."""
        try:
            # Remove markdown code blocks first
            text = response.strip()
            
            if '```json' in text:
                start_idx = text.find('```json') + 7
                end_idx = text.find('```', start_idx)
                if end_idx != -1:
                    text = text[start_idx:end_idx].strip()
            elif '```' in text:
                start_idx = text.find('```') + 3
                end_idx = text.find('```', start_idx)
                if end_idx != -1:
                    text = text[start_idx:end_idx].strip()
            
            # Find the first opening brace
            start_idx = text.find('{')
            if start_idx == -1:
                return None
                
            # Track braces to find the end of the JSON object
            brace_count = 0
            in_string = False
            escape_next = False
            end_idx = -1
            
            for i in range(start_idx, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
            
            if end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                # Validate it's proper JSON by attempting to parse
                json.loads(json_text)
                return json_text
            else:
                logger.warning("Could not find complete JSON object in response")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"Extracted text is not valid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            return None

    def _parse_json_with_fallbacks(self, response: str, description: str = "analysis") -> Optional[Dict[str, Any]]:
        """Parse JSON with multiple fallback strategies and evaluate arithmetic expressions."""
        attempts = [
            ("Direct JSON extraction", lambda: self._extract_json_from_response(response)),
            ("Fixed JSON", lambda: self._fix_json_issues(self._extract_json_from_response(response) or "")),
            ("Regex-based extraction", lambda: self._extract_json_regex_fallback(response)),
        ]
        
        for attempt_name, extract_func in attempts:
            try:
                json_text = extract_func()
                if json_text:
                    # Try to parse the JSON
                    data = json.loads(json_text)
                    
                    # Evaluate any arithmetic expressions in the data
                    data = evaluate_expressions_in_dict(data)
                    
                    logger.info(f"Successfully parsed {description} JSON using {attempt_name}")
                    return data
            except json.JSONDecodeError as e:
                logger.debug(f"{attempt_name} failed for {description}: {e}")
                continue
            except Exception as e:
                logger.debug(f"{attempt_name} error for {description}: {e}")
                continue
        
        logger.error(f"All JSON parsing attempts failed for {description}")
        return None
    
    def _extract_json_regex_fallback(self, response: str) -> Optional[str]:
        """Extract JSON using regex as a last resort."""
        try:
            # Look for JSON-like patterns
            import re
            
            # Find the most JSON-like content between braces
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                # Try to clean and validate each match
                try:
                    cleaned = self._fix_json_issues(match)
                    json.loads(cleaned)  # Validate
                    return cleaned
                except:
                    continue
            
            return None
        except Exception as e:
            logger.error(f"Regex fallback extraction failed: {e}")
            return None

    def _extract_partial_benchmarking(self, response: str) -> Dict[str, Any]:
        """Extract partial benchmarking data when JSON parsing fails."""
        try:
            # Default structure
            partial_data = {
                "sector_comparison": {},
                "peer_companies": [],
                "market_position": "unknown",
                "competitive_advantages": [],
                "competitive_disadvantages": [],
                "industry_trends": []
            }
            
            # Try to extract key information using regex and text parsing
            import re
            
            # Extract market position
            market_pos_match = re.search(r'"market_position":\s*"([^"]+)"', response)
            if market_pos_match:
                partial_data["market_position"] = market_pos_match.group(1)
            
            # Extract competitive advantages
            adv_matches = re.findall(r'"competitive_advantages":\s*\[(.*?)\]', response, re.DOTALL)
            if adv_matches:
                advantages = re.findall(r'"([^"]+)"', adv_matches[0])
                partial_data["competitive_advantages"] = advantages[:5]  # Limit to 5
            
            # Extract competitive disadvantages
            disadv_matches = re.findall(r'"competitive_disadvantages":\s*\[(.*?)\]', response, re.DOTALL)
            if disadv_matches:
                disadvantages = re.findall(r'"([^"]+)"', disadv_matches[0])
                partial_data["competitive_disadvantages"] = disadvantages[:5]  # Limit to 5
            
            # Extract industry trends
            trends_matches = re.findall(r'"industry_trends":\s*\[(.*?)\]', response, re.DOTALL)
            if trends_matches:
                trends = re.findall(r'"([^"]+)"', trends_matches[0])
                partial_data["industry_trends"] = trends[:5]  # Limit to 5
            
            logger.info(f"Extracted partial benchmarking data: {len(partial_data['competitive_advantages'])} advantages, {len(partial_data['competitive_disadvantages'])} disadvantages")
            return partial_data
            
        except Exception as e:
            logger.error(f"Error extracting partial benchmarking data: {e}")
            return {
                "sector_comparison": {},
                "peer_companies": [],
                "market_position": "unknown",
                "competitive_advantages": [],
                "competitive_disadvantages": [],
                "industry_trends": []
            }


    def _extract_partial_financial_data(self, response: str) -> Dict[str, Any]:
        """Extract partial financial data when JSON parsing fails."""
        try:
            # Initialize with default values
            partial_data = {
                "extraction_confidence": 0.3,  # Low confidence for partial extraction
                "data_completeness": 0.0
            }
            
            # Define patterns for common financial metrics (including expressions)
            # Pattern matches: numbers, arithmetic expressions, or null
            value_pattern = r'([0-9]*\.?[0-9]+|null|"[\d.\s+\-*/()]+"|[\d.\s+\-*/()]+)'
            
            patterns = {
                "trailing_pe_ratio": rf'"trailing_pe_ratio":\s*{value_pattern}',
                "forward_pe_ratio": rf'"forward_pe_ratio":\s*{value_pattern}',
                "price_to_book_ratio": rf'"price_to_book_ratio":\s*{value_pattern}',
                "price_to_sales_ratio": rf'"price_to_sales_ratio":\s*{value_pattern}',
                "ev_to_ebitda_ratio": rf'"ev_to_ebitda_ratio":\s*{value_pattern}',
                "gross_profit_margin": rf'"gross_profit_margin":\s*{value_pattern}',
                "operating_profit_margin": rf'"operating_profit_margin":\s*{value_pattern}',
                "net_profit_margin": rf'"net_profit_margin":\s*{value_pattern}',
                "return_on_equity": rf'"return_on_equity":\s*{value_pattern}',
                "return_on_assets": rf'"return_on_assets":\s*{value_pattern}',
                "current_ratio": rf'"current_ratio":\s*{value_pattern}',
                "quick_ratio": rf'"quick_ratio":\s*{value_pattern}',
                "debt_to_equity_ratio": rf'"debt_to_equity_ratio":\s*{value_pattern}',
                "debt_to_assets_ratio": rf'"debt_to_assets_ratio":\s*{value_pattern}',
                "interest_coverage_ratio": rf'"interest_coverage_ratio":\s*{value_pattern}',
                "asset_turnover_ratio": rf'"asset_turnover_ratio":\s*{value_pattern}',
                "revenue_growth_3yr": rf'"revenue_growth_3yr":\s*{value_pattern}',
                "profit_growth_3yr": rf'"profit_growth_3yr":\s*{value_pattern}',
                "ebitda_growth_3yr": rf'"ebitda_growth_3yr":\s*{value_pattern}',
            }
            
            extracted_count = 0
            
            # Extract each metric using regex
            for field_name, pattern in patterns.items():
                match = re.search(pattern, response)
                if match:
                    value_str = match.group(1).strip().strip('"')
                    
                    if value_str == 'null':
                        partial_data[field_name] = None
                    else:
                        # Check if it's an expression
                        if re.match(r'^[\d.\s+\-*/()]+$', value_str):
                            try:
                                # Try to evaluate as expression
                                evaluated = safe_eval(value_str)
                                partial_data[field_name] = round(evaluated, 2)
                                extracted_count += 1
                                logger.debug(f"Evaluated {field_name}: {value_str} = {partial_data[field_name]}")
                            except:
                                try:
                                    # Fallback to direct float conversion
                                    partial_data[field_name] = float(value_str)
                                    extracted_count += 1
                                except:
                                    partial_data[field_name] = None
                        else:
                            try:
                                # Try direct conversion for simple numbers
                                partial_data[field_name] = float(value_str)
                                extracted_count += 1
                            except:
                                partial_data[field_name] = None
                else:
                    partial_data[field_name] = None
            
            # Calculate data completeness based on extracted fields
            total_fields = len(patterns)
            partial_data["data_completeness"] = extracted_count / total_fields if total_fields > 0 else 0.0
            
            # Increase confidence if we got a reasonable number of metrics
            if extracted_count > 5:
                partial_data["extraction_confidence"] = 0.6
            elif extracted_count > 3:
                partial_data["extraction_confidence"] = 0.4
            
            logger.info(f"Partial extraction: {extracted_count}/{total_fields} metrics extracted")
            return partial_data if extracted_count > 0 else None
            
        except Exception as e:
            logger.error(f"Error extracting partial financial data: {e}")
            return None

    def _extract_partial_ipo_data(self, response: str) -> Dict[str, Any]:
        """Extract partial IPO data when JSON parsing fails."""
        try:
            # Default structure for IPO metrics
            partial_data = {
                "ipo_pricing_analysis": {},
                "underwriter_quality": {},
                "use_of_funds_analysis": {},
                "lock_in_analysis": {},
                "promoter_background": {},
                "business_model_assessment": {},
                "growth_strategy_analysis": {},
                "regulatory_compliance": {}
            }
            
            # Try to extract key information using regex and text parsing
            import re
            
            # Extract price band
            price_match = re.search(r'"price_band":\s*"([^"]+)"', response)
            if price_match:
                partial_data["ipo_pricing_analysis"]["price_band"] = price_match.group(1)
            
            # Extract lead managers
            managers_match = re.search(r'"lead_managers":\s*\[([^\]]+)\]', response)
            if managers_match:
                managers_text = managers_match.group(1)
                # Clean up and extract manager names
                managers = [m.strip('"').strip() for m in managers_text.split(',')]
                partial_data["underwriter_quality"]["lead_managers"] = managers
            
            # Extract sustainability
            sustainability_match = re.search(r'"sustainability":\s*"([^"]+)"', response)
            if sustainability_match:
                partial_data["business_model_assessment"]["sustainability"] = sustainability_match.group(1)
            
            logger.info("Extracted partial IPO data using regex fallback")
            return partial_data
            
        except Exception as e:
            logger.error(f"Error extracting partial IPO data: {e}")
            return None

    def find_similar_ipos(self, query_text: str, sector: str = "", n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar IPOs based on text similarity and sector."""
        if not self.use_vector_db:
            return []
        
        try:
            similar_ipos = []
            
            # Search across all collections for similar content
            for collection_name, collection in self.collections.items():
                try:
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=n_results,
                        where={"sector": sector} if sector else None
                    )
                    
                    if results and results['documents'] and results['metadatas']:
                        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                            if metadata.get('company') not in [ipo['company'] for ipo in similar_ipos]:
                                similar_ipos.append({
                                    'company': metadata.get('company'),
                                    'sector': metadata.get('sector'),
                                    'ipo_date': metadata.get('ipo_date'),
                                    'similarity_score': 1.0 - (i * 0.1),  # Rough similarity scoring
                                    'chunk_type': metadata.get('chunk_type'),
                                    'collection': collection_name
                                })
                
                except Exception as e:
                    logger.warning(f"Error querying collection {collection_name} for similar IPOs: {e}")
                    continue
            
            # Sort by similarity score and return top results
            similar_ipos.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_ipos[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar IPOs: {e}")
            return []

    def get_vector_db_stats(self) -> Dict[str, Any]:
            """Get statistics about the vector database contents."""
            if not self.use_vector_db:
                return {'enabled': False}
            
            try:
                stats = {
                    'enabled': True,
                    'collections': {},
                    'total_chunks': 0,
                    'unique_companies': set(),
                    'sectors_covered': set()
                }
                
                for collection_name, collection in self.collections.items():
                    try:
                        # Get collection count
                        count_result = collection.count()
                        
                        # Get sample metadata
                        sample_results = collection.get(limit=100)
                        
                        collection_stats = {
                            'total_documents': count_result,
                            'companies': set(),
                            'sectors': set()
                        }
                        
                        if sample_results and sample_results['metadatas']:
                            for metadata in sample_results['metadatas']:
                                if metadata.get('company'):
                                    collection_stats['companies'].add(metadata['company'])
                                    stats['unique_companies'].add(metadata['company'])
                                if metadata.get('sector'):
                                    collection_stats['sectors'].add(metadata['sector'])
                                    stats['sectors_covered'].add(metadata['sector'])
                        
                        collection_stats['unique_companies'] = len(collection_stats['companies'])
                        collection_stats['unique_sectors'] = len(collection_stats['sectors'])
                        
                        stats['collections'][collection_name] = collection_stats
                        stats['total_chunks'] += count_result
                    
                    except Exception as e:
                        logger.warning(f"Error getting stats for collection {collection_name}: {e}")
                        stats['collections'][collection_name] = {'error': str(e)}
                
                stats['unique_companies'] = len(stats['unique_companies'])
                stats['sectors_covered'] = list(stats['sectors_covered'])
                
                return stats
                
            except Exception as e:
                logger.error(f"Error getting vector DB stats: {e}")
                return {'enabled': True, 'error': str(e)}

    def clear_company_data(self, company_name: str) -> bool:
        """Clear all data for a specific company from vector database."""
        if not self.use_vector_db:
            return False
        
        try:
            deleted_count = 0
            
            for collection_name, collection in self.collections.items():
                try:
                    # Get documents for this company
                    results = collection.get(
                        where={"company": company_name}
                    )
                    
                    if results and results['ids']:
                        # Delete the documents
                        collection.delete(ids=results['ids'])
                        deleted_count += len(results['ids'])
                        logger.info(f"Deleted {len(results['ids'])} chunks for {company_name} from {collection_name}")
                
                except Exception as e:
                    logger.warning(f"Error clearing data from {collection_name}: {e}")
                    continue
            
            logger.info(f"Successfully deleted {deleted_count} total chunks for {company_name}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error clearing company data: {e}")
            return False

    def search_companies_by_text(self, search_query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for companies based on text content similarity."""
        if not self.use_vector_db:
            return []
        
        try:
            search_results = []
            
            for collection_name, collection in self.collections.items():
                try:
                    results = collection.query(
                        query_texts=[search_query],
                        n_results=n_results
                    )
                    
                    if results and results['documents'] and results['metadatas']:
                        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                            search_results.append({
                                'company': metadata.get('company'),
                                'sector': metadata.get('sector'),
                                'ipo_date': metadata.get('ipo_date'),
                                'chunk_type': metadata.get('chunk_type'),
                                'collection': collection_name,
                                'relevance_score': 1.0 - (i * 0.05),  # Rough relevance scoring
                                'preview': doc[:200] + "..." if len(doc) > 200 else doc
                            })
                
                except Exception as e:
                    logger.warning(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort by relevance and remove duplicates
            seen_companies = set()
            unique_results = []
            
            for result in sorted(search_results, key=lambda x: x['relevance_score'], reverse=True):
                company_key = (result['company'], result['chunk_type'])
                if company_key not in seen_companies:
                    seen_companies.add(company_key)
                    unique_results.append(result)
            
            return unique_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching companies by text: {e}")
            return []

    def get_sector_insights(self, sector: str) -> Dict[str, Any]:
        """Get insights about a specific sector from stored IPO data."""
        if not self.use_vector_db or not sector:
            return {}
        
        try:
            insights = {
                'sector': sector,
                'total_companies': 0,
                'common_themes': [],
                'avg_metrics': {},
                'risk_patterns': []
            }
            
            # Count companies in sector
            companies_in_sector = set()
            
            for collection_name, collection in self.collections.items():
                try:
                    # Get all documents for this sector
                    results = collection.get(
                        where={"sector": sector}
                    )
                    
                    if results and results['metadatas']:
                        for metadata in results['metadatas']:
                            if metadata.get('company'):
                                companies_in_sector.add(metadata['company'])
                
                except Exception as e:
                    logger.warning(f"Error getting sector insights from {collection_name}: {e}")
                    continue
            
            insights['total_companies'] = len(companies_in_sector)
            
            # Get common themes using LLM if we have enough data
            if insights['total_companies'] > 2:
                sector_query = f"{sector} industry trends common patterns"
                context_chunks = self.retrieve_relevant_context(
                    sector_query, 
                    chunk_type="all", 
                    n_results=10
                )
                
                if context_chunks:
                    # Use LLM to analyze common themes
                    theme_prompt = f"""
                    Analyze these {sector} sector IPO documents and identify:
                    1. Common business themes
                    2. Typical risk patterns  
                    3. Average performance indicators
                    
                    Return JSON only:
                    {{
                        "common_themes": ["theme1", "theme2", "..."],
                        "risk_patterns": ["risk1", "risk2", "..."],
                        "performance_insights": ["insight1", "insight2", "..."]
                    }}
                    
                    Documents: {' '.join(context_chunks[:3])}
                    """
                    
                    try:
                        theme_response = self._call_llm(theme_prompt, max_tokens=800, temperature=0.2)
                        if theme_response:
                            # Parse JSON response
                            json_text = theme_response.strip()
                            if json_text.startswith('```json'):
                                json_text = json_text[7:]
                            if json_text.startswith('```'):
                                json_text = json_text[3:]
                            if json_text.endswith('```'):
                                json_text = json_text[:-3]
                            
                            theme_data = json.loads(json_text.strip())
                            insights.update(theme_data)
                            
                    except Exception as e:
                        logger.warning(f"Error analyzing sector themes: {e}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting sector insights: {e}")
            return {}
    
    def clear_vector_database(self):
        """
        Clear all stored documents from the vector database.
        """
        if not self.use_vector_db:
            logger.warning("Vector database not available, skipping clear operation")
            return
        
        try:
            # Get collection stats before clearing
            stats = self.get_vector_db_stats()
            total_docs = stats.get('total_chunks', 0)
            logger.info(f"Clearing vector database before storing new PDF. Current stats: {total_docs} documents")
            
            if total_docs > 0:
                # Clear all collections
                for collection_name, collection in self.collections.items():
                    try:
                        # Get all documents in the collection
                        all_docs = collection.get()
                        if all_docs and all_docs['ids']:
                            collection.delete(ids=all_docs['ids'])
                            logger.info(f"Cleared {len(all_docs['ids'])} documents from {collection_name} collection")
                    except Exception as e:
                        logger.warning(f"Error clearing collection {collection_name}: {e}")
                        
                logger.info(f"Vector database cleared successfully. Removed {total_docs} total documents.")
            else:
                logger.info("Vector database is already empty, no need to clear.")
                
        except Exception as e:
            logger.error(f"Error clearing vector database: {e}")
            raise

# Enhanced integration functions
def integrate_llm_analysis(company_name: str, pdf_text: str, sector: str = "", 
                        provider: str = "openai", pdf_path: str = None, 
                        use_vector_db: bool = True) -> Dict[str, Any]:
    """
    Integrate LLM-powered analysis into existing IPO data structure with vector DB and web search support.
    """
    analyzer = LLMProspectusAnalyzer(provider=provider, use_vector_db=use_vector_db)
    
    # Perform comprehensive analysis (includes vector storage)
    financial_metrics, benchmarking, ipo_specifics = analyzer.analyze_prospectus_comprehensive(
        pdf_text, company_name, sector, pdf_path
    )
    
    # Search Brave for additional context
    logger.info(f"Searching Brave API for additional context on {company_name}")
    brave_results = analyzer.search_brave_for_ipo_context(company_name, max_results=5)
    web_context = analyzer._format_brave_context(brave_results) if brave_results else ""
    
    if web_context:
        logger.info(f"Successfully retrieved web context from {len(brave_results)} sources")
    else:
        logger.warning("No web context retrieved from Brave API")
    
    # Generate investment thesis with web context
    investment_thesis = analyzer.generate_investment_thesis(
        financial_metrics, benchmarking, ipo_specifics, company_name, web_context
    )
    
    # Get additional insights from vector DB if available
    similar_ipos = []
    sector_insights = {}
    vector_db_stats = {}
    
    if analyzer.use_vector_db:
        similar_ipos = analyzer.find_similar_ipos(pdf_text[:500], sector)
        sector_insights = analyzer.get_sector_insights(sector)
        vector_db_stats = analyzer.get_vector_db_stats()
    
    # Structure the results for integration
    return {
        'llm_financial_metrics': financial_metrics,
        'llm_benchmarking': benchmarking,
        'llm_ipo_specifics': ipo_specifics,
        'llm_investment_thesis': investment_thesis,
        'llm_analysis_timestamp': datetime.now().isoformat(),
        'llm_provider': provider,
        'vector_db_enabled': analyzer.use_vector_db,
        'similar_ipos': similar_ipos,
        'sector_insights': sector_insights,
        'vector_db_stats': vector_db_stats
    }


# Utility functions for backward compatibility
def calculate_advanced_ratios(financial_metrics: LLMFinancialMetrics) -> Dict[str, float]:
    """Calculate additional ratios from LLM-extracted data."""
    ratios = {}
    
    if financial_metrics.trailing_pe_ratio:
        ratios['pe_ratio'] = financial_metrics.trailing_pe_ratio
    
    if financial_metrics.return_on_equity:
        ratios['roe'] = financial_metrics.return_on_equity
        
    if financial_metrics.debt_to_equity_ratio:
        ratios['debt_equity'] = financial_metrics.debt_to_equity_ratio
        
    if financial_metrics.current_ratio:
        ratios['current_ratio'] = financial_metrics.current_ratio
        
    return ratios


def get_peer_comparison_summary(benchmarking: BenchmarkingAnalysis) -> Dict[str, Any]:
    """Get summarized peer comparison data."""
    return {
        'market_position': benchmarking.market_position,
        'num_peers_identified': len(benchmarking.peer_companies),
        'competitive_advantages_count': len(benchmarking.competitive_advantages),
        'industry_trends_count': len(benchmarking.industry_trends),
        'sector_performance': benchmarking.sector_comparison
    }
