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
    
    def _chunk_document_semantic(self, text: str, chunk_size: int = 2000, 
                                 similarity_threshold: float = 0.5) -> List[str]:
        """
        Split document into semantically coherent chunks using embedding-based similarity.
        
        Args:
            text: The document text to chunk
            chunk_size: Target size for each chunk
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
        
        Returns:
            List of semantically coherent text chunks
        """
        try:
            sentences = self._split_into_sentences(text)
            
            if not sentences or len(sentences) < 2:
                return [text] if text.strip() else []
            
            if not self._can_generate_embeddings():
                logger.info("Falling back to recursive chunking")
                return self._chunk_document_recursive(text, chunk_size)
            
            logger.info(f"Generating embeddings for {len(sentences)} sentences...")
            embeddings = self._generate_sentence_embeddings(sentences)
            
            if embeddings is None:
                return self._chunk_document_recursive(text, chunk_size)
            
            chunks = self._group_sentences_by_similarity(
                sentences, embeddings, chunk_size, similarity_threshold
            )
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self._chunk_document_recursive(text, chunk_size)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling of abbreviations."""
        # Handle common abbreviations to avoid false sentence breaks
        text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs").replace("Ms.", "Ms")
        text = text.replace("Dr.", "Dr").replace("Prof.", "Prof")
        text = text.replace("Inc.", "Inc").replace("Ltd.", "Ltd").replace("Corp.", "Corp")
        text = text.replace("Co.", "Co").replace("vs.", "vs")
        text = text.replace("Rs.", "Rs").replace("Cr.", "Cr").replace("Mn.", "Mn")
        text = text.replace("Fig.", "Fig").replace("Vol.", "Vol").replace("No.", "No")
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def _can_generate_embeddings(self) -> bool:
        """Check if we can generate embeddings with the current LLM provider."""
        # OpenAI provides embeddings API
        if self.provider == "openai" and self.client:
            return True
        
        # Gemini has embedding models
        if self.provider == "gemini" and GEMINI_AVAILABLE:
            return True
        
        # For other providers, we could use sentence-transformers as fallback
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _generate_sentence_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of sentences.
        
        Returns:
            numpy array of shape (num_sentences, embedding_dim) or None if failed
        """
        try:
            if self.provider == "openai" and self.client:
                return self._generate_openai_embeddings(sentences)
            
            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                return self._generate_gemini_embeddings(sentences)
            
            else:
                # Try to use sentence-transformers as fallback
                return self._generate_sentence_transformer_embeddings(sentences)
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _generate_openai_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using OpenAI's embedding API."""
        try:
            # Use the text-embedding-3-small model (cost-effective and fast)
            embeddings = []
            
            # Process in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            return None
    
    def _generate_gemini_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using Google's Gemini embedding API."""
        try:
            embeddings = []
            
            # Use Gemini's embedding model
            for sentence in sentences:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=sentence,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Gemini embeddings error: {e}")
            return None
    
    def _generate_sentence_transformer_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using sentence-transformers (local fallback)."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model
            if not hasattr(self, '_embedding_model'):
                logger.info("Loading sentence-transformer model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self._embedding_model.encode(sentences, show_progress_bar=False)
            return np.array(embeddings)
            
        except ImportError:
            logger.warning("sentence-transformers not available for local embeddings")
            return None
        except Exception as e:
            logger.error(f"Sentence-transformer error: {e}")
            return None
    
    def _group_sentences_by_similarity(self, sentences: List[str], 
                                       embeddings: np.ndarray,
                                       target_chunk_size: int,
                                       similarity_threshold: float) -> List[str]:
        """
        Group sentences into chunks based on semantic similarity.
        
        Algorithm:
        1. Start with the first sentence in a chunk
        2. Add subsequent sentences if they are semantically similar
        3. Start a new chunk when similarity drops below threshold or size limit reached
        """
        chunks = []
        current_chunk = []
        current_chunk_text = ""
        current_chunk_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            sentence_length = len(sentence)
            
            # Start first chunk
            if not current_chunk:
                current_chunk.append(sentence)
                current_chunk_text = sentence
                current_chunk_embedding = embedding
                continue
            
            # Calculate similarity with current chunk
            similarity = self._cosine_similarity(current_chunk_embedding, embedding)
            
            # Decide whether to add to current chunk or start new one
            would_exceed_size = len(current_chunk_text) + sentence_length > target_chunk_size * 1.5
            is_dissimilar = similarity < similarity_threshold
            
            if would_exceed_size or (is_dissimilar and len(current_chunk_text) > target_chunk_size * 0.5):
                # Finalize current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_text = sentence
                current_chunk_embedding = embedding
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_chunk_text += " " + sentence
                
                # Update chunk embedding as running average
                current_chunk_embedding = (current_chunk_embedding * (len(current_chunk) - 1) + embedding) / len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
        

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
    
    def _chunk_document_semantic(self, text: str, chunk_size: int = 2000, 
                                 similarity_threshold: float = 0.5) -> List[str]:
        """
        Split document into semantically coherent chunks using embedding-based similarity.
        
        Args:
            text: The document text to chunk
            chunk_size: Target size for each chunk
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
        
        Returns:
            List of semantically coherent text chunks
        """
        try:
            sentences = self._split_into_sentences(text)
            
            if not sentences or len(sentences) < 2:
                return [text] if text.strip() else []
            
            if not self._can_generate_embeddings():
                logger.info("Falling back to recursive chunking")
                return self._chunk_document_recursive(text, chunk_size)
            
            logger.info(f"Generating embeddings for {len(sentences)} sentences...")
            embeddings = self._generate_sentence_embeddings(sentences)
            
            if embeddings is None:
                return self._chunk_document_recursive(text, chunk_size)
            
            chunks = self._group_sentences_by_similarity(
                sentences, embeddings, chunk_size, similarity_threshold
            )
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self._chunk_document_recursive(text, chunk_size)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling of abbreviations."""
        # Handle common abbreviations to avoid false sentence breaks
        text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs").replace("Ms.", "Ms")
        text = text.replace("Dr.", "Dr").replace("Prof.", "Prof")
        text = text.replace("Inc.", "Inc").replace("Ltd.", "Ltd").replace("Corp.", "Corp")
        text = text.replace("Co.", "Co").replace("vs.", "vs")
        text = text.replace("Rs.", "Rs").replace("Cr.", "Cr").replace("Mn.", "Mn")
        text = text.replace("Fig.", "Fig").replace("Vol.", "Vol").replace("No.", "No")
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def _can_generate_embeddings(self) -> bool:
        """Check if we can generate embeddings with the current LLM provider."""
        # OpenAI provides embeddings API
        if self.provider == "openai" and self.client:
            return True
        
        # Gemini has embedding models
        if self.provider == "gemini" and GEMINI_AVAILABLE:
            return True
        
        # For other providers, we could use sentence-transformers as fallback
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _generate_sentence_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of sentences.
        
        Returns:
            numpy array of shape (num_sentences, embedding_dim) or None if failed
        """
        try:
            if self.provider == "openai" and self.client:
                return self._generate_openai_embeddings(sentences)
            
            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                return self._generate_gemini_embeddings(sentences)
            
            else:
                # Try to use sentence-transformers as fallback
                return self._generate_sentence_transformer_embeddings(sentences)
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _generate_openai_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using OpenAI's embedding API."""
        try:
            # Use the text-embedding-3-small model (cost-effective and fast)
            embeddings = []
            
            # Process in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            return None
    
    def _generate_gemini_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using Google's Gemini embedding API."""
        try:
            embeddings = []
            
            # Use Gemini's embedding model
            for sentence in sentences:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=sentence,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Gemini embeddings error: {e}")
            return None
    
    def _generate_sentence_transformer_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using sentence-transformers (local fallback)."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model
            if not hasattr(self, '_embedding_model'):
                logger.info("Loading sentence-transformer model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self._embedding_model.encode(sentences, show_progress_bar=False)
            return np.array(embeddings)
            
        except ImportError:
            logger.warning("sentence-transformers not available for local embeddings")
            return None
        except Exception as e:
            logger.error(f"Sentence-transformer error: {e}")
            return None
    
    def _group_sentences_by_similarity(self, sentences: List[str], 
                                       embeddings: np.ndarray,
                                       target_chunk_size: int,
                                       similarity_threshold: float) -> List[str]:
        """
        Group sentences into chunks based on semantic similarity.
        
        Algorithm:
        1. Start with the first sentence in a chunk
        2. Add subsequent sentences if they are semantically similar
        3. Start a new chunk when similarity drops below threshold or size limit reached
        """
        chunks = []
        current_chunk = []
        current_chunk_text = ""
        current_chunk_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            sentence_length = len(sentence)
            
            # Start first chunk
            if not current_chunk:
                current_chunk.append(sentence)
                current_chunk_text = sentence
                current_chunk_embedding = embedding
                continue
            
            # Calculate similarity with current chunk
            similarity = self._cosine_similarity(current_chunk_embedding, embedding)
            
            # Decide whether to add to current chunk or start new one
            would_exceed_size = len(current_chunk_text) + sentence_length > target_chunk_size * 1.5
            is_dissimilar = similarity < similarity_threshold
            
            if would_exceed_size or (is_dissimilar and len(current_chunk_text) > target_chunk_size * 0.5):
                # Finalize current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_text = sentence
                current_chunk_embedding = embedding
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_chunk_text += " " + sentence
                
                # Update chunk embedding as running average
                current_chunk_embedding = (current_chunk_embedding * (len(current_chunk) - 1) + embedding) / len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
        

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
    
    def _chunk_document_semantic(self, text: str, chunk_size: int = 2000, 
                                 similarity_threshold: float = 0.5) -> List[str]:
        """
        Split document into semantically coherent chunks using embedding-based similarity.
        
        Args:
            text: The document text to chunk
            chunk_size: Target size for each chunk
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
        
        Returns:
            List of semantically coherent text chunks
        """
        try:
            sentences = self._split_into_sentences(text)
            
            if not sentences or len(sentences) < 2:
                return [text] if text.strip() else []
            
            if not self._can_generate_embeddings():
                logger.info("Falling back to recursive chunking")
                return self._chunk_document_recursive(text, chunk_size)
            
            logger.info(f"Generating embeddings for {len(sentences)} sentences...")
            embeddings = self._generate_sentence_embeddings(sentences)
            
            if embeddings is None:
                return self._chunk_document_recursive(text, chunk_size)
            
            chunks = self._group_sentences_by_similarity(
                sentences, embeddings, chunk_size, similarity_threshold
            )
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self._chunk_document_recursive(text, chunk_size)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling of abbreviations."""
        # Handle common abbreviations to avoid false sentence breaks
        text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs").replace("Ms.", "Ms")
        text = text.replace("Dr.", "Dr").replace("Prof.", "Prof")
        text = text.replace("Inc.", "Inc").replace("Ltd.", "Ltd").replace("Corp.", "Corp")
        text = text.replace("Co.", "Co").replace("vs.", "vs")
        text = text.replace("Rs.", "Rs").replace("Cr.", "Cr").replace("Mn.", "Mn")
        text = text.replace("Fig.", "Fig").replace("Vol.", "Vol").replace("No.", "No")
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def _can_generate_embeddings(self) -> bool:
        """Check if we can generate embeddings with the current LLM provider."""
        # OpenAI provides embeddings API
        if self.provider == "openai" and self.client:
            return True
        
        # Gemini has embedding models
        if self.provider == "gemini" and GEMINI_AVAILABLE:
            return True
        
        # For other providers, we could use sentence-transformers as fallback
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _generate_sentence_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of sentences.
        
        Returns:
            numpy array of shape (num_sentences, embedding_dim) or None if failed
        """
        try:
            if self.provider == "openai" and self.client:
                return self._generate_openai_embeddings(sentences)
            
            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                return self._generate_gemini_embeddings(sentences)
            
            else:
                # Try to use sentence-transformers as fallback
                return self._generate_sentence_transformer_embeddings(sentences)
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _generate_openai_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using OpenAI's embedding API."""
        try:
            # Use the text-embedding-3-small model (cost-effective and fast)
            embeddings = []
            
            # Process in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            return None
    
    def _generate_gemini_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using Google's Gemini embedding API."""
        try:
            embeddings = []
            
            # Use Gemini's embedding model
            for sentence in sentences:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=sentence,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Gemini embeddings error: {e}")
            return None
    
    def _generate_sentence_transformer_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using sentence-transformers (local fallback)."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model
            if not hasattr(self, '_embedding_model'):
                logger.info("Loading sentence-transformer model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self._embedding_model.encode(sentences, show_progress_bar=False)
            return np.array(embeddings)
            
        except ImportError:
            logger.warning("sentence-transformers not available for local embeddings")
            return None
        except Exception as e:
            logger.error(f"Sentence-transformer error: {e}")
            return None
    
    def _group_sentences_by_similarity(self, sentences: List[str], 
                                       embeddings: np.ndarray,
                                       target_chunk_size: int,
                                       similarity_threshold: float) -> List[str]:
        """
        Group sentences into chunks based on semantic similarity.
        
        Algorithm:
        1. Start with the first sentence in a chunk
        2. Add subsequent sentences if they are semantically similar
        3. Start a new chunk when similarity drops below threshold or size limit reached
        """
        chunks = []
        current_chunk = []
        current_chunk_text = ""
        current_chunk_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            sentence_length = len(sentence)
            
            # Start first chunk
            if not current_chunk:
                current_chunk.append(sentence)
                current_chunk_text = sentence
                current_chunk_embedding = embedding
                continue
            
            # Calculate similarity with current chunk
            similarity = self._cosine_similarity(current_chunk_embedding, embedding)
            
            # Decide whether to add to current chunk or start new one
            would_exceed_size = len(current_chunk_text) + sentence_length > target_chunk_size * 1.5
            is_dissimilar = similarity < similarity_threshold
            
            if would_exceed_size or (is_dissimilar and len(current_chunk_text) > target_chunk_size * 0.5):
                # Finalize current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_text = sentence
                current_chunk_embedding = embedding
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_chunk_text += " " + sentence
                
                # Update chunk embedding as running average
                current_chunk_embedding = (current_chunk_embedding * (len(current_chunk) - 1) + embedding) / len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0