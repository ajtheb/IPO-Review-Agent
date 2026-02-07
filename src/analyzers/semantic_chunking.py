"""
Semantic Chunking Module for IPO Prospectus Analysis

This module provides semantic chunking functionality using embedding-based
cosine similarity to create more meaningful document chunks.
"""

import re
import numpy as np
from typing import List, Optional
from loguru import logger

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticChunker:
    """Handles semantic chunking of documents using embeddings."""
    
    def __init__(self, provider: str = "sentence-transformers", client=None):
        """
        Initialize the semantic chunker.
        
        Args:
            provider: Embedding provider ('openai', 'gemini', 'sentence-transformers')
            client: Optional pre-configured client for the provider
        """
        self.provider = provider
        self.client = client
        self.embedding_model = None
        
        # Load sentence-transformers model if using that provider
        if provider == "sentence-transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading sentence-transformer model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence-transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model: {e}")
    
    def chunk_text_semantic(self, text: str, 
                           chunk_size: int = 2000,
                           similarity_threshold: float = 0.5) -> List[str]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: The document text to chunk
            chunk_size: Target size for each chunk (approximate)
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
        
        Returns:
            List of semantically coherent text chunks
        """
        try:
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            if not sentences or len(sentences) < 2:
                return [text] if text.strip() else []
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(sentences)} sentences...")
            embeddings = self._generate_embeddings(sentences)
            
            if embeddings is None or len(embeddings) != len(sentences):
                logger.warning("Failed to generate embeddings, using fallback chunking")
                return self._fallback_chunking(text, chunk_size)
            
            # Group sentences by similarity
            chunks = self._group_by_similarity(
                sentences, embeddings, chunk_size, similarity_threshold
            )
            
            avg_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            logger.info(f"Created {len(chunks)} semantic chunks (avg size: {avg_size:.0f} chars)")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self._fallback_chunking(text, chunk_size)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling common abbreviations."""
        # Handle common abbreviations to avoid false sentence breaks
        abbreviations = {
            "Mr.": "Mr", "Mrs.": "Mrs", "Ms.": "Ms", "Dr.": "Dr", "Prof.": "Prof",
            "Inc.": "Inc", "Ltd.": "Ltd", "Corp.": "Corp", "Co.": "Co",
            "Rs.": "Rs", "Cr.": "Cr", "Mn.": "Mn",
            "Fig.": "Fig", "Vol.": "Vol", "No.": "No", "vs.": "vs"
        }
        
        for abbrev, replacement in abbreviations.items():
            text = text.replace(abbrev, replacement)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences (min length 10 chars)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def _generate_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a list of sentences."""
        try:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                return self._generate_openai_embeddings(sentences)
            
            elif self.provider == "gemini" and GEMINI_AVAILABLE:
                return self._generate_gemini_embeddings(sentences)
            
            elif self.provider == "sentence-transformers" and self.embedding_model:
                return self._generate_transformer_embeddings(sentences)
            
            else:
                logger.warning(f"Provider '{self.provider}' not available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _generate_openai_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using OpenAI's embedding API."""
        try:
            embeddings = []
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
        """Generate embeddings using Gemini's embedding API."""
        try:
            embeddings = []
            
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
    
    def _generate_transformer_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using sentence-transformers."""
        try:
            embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Sentence-transformer error: {e}")
            return None
    
    def _group_by_similarity(self, sentences: List[str], 
                            embeddings: np.ndarray,
                            target_size: int,
                            threshold: float) -> List[str]:
        """
        Group sentences into chunks based on semantic similarity.
        
        Algorithm:
        1. Start with first sentence
        2. Add subsequent sentences if semantically similar
        3. Start new chunk when similarity drops or size limit reached
        """
        chunks = []
        current_sentences = []
        current_text = ""
        current_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            # Start first chunk
            if not current_sentences:
                current_sentences.append(sentence)
                current_text = sentence
                current_embedding = embedding
                continue
            
            # Calculate similarity with current chunk
            similarity = self._cosine_similarity(current_embedding, embedding)
            
            # Check if we should start a new chunk
            would_exceed = len(current_text) + len(sentence) > target_size * 1.5
            is_dissimilar = similarity < threshold
            has_min_size = len(current_text) > target_size * 0.5
            
            if would_exceed or (is_dissimilar and has_min_size):
                # Finalize current chunk
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_text = sentence
                current_embedding = embedding
            else:
                # Add to current chunk
                current_sentences.append(sentence)
                current_text += " " + sentence
                
                # Update embedding as running average
                n = len(current_sentences)
                current_embedding = (current_embedding * (n - 1) + embedding) / n
        
        # Add final chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _fallback_chunking(self, text: str, chunk_size: int) -> List[str]:
        """Simple fallback chunking when embeddings are not available."""
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]


# Standalone function for easy use
def chunk_text_semantically(text: str, 
                            chunk_size: int = 2000,
                            similarity_threshold: float = 0.5,
                            provider: str = "sentence-transformers") -> List[str]:
    """
    Convenience function to chunk text semantically.
    
    Args:
        text: Document text to chunk
        chunk_size: Target chunk size in characters
        similarity_threshold: Similarity threshold for grouping (0.0-1.0)
        provider: Embedding provider to use
    
    Returns:
        List of semantic chunks
    """
    chunker = SemanticChunker(provider=provider)
    return chunker.chunk_text_semantic(text, chunk_size, similarity_threshold)
