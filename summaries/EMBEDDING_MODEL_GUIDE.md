# Embedding Model Configuration Guide

## Overview

The IPO Review Agent uses **BAAI/bge-small-en-v1.5** as the embedding model for semantic search and vector database operations. This model provides high-quality embeddings optimized for retrieval and semantic similarity tasks.

## Why BAAI/bge-small-en-v1.5?

### Advantages
1. **High Quality**: State-of-the-art performance on semantic search benchmarks
2. **Efficient**: Small model size (134M parameters) with 384-dimensional embeddings
3. **Fast**: Quick inference for real-time applications
4. **Multilingual**: Supports multiple languages (useful for international IPOs)
5. **Optimized for Retrieval**: Specifically designed for information retrieval tasks
6. **Open Source**: Free to use with no API costs

### Performance Metrics
- **Model Size**: ~134M parameters
- **Embedding Dimension**: 384
- **MTEB Score**: 62.37 (Average across tasks)
- **Retrieval Performance**: Top-tier results on retrieval benchmarks
- **Speed**: ~1000 sentences/second on CPU

## Configuration

### Installation

The embedding model is automatically installed with the project dependencies:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (includes sentence-transformers)
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
sentence-transformers>=2.2.0  # Required for BAAI/bge-small-en-v1.5 embeddings
```

### Code Configuration

The embedding model is configured in `src/analyzers/llm_prospectus_analyzer.py`:

```python
from chromadb.utils import embedding_functions

class LLMProspectusAnalyzer:
    def __init__(self, provider: str = "openai", use_vector_db: bool = True, 
                 db_path: str = "./ipo_vector_db"):
        # ... other initialization ...
        
        if self.use_vector_db:
            # Create embedding function with BAAI/bge-small-en-v1.5
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Initialize ChromaDB with the embedding function
            self.vector_client = chromadb.PersistentClient(path=db_path)
            self._setup_vector_collections()
```

All vector collections (prospectus, web content, search results, scraped content) use this same embedding function for consistency.

## Verification

### Test the Configuration

Run the validation test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run embedding model tests
python tests/test_embedding_model.py
```

Expected output:
```
✅ TEST 1: Embedding Model Installation - PASSED
✅ TEST 2: ChromaDB Embedding Integration - PASSED
✅ TEST 3: Analyzer Embedding Configuration - PASSED
✅ TEST 4: Vector Similarity Search - PASSED

🎉 ALL TESTS PASSED - Embedding model is properly configured!
```

### Manual Verification

```python
from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
import os

# Set API key
os.environ['GEMINI_API_KEY'] = 'your_key_here'

# Initialize analyzer
analyzer = LLMProspectusAnalyzer(
    provider="gemini",
    use_vector_db=True
)

# Check embedding function
if analyzer.embedding_function:
    test_text = ["Test financial analysis"]
    embeddings = analyzer.embedding_function(test_text)
    print(f"Embedding dimension: {len(embeddings[0])}")  # Should be 384
    print("✅ Using BAAI/bge-small-en-v1.5")
```

## Usage in Vector Database

### Automatic Usage

The embedding model is used automatically when:

1. **Storing Documents**:
```python
analyzer.chunk_and_store_prospectus(
    pdf_text=prospectus_text,
    company_name="ABC Company",
    sector="Technology"
)
# Automatically generates embeddings for each chunk
```

2. **Retrieving Documents**:
```python
prospectus_chunks, web_chunks = analyzer.retrieve_relevant_chunks_for_thesis(
    company_name="ABC Company",
    sector="Technology",
    n_prospectus=10,
    n_web=10
)
# Query is embedded and matched against stored embeddings
```

### Behind the Scenes

When you store a document:
1. Text is chunked into segments
2. Each chunk is embedded using BAAI/bge-small-en-v1.5
3. 384-dimensional vectors are stored in ChromaDB
4. Metadata is associated with each vector

When you query:
1. Query text is embedded using the same model
2. Vector similarity search finds closest matches
3. Top-N most relevant chunks are returned
4. Semantic relevance ensures quality results

## Performance Optimization

### Model Caching

The model is cached after first load:
- **First Run**: Downloads model (~130MB) and loads into memory
- **Subsequent Runs**: Uses cached model (instant loading)
- **Cache Location**: `~/.cache/huggingface/hub/`

### Batch Processing

For efficiency, embed multiple texts together:

```python
# Efficient: Single batch
texts = ["text1", "text2", "text3"]
embeddings = embedding_function(texts)  # Fast

# Inefficient: Multiple calls
for text in texts:
    embedding = embedding_function([text])  # Slower
```

### Hardware Acceleration

The model can use GPU if available:
- CPU: ~1000 sentences/second
- GPU: ~5000+ sentences/second

Install GPU support:
```bash
# For CUDA (NVIDIA)
pip install sentence-transformers[gpu]
```

## Comparison with Other Models

| Model | Dimension | Size | MTEB Score | Use Case |
|-------|-----------|------|------------|----------|
| **BAAI/bge-small-en-v1.5** | 384 | 134M | 62.37 | ⭐ Recommended (balanced) |
| BAAI/bge-base-en-v1.5 | 768 | 435M | 63.55 | Higher accuracy, slower |
| BAAI/bge-large-en-v1.5 | 1024 | 1.34B | 64.23 | Best accuracy, slowest |
| all-MiniLM-L6-v2 | 384 | 22M | 58.04 | Faster, lower quality |
| OpenAI text-embedding-3-small | 1536 | - | ~65 | API-based, costs money |

**Why we chose bge-small-en-v1.5:**
- ✅ Best balance of quality and speed
- ✅ No API costs
- ✅ Runs locally
- ✅ Optimized for retrieval tasks
- ✅ 384 dimensions (efficient storage)

## Troubleshooting

### Issue: Model download fails

**Solution**: Check internet connection and try manual download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
```

### Issue: Out of memory

**Solutions**:
1. Reduce batch size when embedding
2. Clear ChromaDB collections when switching IPOs
3. Use smaller model (all-MiniLM-L6-v2)

### Issue: Slow embedding generation

**Solutions**:
1. Enable GPU support (if available)
2. Increase batch size for bulk operations
3. Cache frequently used embeddings

### Issue: Poor semantic search results

**Solutions**:
1. Verify model is loaded correctly (run test suite)
2. Adjust chunk size (default: 2000 chars)
3. Increase number of retrieved chunks (n_prospectus, n_web)
4. Check query phrasing (use financial terminology)

## Advanced Configuration

### Using a Different Model

If you need to switch models:

```python
# In llm_prospectus_analyzer.py, replace:
self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"  # Change this
)

# Options:
# - "BAAI/bge-base-en-v1.5" (more accurate, slower)
# - "BAAI/bge-large-en-v1.5" (best accuracy, slowest)
# - "all-MiniLM-L6-v2" (faster, less accurate)
```

⚠️ **Important**: Changing models requires clearing existing vector DB:
```python
analyzer.clear_vector_database()  # Clear old embeddings
# Re-chunk and store with new model
```

### Custom Embedding Function

For advanced use cases:

```python
from sentence_transformers import SentenceTransformer

# Create custom model wrapper
class CustomEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

# Use in analyzer
self.embedding_function = CustomEmbedding("BAAI/bge-small-en-v1.5")
```

## Best Practices

1. ✅ **Keep Same Model**: Don't change models mid-analysis
2. ✅ **Clear DB**: Clear vector DB when switching models
3. ✅ **Batch Process**: Embed multiple texts together for efficiency
4. ✅ **Monitor Performance**: Check embedding quality in results
5. ✅ **Cache Model**: Let the model cache on first run
6. ✅ **Test First**: Run validation tests after installation

## Model Details

### BGE (BAAI General Embedding) Family

**Developer**: Beijing Academy of Artificial Intelligence (BAAI)

**Training Data**:
- C-MTEB (Chinese Multi-Task Embedding Benchmark)
- Massive Text Embedding Benchmark (MTEB)
- Diverse retrieval and semantic similarity tasks

**Architecture**:
- Based on BERT/RoBERTa
- Fine-tuned specifically for retrieval
- Mean pooling with normalization
- Optimized for cosine similarity

**License**: MIT License (free for commercial use)

### Performance on Financial Data

While not specifically trained on financial data, bge-small-en-v1.5 performs excellently on:
- ✅ Financial terminology (revenue, profit, margins, etc.)
- ✅ Corporate documents (prospectuses, reports)
- ✅ Numerical data in context
- ✅ Risk and opportunity analysis
- ✅ Market sentiment and trends

## References

- **Model Card**: https://huggingface.co/BAAI/bge-small-en-v1.5
- **Paper**: "C-Pack: Packaged Resources To Advance General Chinese Embedding"
- **Benchmark**: Massive Text Embedding Benchmark (MTEB)
- **Documentation**: sentence-transformers library

## Support

For issues with the embedding model:
1. Run `python tests/test_embedding_model.py`
2. Check logs for embedding dimension (should be 384)
3. Verify sentence-transformers installation
4. Review ChromaDB configuration

---

**Your IPO Review Agent is powered by state-of-the-art embeddings!** 🚀
