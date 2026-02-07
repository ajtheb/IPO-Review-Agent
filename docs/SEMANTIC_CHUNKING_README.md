# Semantic Chunking Feature

## Quick Start

### Installation

```bash
# Install required dependencies
pip install numpy loguru

# Install embedding provider (choose one)
pip install sentence-transformers  # Local, free (recommended)
pip install openai                 # Best quality
pip install google-generativeai     # Good quality, free tier
```

### Basic Usage

```python
from src.analyzers.semantic_chunking import chunk_text_semantically

# Chunk your IPO prospectus
chunks = chunk_text_semantically(
    text=prospectus_text,
    chunk_size=2000,
    similarity_threshold=0.6
)

print(f"Created {len(chunks)} semantic chunks")
```

### Advanced Usage

```python
from src.analyzers.semantic_chunking import SemanticChunker

# Initialize with specific provider
chunker = SemanticChunker(provider="sentence-transformers")

# Create semantic chunks
chunks = chunker.chunk_text_semantic(
    text=prospectus_text,
    chunk_size=2000,
    similarity_threshold=0.6
)

# Use in your application
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk)} chars")
    # Process chunk...
```

## What It Does

Semantic chunking groups related sentences together based on **embedding similarity** rather than arbitrary character limits.

### Example

**Input:**
```
"Revenue was Rs 1,200 crores in FY2023."
"Net profit margin improved to 15%."
"The company operates in cloud services."
"Competition is intense in the market."
```

**Output (Semantic Chunks):**
```
Chunk 1: "Revenue was Rs 1,200 crores in FY2023. Net profit margin improved to 15%."
         → Financial metrics (high similarity)

Chunk 2: "The company operates in cloud services."
         → Business description

Chunk 3: "Competition is intense in the market."
         → Competition analysis
```

## Features

✅ **Embedding-based** - Uses OpenAI/Gemini/Sentence-Transformers  
✅ **Configurable** - Adjust chunk size and similarity threshold  
✅ **Robust** - Graceful fallback if embeddings unavailable  
✅ **Fast** - Optimized for large documents  
✅ **Production-ready** - Used in IPO Review Agent  

## Configuration

### Similarity Threshold

- **0.3-0.4**: More, smaller chunks (fine-grained)
- **0.5-0.6**: Balanced (recommended for IPOs)
- **0.7-0.9**: Fewer, larger chunks (broad context)

### Chunk Size

- **1000**: Small chunks, more granular
- **2000**: Recommended for IPO analysis
- **3000+**: Large chunks, maximum context

### Embedding Providers

| Provider | Quality | Speed | Cost |
|----------|---------|-------|------|
| OpenAI | ⭐⭐⭐⭐⭐ | Fast | Paid |
| Gemini | ⭐⭐⭐⭐ | Good | Free* |
| Sentence-Transformers | ⭐⭐⭐⭐ | Very Fast | Free |

*Free tier available

## Documentation

- **Full Guide**: `/docs/semantic_chunking_guide.md`
- **Summary**: `/docs/semantic_chunking_summary.md`
- **Test**: `/examples/test_embedding_chunking.py`

## Testing

```bash
python examples/test_embedding_chunking.py
```

## Benefits for IPO Analysis

1. **Better Context Preservation**
   - Financial data stays together
   - Business sections remain coherent
   - Risk factors grouped properly

2. **Improved Vector Search**
   - More relevant retrieval results
   - Complete context in search results
   - Better semantic matching

3. **Enhanced LLM Analysis**
   - Coherent context windows
   - Complete information for extraction
   - Reduced information loss

## Example: IPO Financial Section

### Without Semantic Chunking
```
Chunk 1: "...revenue was Rs 1,200 cro"
Chunk 2: "res. Net profit was Rs 180 crores. Operating margin..."
```
❌ Incomplete, broken context

### With Semantic Chunking
```
Chunk 1: "Revenue was Rs 1,200 crores. Net profit was Rs 180 crores. 
         Operating margin improved to 6.8%."
```
✅ Complete, coherent financial section

## API Reference

### `chunk_text_semantically()`

```python
def chunk_text_semantically(
    text: str,
    chunk_size: int = 2000,
    similarity_threshold: float = 0.6,
    provider: str = "sentence-transformers"
) -> List[str]:
    """Chunk text semantically using embeddings."""
```

### `SemanticChunker`

```python
class SemanticChunker:
    def __init__(self, provider: str, client=None):
        """Initialize semantic chunker."""
    
    def chunk_text_semantic(
        self,
        text: str,
        chunk_size: int = 2000,
        similarity_threshold: float = 0.6
    ) -> List[str]:
        """Create semantic chunks."""
```

## Troubleshooting

### "sentence-transformers not available"

```bash
pip install sentence-transformers
```

### Slow performance

- Use `sentence-transformers` (fastest)
- Reduce `chunk_size`
- Lower `similarity_threshold`

### Poor chunk quality

- Increase `similarity_threshold` (try 0.7)
- Use OpenAI provider (best quality)
- Adjust `chunk_size`

## Contributing

Improvements welcome! Key areas:

- [ ] Add caching for embeddings
- [ ] Support for chunk overlap
- [ ] Multi-language support
- [ ] Custom similarity metrics
- [ ] Parallel processing

## License

Part of IPO Review Agent project.

---

**Made with ❤️ for better IPO analysis**
