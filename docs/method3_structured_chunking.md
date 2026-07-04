# Method 3: Structured PDF Chunker Integration

## Overview

Method 3 enhances the IPO Review Agent's document processing capabilities by integrating the **StructuredPDFChunker** - an intelligent chunking system that creates semantically meaningful document chunks with rich metadata.

## Key Features

### 1. **Section-Aware Chunking**
- Automatically identifies document structure (sections, subsections)
- Preserves hierarchical relationships
- Maintains context boundaries

### 2. **Content Classification**
- Financial data sections
- Business and operations
- Risk factors
- Legal and regulatory
- Market and industry
- Management information

### 3. **Enhanced Metadata**
Each chunk includes:
- **Section/Subsection titles**: Full context of where content comes from
- **Page numbers**: Track exact location in source document
- **Content type**: Automatic classification (financial, business, risk, etc.)
- **Categories**: Multi-label classification for cross-referencing
- **Table detection**: Flags chunks containing tabular data
- **Numeric data detection**: Identifies chunks with financial metrics
- **Word/character counts**: Size metrics for better chunking strategies

## Architecture

```
PDF Document
    ↓
StructuredPDFChunker (scripts/structured_pdf_chunker.py)
    ↓
Categorized Chunks with Metadata
    ↓
LLMProspectusAnalyzer (src/analyzers/llm_prospectus_analyzer.py)
    ↓
Vector Database Storage (ChromaDB)
    ↓
Semantic Retrieval with Enhanced Context
    ↓
Investment Thesis Generation
```

## Usage

### Basic Usage

```python
from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

# Initialize analyzer with vector DB support
analyzer = LLMProspectusAnalyzer(
    provider="gemini",
    use_vector_db=True
)

# Process PDF with structured chunking (Method 3)
analyzer.chunk_and_store_prospectus_structured(
    pdf_path="path/to/prospectus.pdf",
    company_name="Company Name",
    sector="Industry Sector",
    use_structured=True  # Enable Method 3
)

# Retrieve relevant chunks for analysis
prospectus_chunks, web_chunks = analyzer.retrieve_relevant_chunks_for_thesis(
    company_name="Company Name",
    sector="Industry Sector",
    n_prospectus=15,
    n_web=5
)

# Generate investment thesis with structured context
thesis = analyzer.generate_investment_thesis(
    financial_metrics=metrics,
    benchmarking=benchmarking,
    ipo_specifics=ipo_specifics,
    company_name="Company Name",
    sector="Industry Sector"
)
```

### Advanced Usage: Direct Structured Chunking

```python
# Extract structured chunks without vector DB storage
categorized_chunks = analyzer._chunk_document_structured(
    pdf_path="path/to/prospectus.pdf",
    company_name="Company Name"
)

# Access chunks by category
financial_chunks = categorized_chunks['financial']
business_chunks = categorized_chunks['business']
risk_chunks = categorized_chunks['risk']

# Examine metadata
for chunk_data in financial_chunks:
    text = chunk_data['text']
    metadata = chunk_data['metadata']
    
    print(f"Section: {metadata['section']}")
    print(f"Pages: {metadata['pages']}")
    print(f"Has tables: {metadata['has_tables']}")
    print(f"Word count: {metadata['word_count']}")
```

## Chunk Categories

### Primary Categories

| Category | Description | Vector DB Collection |
|----------|-------------|---------------------|
| `financial` | Revenue, profit, balance sheet data | financial_sections |
| `business` | Operations, products, services | competitive_sections |
| `risk` | Risk factors, warnings | prospectus_chunks |
| `legal` | Regulatory, compliance, litigation | prospectus_chunks |
| `market` | Industry trends, competition | competitive_sections |
| `management` | Directors, governance | competitive_sections |
| `general` | Other content | prospectus_chunks |

### Content Type Classification

The structured chunker automatically classifies content based on:
- **Keywords**: Financial terms, risk-related phrases, business terms
- **Section headers**: Recognized prospectus sections
- **Content patterns**: Numeric data, tables, regulatory language

## Metadata Schema

```python
{
    'text': str,  # The chunk content
    'metadata': {
        'categories': List[str],        # Multi-label categories
        'content_type': str,            # Primary content type
        'section': str,                 # Main section title
        'subsection': str,              # Subsection title (if any)
        'pages': List[int],             # Page numbers spanned
        'page_start': int,              # First page
        'page_end': int,                # Last page
        'chunk_id': int,                # Unique chunk identifier
        'char_count': int,              # Character count
        'word_count': int,              # Word count
        'has_tables': bool,             # Contains tabular data
        'has_numbers': bool,            # Contains numeric data
        'source': 'structured_chunker'  # Processing method marker
    }
}
```

## Benefits Over Simple Text Splitting

### Method 1 & 2: Basic Text Splitting
- ❌ Breaks at arbitrary character/sentence boundaries
- ❌ No section awareness
- ❌ Limited metadata
- ❌ Context loss across chunks

### Method 3: Structured Chunking
- ✅ Respects document structure
- ✅ Preserves section context
- ✅ Rich metadata for filtering
- ✅ Better semantic coherence
- ✅ Improved retrieval accuracy
- ✅ Enhanced LLM context quality

## Integration with Vector Database

Method 3 stores structured chunks in ChromaDB with enhanced metadata:

```python
# Metadata stored in vector DB
{
    "company": "Company Name",
    "sector": "Industry Sector",
    "chunk_type": "financial",         # Mapped category
    "chunk_index": 0,
    "ipo_date": "2024-01-15",
    "timestamp": "2024-01-15T10:30:00",
    "categories": ["financial", "business"],
    "content_type": "financial",
    "section": "FINANCIAL INFORMATION",
    "subsection": "Revenue Analysis",
    "pages": [45, 46, 47],
    "page_start": 45,
    "page_end": 47,
    "char_count": 2500,
    "word_count": 450,
    "has_tables": true,
    "has_numbers": true,
    "source": "structured_chunker"
}
```

### Retrieval Advantages

1. **Metadata Filtering**: Query specific sections or page ranges
2. **Content Type Targeting**: Retrieve only financial or risk chunks
3. **Page Tracking**: Cite exact prospectus pages in analysis
4. **Table Awareness**: Prioritize chunks with tabular data
5. **Section Context**: Understand chunk provenance

## Performance Considerations

### Chunk Size Configuration

```python
chunker.process_document(
    min_chunk_size=200,   # Minimum chunk size
    max_chunk_size=2000   # Maximum chunk size (configurable)
)
```

**Recommendations:**
- **Financial sections**: Smaller chunks (1000-1500) for precise data
- **Business descriptions**: Larger chunks (2000-3000) for context
- **Risk factors**: Medium chunks (1500-2000) for complete statements

### Processing Time

For a typical 200-page IPO prospectus:
- **Method 1/2**: ~5-10 seconds (simple text splitting)
- **Method 3**: ~20-30 seconds (structure extraction + classification)

**Trade-off**: Slightly longer processing time for significantly better chunk quality.

## Fallback Mechanism

Method 3 includes automatic fallback:

```python
# If structured chunking fails or is unavailable
# - Automatically falls back to Method 2 (recursive text splitting)
# - Logs warning but continues processing
# - Maintains compatibility
```

## Testing

Use the provided test script to validate integration:

```bash
python test_method3_integration.py
```

**Test Coverage:**
- ✅ Structured chunk extraction
- ✅ Category classification
- ✅ Metadata preservation
- ✅ Vector DB storage
- ✅ Semantic retrieval
- ✅ Investment thesis generation

## Troubleshooting

### Issue: "Structured PDF Chunker not available"

**Solution:**
```bash
# Ensure structured_pdf_chunker.py is in scripts/ directory
ls scripts/structured_pdf_chunker.py

# Verify dependencies
pip install PyPDF2 pdfplumber
```

### Issue: No chunks extracted

**Possible causes:**
1. PDF is encrypted or corrupted
2. PDF contains only images (scanned document)
3. Insufficient permissions to read file

**Solutions:**
- Try different PDF reader library (PyPDF2 vs pdfplumber)
- Use OCR for scanned documents
- Check file permissions

### Issue: Missing metadata fields

**Solution:**
- Ensure using latest version of StructuredPDFChunker
- Check that `process_document()` is called before accessing chunks
- Verify chunker initialization parameters

## Best Practices

1. **Pre-processing**
   - Validate PDF is text-based (not scanned images)
   - Check file size (large files may need more processing time)
   - Ensure file path is absolute

2. **Configuration**
   - Use larger `max_chunk_size` for narrative sections
   - Use smaller sizes for dense financial tables
   - Adjust based on LLM context window

3. **Retrieval**
   - Leverage metadata filters for targeted queries
   - Use category-specific queries for better results
   - Combine prospectus and web chunks for comprehensive context

4. **Storage**
   - Clear vector DB before processing new document
   - Use company name + timestamp for unique identifiers
   - Backup structured chunks to disk for recovery

## Future Enhancements

- [ ] PDF page extraction for visual context
- [ ] Table structure preservation
- [ ] Chart and graph detection
- [ ] Multi-document cross-referencing
- [ ] Importance scoring for chunks
- [ ] Automatic summary generation per section
- [ ] Relationship mapping between sections

## References

- **StructuredPDFChunker**: `scripts/structured_pdf_chunker.py`
- **Integration code**: `src/analyzers/llm_prospectus_analyzer.py` (lines 966-1200)
- **Test script**: `test_method3_integration.py`
- **Vector DB setup**: ChromaDB with BAAI/bge-small-en-v1.5 embeddings

## Support

For issues or questions:
1. Check logs in `app.log` or `cli.log`
2. Run test script with verbose logging
3. Verify structured chunker availability
4. Review metadata schema compatibility

---

**Version**: 1.0  
**Last Updated**: 2024-01-15  
**Status**: Production Ready ✅
