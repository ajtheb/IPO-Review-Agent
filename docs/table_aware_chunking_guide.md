# Table-Aware Chunking - Quick Reference Guide

## Overview
The IPO Review Agent now intelligently detects and prioritizes financial tables in prospectus documents for better data extraction.

## Key Features

### 🎯 What Gets Detected
- ✅ Financial tables (P&L, Balance Sheet, Ratios)
- ✅ Multi-year data (FY2023, FY2022, FY2021, etc.)
- ✅ Metric density (concentration of financial numbers)
- ✅ Table dimensions (rows and columns)

### 📊 Prioritization System
**Score Calculation:**
```
Score = 100 (if financial table)
      + 50  (if any table)
      + 30  (if multi-year data)
      + metric_density * 20
      + table_rows * 0.5
```

## Usage

### Basic Usage (Auto-enabled)
```python
from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

# Initialize with vector DB
analyzer = LLMProspectusAnalyzer(provider="gemini", use_vector_db=True)

# Chunk and store (table detection is automatic)
analyzer.chunk_and_store_prospectus(
    pdf_text=prospectus_text,
    company_name="Company Name",
    sector="Sector"
)

# Retrieve with table prioritization (default behavior)
chunks = analyzer.retrieve_relevant_context(
    query="financial metrics revenue profit",
    chunk_type="financial",
    n_results=3
)
```

### Advanced: Retrieve Only Tables
```python
# Get ONLY chunks containing financial tables
table_chunks = analyzer.retrieve_table_chunks(
    query="financial statements P&L balance sheet",
    chunk_type="financial",
    n_results=5,
    only_financial_tables=True  # Filter non-table chunks
)

# Process table chunks
for chunk in table_chunks:
    doc = chunk['document']  # The actual text
    meta = chunk['metadata']  # All metadata
    
    print(f"Table: {meta['table_type']}")
    print(f"Years: {meta['financial_years']}")
    print(f"Rows: {meta['table_row_count']}")
    print(f"Density: {meta['metric_density']}")
```

### Disable Table Prioritization (if needed)
```python
# Retrieve without table prioritization
chunks = analyzer.retrieve_relevant_context(
    query="company overview",
    chunk_type="all",
    n_results=3,
    prioritize_tables=False  # Standard retrieval
)
```

## Metadata Fields

Each chunk now includes these table-aware metadata fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `contains_financial_table` | bool | Highest priority flag | `True` |
| `contains_table` | bool | Any table structure | `True` |
| `table_type` | str | Type classification | `'financial_table'` |
| `table_row_count` | int | Number of rows | `17` |
| `table_column_count` | int | Estimated columns | `3` |
| `has_multi_year_data` | bool | Multi-year presence | `True` |
| `financial_years` | str | Comma-separated years | `'2023,2022,2021'` |
| `metric_density` | float | Metrics per 100 chars | `1.000` |
| `chunk_length` | int | Character count | `1543` |

## Examples

### Example 1: Financial Metric Extraction
```python
# Get chunks with actual financial tables
table_chunks = analyzer.retrieve_table_chunks(
    query="revenue profit margin EBITDA growth",
    chunk_type="financial",
    n_results=5,
    only_financial_tables=True
)

# Build context from tables
context = "\n\n---TABLE---\n\n".join([c['document'] for c in table_chunks])

# Pass to LLM for extraction
# The LLM now sees actual tables with numbers instead of descriptions
```

### Example 2: Multi-Year Analysis
```python
# Find chunks with 3+ years of data
multi_year_chunks = [
    c for c in analyzer.retrieve_table_chunks(
        query="historical financial performance",
        chunk_type="financial",
        n_results=10
    )
    if c['metadata'].get('has_multi_year_data', False)
]

print(f"Found {len(multi_year_chunks)} chunks with multi-year data")
```

### Example 3: Check Chunking Statistics
```python
# After chunking, check what was detected
analyzer.chunk_and_store_prospectus(pdf_text, company_name, sector)

# Console output will show:
# Table Statistics:
#   - Chunks with tables: 5
#   - Chunks with financial tables: 3
#   - Chunks with multi-year data: 3
#   - High-density chunks (>0.5): 7
```

## Debugging

### Check Saved Chunks
All chunks are saved with metadata to `context_chunks/<Company_Name>/`

```bash
# View chunk metadata
cat context_chunks/Vidya_Wires/all_prospectus_chunks_*.txt | head -50
```

### Enable Verbose Logging
```python
from loguru import logger
logger.enable("src.analyzers.llm_prospectus_analyzer")

# Now you'll see:
# "Chunk prioritization (top 5):"
# "  Rank 1: score=208.5, table=True, multi_year=True, density=1.00"
```

### Test Table Detection
```python
# Test on a sample text
test_text = """
Particulars           FY2023    FY2022    FY2021
Revenue              1,234.56   987.23    756.89
EBITDA                234.56   189.45    145.23
"""

table_info = analyzer._detect_table_structure(test_text)
print(f"Contains Table: {table_info['contains_table']}")
print(f"Table Type: {table_info['table_type']}")
print(f"Years: {table_info['financial_years']}")
```

## Performance Tips

### 1. Use Appropriate n_results
```python
# For financial extraction, 3-5 table chunks are usually sufficient
table_chunks = analyzer.retrieve_table_chunks(
    query="financial metrics",
    n_results=5  # Don't request too many
)
```

### 2. Target Specific Collections
```python
# Don't search all collections if you only need financial data
chunks = analyzer.retrieve_relevant_context(
    query="revenue profit",
    chunk_type="financial",  # Not "all"
    n_results=3
)
```

### 3. Use Table-Only Retrieval When Needed
```python
# For metric extraction, use table-only retrieval
table_chunks = analyzer.retrieve_table_chunks(...)  # More focused

# For general analysis, use standard retrieval
all_chunks = analyzer.retrieve_relevant_context(...)  # Broader context
```

## Expected Improvements

### Before Table-Aware Chunking ❌
```
Retrieved chunks:
1. "The company has shown consistent revenue growth..."
2. "Business model focuses on manufacturing..."
3. "Industry outlook remains positive..."
```
**Problem:** Descriptive text, no actual numbers

### After Table-Aware Chunking ✅
```
Retrieved chunks:
1. [Financial Table] Revenue FY23: 11,884.89, FY22: 9,456.78...
2. [Ratio Table] Net Margin: 4.51%, ROE: 18.45%...
3. [Balance Sheet] Total Assets FY23: 8,993.00...
```
**Result:** Actual financial data with numbers

## Troubleshooting

### Issue: No Tables Detected
**Check:**
1. Does the prospectus actually have tables?
2. Are tables formatted with numeric columns?
3. Is text extraction quality good?

**Solution:**
```python
# Manually inspect chunks
with open('context_chunks/<Company>/all_prospectus_chunks_*.txt') as f:
    content = f.read()
    print(content[:500])  # Check first chunk
```

### Issue: Wrong Table Type Detected
**Check:**
1. Are financial keywords present?
2. Is metric density calculation working?

**Solution:**
```python
# Test classification
chunk = "..."  # Your chunk text
chunk_type = analyzer._classify_chunk(chunk)
table_info = analyzer._detect_table_structure(chunk)
density = analyzer._calculate_metric_density(chunk)

print(f"Type: {chunk_type}")
print(f"Table Info: {table_info}")
print(f"Density: {density}")
```

### Issue: Tables Split Across Chunks
**Adjust chunk size:**
```python
# In _chunk_document_recursive(), increase chunk_size
chunks = self._chunk_document_recursive(
    pdf_text, 
    chunk_size=3000,  # Larger chunks
    overlap=300
)
```

## FAQs

**Q: Does this work with all PDF formats?**
A: Works with any text-based PDF. OCR PDFs may have formatting issues.

**Q: Can I disable table detection?**
A: Set `prioritize_tables=False` in `retrieve_relevant_context()`

**Q: What if my prospectus has no tables?**
A: System gracefully falls back to standard chunking and retrieval.

**Q: Does this slow down processing?**
A: Minimal impact (<5% overhead). Table detection is fast regex-based.

**Q: Can I adjust detection sensitivity?**
A: Yes, modify thresholds in `_detect_table_structure()` method.

---

**Last Updated:** March 2, 2026
**Version:** 1.0
