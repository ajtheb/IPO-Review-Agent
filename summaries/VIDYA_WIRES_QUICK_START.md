# Quick Start: Analyzing Vidya Wires IPO

## Prerequisites

1. **GROQ API Key** must be set:
   ```bash
   export GROQ_API_KEY='your_groq_api_key_here'
   ```

2. **Python Environment** activated:
   ```bash
   source .venv/bin/activate  # If using virtual environment
   ```

## Option 1: Run Test Script (Recommended)

This comprehensive test will verify everything is working:

```bash
python test_vidya_wires_analysis.py
```

**What it does:**
- ✅ Checks for GROQ_API_KEY
- ✅ Initializes LLM analyzer
- ✅ Searches for Vidya Wires prospectus
- ✅ Extracts financial metrics using LLM
- ✅ Saves context chunks to `context_chunks/Vidya Wires/`
- ✅ Displays extraction summary

**Expected Output:**
```
================================================================================
VIDYA WIRES IPO - LLM ANALYSIS & CONTEXT CHUNK SAVING TEST
================================================================================

✓ GROQ_API_KEY found
✓ Analyzer initialized successfully
✓ Parser initialized successfully
✓ Found 1 prospectus result(s)
✓ Prospectus text loaded (XXX,XXX characters)
✓ Financial metrics extraction completed
✓ Context chunks saved successfully!
  Directory: /Users/.../context_chunks/Vidya Wires
  Text chunks: X files
  Metadata files: X files
```

## Option 2: Run Streamlit App

Launch the full IPO Review Agent interface:

```bash
streamlit run app.py
```

Then:
1. Enter "Vidya Wires" in the company search
2. Click "Analyze IPO"
3. Wait for analysis to complete
4. Check `context_chunks/Vidya Wires/` directory

## Option 3: Python Script

```python
from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusParser

# Initialize
analyzer = LLMProspectusAnalyzer(provider='groq', use_vector_db=False)
parser = EnhancedProspectusParser()

# Search and analyze
results = parser.search_prospectus("Vidya Wires")
if results:
    prospectus = results[0]
    metrics = analyzer._extract_financial_metrics(
        prospectus_text=prospectus['full_text'],
        company_name="Vidya Wires"
    )
    print(f"Extracted {len(metrics)} metrics")
```

## Verify Context Chunks Were Saved

```bash
ls -lh context_chunks/Vidya\ Wires/
```

**You should see:**
- `prospectus_full.txt` - Complete prospectus text
- `prospectus_metadata.json` - Analysis metadata
- `financial_metrics_chunks_*.txt` - Relevant financial chunks
- `financial_metrics_metadata_*.json` - Chunk metadata

## Check Extraction Results

The test script will display:
- Total metrics extracted
- Data completeness percentage
- Extraction confidence score
- Key financial metrics (revenue, profit, growth rates)

## Troubleshooting

### No Prospectus Found

If you see "No prospectus found for 'Vidya Wires'":

```python
from src.data_sources.enhanced_prospectus_parser import EnhancedProspectusParser

parser = EnhancedProspectusParser()
parser.fetch_and_parse_prospectus("Vidya Wires")
```

### API Key Error

If you see "GROQ_API_KEY not found":

```bash
export GROQ_API_KEY='your_key_here'
```

### Import Errors

Make sure you're using the virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## What Gets Saved

### Context Chunks Directory Structure

```
context_chunks/
└── Vidya Wires/
    ├── prospectus_full.txt          # Complete prospectus text
    ├── prospectus_metadata.json     # Company info, analysis metadata
    ├── financial_metrics_chunks_1.txt   # Financial statement chunks
    ├── financial_metrics_metadata_1.json
    ├── financial_metrics_chunks_2.txt
    └── financial_metrics_metadata_2.json
```

### Metadata Includes

- Company name
- Total chunks
- Chunk size
- Analysis type
- Timestamp
- Token counts
- Section information

## Performance Notes

- **Analysis Time:** 30-60 seconds (depends on prospectus size)
- **Context Window:** Optimized for Groq's 8K token limit
- **Chunk Size:** 3000 characters with 500-character overlap
- **Chunks Retrieved:** 3 high-quality chunks per query

## Recent Fixes Applied

✅ **Syntax Error Fixed** - Line 1807 in `llm_prospectus_analyzer.py`  
✅ **Context Chunk Saving** - Integrated into financial metrics extraction  
✅ **Groq Optimization** - Reduced context size for faster processing  
✅ **Table-Aware Chunking** - Prioritizes financial statement tables  

See `SYNTAX_ERROR_FIX.md` for detailed fix information.
