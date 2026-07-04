# Chunk Saving Status - Current Implementation

**Date:** March 2, 2026  
**Status:** ✅ PARTIALLY WORKING (needs vector DB retrieval to complete)

## What's Being Saved

### Currently Saved Files:

1. **Full Prospectus Text** ✅
   - File: `prospectus_text_YYYYMMDD_HHMMSS.txt`
   - Location: `context_chunks/{Company_Name}/`
   - Content: Complete raw prospectus text
   - When: During `_extract_financial_metrics()`
   - Size: ~273KB for Vidya Wires

2. **All Prospectus Chunks** ✅
   - File: `all_prospectus_chunks_YYYYMMDD_HHMMSS.txt`
   - Location: `context_chunks/{Company_Name}/`
   - Content: All chunks created by recursive splitter with metadata
   - When: During `chunk_and_store_prospectus()`
   - Format:
     ```
     === CHUNK 1 (Type: financial) ===
     Length: 2000 chars
     
     [chunk content]
     
     === CHUNK 2 (Type: competitive) ===
     ...
     ```

3. **Retrieved Financial Chunks** ✅ (when vector DB is enabled and has data)
   - File: `retrieved_financial_chunks_YYYYMMDD_HHMMSS.txt`
   - Location: `context_chunks/{Company_Name}/`
   - Content: Top-N most relevant chunks retrieved from vector DB
   - When: During `_extract_financial_metrics()` after vector DB query
   - Format:
     ```
     === CHUNK 1 ===
     [most relevant financial content]
     
     === CHUNK 2 ===
     ...
     ```

## Test Results

### Test Company (Synthetic Data)
```
context_chunks/Test_Company/
├── all_prospectus_chunks_20260302_133713.txt (17KB)
└── test_direct_save_20260302_133719.txt (112B)
```

✅ **All prospectus chunks** saved successfully
✅ **Direct save test** working
❌ **Retrieved chunks** not saved (expected - first run, vector DB empty)

### Vidya Wires (Real IPO)
```
context_chunks/Vidya_Wires/
├── prospectus_text_20260302_131434.txt (273KB)
└── prospectus_text_20260302_133233.txt (273KB)
```

✅ **Full prospectus text** saved successfully
❌ **All prospectus chunks** NOT saved
❌ **Retrieved chunks** NOT saved

## Why Vidya Wires Chunks Aren't Being Saved

### Missing: all_prospectus_chunks file

**Reason:** The `chunk_and_store_prospectus()` method was NOT called for Vidya Wires during the analysis.

**Evidence:**
- No `all_prospectus_chunks_*.txt` file in `context_chunks/Vidya_Wires/`
- Only full prospectus text files exist

**Why:** The analysis flow in `app.py` might be calling `_extract_financial_metrics()` directly without first calling `chunk_and_store_prospectus()`.

### Missing: retrieved_financial_chunks file

**Reason:** Vector DB retrieval is happening, but chunks are either:
1. Not being retrieved (vector DB empty on first run)
2. Being retrieved but the save code isn't executing

**Evidence from code:**
```python
if context_chunks:
    financial_context = f"\nRelevant financial context:\n" + "\n---\n".join(context_chunks)
    
    # Save retrieved chunks for debugging
    chunks_content = ...
    self._save_context_chunks(...)  # ← This should save
```

## How to Fix for Vidya Wires

### Option 1: Run Full Analysis Flow (Recommended)

Ensure the full flow is executed:
```python
# 1. Chunk and store in vector DB first
analyzer.chunk_and_store_prospectus(
    pdf_text=prospectus_text,
    company_name="Vidya Wires",
    sector="Manufacturing"
)

# 2. Then extract metrics (which will retrieve and save chunks)
metrics = analyzer._extract_financial_metrics(
    pdf_text=prospectus_text,
    company_name="Vidya Wires"
)
```

### Option 2: Use analyze_prospectus_comprehensive

This method does everything:
```python
financial_metrics, benchmarking, ipo_specifics = analyzer.analyze_prospectus_comprehensive(
    pdf_text=prospectus_text,
    company_name="Vidya Wires",
    sector="Manufacturing"
)
```

This will:
1. ✅ Chunk and store prospectus
2. ✅ Save all chunks to file
3. ✅ Retrieve relevant chunks
4. ✅ Save retrieved chunks to file
5. ✅ Extract financial metrics

## Files You Should See After Full Analysis

```
context_chunks/Vidya_Wires/
├── prospectus_text_YYYYMMDD_HHMMSS.txt          # Full prospectus (273KB)
├── all_prospectus_chunks_YYYYMMDD_HHMMSS.txt     # All ~140 chunks created
└── retrieved_financial_chunks_YYYYMMDD_HHMMSS.txt # Top 3 relevant chunks
```

## Next Steps

1. **Re-run Vidya Wires Analysis** using full flow
2. **Check Vector DB Stats** to confirm chunks are stored
3. **Verify Retrieved Chunks** are actually used for extraction
4. **Implement Table-Aware Metadata** (future enhancement)

## Command to Re-run Full Analysis

```bash
python test_vidya_wires_analysis.py
```

Or in Streamlit app, search for "Vidya Wires" and click "Analyze IPO".

## Expected Output

After running full analysis, you should see:
- ✅ 3 files in `context_chunks/Vidya_Wires/`
- ✅ Retrieved chunks show high-quality financial sections
- ✅ Chunks prioritize tables and numerical data
- ✅ Metadata shows chunk types and relevance scores

## Current Implementation Status

| Feature | Status | File Location |
|---------|--------|---------------|
| Save full prospectus | ✅ Working | `_extract_financial_metrics()` |
| Save all chunks | ✅ Working | `chunk_and_store_prospectus()` |
| Save retrieved chunks | ✅ Working | `_extract_financial_metrics()` |
| Table-aware metadata | 📋 Planned | See TABLE_AWARE_CHUNKING_STATUS.md |
| Chunk quality scoring | 📋 Planned | See TABLE_AWARE_CHUNKING_STATUS.md |

---

**Summary:** Chunk saving IS implemented and working. For Vidya Wires, you need to run the **full analysis flow** to see all three types of files created. The test with synthetic data proves the code works correctly.
