# Chunk Retrieval Increase for Investment Thesis

## Summary
Increased the number of chunks retrieved in the multi-query retrieval strategy to provide richer context for the investment thesis generation.

## Changes Made

### 1. Financial Metrics Extraction (`_extract_financial_metrics`)
Increased chunk retrieval across 5 specialized queries:

| Query Type | Previous | New | Increase |
|------------|----------|-----|----------|
| P&L Statement | 2 | **4** | +100% |
| Financial Ratios | 2 | **3** | +50% |
| Balance Sheet | 2 | **3** | +50% |
| EBITDA Data | 2 | **3** | +50% |
| Liquidity Ratios | 2 | **3** | +50% |
| **Total** | **10** | **16** | **+60%** |

### 2. Benchmarking Analysis (`_perform_benchmarking_analysis`)
Increased chunk retrieval across 3 specialized queries:

| Query Type | Previous | New | Increase |
|------------|----------|-----|----------|
| Market Position | 2 | **3** | +50% |
| Competitive Advantages | 2 | **3** | +50% |
| Industry Trends | 2 | **3** | +50% |
| **Total** | **6** | **9** | **+50%** |

### 3. IPO Specifics Analysis (`_analyze_ipo_specifics`)
Increased chunk retrieval across 3 specialized queries:

| Query Type | Previous | New | Increase |
|------------|----------|-----|----------|
| IPO Pricing | 2 | **3** | +50% |
| Use of Funds | 2 | **3** | +50% |
| Underwriters | 1 | **2** | +100% |
| **Total** | **5** | **8** | **+60%** |

## Overall Impact

### Total Chunks Retrieved
- **Before**: 21 chunks (10 + 6 + 5)
- **After**: 33 chunks (16 + 9 + 8)
- **Increase**: +12 chunks (+57% overall)

### Investment Thesis Generation
- **Max Tokens**: Increased from 1,200 to 2,000 (+67%)
- **Reason**: Accommodate richer, more detailed analysis with additional context

### Expected Benefits

1. **Richer Financial Context**
   - More P&L data for revenue and profitability analysis
   - Better coverage of financial ratios (especially EBITDA margin, current ratio)
   - More comprehensive balance sheet information

2. **Better Competitive Analysis**
   - More context on market positioning
   - Improved understanding of competitive advantages
   - Richer industry trend information

3. **Enhanced IPO Insights**
   - More details on IPO pricing and valuation
   - Better coverage of fund utilization plans
   - Additional underwriter information

4. **Improved Investment Thesis Quality**
   - More data-backed assertions
   - Better coverage of all analysis sections
   - Reduced instances of "data not available"
   - More specific figures and metrics to cite

### Context Size Impact

Estimated context length increase:
- **Before**: ~15,000-20,000 characters
- **After**: ~24,000-32,000 characters
- **Still well within LLM token limits** (most models support 100K+ tokens)

## Deduplication

All retrieval strategies use hash-based deduplication to:
- Avoid duplicate chunks from overlapping queries
- Ensure unique content in the final context
- Optimize token usage

## Monitoring

To validate the improvements:
1. Check the `context_chunks/` folder for saved retrieval results
2. Look for increased data completeness in financial metrics
3. Review investment thesis for more specific citations
4. Monitor extraction confidence scores

## Next Steps

1. **Test with sample prospectus**: Run analysis on Vidya Wires or similar IPO
2. **Measure improvements**: Compare data completeness before/after
3. **Adjust if needed**: Fine-tune n_results if context becomes too large or sparse
4. **Document results**: Update metrics on extraction accuracy

## Files Modified

- `/Users/apoorvjain/Projects/IPO Review Agent/src/analyzers/llm_prospectus_analyzer.py`
  - `_extract_financial_metrics()`: Lines ~1220-1260
  - `_perform_benchmarking_analysis()`: Lines ~1513-1533
  - `_analyze_ipo_specifics()`: Lines ~1658-1678

## Testing Commands

```bash
# Test with Vidya Wires prospectus
python analyze_vidya_wires_full.py

# Check retrieved chunks
ls -lh context_chunks/Vidya_Wires/

# Validate chunk counts
grep "unique_chunks_retrieved" context_chunks/Vidya_Wires/retrieved_financial_chunks_*.txt
```

---

**Date**: March 3, 2026  
**Status**: ✅ Implemented and ready for testing
