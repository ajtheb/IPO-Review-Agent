# LLM Provider Optimizations

## Overview

The IPO Review Agent now includes intelligent optimizations for different LLM providers to handle their specific rate limits, token constraints, and pricing models.

## Provider-Specific Configurations

### 1. **Groq** (Cost-Effective, Rate-Limited)

**Characteristics:**
- **Token Limits**: 6,000 TPM (tokens per minute) for 8B models
- **Context Window**: 8,192 tokens (8B instant), 32,768 tokens (70B versatile)
- **Pricing**: Free tier / very low cost
- **Speed**: Very fast inference

**Optimizations Applied:**
```python
# Investment Thesis Generation
n_prospectus = 30          # Reduced from 50
n_web = 15                 # Reduced from 30
prospectus_chunks_to_use = 15  # Reduced from 25
web_chunks_to_use = 8      # Reduced from 15
chars_per_prospectus_chunk = 800   # Reduced from 1200
chars_per_web_chunk = 600  # Reduced from 1000

# Max tokens for output
max_tokens = 1000 (8B model)  # Reduced from 1200-1500
```

**Model Fallback Chain:**
1. `llama-3.1-70b-versatile` (Primary - higher rate limits)
2. `llama-3.1-8b-instant` (Backup - faster but lower limits)

**Error Handling:**
- Rate limit (413) errors → Try next model
- Context length errors → Automatic fallback
- Decommissioned models → Skip to next available

### 2. **OpenAI** (High Quality, Higher Cost)

**Characteristics:**
- **Token Limits**: 90,000 TPM (GPT-4), 10,000,000 TPM (GPT-3.5)
- **Context Window**: 8,192 tokens (GPT-4), 16,385 tokens (GPT-4-turbo)
- **Pricing**: $0.01-0.03 per 1K tokens
- **Speed**: Moderate

**Optimizations:**
```python
# Full context available
n_prospectus = 50
n_web = 30
prospectus_chunks_to_use = 25
web_chunks_to_use = 15
chars_per_prospectus_chunk = 1200
chars_per_web_chunk = 1000
max_tokens = 1500
```

### 3. **Anthropic Claude** (Balanced)

**Characteristics:**
- **Token Limits**: 40,000 TPM (Claude-3-Sonnet)
- **Context Window**: 200,000 tokens
- **Pricing**: $0.003-0.015 per 1K tokens
- **Speed**: Moderate to fast

**Optimizations:**
```python
# Full context available (same as OpenAI)
n_prospectus = 50
n_web = 30
max_tokens = 1500
```

### 4. **Google Gemini** (Large Context)

**Characteristics:**
- **Token Limits**: Variable based on tier
- **Context Window**: 1,000,000 tokens (Gemini Pro)
- **Pricing**: Free tier available, then $0.00025-0.0005 per 1K tokens
- **Speed**: Fast

**Optimizations:**
```python
# Can handle very large context
n_prospectus = 50
n_web = 30
max_tokens = 1500
```

## Context Size Comparison

### Estimated Token Counts

**Groq Configuration:**
- Prospectus: 15 chunks × 800 chars = ~12,000 chars (~3,000 tokens)
- Web: 8 chunks × 600 chars = ~4,800 chars (~1,200 tokens)
- Structured data (metrics, benchmarking, IPO specifics): ~1,500 tokens
- **Total Input**: ~5,700 tokens
- **Output**: 1,000 tokens
- **Grand Total**: ~6,700 tokens (fits within limits with retry logic)

**OpenAI/Anthropic/Gemini Configuration:**
- Prospectus: 25 chunks × 1,200 chars = ~30,000 chars (~7,500 tokens)
- Web: 15 chunks × 1,000 chars = ~15,000 chars (~3,750 tokens)
- Structured data: ~1,500 tokens
- **Total Input**: ~12,750 tokens
- **Output**: 1,500 tokens
- **Grand Total**: ~14,250 tokens (comfortable margin)

## Automatic Provider Detection

The system automatically adjusts chunk sizes based on the provider:

```python
if self.provider == "groq":
    # Use reduced context
    n_prospectus = 30
    n_web = 15
    # ... reduced sizes
    logger.info("Using reduced context for Groq provider")
else:
    # Use full context for other providers
    n_prospectus = 50
    n_web = 30
    # ... full sizes
```

## Error Handling Strategy

### Groq-Specific Errors

1. **Rate Limit (413)**
   ```
   Error code: 413 - Request too large for model
   ```
   **Action**: Automatically switch to 70B model (higher limits) or reduce context

2. **Context Length Exceeded**
   ```
   Error code: 400 - Context length exceeded
   ```
   **Action**: Fall back to next model with larger context window

3. **Decommissioned Model**
   ```
   Model not found or decommissioned
   ```
   **Action**: Skip to next available model in chain

### Retry Logic

```python
for model_name in models_to_try:
    try:
        # Attempt with current model
        response = self.client.chat.completions.create(...)
        return response  # Success!
    except RateLimitError:
        continue  # Try next model
    except ContextLengthError:
        continue  # Try next model
```

## Performance Metrics

### Groq (Optimized)
- ✅ **Speed**: ~1-2 seconds per request
- ✅ **Cost**: Free / Very low
- ⚠️ **Context**: Limited (6,000 tokens)
- ✅ **Quality**: Good for structured extraction

### OpenAI GPT-4
- ⚠️ **Speed**: ~10-15 seconds per request
- ❌ **Cost**: Higher ($0.01-0.03 per 1K tokens)
- ✅ **Context**: Large (8K-128K tokens)
- ✅ **Quality**: Excellent

### Anthropic Claude
- ✅ **Speed**: ~5-8 seconds per request
- ✅ **Cost**: Moderate ($0.003-0.015 per 1K tokens)
- ✅ **Context**: Very large (200K tokens)
- ✅ **Quality**: Excellent

### Google Gemini
- ✅ **Speed**: ~3-5 seconds per request
- ✅ **Cost**: Very low (free tier available)
- ✅ **Context**: Massive (1M tokens)
- ✅ **Quality**: Very good

## Usage Recommendations

### For Development/Testing
**Use Groq** (llama-3.1-70b-versatile)
- Fast iteration
- No cost concerns
- Sufficient quality for testing

### For Production (Cost-Sensitive)
**Use Groq** (with fallback chain) or **Gemini**
- Best price/performance ratio
- Good quality for structured data extraction
- Fast response times

### For Production (Quality-Critical)
**Use OpenAI GPT-4** or **Anthropic Claude**
- Best analysis quality
- More nuanced understanding
- Better handling of complex financial terminology

### For High-Volume Processing
**Use Groq** or **Gemini**
- No rate limit concerns (with Gemini)
- Batch processing support
- Lower costs at scale

## Configuration Example

```python
# Initialize with provider-aware configuration
analyzer = LLMProspectusAnalyzer(
    provider="groq",  # Auto-optimizes for Groq limits
    use_vector_db=True
)

# Generate thesis - automatically adjusts context size
thesis = analyzer.generate_investment_thesis(
    financial_metrics=metrics,
    benchmarking=benchmarking,
    ipo_specifics=ipo_specifics,
    company_name="Company Name",
    sector="Technology"
)
```

## Monitoring and Logging

The system logs provider-specific information:

```
INFO - Using reduced context for Groq provider (token limit: 6000)
INFO - Calling Groq model llama-3.1-70b-versatile with max_tokens=1000
WARNING - Groq model llama-3.1-8b-instant rate limit exceeded, trying next model...
INFO - Groq model llama-3.1-70b-versatile succeeded
```

## Future Enhancements

1. **Dynamic Token Budgeting**
   - Real-time token counting
   - Adaptive chunk sizing based on available budget

2. **Intelligent Model Selection**
   - Cost vs. quality trade-offs
   - Automatic provider switching based on task complexity

3. **Caching Layer**
   - Cache common analyses
   - Reduce API calls for similar queries

4. **Batch Processing**
   - Queue multiple IPO analyses
   - Optimize for rate limits across batches

## Troubleshooting

### Issue: "Request too large" with Groq
**Solution**: The system should auto-switch to the 70B model. If it persists:
- Check that the 70B model is available in your account
- Verify the model names are correct
- Ensure the fallback chain is properly configured

### Issue: Slow response times
**Solution**: 
- Use Groq or Gemini for faster responses
- Reduce chunk sizes further if needed
- Enable caching for repeated queries

### Issue: Poor analysis quality with Groq
**Solution**:
- Switch to OpenAI or Anthropic for complex analyses
- Use Groq for initial screening, then OpenAI for detailed analysis
- Increase chunk quality by better semantic search

---

**Version**: 1.1  
**Last Updated**: 2026-02-17  
**Status**: Production Ready ✅
