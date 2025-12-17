# API Configuration Guide - IPO Review Agent

This guide explains how to configure API keys for enhanced functionality of the IPO Review Agent.

## 🔑 API Keys Overview

The IPO Review Agent can work **without any API keys**, but additional API keys enhance the analysis with:
- Real-time news and market sentiment data
- Financial data from multiple sources
- Comprehensive market intelligence

## 📋 Required vs Optional APIs

### ✅ **Works Without API Keys:**
- Enhanced prospectus parsing (SEBI filings)
- Basic financial analysis
- Risk assessment algorithms  
- Business model analysis
- IPO recommendation engine
- Web interface and CLI

### 🌟 **Enhanced with API Keys:**
- News sentiment analysis
- Real-time market data
- Financial data APIs
- Comprehensive market intelligence

## 🔧 API Configuration

### 1. Create Environment File
Create a `.env` file in the project root:

```bash
cd "/Users/apoorvjain/Projects/IPO Review Agent"
cp .env.example .env
```

### 2. Configure API Keys (Optional)

#### News API (for news sentiment analysis)
```bash
# Get free API key from https://newsapi.org/
NEWS_API_KEY=your_news_api_key_here
```

**Benefits:**
- Company-specific news analysis
- Market sentiment tracking
- Sector news intelligence
- Enhanced risk assessment

#### Alpha Vantage (for financial data)
```bash
# Get free API key from https://www.alphavantage.co/
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

**Benefits:**
- Real-time stock data
- Financial statements
- Technical indicators
- Market benchmarks

#### Finnhub (for market data)
```bash
# Get free API key from https://finnhub.io/
FINNHUB_API_KEY=your_finnhub_key_here
```

**Benefits:**
- IPO calendar data
- Market statistics
- Company financials
- Economic indicators

### 3. Complete .env File Example
```bash
# News and sentiment analysis
NEWS_API_KEY=your_news_api_key_here

# Financial data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# Optional: Custom configuration
PROSPECTUS_CACHE_HOURS=24
MIN_QUALITY_THRESHOLD=0.3
```

## 🚀 Getting API Keys

### News API (Free Tier: 1000 requests/day)
1. Visit https://newsapi.org/
2. Sign up for free account
3. Get API key from dashboard
4. Add to `.env` file

### Alpha Vantage (Free Tier: 500 requests/day)
1. Visit https://www.alphavantage.co/
2. Click "Get Free API Key"
3. Sign up and get key
4. Add to `.env` file

### Finnhub (Free Tier: 60 calls/minute)
1. Visit https://finnhub.io/
2. Sign up for free account
3. Get API key from dashboard
4. Add to `.env` file

## 🧪 Testing API Configuration

### Test All APIs
```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

apis = {
    'News API': os.getenv('NEWS_API_KEY'),
    'Alpha Vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'Finnhub': os.getenv('FINNHUB_API_KEY')
}

print('API Configuration Status:')
for name, key in apis.items():
    status = '✅ Configured' if key else '❌ Not configured'
    print(f'{name}: {status}')

configured_count = sum(1 for key in apis.values() if key)
print(f'\nConfigured APIs: {configured_count}/3')

if configured_count == 0:
    print('ℹ️  No APIs configured - using basic functionality')
elif configured_count < 3:
    print('⚠️  Partial API configuration - some features limited')
else:
    print('🎉 All APIs configured - full functionality available')
"
```

### Test IPO Analysis with APIs
```bash
python -c "
from src.data_sources import DataSourceManager
import json

print('Testing IPO analysis with current API configuration...')
manager = DataSourceManager()

# Test data collection
ipo_details = {'sector': 'Technology', 'exchange': 'NSE'}
data = manager.collect_ipo_data('Test Company', ipo_details)

print(f'News articles: {len(data.get(\"company_news\", []))}')
print(f'Market news: {len(data.get(\"market_news\", []))}')
print(f'Sector news: {len(data.get(\"sector_news\", []))}')
print(f'IPO data: {len(data.get(\"upcoming_ipos\", []))}')

if any(len(data.get(key, [])) > 0 for key in ['company_news', 'market_news', 'sector_news']):
    print('✅ API-based data sources working')
else:
    print('ℹ️  Using basic functionality (no API data)')
"
```

## 🔒 Security Best Practices

### 1. Environment Variables Only
```bash
# ✅ Good - use environment variables
NEWS_API_KEY=your_key_here

# ❌ Bad - never hardcode in source files
api_key = "your_key_here"
```

### 2. Git Ignore
Ensure `.env` is in `.gitignore`:
```bash
echo ".env" >> .gitignore
```

### 3. Key Rotation
- Rotate API keys regularly
- Use different keys for development/production
- Monitor API usage and set alerts

## 📊 Functionality Comparison

### Without API Keys:
✅ Enhanced prospectus parsing  
✅ Financial data extraction from SEBI filings  
✅ Business model analysis  
✅ Risk factor assessment  
✅ Investment recommendations  
✅ Web interface and CLI  
❌ Real-time news analysis  
❌ Market sentiment tracking  
❌ Live financial data  

### With All API Keys:
✅ All basic functionality  
✅ Real-time news sentiment analysis  
✅ Market trend analysis  
✅ Live financial data integration  
✅ Comprehensive market intelligence  
✅ Enhanced investment recommendations  

## 🚫 Troubleshooting API Issues

### Common Problems:

#### 1. "API key not configured" messages
```bash
# This is normal if you haven't set up API keys
# The system works fine without them
2025-12-02 20:02:46.785 | INFO | News API key not configured - returning empty news list
```

**Solution:** Either configure API keys or ignore these messages (system works without them).

#### 2. API rate limits exceeded
```bash
# You'll see HTTP 429 errors
Error: Rate limit exceeded for News API
```

**Solution:** 
- Wait for rate limit reset
- Upgrade to paid API plan
- Use caching to reduce API calls

#### 3. Invalid API key errors
```bash
# HTTP 401/403 errors
Error: Invalid API key for Alpha Vantage
```

**Solution:**
- Verify API key is correct
- Check if key has proper permissions
- Regenerate key if necessary

### Debug API Configuration:
```bash
python examples/debug_api_config.py
```

## 💡 Recommendations

### For Basic Usage:
- **No API keys needed** - the system works great without them
- Focus on prospectus analysis and business evaluation
- Use manual data entry for critical financial metrics

### For Enhanced Analysis:
- **Start with News API** - biggest impact on analysis quality
- **Add Alpha Vantage** for financial data enhancement
- **Finnhub optional** - adds market intelligence

### For Production:
- **All APIs recommended** for comprehensive analysis
- Set up monitoring and alerting for API usage
- Configure backup data sources
- Implement proper error handling and logging

## 🎯 Quick Start Commands

```bash
# 1. No configuration needed - basic functionality
python app.py

# 2. With News API only
echo "NEWS_API_KEY=your_key" >> .env
python app.py

# 3. Full configuration
echo "NEWS_API_KEY=your_news_key" >> .env
echo "ALPHA_VANTAGE_API_KEY=your_av_key" >> .env  
echo "FINNHUB_API_KEY=your_finnhub_key" >> .env
python app.py
```

The IPO Review Agent is designed to provide valuable analysis regardless of API configuration, with APIs enhancing rather than enabling the core functionality.
