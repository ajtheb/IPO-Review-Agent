# 🇮🇳 Indian IPO Review Agent

An intelligent IPO analysis system specifically designed for the **Indian stock market** (NSE & BSE). Provides comprehensive investment insights for companies planning to go public in India.

## Features

### 🔍 Financial Analysis
- Revenue growth and profitability analysis
- Financial ratios and metrics calculation
- Comparative industry analysis
- Valuation assessment

### 📈 Technical Indicators
- Gross Profit Margin (GPM) analysis
- Listing gains prediction models
- Long-term investment scoring
- Market trend analysis

### 📰 Market Intelligence
- Real-time news sentiment analysis
- Market trend identification
- Sector performance tracking
- Economic indicators impact assessment

### 📊 Comprehensive Reports
- Detailed IPO review reports
- Clear strengths and risks summary
- Investment recommendations
- Visual data presentations

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (see `.env.example`)
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Configuration

Create a `.env` file with your API keys:

### Basic Financial Data APIs:
- `ALPHA_VANTAGE_API_KEY`: For financial data
- `NEWS_API_KEY`: For news analysis
- `FINNHUB_API_KEY`: For market data

### LLM APIs (for Enhanced Analysis):
- `OPENAI_API_KEY`: For OpenAI GPT models
- `ANTHROPIC_API_KEY`: For Anthropic Claude models
- `GROQ_API_KEY`: For Groq LLM models
- `GEMINI_API_KEY`: For Google Gemini models

**Note**: You need at least one LLM API key to use the enhanced prospectus analysis features.

## Usage

### For Indian IPO Analysis:

1. **Web Interface**: 
   ```bash
   streamlit run app.py
   ```
   - Enter company name (no stock symbol needed for IPOs)
   - Set IPO price range in ₹ (Indian Rupees)
   - Select appropriate sector
   - Get comprehensive analysis

2. **CLI Mode**: 
   ```bash
   python cli.py "Zomato Limited" --sector Technology --price-min 72 --price-max 76
   ```

3. **Python API**:
   ```python
   from app import IPOReviewAgent
   
   agent = IPOReviewAgent()
   ipo_details = {
       'price_range': (100, 120),
       'sector': 'Technology',
       'exchange': 'NSE & BSE'
   }
   report = agent.analyze_ipo("Company Name", ipo_details)
   ```

## Project Structure

```
├── src/
│   ├── analyzers/          # Core analysis modules
│   ├── data_sources/       # Data collection components
│   ├── models/            # Data models and schemas
│   ├── reports/           # Report generation
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── tests/                # Test suite
├── examples/             # Example usage
└── docs/                 # Documentation
```

## License

MIT License
