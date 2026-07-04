# 🇮🇳 Indian IPO Review Agent

An intelligent IPO analysis system specifically designed for the **Indian stock market** (NSE & BSE). Provides comprehensive investment insights for companies planning to go public in India.

## Features

### 🤖 Intelligent Vector Search (NEW!)
- **Smart Context Retrieval**: Uses ChromaDB with `BAAI/bge-small-en-v1.5` embeddings
- **90-95% Context Reduction**: Only retrieves most relevant information
- **Semantic Search**: Finds related content even with different wording
- **Validated Quality**: All tests passing (run `python tests/test_embedding_model.py`)
- **📖 See [summaries/EMBEDDING_MODEL_GUIDE.md](summaries/EMBEDDING_MODEL_GUIDE.md) for details**
- **⚡ Quick Start: [summaries/VECTOR_DB_QUICKSTART.md](summaries/VECTOR_DB_QUICKSTART.md)**

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

### Quick Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "IPO Review Agent"
   ```

2. **Run the setup script**
   ```bash
   bash setup_venv.sh
   ```
   This will:
   - Create a Python virtual environment (`.venv`)
   - Install all required dependencies
   - Set up the development environment

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the application**
   ```bash
   bash run_app.sh
   ```

### Manual Setup

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Run application
streamlit run app.py
```

## Configuration

Create a `.env` file with your API keys:

### Basic Financial Data APIs:
- `ALPHA_VANTAGE_API_KEY`: For financial data
- `NEWS_API_KEY`: For news analysis

### Required LLM & Search APIs:
- `GROQ_API_KEY`: **[REQUIRED]** For fast LLM-based GMP extraction
- `BRAVE_API_KEY`: **[REQUIRED]** For web search and real-time GMP data

### Optional LLM APIs (for Enhanced Analysis):
- `OPENAI_API_KEY`: For OpenAI GPT models
- `ANTHROPIC_API_KEY`: For Anthropic Claude models
- `GEMINI_API_KEY`: For Google Gemini models

**Note**: The system now uses **Groq** as the primary LLM provider and **Brave Search API** for real-time market data. Both are required for GMP (Grey Market Premium) extraction.

## Usage

### Using Wrapper Scripts (Recommended)

All scripts automatically use the `.venv` virtual environment:

1. **Web Interface**: 
   ```bash
   bash run_app.sh
   ```

2. **Run Tests**:
   ```bash
   bash run_test.sh examples/test_brave_search_gmp.py
   bash run_test.sh examples/test_llm_gmp_extraction.py
   ```

### Manual Usage (with venv activated)

1. **Web Interface**: 
   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```

2. **CLI Mode**: 
   ```bash
   source .venv/bin/activate
   python cli.py "Zomato Limited"
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

### Testing GMP Extraction

Test the Groq + Brave Search integration:
```bash
bash run_test.sh examples/test_brave_search_gmp.py
```

This will:
- Fetch GMP data using Brave Search API
- Extract GMP values using Groq LLM
- Save all chunks and search results to `gmp_chunks/`
- Display formatted results

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
