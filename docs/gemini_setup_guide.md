# Gemini API Key Configuration Guide

## 🔑 How to Get Your Gemini API Key

### Step 1: Visit Google AI Studio
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account

### Step 2: Create API Key
1. Click "Create API Key" button
2. Choose to create in new project or existing project
3. Copy the generated API key

### Step 3: Add to Your .env File
```bash
# Add this line to your .env file
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### Step 4: Verify Configuration
```bash
# Run the test script to verify
python test_gemini_config.py
```

## 🚀 Using Gemini in IPO Review Agent

### In Streamlit App:
1. Run: `streamlit run app.py`
2. Go to "IPO Analysis" tab
3. In the sidebar, under "LLM Configuration"
4. Select "Google Gemini" from the dropdown
5. Enable "Enhanced LLM Analysis"

### Key Benefits of Gemini:
- **Advanced Reasoning**: Excellent for complex financial document analysis
- **Large Context Window**: Can process longer prospectus documents
- **Multimodal Capabilities**: Future support for chart/graph analysis
- **Cost Effective**: Competitive pricing compared to other models

## 📊 Gemini Model Used
- **Model**: `gemini-1.5-pro`
- **Context Length**: Up to 2M tokens
- **Best For**: Complex financial analysis, long document processing

## 🔧 Troubleshooting

### Common Issues:
1. **API Key Invalid**: Ensure you copied the full key without spaces
2. **Quota Exceeded**: Check your Google AI Studio usage limits
3. **Region Restrictions**: Gemini may not be available in all regions

### Test Your Configuration:
```bash
# Quick test
python test_gemini_config.py

# Full LLM analysis test (if other dependencies are working)
python -c "
from dotenv import load_dotenv
load_dotenv()
print('✅ Gemini configuration loaded successfully')
"
```

## 💡 Pro Tips

1. **API Limits**: Start with smaller analyses to understand your usage patterns
2. **Backup Provider**: Configure multiple LLM providers for reliability
3. **Cost Management**: Monitor your Google AI Studio usage dashboard
4. **Performance**: Gemini excels at extracting complex financial relationships

## 🔗 Additional Resources
- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Pricing Information](https://ai.google.dev/pricing)
