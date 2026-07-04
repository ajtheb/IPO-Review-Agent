# PDF Chunker Integration - User Guide

## 🎯 Overview

The **Structured PDF Chunker** has been integrated into the IPO Review Agent web application, allowing you to upload and intelligently segment IPO prospectus documents directly through the browser.

## 🚀 Quick Start

### 1. Access the PDF Chunker
- Open the IPO Review Agent app: `streamlit run app.py`
- Navigate to the **"📑 PDF Chunker"** tab

### 2. Upload Your Document
- Click the file uploader
- Select your IPO prospectus PDF (DRHP or final prospectus)
- Wait for upload confirmation

### 3. Configure Parameters (Optional)
- **Minimum Chunk Size**: 200 characters (default)
- **Maximum Chunk Size**: 2000 characters (default)
- **Save Individual Files**: Create separate JSON files for each chunk

### 4. Process & Chunk
- Click **"🚀 Process & Chunk PDF"**
- Watch the progress bar
- Review the results

## 📊 What You Get

### Output Files
1. **all_chunks.json** - All chunks in a single JSON file
2. **structure_index.json** - Document hierarchy and structure
3. **metadata.json** - Processing statistics
4. **summary_report.txt** - Human-readable analysis
5. **chunks/** directory - Individual chunk files (optional)

### Chunk Structure
Each chunk includes:
```json
{
  "chunk_id": 1,
  "text": "Chunk content here...",
  "metadata": {
    "section": "RISK FACTORS",
    "subsection": "Business Risks",
    "pages": [30, 31],
    "categories": ["risk", "business"],
    "has_tables": false,
    "has_numbers": true,
    "char_count": 1250,
    "line_count": 15,
    "content_type": "risk"
  },
  "tables": []
}
```

## 🎯 Use Cases

### 1. Financial Analysis
- Extract financial statements
- Identify key metrics and ratios
- Track performance across years

### 2. LLM-Powered Analysis
- Feed chunks to LLMs for deep analysis
- Generate investment theses
- Extract structured financial data

### 3. Risk Assessment
- Identify risk factors
- Categorize by risk type
- Analyze risk mitigation strategies

### 4. Competitive Analysis
- Extract peer company comparisons
- Identify competitive advantages
- Analyze market positioning

### 5. Data Extraction
- Extract tables and financial data
- Identify key terms and metrics
- Create structured datasets

## 📈 Example: Vidya Wires Limited

**Document**: Vidya Wires Limited DRHP (420 pages)

**Results**:
- **3,468 total chunks** created
- **1,447 financial chunks** (41.7%)
- **838 chunks with tables** (24.2%)
- **Content categories**: Financial, Business, Risk, Legal, etc.

**Category Distribution**:
- Balance Sheet: 668 chunks (46.2%)
- Financial Ratios: 426 chunks (29.4%)
- Profit & Loss: 271 chunks (18.7%)
- Revenue: 142 chunks (9.8%)
- Tax: 158 chunks (10.9%)

## ⚙️ Configuration Options

### Chunk Size Guidelines

**Minimum Chunk Size (200-500 chars)**:
- **200**: Maximum granularity, more chunks
- **300**: Balanced approach (recommended)
- **500**: Larger context per chunk

**Maximum Chunk Size (1000-5000 chars)**:
- **1000**: Fine-grained analysis
- **2000**: Balanced (recommended)
- **3000**: More context, fewer chunks
- **5000**: Maximum context per chunk

### When to Adjust

**Increase chunk size** if:
- You need more context per chunk
- Working with narrative-heavy documents
- Feeding to LLMs with large context windows

**Decrease chunk size** if:
- You need fine-grained analysis
- Working with structured data (tables, lists)
- Creating embeddings for vector databases

## 🔧 Technical Details

### Supported Document Types
- ✅ Text-based PDFs (searchable)
- ✅ SEBI DRHP documents
- ✅ IPO prospectus documents
- ✅ Financial reports
- ❌ Scanned PDFs (OCR not supported yet)

### Section Detection
The chunker automatically identifies:
- Main sections (RISK FACTORS, BUSINESS OVERVIEW, etc.)
- Subsections (numbered and lettered)
- Financial statements
- Tables and structured data

### Content Classification
Chunks are categorized as:
- **financial**: Revenue, profit, balance sheets
- **business**: Operations, products, services
- **risk**: Risk factors and mitigation
- **legal**: Regulations, compliance, litigation
- **management**: Directors, key personnel
- **market**: Industry analysis, competition
- **general**: Other content

## 💡 Best Practices

### 1. Document Preparation
- Use the latest DRHP version
- Ensure PDF is text-based (not scanned)
- Check file size (< 50MB recommended)

### 2. Parameter Selection
- Start with default settings
- Adjust based on document structure
- Review sample chunks before full analysis

### 3. Output Organization
- Keep chunks organized by company
- Use descriptive folder names
- Archive old analyses

### 4. Integration with Analysis
- Load chunks for LLM analysis
- Filter by category for focused analysis
- Use metadata for selective processing

## 🐛 Troubleshooting

### "Chunker module not found"
**Solution**: Ensure `scripts/structured_pdf_chunker.py` exists in your project directory.

### "No text extracted"
**Problem**: PDF is scanned or image-based
**Solution**: Use OCR preprocessing or text-based PDFs

### "Too many small chunks"
**Solution**: Increase minimum chunk size parameter

### "Chunks too large for LLM"
**Solution**: Decrease maximum chunk size parameter

### "Processing takes too long"
**Problem**: Large PDF (> 500 pages)
**Solution**: Be patient, or process sections separately

## 📚 Next Steps

After chunking:
1. **Analyze Chunks**: Use the analysis script to identify financial information
2. **Feed to LLM**: Use chunks for AI-powered analysis
3. **Extract Metrics**: Build automated financial metric extraction
4. **Create Database**: Store chunks in vector database for semantic search
5. **Generate Reports**: Create investment analysis reports

## 🔗 Related Files

- **Chunker Script**: `scripts/structured_pdf_chunker.py`
- **Analysis Script**: `scripts/analyze_financial_chunks.py`
- **Example Output**: `structured_chunks/Vidya Wires/`
- **Main App**: `app.py`

## 📞 Support

For issues or questions:
1. Check the error details in the expander
2. Review logs in the console
3. Check if required packages are installed (`pip install -r requirements.txt`)
4. Ensure input PDF is valid and text-based

---

**Last Updated**: February 17, 2026
**Version**: 1.0
**Author**: IPO Review Agent Team
