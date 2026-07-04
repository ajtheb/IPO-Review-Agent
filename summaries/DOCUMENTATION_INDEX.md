# IPO Review Agent - Documentation Index

This document provides a comprehensive index of all documentation in the IPO Review Agent project.

## Quick Start
- **[README.md](README.md)** - Main project README with setup and usage instructions

## Core Documentation (`/docs`)

### API & Configuration
- **[api_configuration_guide.md](docs/api_configuration_guide.md)** - API configuration and setup guide
- **[gemini_setup_guide.md](docs/gemini_setup_guide.md)** - Google Gemini API setup
- **[llm_provider_optimizations.md](docs/llm_provider_optimizations.md)** - LLM provider optimization strategies

### Prospectus & PDF Processing
- **[prospectus_integration_implementation.md](docs/prospectus_integration_implementation.md)** - Prospectus integration implementation
- **[prospectus_integration_summary.md](docs/prospectus_integration_summary.md)** - Prospectus integration summary
- **[enhanced_prospectus_guide.md](docs/enhanced_prospectus_guide.md)** - Enhanced prospectus parsing guide
- **[sebi_prospectus_fetching_guide.md](docs/sebi_prospectus_fetching_guide.md)** - SEBI prospectus fetching guide
- **[pdf_chunker_guide.md](docs/pdf_chunker_guide.md)** - PDF chunking strategies guide
- **[pdf_chunker_implementation.md](docs/pdf_chunker_implementation.md)** - PDF chunker implementation details
- **[robust_extraction_guide.md](docs/robust_extraction_guide.md)** - Robust data extraction guide

### Chunking & Semantic Analysis
- **[structured_chunking_guide.md](docs/structured_chunking_guide.md)** - Structured chunking guide
- **[method3_structured_chunking.md](docs/method3_structured_chunking.md)** - Method 3 structured chunking
- **[semantic_chunking_guide.md](docs/semantic_chunking_guide.md)** - Semantic chunking guide
- **[semantic_chunking_summary.md](docs/semantic_chunking_summary.md)** - Semantic chunking summary
- **[SEMANTIC_CHUNKING_README.md](docs/SEMANTIC_CHUNKING_README.md)** - Semantic chunking README

### Vector Database & Retrieval
- **[vector_db_retrieval_implementation.md](docs/vector_db_retrieval_implementation.md)** - Vector DB retrieval implementation
- **[enhanced_context_retrieval_guide.md](docs/enhanced_context_retrieval_guide.md)** - Enhanced context retrieval guide
- **[enhanced_retrieval_validation.md](docs/enhanced_retrieval_validation.md)** - Enhanced retrieval validation

### GMP (Grey Market Premium) System
- **[GMP_QUICK_START.md](docs/GMP_QUICK_START.md)** - GMP system quick start
- **[GMP_SYSTEM_STATUS.md](docs/GMP_SYSTEM_STATUS.md)** - GMP system status
- **[GMP_INVESTIGATION_COMPLETE.md](docs/GMP_INVESTIGATION_COMPLETE.md)** - GMP investigation complete
- **[GMP_FINAL_STATUS_REPORT.md](docs/GMP_FINAL_STATUS_REPORT.md)** - GMP final status report
- **[GMP_DYNAMIC_CONTENT_ANALYSIS.md](docs/GMP_DYNAMIC_CONTENT_ANALYSIS.md)** - GMP dynamic content analysis
- **[GMP_ANALYSIS_SUMMARY.md](docs/GMP_ANALYSIS_SUMMARY.md)** - GMP analysis summary
- **[LLM_GMP_EXTRACTION.md](docs/LLM_GMP_EXTRACTION.md)** - LLM-based GMP extraction
- **[gmp_fetcher_guide.md](docs/gmp_fetcher_guide.md)** - GMP fetcher guide
- **[gmp_fetcher_summary.md](docs/gmp_fetcher_summary.md)** - GMP fetcher summary

### Web Scraping
- **[WEBSITE_SCRAPING_GUIDE.md](docs/WEBSITE_SCRAPING_GUIDE.md)** - Website scraping guide

### Indian IPO Specific
- **[indian_ipo_updates.md](docs/indian_ipo_updates.md)** - Indian IPO system updates

### Project Status
- **[comprehensive_guide.md](docs/comprehensive_guide.md)** - Comprehensive project guide
- **[PROJECT_COMPLETION_SUMMARY.md](docs/PROJECT_COMPLETION_SUMMARY.md)** - Project completion summary
- **[LLM_SOLUTION_SUMMARY.md](docs/LLM_SOLUTION_SUMMARY.md)** - LLM solution summary

## Summary Documents (`/summaries`)

For detailed information about implementation notes, bug fixes, and optimization summaries, see:
- **[summaries/README.md](summaries/README.md)** - Index of all summary documents

### Key Summaries

#### Groq Integration & Optimization
- [GROQ_QUICK_REFERENCE.md](summaries/GROQ_QUICK_REFERENCE.md)
- [GROQ_OPTIMIZATION_COMPLETE.md](summaries/GROQ_OPTIMIZATION_COMPLETE.md)
- [ANTI_HALLUCINATION_PROMPTS.md](summaries/ANTI_HALLUCINATION_PROMPTS.md)

#### Vector Database & Embeddings
- [VECTOR_DB_QUICKSTART.md](summaries/VECTOR_DB_QUICKSTART.md)
- [EMBEDDING_MODEL_GUIDE.md](summaries/EMBEDDING_MODEL_GUIDE.md)
- [ENHANCED_RETRIEVAL_SUMMARY.md](summaries/ENHANCED_RETRIEVAL_SUMMARY.md)

#### System Status
- [SYSTEM_STATUS_REPORT.md](summaries/SYSTEM_STATUS_REPORT.md)
- [SOLUTION_COMPLETE.md](summaries/SOLUTION_COMPLETE.md)

## Context Data (`/context`)
- **[context/README.md](context/README.md)** - Context data storage for GMP and other real-time data

## Development

### Coding Instructions
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI assistant coding guidelines

### Tests
- Located in `/tests` directory
- Run tests using: `python -m pytest tests/`

### Examples
- Located in `/examples` directory
- Includes practical demos and test scripts

## Navigation Tips

1. **New to the project?** Start with [README.md](README.md)
2. **Setting up APIs?** Check [docs/api_configuration_guide.md](docs/api_configuration_guide.md)
3. **Working with prospectus?** See [docs/enhanced_prospectus_guide.md](docs/enhanced_prospectus_guide.md)
4. **Optimizing LLMs?** Review [summaries/GROQ_OPTIMIZATION_COMPLETE.md](summaries/GROQ_OPTIMIZATION_COMPLETE.md)
5. **Understanding vector search?** Read [summaries/VECTOR_DB_QUICKSTART.md](summaries/VECTOR_DB_QUICKSTART.md)
6. **Bug fixes and resolutions?** Browse [summaries/](summaries/)

## Contributing

When adding new documentation:
1. Place comprehensive guides in `/docs`
2. Place implementation notes and summaries in `/summaries`
3. Update this index file
4. Update the respective README files in each directory
