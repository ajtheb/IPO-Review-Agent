<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# IPO Review Agent - Copilot Instructions

This is a financial analysis application for IPO (Initial Public Offering) review and investment recommendations.

## Project Context
- **Domain**: Financial Technology (FinTech)
- **Purpose**: Automated IPO analysis and investment recommendations
- **Tech Stack**: Python, Streamlit, pandas, financial APIs

## Code Guidelines
1. **Financial Data Handling**: Always validate financial data and handle missing values appropriately
2. **API Integration**: Use proper error handling for external API calls (Alpha Vantage, News API, etc.)
3. **Security**: Never hardcode API keys - use environment variables
4. **Documentation**: Include docstrings for all financial calculation methods
5. **Testing**: Write unit tests for all analysis functions

## Key Components
- Financial analyzers for revenue, profit, and ratio analysis
- Market intelligence modules for news sentiment and trend analysis
- Risk assessment algorithms
- Report generation with clear recommendations
- Web interface using Streamlit

## Best Practices
- Use type hints for all function parameters and return values
- Implement proper logging for analysis steps
- Handle edge cases in financial calculations (division by zero, negative values)
- Ensure all monetary values are properly formatted
- Validate input data before processing

## Analysis Focus Areas
1. Revenue growth patterns and sustainability
2. Profit margin analysis and trends
3. Business model strengths and competitive advantages
4. Risk factors and market conditions
5. Valuation metrics and listing gain potential
