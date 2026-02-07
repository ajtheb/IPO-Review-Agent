#!/usr/bin/env python3
"""
Test Brave Search API for GMP Extraction with Analysis

This script demonstrates using Brave Search API to find current GMP data
for Indian IPOs and generates comprehensive analysis using Groq LLM.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq
from src.data_sources.llm_gmp_extractor import LLMGMPExtractor


def generate_gmp_analysis(company_name: str, context_chunks: list, groq_client: Groq) -> str:
    """
    Generate comprehensive GMP analysis using Groq API.
    
    Args:
        company_name: Name of the company
        context_chunks: List of relevant text chunks from web scraping
        groq_client: Groq API client
        
    Returns:
        Comprehensive GMP analysis text
    """
    # Combine chunks into context
    combined_context = "\n\n".join(context_chunks[:5])  # Use top 5 chunks
    
    analysis_prompt = f"""You are a financial analyst specializing in Indian IPO market analysis. Analyze the Grey Market Premium (GMP) data for {company_name} based on the following information:

CONTEXT FROM WEB SOURCES:
{combined_context}

Please provide a comprehensive GMP analysis covering:

1. **Current GMP Status**
   - Current GMP price and percentage
   - Issue price and expected listing price
   - Whether GMP is positive, negative, or neutral

2. **Market Sentiment Analysis**
   - What does the GMP indicate about market demand?
   - Is the IPO oversubscribed or undersubscribed?
   - Investor confidence level

3. **Listing Gain Potential**
   - Expected listing gains based on GMP
   - Risk-reward assessment
   - Comparison with similar IPOs if mentioned

4. **IPO Timeline & Status**
   - Current IPO status (Open/Upcoming/Closed/Listed)
   - Important dates (opening, closing, listing)
   - Time-sensitive insights

5. **Investment Recommendation**
   - Should investors apply for this IPO?
   - Grey market trends (rising/falling)
   - Risk factors to consider

6. **Key Takeaways**
   - 3-5 bullet points summarizing the analysis
   - Action items for potential investors

Format the response in clear, professional language suitable for investment decision-making. Use ₹ symbol for prices and % for percentages. Be specific with numbers where available.

If the context doesn't contain sufficient GMP data, clearly state what information is missing."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst specializing in Indian IPO markets and grey market premium analysis."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating analysis: {e}"


def test_brave_search_gmp():
    """Test GMP extraction using Brave Search API with analysis."""
    
    print("\n" + "="*80)
    print("🔍 BRAVE SEARCH API - GMP EXTRACTION & ANALYSIS TEST")
    print("="*80 + "\n")
    
    # Check API keys
    groq_key = os.getenv('GROQ_API_KEY')
    brave_key = os.getenv('BRAVE_API_KEY')
    
    if not groq_key:
        print("❌ GROQ_API_KEY not found in environment")
        print("   Set it in .env file: GROQ_API_KEY=your_key_here")
        return
    
    if not brave_key:
        print("❌ BRAVE_API_KEY not found in environment")
        print("   Get your free API key at: https://api.search.brave.com/")
        print("   Set it in .env file: BRAVE_API_KEY=your_key_here")
        return
    
    print(f"✅ GROQ_API_KEY found: {groq_key[:10]}...")
    print(f"✅ BRAVE_API_KEY found: {brave_key[:10]}...\n")
    
    # Test company
    company_name = "Biopol Chemicals"
    
    # Initialize Groq client for analysis
    groq_client = Groq(api_key=groq_key)
    
    # Initialize extractor with Brave Search enabled
    print("🚀 Initializing LLM GMP Extractor with Brave Search...")
    extractor = LLMGMPExtractor(provider="groq", use_brave_search=True)
    print("✅ Extractor initialized\n")
    
    print("\n" + "="*80)
    print(f"Testing: {company_name}")
    print("="*80)
    
    try:
        # Step 1: Search Brave and scrape websites
        print("\n📡 Step 1: Searching Brave API and scraping websites...")
        search_results = extractor.search_gmp_with_brave(company_name, max_results=5)
        
        if not search_results:
            print("❌ No search results found")
            return
        
        print(f"✅ Found {len(search_results)} search results")
        
        # Step 2: Scrape and extract content from websites
        print("\n🌐 Step 2: Scraping website content...")
        scraped_chunks = []
        
        for i, result in enumerate(search_results[:3], 1):
            url = result.get('url')
            print(f"\n   [{i}/3] Scraping: {url}")
            
            html_content = extractor.scrape_url_content(url)
            if html_content:
                text_content = extractor.extract_text_from_html(html_content)
                
                # Save scraped content
                extractor.save_scraped_content(
                    company_name=company_name,
                    url=url,
                    html_content=html_content,
                    text_content=text_content,
                    folder="gmp_chunks"
                )
                
                # Add to chunks (limit size for LLM)
                max_chunk_size = 5000
                if len(text_content) > max_chunk_size:
                    text_content = text_content[:max_chunk_size]
                
                scraped_chunks.append(f"Source: {url}\n{text_content}")
                print(f"   ✅ Scraped {len(text_content)} characters")
            else:
                print(f"   ⚠️  Failed to scrape")
        
        if not scraped_chunks:
            print("\n❌ No content scraped from websites")
            return
        
        print(f"\n✅ Successfully scraped {len(scraped_chunks)} websites")
        
        # Step 3: Extract structured GMP data
        print("\n📊 Step 3: Extracting structured GMP data...")
        result = extractor.extract_gmp_from_brave_results(
            company_name=company_name,
            search_results=search_results,
            scrape_websites=False,  # Already scraped above
            save_scraped=False
        )
        
        # Display structured results
        print("\n" + "="*80)
        print("📊 STRUCTURED GMP DATA")
        print("="*80)
        
        print(f"\n🏢 Company: {result.get('company_name')}")
        print(f"📊 Status: {result.get('status')}")
        
        if result.get('status') == 'success':
            print(f"\n💰 GMP Price: ₹{result.get('gmp_price', 'N/A')}")
            print(f"📈 GMP Percentage: {result.get('gmp_percentage', 'N/A')}%")
            print(f"💵 Issue Price: ₹{result.get('issue_price', 'N/A')}")
            print(f"🎯 Expected Listing: ₹{result.get('expected_listing_price', 'N/A')}")
            print(f"💹 Estimated Gain: {result.get('estimated_listing_gain', 'N/A')}%")
            
            if result.get('ipo_status'):
                print(f"\n📅 IPO Status: {result['ipo_status']}")
            if result.get('open_date'):
                print(f"📆 Open Date: {result['open_date']}")
            if result.get('close_date'):
                print(f"📆 Close Date: {result['close_date']}")
            if result.get('listing_date'):
                print(f"📆 Listing Date: {result['listing_date']}")
            
            if result.get('confidence'):
                print(f"\n🎯 Confidence: {result['confidence']}")
        else:
            print(f"\n⚠️  {result.get('message', 'No data available')}")
        
        # Step 4: Generate comprehensive analysis
        print("\n" + "="*80)
        print("🤖 Step 4: Generating Comprehensive GMP Analysis with Groq...")
        print("="*80 + "\n")
        
        analysis = generate_gmp_analysis(company_name, scraped_chunks, groq_client)
        
        print("\n" + "="*80)
        print("� COMPREHENSIVE GMP ANALYSIS")
        print("="*80 + "\n")
        print(analysis)
        
        # Step 5: Save analysis to file
        print("\n" + "="*80)
        print("💾 Step 5: Saving Analysis to File...")
        print("="*80)
        
        from datetime import datetime
        import re
        
        os.makedirs("gmp_chunks", exist_ok=True)
        safe_name = re.sub(r'[^\w\s-]', '_', company_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"gmp_chunks/{safe_name}_analysis_{timestamp}.txt"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"GMP Analysis for {company_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            
            f.write("STRUCTURED DATA:\n")
            f.write(f"  GMP Price: ₹{result.get('gmp_price', 'N/A')}\n")
            f.write(f"  GMP %: {result.get('gmp_percentage', 'N/A')}%\n")
            f.write(f"  Issue Price: ₹{result.get('issue_price', 'N/A')}\n")
            f.write(f"  Expected Listing: ₹{result.get('expected_listing_price', 'N/A')}\n")
            f.write(f"  IPO Status: {result.get('ipo_status', 'N/A')}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            f.write("COMPREHENSIVE ANALYSIS:\n")
            f.write(analysis)
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"Sources: {len(scraped_chunks)} websites scraped\n")
            for i, result_item in enumerate(search_results[:3], 1):
                f.write(f"  {i}. {result_item['url']}\n")
        
        print(f"✅ Analysis saved to: {analysis_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ BRAVE SEARCH + ANALYSIS TEST COMPLETE")
    print("="*80)
    print(f"""
📁 Check the gmp_chunks/ folder for:
   - Brave search results
   - Scraped HTML files
   - Extracted text content
   - Comprehensive GMP analysis ({safe_name}_analysis_*.txt)

💡 This workflow:
   ✅ Searches Brave for latest GMP data
   ✅ Scrapes actual website content
   ✅ Extracts structured GMP data
   ✅ Generates comprehensive analysis using Groq
   ✅ Saves everything for review
    """)


if __name__ == "__main__":
    try:
        test_brave_search_gmp()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
