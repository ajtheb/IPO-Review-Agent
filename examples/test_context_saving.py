"""
Quick test to verify context chunks are being saved.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

def test_context_saving():
    """Test that context chunks are saved to disk."""
    
    # Initialize analyzer
    analyzer = LLMProspectusAnalyzer(provider="groq", use_vector_db=False)
    
    # Save a test chunk
    test_company = "Test Company"
    test_content = "This is test financial data for Test Company. Revenue: $100M, Profit: $20M"
    
    analyzer._save_context_chunks(
        company_name=test_company,
        context_type="test_data",
        content=test_content,
        metadata={"test": True, "length": len(test_content)}
    )
    
    # Check if file was created
    context_dir = Path("context_chunks") / "Test_Company"
    
    if context_dir.exists():
        files = list(context_dir.glob("test_data_*.txt"))
        if files:
            print(f"✅ SUCCESS: Context file created at {files[0]}")
            print(f"   Directory: {context_dir}")
            print(f"   Files in directory: {len(list(context_dir.glob('*')))}")
            
            # Read and display content
            with open(files[0], 'r') as f:
                content = f.read()
                print(f"\n📄 File content preview:")
                print(content[:200] + "..." if len(content) > 200 else content)
            
            return True
        else:
            print(f"❌ FAILED: No files found in {context_dir}")
            return False
    else:
        print(f"❌ FAILED: Directory not created: {context_dir}")
        return False

if __name__ == "__main__":
    print("🧪 Testing context chunks saving functionality...\n")
    success = test_context_saving()
    
    if success:
        print("\n✅ Test passed! Context chunks will be saved for Vidya Wires analysis.")
        print("\n📁 Expected structure:")
        print("   context_chunks/")
        print("   └── Vidya_Wires/")
        print("       ├── prospectus_text_20260302_HHMMSS.txt")
        print("       ├── brave_search_results_20260302_HHMMSS.txt (if Brave Search enabled)")
        print("       └── web_scraped_url_X_20260302_HHMMSS.txt (if web scraping enabled)")
    else:
        print("\n❌ Test failed! Check the error messages above.")
