#!/usr/bin/env python3
"""
SEBI Extraction Session Manager

Utility script to manage, monitor, and analyze SEBI extraction sessions.
Provides tools to check session status, merge data, and generate reports.

Author: IPO Review Agent
Date: 2024
"""

import json
import pandas as pd
from pathlib import Path
import os
import sys
from datetime import datetime
import argparse

class SessionManager:
    """Manage SEBI extraction sessions and data."""
    
    def __init__(self):
        self.session_file = 'sebi_session.json'
        self.download_session_file = 'download_session.json'
    
    def load_session_data(self):
        """Load extraction session data."""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
        return None
    
    def load_download_data(self):
        """Load download session data."""
        try:
            if os.path.exists(self.download_session_file):
                with open(self.download_session_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading download session: {e}")
        return None
    
    def show_session_status(self):
        """Show current session status."""
        print("📊 SEBI Extraction Session Status")
        print("=================================")
        
        # Extraction session
        session_data = self.load_session_data()
        if session_data:
            print(f"\n📈 Extraction Session:")
            print(f"  Last page processed: {session_data.get('last_page', 'None')}")
            print(f"  Records extracted: {len(session_data.get('extracted_data', []))}")
            print(f"  Expected total records: {session_data.get('total_records_expected', 'Unknown')}")
            print(f"  Last update: {session_data.get('last_update', 'Unknown')}")
            
            # Show statistics if available
            stats = session_data.get('stats', {})
            if stats:
                print(f"  Statistics:")
                print(f"    Pages processed: {stats.get('pages_processed', 0)}")
                print(f"    Errors encountered: {stats.get('errors_encountered', 0)}")
                print(f"    Retries attempted: {stats.get('retries_attempted', 0)}")
                print(f"    Session recoveries: {stats.get('session_recoveries', 0)}")
        else:
            print(f"\n📈 Extraction Session: No session file found")
        
        # Download session
        download_data = self.load_download_data()
        if download_data:
            print(f"\n📥 Download Session:")
            print(f"  Files downloaded: {len(download_data.get('downloaded_files', []))}")
            print(f"  Failed downloads: {len(download_data.get('failed_downloads', []))}")
            print(f"  Last update: {download_data.get('last_update', 'Unknown')}")
            
            # Show download statistics if available
            stats = download_data.get('stats', {})
            if stats:
                print(f"  Statistics:")
                print(f"    Total attempted: {stats.get('total_attempted', 0)}")
                print(f"    Successful: {stats.get('successful_downloads', 0)}")
                print(f"    Failed: {stats.get('failed_downloads', 0)}")
                print(f"    Skipped existing: {stats.get('skipped_existing', 0)}")
                print(f"    Bytes downloaded: {stats.get('bytes_downloaded', 0):,}")
        else:
            print(f"\n📥 Download Session: No session file found")
    
    def list_data_files(self):
        """List available data files."""
        print("\n📁 Available Data Files:")
        print("=" * 40)
        
        # Excel files
        excel_files = list(Path('.').glob('*sebi_ipo_documents*.xlsx'))
        if excel_files:
            print("\n📊 Excel Files:")
            for file in sorted(excel_files, key=os.path.getctime, reverse=True):
                size = file.stat().st_size
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"  • {file.name}")
                print(f"    Size: {size:,} bytes")
                print(f"    Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Try to read record count
                try:
                    df = pd.read_excel(file)
                    print(f"    Records: {len(df):,}")
                except:
                    print(f"    Records: Unable to read")
                print()
        
        # PDF files
        pdf_files = list(Path('.').glob('**/*.pdf', ))
        pdf_dirs = ['downloaded_documents', 'ipo_documents']
        
        for pdf_dir in pdf_dirs:
            if Path(pdf_dir).exists():
                pdf_files.extend(Path(pdf_dir).glob('*.pdf'))
        
        if pdf_files:
            print(f"\n📄 PDF Documents: {len(pdf_files)} files")
            total_size = sum(f.stat().st_size for f in pdf_files)
            print(f"    Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    def analyze_extraction_data(self, file_path=None):
        """Analyze extraction data."""
        if not file_path:
            # Find most recent Excel file
            excel_files = list(Path('.').glob('*sebi_ipo_documents*.xlsx'))
            if not excel_files:
                print("❌ No data files found")
                return
            file_path = max(excel_files, key=os.path.getctime)
        
        try:
            df = pd.read_excel(file_path)
            print(f"\n📊 Data Analysis: {file_path}")
            print("=" * 50)
            
            print(f"Total records: {len(df):,}")
            
            # Date range analysis
            if 'Date' in df.columns:
                print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Company analysis
            if 'Company' in df.columns:
                unique_companies = df['Company'].nunique()
                print(f"Unique companies: {unique_companies:,}")
                
                print(f"\nTop 10 companies by document count:")
                company_counts = df['Company'].value_counts().head(10)
                for company, count in company_counts.items():
                    print(f"  • {company}: {count} documents")
            
            # Document type analysis
            if 'Type' in df.columns:
                print(f"\nDocument types:")
                type_counts = df['Type'].value_counts()
                for doc_type, count in type_counts.items():
                    print(f"  • {doc_type}: {count} documents")
            
            # Link availability
            if 'Doc_Link' in df.columns:
                links_available = df['Doc_Link'].notna().sum()
                print(f"\nDocuments with links: {links_available:,} ({links_available/len(df)*100:.1f}%)")
            
            # Page distribution
            if 'Page' in df.columns:
                page_stats = df['Page'].describe()
                print(f"\nPage distribution:")
                print(f"  Pages covered: {df['Page'].min()} to {df['Page'].max()}")
                print(f"  Records per page: {len(df) / df['Page'].nunique():.1f} average")
        
        except Exception as e:
            print(f"❌ Error analyzing data: {e}")
    
    def merge_session_data(self):
        """Merge data from session file with Excel files."""
        session_data = self.load_session_data()
        if not session_data or not session_data.get('extracted_data'):
            print("❌ No session data to merge")
            return
        
        # Convert session data to DataFrame
        session_df = pd.DataFrame(session_data['extracted_data'])
        
        print(f"📊 Session data contains {len(session_df):,} records")
        
        # Create merged file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"merged_sebi_data_{timestamp}.xlsx"
        
        session_df.to_excel(output_file, index=False)
        print(f"✅ Merged data saved to: {output_file}")
        
        return output_file
    
    def clean_session_files(self):
        """Clean old session files."""
        files_to_clean = [self.session_file, self.download_session_file]
        
        for file in files_to_clean:
            if os.path.exists(file):
                backup_name = f"{file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(file, backup_name)
                print(f"🔄 Backed up {file} to {backup_name}")
        
        print("✅ Session files cleaned")
    
    def export_failed_downloads(self):
        """Export list of failed downloads for retry."""
        download_data = self.load_download_data()
        if not download_data or not download_data.get('failed_downloads'):
            print("❌ No failed downloads to export")
            return
        
        failed_df = pd.DataFrame(download_data['failed_downloads'])
        output_file = f"failed_downloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        failed_df.to_excel(output_file, index=False)
        print(f"✅ Failed downloads exported to: {output_file}")
        print(f"📊 {len(failed_df)} failed downloads recorded")
    
    def generate_progress_report(self):
        """Generate comprehensive progress report."""
        print("\n📋 SEBI Extraction Progress Report")
        print("=" * 50)
        
        # Current status
        self.show_session_status()
        
        # File analysis
        self.list_data_files()
        
        # Data analysis
        self.analyze_extraction_data()
        
        # Recommendations
        print("\n🎯 Recommendations:")
        session_data = self.load_session_data()
        download_data = self.load_download_data()
        
        if session_data:
            extracted = len(session_data.get('extracted_data', []))
            expected = session_data.get('total_records_expected', 0)
            
            if expected > 0 and extracted < expected:
                remaining = expected - extracted
                print(f"  • Continue extraction: {remaining:,} records remaining")
            else:
                print(f"  • ✅ Extraction appears complete")
        
        if download_data:
            downloaded = len(download_data.get('downloaded_files', []))
            failed = len(download_data.get('failed_downloads', []))
            
            if failed > 0:
                print(f"  • Retry failed downloads: {failed} documents failed")
            
            if downloaded > 0:
                print(f"  • ✅ {downloaded} documents successfully downloaded")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='SEBI Extraction Session Manager')
    parser.add_argument('command', choices=['status', 'analyze', 'merge', 'clean', 'failed', 'report'], 
                       help='Command to execute')
    parser.add_argument('--file', '-f', help='Specific file to analyze')
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.command == 'status':
        manager.show_session_status()
        manager.list_data_files()
    
    elif args.command == 'analyze':
        manager.analyze_extraction_data(args.file)
    
    elif args.command == 'merge':
        manager.merge_session_data()
    
    elif args.command == 'clean':
        confirm = input("Clean session files? This will backup current sessions. (y/n): ")
        if confirm.lower() == 'y':
            manager.clean_session_files()
    
    elif args.command == 'failed':
        manager.export_failed_downloads()
    
    elif args.command == 'report':
        manager.generate_progress_report()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("📊 SEBI Session Manager")
        print("======================")
        print("Available commands:")
        print("  python session_manager.py status   - Show session status")
        print("  python session_manager.py analyze  - Analyze extraction data")
        print("  python session_manager.py merge    - Merge session data to Excel")
        print("  python session_manager.py clean    - Clean session files")
        print("  python session_manager.py failed   - Export failed downloads")
        print("  python session_manager.py report   - Generate full report")
        
        # Quick status by default
        manager = SessionManager()
        manager.show_session_status()
        manager.list_data_files()
    else:
        main()
