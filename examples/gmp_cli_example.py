"""
CLI Integration Example for GMP Fetcher

This script demonstrates how to integrate GMP fetching into the CLI application.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.data_sources.gmp_fetcher import GMPFetcher
from loguru import logger
import argparse


def cli_gmp_fetch(company_name: str, use_cache: bool = True, format_output: bool = True):
    """
    Fetch GMP for a company via CLI.
    
    Args:
        company_name: Name of the company
        use_cache: Whether to use cached data
        format_output: Whether to show formatted report or raw data
    """
    logger.info(f"Fetching GMP for {company_name}...")
    
    fetcher = GMPFetcher(cache_duration_hours=6)
    gmp_data = fetcher.get_gmp(company_name, use_cache=use_cache)
    
    if format_output:
        # Show formatted report
        report = fetcher.format_gmp_report(gmp_data)
        print(report)
    else:
        # Show raw data
        print("\n📊 GMP Data (Raw):")
        print("-" * 60)
        for key, value in gmp_data.items():
            if key != 'message':
                print(f"{key}: {value}")
    
    return gmp_data


def cli_gmp_compare(company_names: list):
    """
    Compare GMP for multiple companies.
    
    Args:
        company_names: List of company names
    """
    logger.info(f"Comparing GMP for {len(company_names)} companies...")
    
    fetcher = GMPFetcher(cache_duration_hours=6)
    results = fetcher.get_multiple_gmp(company_names)
    
    # Sort by GMP percentage
    sorted_results = sorted(
        [(name, data) for name, data in results.items() if data['status'] == 'active'],
        key=lambda x: x[1].get('gmp_percentage', 0),
        reverse=True
    )
    
    print("\n" + "="*80)
    print("  📊 GMP COMPARISON REPORT")
    print("="*80 + "\n")
    
    if not sorted_results:
        print("⚠️  No GMP data available for any of the companies")
        return
    
    print(f"{'Rank':<6} {'Company':<30} {'GMP (₹)':<12} {'GMP %':<12} {'Expected Gain':<15}")
    print("-" * 80)
    
    for rank, (company, data) in enumerate(sorted_results, 1):
        gmp_price = data.get('gmp_price', 0)
        gmp_pct = data.get('gmp_percentage', 0)
        gain = data.get('estimated_listing_gain', 0)
        
        # Add emoji based on GMP percentage
        if gmp_pct > 50:
            emoji = "🔥"
        elif gmp_pct > 30:
            emoji = "✅"
        elif gmp_pct > 10:
            emoji = "📊"
        else:
            emoji = "⚠️ "
        
        print(f"{rank:<6} {company:<30} ₹{gmp_price:<10.2f} {gmp_pct:<11.2f}% {emoji} {gain:.2f}%")
    
    # Show companies with no data
    no_data = [name for name, data in results.items() if data['status'] != 'active']
    if no_data:
        print("\n⚠️  No data available for:")
        for company in no_data:
            print(f"   - {company}")
    
    print("\n" + "="*80)


def cli_gmp_monitor(company_names: list, threshold: float = 30.0, interval: int = 3600):
    """
    Monitor GMP for companies and alert when threshold is crossed.
    
    Args:
        company_names: List of company names to monitor
        threshold: GMP percentage threshold for alerts
        interval: Check interval in seconds (default: 1 hour)
    """
    import time
    from datetime import datetime
    
    logger.info(f"Starting GMP monitor for {len(company_names)} companies")
    logger.info(f"Alert threshold: {threshold}%")
    logger.info(f"Check interval: {interval} seconds")
    
    fetcher = GMPFetcher(cache_duration_hours=1)  # Short cache for monitoring
    
    print("\n" + "="*80)
    print("  🔔 GMP MONITORING STARTED")
    print("="*80)
    print(f"Monitoring: {', '.join(company_names)}")
    print(f"Alert Threshold: {threshold}%")
    print(f"Check Interval: {interval}s")
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        check_count = 0
        while True:
            check_count += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n[Check #{check_count}] {timestamp}")
            print("-" * 60)
            
            for company in company_names:
                gmp_data = fetcher.get_gmp(company, use_cache=False)
                
                if gmp_data['status'] == 'active':
                    gmp_pct = gmp_data['gmp_percentage']
                    gmp_price = gmp_data['gmp_price']
                    
                    if gmp_pct >= threshold:
                        print(f"🔥 ALERT: {company} - GMP {gmp_pct:.2f}% (₹{gmp_price})")
                    else:
                        print(f"   {company}: GMP {gmp_pct:.2f}% (₹{gmp_price})")
                else:
                    print(f"   {company}: No data")
            
            print(f"\nNext check in {interval}s...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        return


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GMP Fetcher CLI - Grey Market Premium for Indian IPOs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch GMP for a single company
  python gmp_cli_example.py fetch "Vidya Wires"
  
  # Fetch without using cache
  python gmp_cli_example.py fetch "Vidya Wires" --no-cache
  
  # Compare multiple companies
  python gmp_cli_example.py compare "Vidya Wires" "Akums Drugs" "DAM Capital"
  
  # Monitor companies with alerts
  python gmp_cli_example.py monitor "Vidya Wires" "Akums Drugs" --threshold 40
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch GMP for a single company')
    fetch_parser.add_argument('company', help='Company name')
    fetch_parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    fetch_parser.add_argument('--raw', action='store_true', help='Show raw data instead of formatted report')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare GMP for multiple companies')
    compare_parser.add_argument('companies', nargs='+', help='Company names')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor GMP with alerts')
    monitor_parser.add_argument('companies', nargs='+', help='Company names')
    monitor_parser.add_argument('--threshold', type=float, default=30.0, help='Alert threshold (default: 30%%)')
    monitor_parser.add_argument('--interval', type=int, default=3600, help='Check interval in seconds (default: 3600)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Execute command
    if args.command == 'fetch':
        cli_gmp_fetch(
            args.company,
            use_cache=not args.no_cache,
            format_output=not args.raw
        )
    
    elif args.command == 'compare':
        cli_gmp_compare(args.companies)
    
    elif args.command == 'monitor':
        cli_gmp_monitor(
            args.companies,
            threshold=args.threshold,
            interval=args.interval
        )


if __name__ == "__main__":
    main()
