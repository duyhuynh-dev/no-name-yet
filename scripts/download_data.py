#!/usr/bin/env python
"""
Script to download and prepare market data for the HFT simulator.

Usage:
    python scripts/download_data.py --symbol SPY --source yfinance
    python scripts/download_data.py --symbol BTC/USDT --source ccxt --exchange kraken
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Download market data')
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Symbol to download (e.g., SPY, AAPL, BTC/USDT)')
    parser.add_argument('--source', type=str, default='yfinance',
                        choices=['yfinance', 'ccxt'],
                        help='Data source')
    parser.add_argument('--exchange', type=str, default='kraken',
                        help='Exchange for ccxt (e.g., binance, kraken, coinbase)')
    parser.add_argument('--train-start', type=str, default='2018-01-01',
                        help='Training start date')
    parser.add_argument('--train-end', type=str, default='2022-12-31',
                        help='Training end date')
    parser.add_argument('--test-start', type=str, default='2023-01-01',
                        help='Test start date')
    parser.add_argument('--test-end', type=str, default='2023-12-31',
                        help='Test end date')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1h, etc.)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save data to disk')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    logger.info(f"Downloading {args.symbol} from {args.source}")
    logger.info(f"Date range: {args.train_start} to {args.test_end}")
    
    try:
        # Create pipeline
        pipeline = DataPipeline(
            source=args.source,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            data_dir=args.output_dir
        )
        
        # Run pipeline
        if args.source == 'yfinance':
            data = pipeline.run(args.symbol, interval=args.interval)
        else:
            data = pipeline.run(args.symbol, exchange=args.exchange, timeframe=args.interval)
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA DOWNLOAD COMPLETE")
        print("=" * 60)
        
        for name, df in data.items():
            if len(df) > 0:
                print(f"\n{name.upper()}:")
                print(f"  Rows: {len(df)}")
                print(f"  Features: {len(df.columns)}")
                print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Save data
        if args.save:
            pipeline.save(path=args.output_dir, symbol=args.symbol.replace('/', '_'))
            print(f"\nData saved to: {args.output_dir}/")
        
        # Print metadata
        print("\nMetadata:")
        summary = pipeline.get_summary()
        print(f"  Source: {summary['source']}")
        print(f"  Features: {len(summary['metadata'].get('features', []))}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

