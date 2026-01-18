#!/usr/bin/env python3
"""
Run Simple Pipeline
===================
Cost-effective pipeline using marker-pdf + Claude Haiku API.

Estimated cost: ~$48 for 16k items
Estimated time: 1-2 days

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY="your-key-here"

    # Process all data
    python run_simple_pipeline.py --data-dir ./data

    # Process with limit
    python run_simple_pipeline.py --data-dir ./data --limit 100

    # Export training data
    python run_simple_pipeline.py --data-dir ./data --export-only
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from pipeline import create_simple_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the simplified paper-to-poster pipeline (Claude Haiku + marker-pdf)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process 100 items
    python run_simple_pipeline.py --data-dir ./data --limit 100

    # Process all with custom concurrency
    python run_simple_pipeline.py --data-dir ./data --concurrency 50

    # Just export existing results
    python run_simple_pipeline.py --data-dir ./data --export-only

Cost estimate: ~$0.003 per poster (Claude Haiku 3)
    100 posters: ~$0.30
    1000 posters: ~$3
    16000 posters: ~$48
        """
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=True,
        help="Directory containing train.csv and data files"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Which dataset to process (default: train)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to process"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (default: 0)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10, lower = less rate limiting)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {data-dir}/simple_pipeline_output)"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing results to JSONL"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "parquet"],
        help="Export format (default: jsonl)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt (for automation)"
    )

    args = parser.parse_args()

    # Check API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.export_only:
        print("ERROR: Anthropic API key required!")
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "simple_pipeline_output"

    # Create pipeline
    print(f"\n{'='*70}")
    print("SIMPLE PIPELINE")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    pipeline = create_simple_pipeline(
        data_dir=str(data_dir),
        api_key=api_key,
        output_dir=str(output_dir)
    )

    if args.export_only:
        # Just export existing results
        export_path = output_dir / f"training_data.{args.export_format}"
        count = pipeline.export_training_data(str(export_path), format=args.export_format)
        print(f"\nExported {count} training examples to {export_path}")
        return

    # Load dataset
    csv_map = {
        "train": "train.csv",
        "test": "test.csv",
        "validation": "validation.csv"
    }
    csv_path = data_dir / csv_map[args.dataset]

    if not csv_path.exists():
        print(f"ERROR: Dataset not found: {csv_path}")
        sys.exit(1)

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")

    # Show cost estimate
    num_items = min(args.limit, len(df) - args.start) if args.limit else len(df) - args.start
    estimated_cost = num_items * 0.003

    print(f"\n{'='*70}")
    print("COST ESTIMATE")
    print(f"{'='*70}")
    print(f"Items to process: {num_items}")
    print(f"Estimated API cost: ${estimated_cost:.2f}")
    print(f"Concurrency: {args.concurrency}")
    print(f"{'='*70}\n")

    # Confirm
    if num_items > 100 and not args.yes:
        response = input(f"Process {num_items} items for ~${estimated_cost:.2f}? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Process
    results, stats = pipeline.process_dataframe(
        df,
        limit=args.limit,
        start_idx=args.start,
        concurrency=args.concurrency
    )

    # Print summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total processed: {stats.completed + stats.failed}")
    print(f"Successful: {stats.completed}")
    print(f"Failed: {stats.failed}")
    print(f"Estimated cost: ${stats.api_cost_estimate:.2f}")
    if stats.start_time and stats.end_time:
        duration = (stats.end_time - stats.start_time).total_seconds()
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Rate: {stats.completed / (duration/60):.1f} items/minute")
    print(f"{'='*70}\n")

    # Export
    export_path = output_dir / f"training_data.{args.export_format}"
    count = pipeline.export_training_data(str(export_path), format=args.export_format)
    print(f"Exported {count} training examples to {export_path}")


if __name__ == "__main__":
    main()
