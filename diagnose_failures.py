#!/usr/bin/env python3
"""
Diagnose poster processing failures.
Run with: ANTHROPIC_API_KEY=your-key python diagnose_failures.py
"""

import asyncio
import os
import sys
import pandas as pd
from pathlib import Path
from collections import Counter

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.claude_poster_processor import ClaudePosterProcessor, ClaudeConfig


def main():
    # Check API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Load failed IDs
    failed_ids_file = Path('/tmp/failed_ids.txt')
    if not failed_ids_file.exists():
        print("ERROR: /tmp/failed_ids.txt not found. Run the analysis first.")
        sys.exit(1)

    with open(failed_ids_file, 'r') as f:
        failed_ids = [line.strip() for line in f.readlines()]

    print(f"Total failed items: {len(failed_ids)}")
    print(f"Testing first 20 to diagnose errors...\n")

    # Load dataset
    df = pd.read_csv('train.csv')
    df['paper_id'] = df['paper_id'].astype(str)

    # Setup processor with longer timeout
    config = ClaudeConfig()
    config.api_key = api_key
    config.timeout = 120  # Increase timeout to 2 minutes
    config.max_retries = 2
    processor = ClaudePosterProcessor(config)

    errors = Counter()
    successes = 0

    async def test_one(paper_id, image_path):
        try:
            doc, layout = await processor.process_poster_async(image_path, paper_id)
            return paper_id, 'SUCCESS', None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            return paper_id, error_type, error_msg

    async def run_tests():
        nonlocal successes
        tasks = []
        test_ids = failed_ids[:20]

        for pid in test_ids:
            row = df[df['paper_id'] == pid]
            if len(row) == 0:
                continue
            img_path = row.iloc[0]['local_image_path']
            if pd.isna(img_path):
                continue
            tasks.append(test_one(pid, img_path))

        results = await asyncio.gather(*tasks)

        print("=== INDIVIDUAL RESULTS ===\n")
        for pid, status, error in results:
            if status == 'SUCCESS':
                successes += 1
                print(f"✓ {pid}: SUCCESS")
            else:
                errors[status] += 1
                print(f"✗ {pid}: {status}")
                print(f"  {error}\n")

    asyncio.run(run_tests())

    print("\n=== SUMMARY ===")
    print(f"Successes: {successes}")
    print(f"Failures by type:")
    for error_type, count in errors.most_common():
        print(f"  {error_type}: {count}")


if __name__ == "__main__":
    main()
