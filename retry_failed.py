#!/usr/bin/env python3
"""
Retry failed poster processing.
Uses improved retry logic and image resizing.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.claude_poster_processor import ClaudePosterProcessor, ClaudeConfig
from pipeline.models import ParsedDocument, PosterLayout, TrainingExample
from pipeline.marker_parser import MarkerParser, MarkerConfig


def main():
    # Check API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Load failed IDs
    failed_ids_file = Path('/tmp/failed_ids.txt')
    if not failed_ids_file.exists():
        print("ERROR: /tmp/failed_ids.txt not found")
        sys.exit(1)

    with open(failed_ids_file, 'r') as f:
        failed_ids = [line.strip() for line in f.readlines()]

    # Skip items already successfully processed
    training_dir = Path('pipeline_output/training_data')
    already_done = set(f.replace('.json', '') for f in os.listdir(training_dir)) if training_dir.exists() else set()
    failed_ids = [fid for fid in failed_ids if fid not in already_done]

    print(f"Total failed items to retry: {len(failed_ids)} (skipped {len(already_done)} already done)")

    # Load dataset
    df = pd.read_csv('train.csv')
    df['paper_id'] = df['paper_id'].astype(str)

    # Setup processor
    config = ClaudeConfig()
    config.api_key = api_key
    processor = ClaudePosterProcessor(config)

    # Output directories
    output_base = Path('pipeline_output')
    poster_desc_dir = output_base / 'poster_descriptions'
    training_dir = output_base / 'training_data'
    poster_desc_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    successes = 0
    failures = 0
    start_time = datetime.now()

    async def process_one(paper_id, image_path, metadata, semaphore):
        async with semaphore:
            try:
                doc, layout = await processor.process_poster_async(image_path, paper_id)
                return paper_id, doc, layout, metadata, None
            except Exception as e:
                return paper_id, None, None, metadata, str(e)

    async def run_batch(items, concurrency=10):
        nonlocal successes, failures
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [process_one(pid, img, meta, semaphore) for pid, img, meta in items]

        # Process with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            result = await coro
            results.append(result)

            paper_id, doc, layout, metadata, error = result
            if error or doc is None or layout is None:
                failures += 1
                continue

            successes += 1

            # Save layout
            layout_path = poster_desc_dir / f"{paper_id}_layout.json"
            layout.save(layout_path)

            # Load parsed paper
            paper_parsed_dir = output_base / 'paper_parsed' / paper_id
            paper_md_path = paper_parsed_dir / 'paper.md'
            paper_markdown = ""
            if paper_md_path.exists():
                with open(paper_md_path, 'r') as f:
                    paper_markdown = f.read()

            # Create training example
            training_example = TrainingExample(
                paper_id=paper_id,
                paper_markdown=paper_markdown,
                paper_abstract=metadata.get("abstract", ""),
                paper_title=metadata.get("title", ""),
                poster_layout=layout,
                poster_markdown=doc.markdown_content,
                figure_matches=[],
                conference=metadata.get("conference", ""),
                year=int(metadata.get("year", 0)) if metadata.get("year") else 0,
                topics=metadata.get("topics", [])
            )

            # Save
            training_path = training_dir / f"{paper_id}.json"
            training_example.save(training_path)

        return results

    # Build items list
    items = []
    for pid in failed_ids:
        row = df[df['paper_id'] == pid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        img_path = r.get('local_image_path')
        if pd.isna(img_path) or not Path(img_path).exists():
            continue

        metadata = {
            'title': r.get('title', ''),
            'abstract': r.get('abstract', ''),
            'conference': r.get('conference', ''),
            'year': r.get('year', 0),
            'topics': r.get('topics', [])
        }
        items.append((pid, img_path, metadata))

    print(f"Valid items to process: {len(items)}")
    print(f"Concurrency: 10")
    print(f"Estimated cost: ${len(items) * 0.003:.2f}")
    print()

    # Run
    asyncio.run(run_batch(items, concurrency=10))

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print("RETRY COMPLETE")
    print(f"{'='*60}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Rate: {successes/(duration/60):.1f} items/minute")
    print(f"Estimated cost: ${successes * 0.003:.2f}")

    # Re-export training data
    print(f"\nRe-exporting training data...")
    examples = []
    for json_file in training_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                example = {
                    "instruction": f"Generate an academic poster for the following research paper.\n\nTitle: {data.get('paper_title', '')}\n\nAbstract: {data.get('paper_abstract', '')}",
                    "input": data.get('paper_markdown', ''),
                    "output": json.dumps({
                        "layout": data.get('poster_layout', {}),
                        "content": data.get('poster_markdown', '')
                    })
                }
                examples.append(example)
        except Exception as e:
            pass

    export_path = output_base / "training_data.jsonl"
    with open(export_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Exported {len(examples)} total training examples to {export_path}")


if __name__ == "__main__":
    main()
