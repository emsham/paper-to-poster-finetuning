#!/usr/bin/env python3
"""
Test comparison between original and simple pipelines.
"""

import os
import sys
import json
import time
from pathlib import Path

# Test data
TEST_PAPER_ID = "19205"
TEST_POSTER_PATH = "images/19205.png"
TEST_PAPER_PATH = "papers/2309.02046v2_A Fast and Provable Algorithm for Sparse Phase Ret.pdf"
TEST_METADATA = {
    "title": "A Fast and Provable Algorithm for Sparse Phase Retrieval",
    "abstract": "We study the sparse phase retrieval problem...",
    "conference": "ICLR",
    "year": 2024,
    "topics": ["Signal Processing", "Optimization", "Algorithms"]
}

def test_simple_pipeline():
    """Test the simple pipeline (marker-pdf + Claude Haiku)."""
    print("\n" + "="*70)
    print("TESTING SIMPLE PIPELINE")
    print("="*70)

    from pipeline import create_simple_pipeline

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        return None

    # Create output directory
    output_dir = Path("test_simple_output")
    output_dir.mkdir(exist_ok=True)

    # Create pipeline
    pipeline = create_simple_pipeline(
        data_dir=".",
        api_key=api_key,
        output_dir=str(output_dir)
    )

    # Process single item
    start_time = time.time()
    result = pipeline.process_single(
        paper_id=TEST_PAPER_ID,
        poster_path=TEST_POSTER_PATH,
        paper_path=TEST_PAPER_PATH,
        metadata=TEST_METADATA
    )
    elapsed = time.time() - start_time

    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    if result.error:
        print(f"Error: {result.error}")

    print(f"Time: {elapsed:.1f} seconds")

    if result.success:
        print(f"\n--- Paper Markdown (first 500 chars) ---")
        print(result.parsed_paper.markdown_content[:500])

        print(f"\n--- Poster Markdown (first 500 chars) ---")
        print(result.parsed_poster.markdown_content[:500])

        print(f"\n--- Poster Layout ---")
        layout = result.poster_layout.to_dict()
        print(f"Orientation: {layout.get('poster', {}).get('orientation')}")
        print(f"Columns: {layout.get('body', {}).get('columns')}")
        print(f"Sections: {len(layout.get('sections', []))}")
        print(f"Reading order: {layout.get('reading_order')}")

        print(f"\n--- Figure Matches ---")
        print(f"Paper figures found: {len(result.parsed_paper.figures)}")
        print(f"Matches: {len(result.figure_matches)}")

    return result


def test_original_pipeline():
    """Test the original pipeline (Dolphin + Qwen VLM)."""
    print("\n" + "="*70)
    print("TESTING ORIGINAL PIPELINE")
    print("="*70)
    print("Note: This requires GPU and may be slow/unavailable")

    try:
        from pipeline import create_pipeline

        # Create output directory
        output_dir = Path("test_original_output")
        output_dir.mkdir(exist_ok=True)

        # Try to create pipeline - this will fail without proper GPU setup
        pipeline = create_pipeline(
            data_dir=".",
            output_dir=str(output_dir)
        )

        # Process single item
        start_time = time.time()
        result = pipeline.process_single(
            paper_id=TEST_PAPER_ID,
            poster_path=TEST_POSTER_PATH,
            paper_path=TEST_PAPER_PATH,
            metadata=TEST_METADATA
        )
        elapsed = time.time() - start_time

        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        if result.error:
            print(f"Error: {result.error}")
        print(f"Time: {elapsed:.1f} seconds")

        return result

    except Exception as e:
        print(f"\nCannot run original pipeline: {e}")
        print("This is expected if GPU models are not available.")
        return None


def compare_results(simple_result, original_result):
    """Compare outputs from both pipelines."""
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    if not simple_result or not simple_result.success:
        print("Simple pipeline failed - cannot compare")
        return

    if not original_result or not original_result.success:
        print("Original pipeline failed/unavailable - showing simple pipeline results only")
        print("\n--- Simple Pipeline Output ---")
        print(f"Paper markdown length: {len(simple_result.parsed_paper.markdown_content)} chars")
        print(f"Poster markdown length: {len(simple_result.parsed_poster.markdown_content)} chars")
        print(f"Figures extracted: {len(simple_result.parsed_paper.figures)}")

        # Save the output for inspection
        output_file = Path("test_simple_output") / "comparison_output.json"
        with open(output_file, 'w') as f:
            json.dump({
                "paper_markdown": simple_result.parsed_paper.markdown_content,
                "poster_markdown": simple_result.parsed_poster.markdown_content,
                "poster_layout": simple_result.poster_layout.to_dict(),
                "figure_count": len(simple_result.parsed_paper.figures)
            }, f, indent=2)
        print(f"\nFull output saved to: {output_file}")
        return

    # Full comparison
    print("\n| Metric | Simple Pipeline | Original Pipeline |")
    print("|--------|-----------------|-------------------|")
    print(f"| Paper markdown length | {len(simple_result.parsed_paper.markdown_content)} | {len(original_result.parsed_paper.markdown_content)} |")
    print(f"| Poster markdown length | {len(simple_result.parsed_poster.markdown_content)} | {len(original_result.parsed_poster.markdown_content)} |")
    print(f"| Figures extracted | {len(simple_result.parsed_paper.figures)} | {len(original_result.parsed_paper.figures)} |")
    print(f"| Figure matches | {len(simple_result.figure_matches)} | {len(original_result.figure_matches)} |")


def main():
    print("="*70)
    print("PIPELINE COMPARISON TEST")
    print("="*70)
    print(f"Test paper: {TEST_PAPER_ID}")
    print(f"Poster: {TEST_POSTER_PATH}")
    print(f"Paper: {TEST_PAPER_PATH}")

    # Check files exist
    if not Path(TEST_POSTER_PATH).exists():
        print(f"ERROR: Poster not found: {TEST_POSTER_PATH}")
        sys.exit(1)
    if not Path(TEST_PAPER_PATH).exists():
        print(f"ERROR: Paper not found: {TEST_PAPER_PATH}")
        sys.exit(1)

    # Test simple pipeline
    simple_result = test_simple_pipeline()

    # Test original pipeline (may fail without GPU)
    original_result = test_original_pipeline()

    # Compare
    compare_results(simple_result, original_result)


if __name__ == "__main__":
    main()
