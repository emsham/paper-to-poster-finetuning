#!/usr/bin/env python3
"""
Paper-to-Poster Pipeline CLI
=============================

Command-line interface for running the paper-to-poster finetuning pipeline.

Usage:
    # Process entire training set (sequential)
    python run_pipeline.py --data-dir ./ --dolphin-model ./hf_model

    # Process with limit
    python run_pipeline.py --data-dir ./ --limit 100 --start 0

    # PARALLEL: Staged processing (memory efficient, single GPU)
    python run_pipeline.py --parallel staged --num-workers 4

    # PARALLEL: Multi-GPU processing (maximum throughput)
    python run_pipeline.py --parallel multi_gpu --gpu-ids 0,1,2

    # PARALLEL: Auto-detect best strategy
    python run_pipeline.py --parallel auto

    # Resume interrupted processing
    python run_pipeline.py --parallel staged --resume

    # Process single paper-poster pair
    python run_pipeline.py --single \
        --paper-id 12345 \
        --poster ./images/12345.png \
        --paper ./papers/paper.pdf

    # Export training data
    python run_pipeline.py --export --export-output training_data.jsonl

Parallel Strategies:
    none      - Sequential processing (default)
    staged    - Process all items through each stage before next
                (memory efficient, only one model loaded at a time)
    multi_gpu - Distribute models across GPUs for pipelining
                (maximum throughput, requires 2+ GPUs)
    auto      - Automatically choose based on available GPUs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Paper-to-Poster Finetuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./",
        help="Root data directory containing train.csv, images/, papers/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to data-dir/pipeline_output)"
    )

    # Model paths
    parser.add_argument(
        "--dolphin-model",
        type=str,
        default="./hf_model",
        help="Path to Dolphin model"
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="VLM model for figure matching and poster description"
    )

    # Processing options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "test", "validation"],
        default="train",
        help="Which dataset split to process"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index in dataset"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed items"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-process all items even if already done"
    )

    # Single item processing
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process a single paper-poster pair"
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        help="Paper ID for single processing"
    )
    parser.add_argument(
        "--poster",
        type=str,
        help="Poster image path for single processing"
    )
    parser.add_argument(
        "--paper",
        type=str,
        help="Paper PDF path for single processing"
    )

    # Export options
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export training data to file"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["jsonl", "json", "parquet"],
        default="jsonl",
        help="Export format"
    )
    parser.add_argument(
        "--export-output",
        type=str,
        default="training_data.jsonl",
        help="Export output path"
    )
    parser.add_argument(
        "--export-tracking",
        type=str,
        default=None,
        help="Export tracking data merged with source CSV to this path"
    )
    parser.add_argument(
        "--show-tracking",
        action="store_true",
        help="Show tracking summary after processing"
    )

    # Parallel processing options
    parser.add_argument(
        "--parallel",
        type=str,
        choices=["none", "staged", "multi_gpu", "auto"],
        default="none",
        help="Parallel processing strategy"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for I/O tasks"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for multi-GPU mode (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, ignore checkpoints"
    )

    # Advanced options
    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=0.10,
        help="Feature matching threshold for VLM candidates"
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use flash attention for VLM models"
    )

    args = parser.parse_args()

    # Import pipeline (delayed to avoid slow imports for --help)
    from pipeline import create_pipeline, PipelineConfig, get_default_config

    # Create configuration
    config = get_default_config(args.data_dir)

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    config.dolphin.model_path = args.dolphin_model
    config.vlm_matcher.model_name = args.vlm_model
    config.vlm_matcher.feature_threshold = args.feature_threshold
    config.vlm_matcher.use_flash_attn = args.use_flash_attn
    config.poster_descriptor.model_name = args.vlm_model
    config.poster_descriptor.use_flash_attn = args.use_flash_attn
    config.skip_existing = args.skip_existing

    # Create pipeline
    from pipeline import PaperToPosterPipeline
    pipeline = PaperToPosterPipeline(config)

    if args.export:
        # Export mode
        pipeline.export_training_data(
            args.export_output,
            format=args.export_format
        )

    elif args.single:
        # Single item mode
        if not all([args.paper_id, args.poster, args.paper]):
            parser.error("--single requires --paper-id, --poster, and --paper")

        result = pipeline.process_single(
            paper_id=args.paper_id,
            poster_path=args.poster,
            paper_path=args.paper
        )

        if result.success:
            print(f"\nSuccess! Training data saved.")
        else:
            print(f"\nFailed: {result.error}")
            sys.exit(1)

    else:
        # Dataset mode
        dataset_map = {
            "train": config.train_path,
            "test": config.test_path,
            "validation": config.validation_path
        }

        csv_path = dataset_map[args.dataset]
        if not csv_path.exists():
            print(f"Error: Dataset not found at {csv_path}")
            sys.exit(1)

        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")

        # Apply limit/start
        if args.limit or args.start > 0:
            df = df.iloc[args.start:args.start + (args.limit or len(df))]
            print(f"Processing rows {args.start} to {args.start + len(df)}")

        # Choose processing strategy
        if args.parallel != "none":
            from pipeline import StagedPipeline, MultiGPUPipeline, create_parallel_pipeline

            # Parse GPU IDs if provided
            gpu_ids = None
            if args.gpu_ids:
                gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

            if args.parallel == "multi_gpu" or (args.parallel == "auto" and gpu_ids and len(gpu_ids) >= 2):
                print(f"\nUsing Multi-GPU pipeline with GPUs: {gpu_ids or 'auto-detect'}")
                parallel_pipeline = MultiGPUPipeline(config, gpu_ids=gpu_ids)
            else:
                print(f"\nUsing Staged pipeline (memory efficient)")
                parallel_pipeline = StagedPipeline(config)

            # Run parallel pipeline
            stats = parallel_pipeline.run(
                df,
                num_workers=args.num_workers,
                resume=args.resume
            )

            print(f"\nCompleted! Stats: {stats}")

            # Export tracking data if requested
            if args.export_tracking:
                df_full = pd.read_csv(csv_path)  # Reload full dataset
                parallel_pipeline.export_tracking(
                    df_full,
                    output_path=args.export_tracking,
                    format="parquet" if args.export_tracking.endswith(".parquet") else "csv"
                )

            # Show tracking summary if requested
            if args.show_tracking:
                summary = parallel_pipeline.get_tracking_summary()
                print("\n" + "="*70)
                print("TRACKING SUMMARY")
                print("="*70)
                for key, value in summary.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")

                # Show pending and failed
                pending = parallel_pipeline.get_pending_items()
                failed = parallel_pipeline.get_failed_items()
                if pending:
                    print(f"\n  Pending items: {len(pending)}")
                if failed:
                    print(f"  Failed items: {len(failed)}")
                    for paper_id in failed[:5]:  # Show first 5
                        print(f"    - {paper_id}")
                    if len(failed) > 5:
                        print(f"    ... and {len(failed) - 5} more")
                print("="*70)

        else:
            # Standard sequential processing
            results = pipeline.process_dataframe(df)

            # Summary
            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            print(f"\nCompleted: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
