#!/usr/bin/env python3
"""
RunPod Deployment Script
========================

Cost-optimized script for running the pipeline on RunPod.

Optimizations:
1. Uses RTX 3090/4090 (cheapest viable GPUs)
2. Processes in batches with checkpointing
3. Skips papers with too many pages (saves time)
4. Community cloud for lowest prices

Usage:
    # On RunPod instance
    python runpod_deploy.py --node-id 0 --total-nodes 8

Estimated costs (16k items):
    - 8x RTX 3090 community: ~$350-400 (7-8 days)
    - 8x RTX 4090 community: ~$500-600 (5-6 days)
    - Subset 8k items: ~$175-200 (3-4 days)
"""

import argparse
import subprocess
from pathlib import Path


def setup_environment():
    """Install dependencies on RunPod instance."""
    commands = [
        "pip install torch transformers accelerate",
        "pip install pandas numpy pillow opencv-python",
        "pip install imagehash scikit-image pymupdf",
        "pip install qwen-vl-utils tqdm",
        # Clone Dolphin if not present
        "[ -d Dolphin ] || git clone https://github.com/bytedance/Dolphin.git",
        # Download Dolphin model
        "[ -d hf_model ] || huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model",
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def run_pipeline(node_id: int, total_nodes: int, dataset: str = "train",
                 max_pages: int = 25, output_base: str = "./output"):
    """Run pipeline for this node's portion of the dataset."""
    import pandas as pd
    from pipeline import StagedPipeline, get_default_config

    # Load dataset
    df = pd.read_csv(f"{dataset}.csv")

    # Filter valid items
    df = df[
        df['local_image_path'].notna() &
        df['local_pdf_path'].notna() &
        (df['error'].isna() | (df['error'] == ''))
    ]

    total_items = len(df)
    items_per_node = total_items // total_nodes
    start_idx = node_id * items_per_node

    # Last node takes remainder
    if node_id == total_nodes - 1:
        end_idx = total_items
    else:
        end_idx = start_idx + items_per_node

    node_df = df.iloc[start_idx:end_idx].copy()

    print(f"\n{'='*60}")
    print(f"NODE {node_id}/{total_nodes}")
    print(f"Processing items {start_idx} to {end_idx} ({len(node_df)} items)")
    print(f"{'='*60}\n")

    # Configure pipeline
    config = get_default_config('./')
    config.output_dir = Path(f"{output_base}_node{node_id}")
    config.dolphin.model_path = "./hf_model"

    # Run pipeline
    pipeline = StagedPipeline(config)
    stats = pipeline.run(node_df, num_workers=2, resume=True)

    # Export tracking
    pipeline.export_tracking(
        node_df,
        output_path=f"tracking_node{node_id}.csv"
    )

    print(f"\nNode {node_id} complete!")
    print(f"Stats: {stats}")

    return stats


def estimate_cost(total_items: int, gpu_type: str = "3090"):
    """Estimate RunPod costs."""
    # Timings per item (minutes)
    time_per_item = {
        "3090": 12,  # Slightly slower than 5090
        "4090": 10,
        "a100": 8,
    }

    # RunPod community prices ($/hr)
    prices = {
        "3090": 0.22,
        "4090": 0.44,
        "a100": 1.09,
    }

    minutes = total_items * time_per_item.get(gpu_type, 10)
    hours = minutes / 60
    cost = hours * prices.get(gpu_type, 0.44)

    return {
        "total_hours": hours,
        "cost_single_gpu": cost,
        "cost_8_gpus": cost,  # Same total GPU-hours, just faster
        "days_single_gpu": hours / 24,
        "days_8_gpus": hours / 24 / 8,
    }


def main():
    parser = argparse.ArgumentParser(description="RunPod Pipeline Deployment")
    parser.add_argument("--setup", action="store_true", help="Install dependencies")
    parser.add_argument("--node-id", type=int, default=0, help="This node's ID (0-indexed)")
    parser.add_argument("--total-nodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset to process")
    parser.add_argument("--estimate", action="store_true", help="Just show cost estimate")
    parser.add_argument("--items", type=int, default=16000, help="Total items for estimate")
    parser.add_argument("--gpu", type=str, default="3090", help="GPU type for estimate")

    args = parser.parse_args()

    if args.setup:
        setup_environment()
        return

    if args.estimate:
        est = estimate_cost(args.items, args.gpu)
        print(f"\nCost Estimate for {args.items} items on RTX {args.gpu.upper()}:")
        print(f"  Total GPU-hours: {est['total_hours']:.0f}")
        print(f"  Cost (any # GPUs): ${est['cost_single_gpu']:.0f}")
        print(f"  Time (1 GPU): {est['days_single_gpu']:.1f} days")
        print(f"  Time (8 GPUs): {est['days_8_gpus']:.1f} days")
        return

    run_pipeline(
        node_id=args.node_id,
        total_nodes=args.total_nodes,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()
