# Paper-to-Poster Finetuning Pipeline

A pipeline for creating training data from academic paper-poster pairs. The goal is to finetune an LLM that can generate academic posters from research papers.

## Overview

This pipeline processes ~16k paper-poster pairs to create training data where:
- **Input**: Research paper (markdown + metadata)
- **Output**: Poster layout (JSON) + poster content (markdown)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PAPER-TO-POSTER PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐     ┌──────────────┐
                    │   Poster     │     │    Paper     │
                    │   (PNG)      │     │    (PDF)     │
                    └──────┬───────┘     └──────┬───────┘
                           │                    │
                           ▼                    ▼
              ┌────────────────────────────────────────────┐
              │              DOLPHIN PARSER                │
              │    (Document Understanding Model)          │
              │                                            │
              │  • Extracts markdown content               │
              │  • Extracts figures, tables, equations     │
              │  • Preserves reading order                 │
              └────────────────┬───────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   │                   ▼
    ┌─────────────┐            │            ┌─────────────┐
    │   Poster    │            │            │    Paper    │
    │  Markdown   │            │            │  Markdown   │
    │  + Figures  │            │            │  + Figures  │
    └──────┬──────┘            │            └──────┬──────┘
           │                   │                   │
           │                   ▼                   │
           │    ┌──────────────────────────┐       │
           │    │    VLM FIGURE MATCHER    │       │
           └───►│     (Qwen3-VL-8B)        │◄──────┘
                │                          │
                │  Pass 1: Feature filter  │
                │  • SIFT keypoints        │
                │  • Perceptual hash       │
                │  • SSIM similarity       │
                │                          │
                │  Pass 2: VLM comparison  │
                │  • Semantic matching     │
                │  • Same/Similar/Different│
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │    POSTER DESCRIPTOR     │
                │     (Qwen3-VL-8B)        │
                │                          │
                │  Extracts JSON layout:   │
                │  • Grid structure        │
                │  • Section positions     │
                │  • Color scheme          │
                │  • Figure placements     │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │    TRAINING EXAMPLE      │
                │                          │
                │  {                       │
                │    input: paper_md,      │
                │    output: {             │
                │      layout: {...},      │
                │      content: poster_md, │
                │      figures: [...]      │
                │    }                     │
                │  }                       │
                └──────────────────────────┘
```

## Requirements

### Hardware
- **Minimum**: 1x GPU with 24GB+ VRAM (RTX 3090/4090/5090)
- **Recommended**: 2-3x GPUs for parallel processing

### Software
```bash
# Python 3.10+
pip install torch transformers accelerate
pip install pandas numpy pillow opencv-python
pip install imagehash scikit-image pymupdf
pip install qwen-vl-utils tqdm
```

### Models
1. **Dolphin** (Document Parser): ByteDance/Dolphin-v2
2. **Qwen3-VL** (Vision-Language): Qwen/Qwen3-VL-8B-Instruct

```bash
# Download Dolphin model
huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model

# Clone Dolphin repo (for utilities)
git clone https://github.com/bytedance/Dolphin.git
```

## Quick Start

### 1. Test the Pipeline

```bash
# Dry run (check setup without loading models)
python test_pipeline.py --dry-run

# Full test on a single sample
python run_pipeline.py --single \
    --paper-id 37030 \
    --poster archive/test_set/37030.png \
    --paper "archive/test_set/paper.pdf" \
    --dolphin-model /path/to/hf_model
```

### 2. Process Dataset

```bash
# Single GPU (staged - memory efficient)
python run_pipeline.py \
    --parallel staged \
    --dolphin-model /path/to/hf_model \
    --dataset train \
    --resume

# Multi-GPU (maximum throughput)
python run_pipeline.py \
    --parallel multi_gpu \
    --gpu-ids 0,1,2 \
    --dolphin-model /path/to/hf_model \
    --dataset train
```

### 3. Export Training Data

```bash
python run_pipeline.py --export --export-output training_data.jsonl
```

## Pipeline Stages

### Stage 1: Dolphin Poster Parsing
Parses poster images to extract:
- Markdown content with structure
- Figures saved as PNG files
- Layout visualization

### Stage 2: Dolphin Paper Parsing
Parses PDF papers (multi-page) to extract:
- Full paper content as markdown
- All figures from each page
- Reading order preserved

### Stage 3: VLM Figure Matching
Matches figures between poster and paper:
- **Pass 1**: Fast feature filtering (SIFT, hash, SSIM)
- **Pass 2**: Qwen3-VL semantic comparison
- Outputs: same/similar/different with confidence

### Stage 4: Poster Layout Description
Extracts structured JSON describing:
- Orientation, aspect ratio, colors
- Column/grid structure
- Section positions and styling
- Figure placements

### Stage 5: Combine
Merges all outputs into training examples.

## Parallel Processing Strategies

### Staged Pipeline (Single GPU)
```
Time ──────────────────────────────────────────────────────►

GPU 0: [Dolphin Poster]──►[Dolphin Paper]──►[VLM Match]──►[VLM Desc]
                          ↓ unload          ↓ unload      ↓ unload
```
- Processes all items through each stage
- Unloads models between stages
- Memory efficient (~16GB VRAM)
- Checkpoints between stages

### Multi-GPU Pipeline (2+ GPUs)
```
Time ──────────────────────────────────────────────────────►

GPU 0: [Dolphin Poster ──────────────────────────────────►]
GPU 1: [Dolphin Paper ───────────────────────────────────►]
GPU 2: [VLM (waits)...][VLM Match + Describe ────────────►]
                       ▲
                       └── starts when items ready
```
- Models run in parallel on different GPUs
- Queue-based pipelining between stages
- ~2-3x faster than staged

## Output Structure

```
pipeline_output/
├── poster_parsed/{paper_id}/
│   ├── markdown/
│   │   ├── {paper_id}.md           # Poster content
│   │   └── figures/                # Extracted figures
│   ├── output_json/                # Raw recognition results
│   └── layout_visualization/       # Visual layout overlay
│
├── paper_parsed/{paper_id}/
│   ├── markdown/
│   │   ├── {paper_id}.md           # Paper content
│   │   └── figures/                # Extracted figures
│   └── ...
│
├── figure_matches/{paper_id}/
│   ├── same/                       # Confirmed same figures
│   ├── similar/                    # Similar figures
│   ├── uncertain/                  # Low confidence matches
│   └── report.json                 # Match details
│
├── poster_descriptions/
│   └── {paper_id}_layout.json      # Poster layout JSON
│
├── training_data/
│   └── {paper_id}.json             # Final training example
│
└── checkpoint.json                 # Resume checkpoint
```

## Training Data Format

Each training example (`training_data/{paper_id}.json`):

```json
{
  "paper_id": "12345",
  "paper_title": "Paper Title",
  "paper_abstract": "Abstract text...",
  "paper_markdown": "# Full paper content...",
  "poster_markdown": "# Poster content...",
  "poster_layout": {
    "poster": {"orientation": "landscape", "aspect_ratio": "16:9"},
    "header": {"height_pct": 10, "title_alignment": "center"},
    "body": {"columns": 3, "column_widths": ["equal"]},
    "sections": [
      {"id": 1, "title": "Introduction", "column": 1, ...}
    ],
    "figures": [...],
    "color_scheme": {"primary": "#0033A0", ...}
  },
  "figure_matches": [
    {"poster_figure": "fig_1", "paper_figure": "page_3_fig_1", ...}
  ]
}
```

For LLM finetuning (JSONL export):
```json
{
  "instruction": "Generate an academic poster...",
  "input": "# Paper markdown content...",
  "output": "{\"layout\": {...}, \"content\": \"...\"}"
}
```

## Configuration

Edit `pipeline/config.py` or pass CLI arguments:

| Config | Default | Description |
|--------|---------|-------------|
| `dolphin.model_path` | `./hf_model` | Path to Dolphin model |
| `vlm_matcher.model_name` | `Qwen/Qwen3-VL-8B-Instruct` | VLM for matching |
| `vlm_matcher.feature_threshold` | `0.10` | Min feature ratio for candidates |
| `skip_existing` | `True` | Skip already processed items |

## CLI Reference

```bash
python run_pipeline.py [OPTIONS]

Dataset Processing:
  --dataset {train,test,validation}  Which split to process
  --limit N                          Process only N items
  --start N                          Start from index N

Parallel Processing:
  --parallel {none,staged,multi_gpu,auto}
  --num-workers N                    I/O workers (default: 4)
  --gpu-ids 0,1,2                    GPUs for multi-GPU mode
  --resume / --no-resume             Resume from checkpoint

Models:
  --dolphin-model PATH               Path to Dolphin model
  --vlm-model NAME                   VLM model name
  --use-flash-attn                   Enable flash attention

Single Item:
  --single --paper-id ID --poster PATH --paper PATH

Export:
  --export --export-output FILE --export-format {jsonl,json,parquet}

Tracking:
  --export-tracking FILE             Export tracking CSV merged with source data
  --show-tracking                    Show processing summary after completion
```

## Processing Tracking

The pipeline tracks processing status for each paper-poster pair:

```python
from pipeline import ProcessingTracker, create_tracker

# Create or load existing tracker
tracker = create_tracker("./pipeline_output")

# Check processing status
pending = tracker.get_pending("poster_parsed")  # Items not yet parsed
failed = tracker.get_failed()                    # Items with errors
summary = tracker.get_summary()                  # Overall statistics

# Export tracking data merged with source CSV
import pandas as pd
df = pd.read_csv("train.csv")
merged = tracker.merge_with_source(df)
merged.to_csv("train_with_tracking.csv")
```

Tracking CSV includes columns:
- `poster_parsed`, `paper_parsed`, `figures_matched`, `layout_extracted`, `training_data_created` (boolean flags)
- `poster_markdown_path`, `paper_markdown_path`, `figure_matches_path`, etc. (relative paths)
- `poster_figure_count`, `paper_figure_count`, `matched_figure_count`, `section_count`
- `error_message`, `started_at`, `completed_at`, `processing_time_seconds`

## Troubleshooting

### CUDA Out of Memory
- Use `--parallel staged` for single GPU
- Reduce batch size in config
- Enable `--use-flash-attn`

### Missing Dependencies
```bash
pip install pymupdf qwen-vl-utils
```

### Resume Failed Job
```bash
python run_pipeline.py --parallel staged --resume
```
Checkpoint saved in `pipeline_output/checkpoint.json`

## Performance

Benchmarks on RTX 5090 (32GB):

| Stage | Time per item |
|-------|---------------|
| Parse poster | ~25s |
| Parse paper (20 pages) | ~7 min |
| Figure matching | ~25s |
| Layout description | ~2 min |

Estimated full dataset (16k items):
- **Staged (1 GPU)**: ~10 min/item → ~111 days
- **Multi-GPU (3x)**: ~3-4 min/item → ~37 days
- **Parallel workers (8x GPUs)**: ~1.5 min/item → ~17 days

## License

This pipeline is for research purposes. Please check licenses for:
- Dolphin: [ByteDance License](https://github.com/bytedance/Dolphin)
- Qwen3-VL: [Qwen License](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
