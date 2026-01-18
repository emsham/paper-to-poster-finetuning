# Original GPU Pipeline Documentation

This document provides detailed documentation for the original GPU-based pipeline that uses ByteDance's Dolphin model and Qwen3-VL for processing paper-poster pairs.

> **Note**: This pipeline requires significant GPU resources (6+ GPUs recommended) and takes much longer than the simple pipeline. For most users, the [simple pipeline](../README.md#simple-pipeline-recommended) using Claude API is recommended.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORIGINAL GPU PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐                        ┌──────────────┐
     │   Poster     │                        │    Paper     │
     │   (PNG)      │                        │    (PDF)     │
     └──────┬───────┘                        └──────┬───────┘
            │                                       │
            ▼                                       ▼
   ┌─────────────────────┐              ┌─────────────────────┐
   │   DOLPHIN PARSER    │              │   DOLPHIN PARSER    │
   │   (Qwen2.5-VL)      │              │   (Qwen2.5-VL)      │
   │                     │              │                     │
   │  • Two-stage parse  │              │  • PDF to images    │
   │  • Layout detection │              │  • Page-by-page     │
   │  • OCR extraction   │              │  • Figure extraction│
   │  • ~25 sec/poster   │              │  • ~7 min/paper     │
   └──────────┬──────────┘              └──────────┬──────────┘
              │                                    │
              ▼                                    ▼
         ┌─────────┐                         ┌─────────┐
         │ Poster  │                         │  Paper  │
         │Markdown │                         │Markdown │
         │+ Figures│                         │+ Figures│
         └────┬────┘                         └────┬────┘
              │                                    │
              └──────────────┬─────────────────────┘
                             │
                             ▼
               ┌──────────────────────────┐
               │    VLM FIGURE MATCHER    │
               │    (Qwen3-VL 8B)         │
               │                          │
               │  • SIFT keypoints        │
               │  • Perceptual hash       │
               │  • SSIM similarity       │
               │  • VLM verification      │
               │  • ~30 sec/pair          │
               └────────────┬─────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │   POSTER DESCRIPTOR      │
               │   (Qwen3-VL 8B)          │
               │                          │
               │  • Layout JSON extraction│
               │  • Color scheme          │
               │  • Section structure     │
               │  • ~2 min/poster         │
               └────────────┬─────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │    TRAINING EXAMPLE      │
               │                          │
               │  {                       │
               │    paper_markdown,       │
               │    poster_markdown,      │
               │    poster_layout,        │
               │    figure_matches        │
               │  }                       │
               └──────────────────────────┘
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 24GB (1x RTX 3090) | 80GB (1x A100) or 4x RTX 4090 |
| System RAM | 32GB | 64GB+ |
| Storage | 100GB | 500GB+ (for outputs) |
| GPUs | 1 (very slow) | 4-6 (parallel processing) |

### Estimated Processing Time

| Configuration | Time per Item | 7,000 Items |
|---------------|---------------|-------------|
| 1x RTX 3090 | ~10 min | ~50 days |
| 4x RTX 4090 | ~3 min | ~15 days |
| 6x A100 80GB | ~1.5 min | ~7 days |

---

## File Structure

```
pipeline/
├── config.py              # Configuration dataclasses
├── models.py              # Data models (shared with simple pipeline)
├── dolphin_parser.py      # Dolphin VLM document parser
├── poster_descriptor.py   # Qwen3-VL layout extraction
├── figure_matcher.py      # VLM-based figure matching
├── pipeline.py            # Main orchestrator (single-threaded)
├── parallel.py            # Multi-GPU parallel processing
├── tracking.py            # Processing status tracker
└── OLD_PIPELINE_GUIDE.md  # This file
```

---

## Component Details

### 1. Configuration (`config.py`)

Defines all settings for the pipeline components:

```python
@dataclass
class DolphinConfig:
    """Settings for Dolphin document parser."""
    model_path: str = "./hf_model"           # Path to Dolphin weights
    batch_size: int = 4                       # Batch size for inference
    max_new_tokens: int = 4096               # Max output tokens
    device: str = "cuda"                      # Device to use

@dataclass
class PosterDescriptorConfig:
    """Settings for Qwen3-VL poster layout extraction."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    do_sample: bool = False
    use_flash_attn: bool = True              # Requires flash-attn package

@dataclass
class VLMMatcherConfig:
    """Settings for VLM figure matching."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_vlm: bool = True                     # True = VLM verification pass
    feature_threshold: float = 0.10          # Min score for candidates
    feature_match_threshold: float = 0.25    # Score for "similar"
```

**Full Configuration Example:**

```python
from pipeline.config import PipelineConfig, get_default_config

# Get default config
config = get_default_config(data_dir="./data")

# Customize
config.dolphin.model_path = "/path/to/dolphin/weights"
config.dolphin.batch_size = 8
config.vlm_matcher.model_name = "Qwen/Qwen3-VL-8B-Instruct"
config.vlm_matcher.use_vlm = True
config.poster_descriptor.use_flash_attn = True
config.output_dir = Path("./output")
config.skip_existing = True
config.save_intermediate = True
```

---

### 2. Dolphin Parser (`dolphin_parser.py`)

ByteDance's Dolphin model for document parsing. Uses Qwen2.5-VL architecture.

#### Class: `DolphinParser`

```python
class DolphinParser:
    def __init__(self, config: DolphinConfig):
        self.config = config
        self.model = None
        self._model_loaded = False
```

#### Method: `load_model()`

Loads the Dolphin model into GPU memory:

```python
def load_model(self):
    """
    Loads:
    - Qwen2_5_VLForConditionalGeneration model
    - AutoProcessor for tokenization
    - Sets bfloat16 precision on CUDA
    """
```

#### Method: `parse_poster(image_path, output_dir, doc_id) -> ParsedDocument`

Parses a poster image using two-stage processing:

**Stage 1: Layout Detection**
```python
# Prompt: "Parse the reading order of this document."
# Output: List of bounding boxes with labels
# Labels: "text", "fig", "tab", "equ", "code"
```

**Stage 2: Content Extraction**
```python
# For each detected element:
# - "fig" → Save as image file
# - "tab" → "Parse the table in the image."
# - "equ" → "Read formula in the image."
# - "code" → "Read code in the image."
# - "text" → "Read text in the image."
```

**Processing Flow:**
```
┌─────────────────┐
│  Poster Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Layout Detection      │
│  "Parse the reading order..."   │
│                                 │
│  Output: [(bbox, label, tags)]  │
│  e.g., [([0,0,100,50], "text", [])]
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2: Content Extraction    │
│                                 │
│  For each bbox:                 │
│  • Crop image region            │
│  • Run type-specific prompt     │
│  • Collect results              │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  MarkdownConverter              │
│  • Combine results              │
│  • Apply reading order          │
│  • Generate markdown            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  ParsedDocument                 │
│  • doc_id: "12345"              │
│  • markdown_content: "# Title..."
│  • figures: [ExtractedFigure]   │
│  • metadata: {...}              │
└─────────────────────────────────┘
```

#### Method: `parse_paper(pdf_path, output_dir, doc_id) -> ParsedDocument`

Parses a multi-page PDF:

```python
def parse_paper(self, pdf_path, output_dir, doc_id):
    # 1. Convert PDF to images (one per page)
    images = convert_pdf_to_images(pdf_path)

    # 2. Process each page
    for page_idx, image in enumerate(images):
        results, figures = self._process_single_image(image, ...)

    # 3. Combine all pages into single ParsedDocument
    return ParsedDocument(...)
```

#### Method: `chat(prompt, image) -> str`

Low-level inference method supporting single and batch processing:

```python
def chat(self, prompt: str, image) -> str:
    """
    Args:
        prompt: Text prompt or list of prompts
        image: PIL Image or list of PIL Images

    Returns:
        Model response(s)

    Supports batched inference for efficiency.
    """
```

---

### 3. Poster Descriptor (`poster_descriptor.py`)

Extracts structured layout JSON from poster images using Qwen3-VL.

#### Class: `PosterDescriptor`

```python
class PosterDescriptor:
    def __init__(self, config: PosterDescriptorConfig):
        self.config = config
        self.model = None
        self.processor = None
```

#### Prompt: `LAYOUT_EXTRACTION_PROMPT`

The prompt instructs the model to extract:

```json
{
  "poster": {
    "orientation": "landscape|portrait",
    "aspect_ratio": "16:9",
    "background": "#FFFFFF"
  },
  "header": {
    "height_pct": 15,
    "background": "#hex|gradient",
    "title_alignment": "left|center",
    "logo_positions": ["left", "right"]
  },
  "footer": {
    "present": true,
    "height_pct": 10,
    "content": "contact|references|qr_code"
  },
  "body": {
    "columns": 3,
    "column_widths": ["equal"] or ["30%", "40%", "30%"],
    "gutter_pct": 2
  },
  "sections": [
    {
      "id": 1,
      "title": "Introduction",
      "column": 1,
      "row_in_column": 1,
      "height_pct": 30,
      "style": {
        "header_bg": "#0066CC",
        "body_bg": "#FFFFFF",
        "border": "#CCCCCC"
      },
      "content_type": "text|bullets|figure|mixed"
    }
  ],
  "figures": [
    {
      "id": 1,
      "section_id": 2,
      "type": "line chart|bar chart|diagram|flowchart",
      "description": "Performance comparison graph"
    }
  ],
  "color_scheme": {
    "primary": "#0066CC",
    "secondary": "#003366",
    "accent": "#FF6600",
    "text": "#000000",
    "background": "#FFFFFF"
  },
  "reading_order": "columns-left-to-right"
}
```

#### Method: `describe_poster(image_path, paper_id) -> PosterLayout`

```python
def describe_poster(self, image_path, paper_id, custom_prompt=None):
    # 1. Load and prepare image
    image = Image.open(image_path).convert("RGB")

    # 2. Format messages for Qwen3-VL
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": LAYOUT_EXTRACTION_PROMPT}
    ]}]

    # 3. Generate layout JSON
    output_text = self.model.generate(...)

    # 4. Parse JSON (handles markdown code blocks)
    json_output = self._parse_json_output(output_text)

    # 5. Return PosterLayout object
    return PosterLayout.from_qwen_output(paper_id, image_path, json_output)
```

---

### 4. Figure Matcher (`figure_matcher.py`)

Matches figures between posters and papers using feature extraction + VLM verification.

#### Class: `VLMFigureMatcher`

```python
class VLMFigureMatcher:
    def __init__(self, config: VLMMatcherConfig):
        self.config = config
        self.model = None  # Qwen3-VL (only loaded if use_vlm=True)
```

#### Two-Pass Matching Algorithm

**Pass 1: Feature-Based Filtering**

```python
def compute_feature_score(self, img1_path, img2_path) -> Tuple[float, float]:
    """
    Computes similarity using:
    1. SIFT keypoints (0-1 score based on matches)
    2. Perceptual hash (hamming distance → similarity)
    3. SSIM (structural similarity)

    Returns: (combined_score, sift_score)
    """

    # SIFT matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    sift_score = len(good_matches) / max(len(kp1), len(kp2))

    # Perceptual hash
    hash1 = imagehash.phash(Image.open(img1_path))
    hash2 = imagehash.phash(Image.open(img2_path))
    hash_score = 1 - (hash1 - hash2) / 64

    # SSIM
    ssim_score = ssim(img1_resized, img2_resized, channel_axis=2)

    # Combined (weighted average)
    return 0.3 * sift_score + 0.3 * hash_score + 0.4 * ssim_score
```

**Pass 2: VLM Verification (if `use_vlm=True`)**

```python
def verify_match_vlm(self, img1_path, img2_path) -> Tuple[str, float]:
    """
    Uses Qwen3-VL to compare two images.

    Prompt: "Compare these two images. Are they the same figure
    or showing the same content? Answer: SAME, SIMILAR, or DIFFERENT"

    Returns: (verdict, confidence)
    """
```

#### Method: `match_figures(poster_figures, paper_figures) -> List[FigureMatch]`

```python
def match_figures(self, poster_figures, paper_figures, output_dir=None):
    matches = []

    # Pass 1: Feature-based candidates
    for poster_fig in poster_figures:
        candidates = []
        for paper_fig in paper_figures:
            score, sift = self.compute_feature_score(
                poster_fig.file_path,
                paper_fig.file_path
            )
            if score > self.config.feature_threshold:
                candidates.append((paper_fig, score))

        # Pass 2: VLM verification (if enabled)
        if self.config.use_vlm and candidates:
            for paper_fig, score in candidates:
                verdict, confidence = self.verify_match_vlm(...)
                if verdict in ["SAME", "SIMILAR"]:
                    matches.append(FigureMatch(...))
        else:
            # Use feature threshold directly
            best = max(candidates, key=lambda x: x[1])
            if best[1] > self.config.feature_match_threshold:
                matches.append(FigureMatch(...))

    return matches
```

---

### 5. Pipeline Orchestrator (`pipeline.py`)

Coordinates all components for single-threaded processing.

#### Class: `PaperToPosterPipeline`

```python
class PaperToPosterPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._dolphin_parser = None      # Lazy loaded
        self._figure_matcher = None      # Lazy loaded
        self._poster_descriptor = None   # Lazy loaded
```

#### Method: `process_single(paper_id, poster_path, paper_path, metadata) -> PipelineResult`

Processes one paper-poster pair through all stages:

```python
def process_single(self, paper_id, poster_path, paper_path, metadata=None):
    result = PipelineResult(paper_id=paper_id)

    # Step 1: Parse poster with Dolphin
    print("[1/4] Parsing poster...")
    result.parsed_poster = self.dolphin_parser.parse_poster(
        poster_path, output_dir, paper_id
    )

    # Step 2: Parse paper with Dolphin
    print("[2/4] Parsing paper...")
    result.parsed_paper = self.dolphin_parser.parse_paper(
        paper_path, output_dir, paper_id
    )

    # Step 3: Match figures using VLM
    print("[3/4] Matching figures...")
    result.figure_matches = self.figure_matcher.match_figures(
        result.parsed_poster.figures,
        result.parsed_paper.figures
    )

    # Step 4: Extract poster layout
    print("[4/4] Extracting layout...")
    result.poster_layout = self.poster_descriptor.describe_poster(
        poster_path, paper_id
    )

    # Create training example
    result.training_example = self._create_training_example(...)

    return result
```

#### Method: `process_dataframe(df, limit, start_idx) -> List[PipelineResult]`

Batch processing from a DataFrame:

```python
def process_dataframe(self, df, limit=None, start_idx=0):
    """
    Expected DataFrame columns:
    - paper_id: Unique identifier
    - local_image_path: Path to poster image
    - local_pdf_path: Path to paper PDF
    - title, abstract, conference, year, topics (optional)
    """
    for idx, row in df.iterrows():
        result = self.process_single(
            row['paper_id'],
            row['local_image_path'],
            row['local_pdf_path'],
            metadata={...}
        )
```

---

### 6. Parallel Processing (`parallel.py`)

Efficient parallel processing for large datasets.

#### Class: `StagedPipeline`

Memory-efficient approach that processes all items through one stage before moving to the next:

```python
class StagedPipeline:
    """
    Stages:
    1. dolphin_poster - Parse all posters
    2. dolphin_paper - Parse all papers
    3. figure_match - Match all figures
    4. poster_desc - Extract all layouts
    5. combine - Create training examples

    Benefits:
    - Only one model loaded at a time
    - Checkpointing between stages
    - Can resume interrupted processing
    """
```

**Usage:**

```python
from pipeline.parallel import StagedPipeline
from pipeline.config import get_default_config

config = get_default_config("./data")
pipeline = StagedPipeline(config)

# Run with checkpointing
stats = pipeline.run(
    df=train_df,
    num_workers=4,      # Parallel I/O workers
    batch_size=8,       # Model batch size
    resume=True         # Resume from checkpoint
)
```

**Stage Execution Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: dolphin_poster                                        │
│  • Load Dolphin model                                           │
│  • Process all posters in parallel batches                      │
│  • Save checkpoint                                              │
│  • Unload model, free GPU memory                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: dolphin_paper                                         │
│  • Load Dolphin model (if not cached)                           │
│  • Process all papers                                           │
│  • Save checkpoint                                              │
│  • Unload model                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: figure_match                                          │
│  • Load Qwen3-VL (if use_vlm=True)                              │
│  • Match figures for all pairs                                  │
│  • Save checkpoint                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: poster_desc                                           │
│  • Load Qwen3-VL                                                │
│  • Extract layout for all posters                               │
│  • Save checkpoint                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: combine                                               │
│  • No model needed                                              │
│  • Combine all outputs into TrainingExamples                    │
│  • Export to JSONL                                              │
└─────────────────────────────────────────────────────────────────┘
```

#### Class: `MultiGPUPipeline`

Distributes models across multiple GPUs for parallel processing:

```python
from pipeline.parallel import MultiGPUPipeline

pipeline = MultiGPUPipeline(
    config,
    gpu_ids=[0, 1, 2, 3]  # Use GPUs 0-3
)
pipeline.run(df)
```

**GPU Distribution Strategy:**

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU 0: Dolphin (poster parsing)                                │
│  GPU 1: Dolphin (paper parsing)                                 │
│  GPU 2: Qwen3-VL (figure matching)                              │
│  GPU 3: Qwen3-VL (layout extraction)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7. Processing Tracker (`tracking.py`)

Tracks processing status and enables resumption.

#### Class: `ProcessingTracker`

```python
class ProcessingTracker:
    """
    Maintains a CSV file with processing status for each item.

    Tracked fields:
    - Stage completion (poster_parsed, paper_parsed, etc.)
    - Output paths (markdown, figures, layout JSON)
    - Counts (figures found, matches made)
    - Timing (started_at, completed_at, duration)
    - Errors (error_message)
    """
```

#### Class: `ProcessingStatus`

```python
@dataclass
class ProcessingStatus:
    paper_id: str

    # Stage completion
    poster_parsed: bool = False
    paper_parsed: bool = False
    figures_matched: bool = False
    layout_extracted: bool = False
    training_data_created: bool = False

    # Output paths
    poster_markdown_path: Optional[str] = None
    paper_markdown_path: Optional[str] = None
    figure_matches_path: Optional[str] = None
    layout_json_path: Optional[str] = None

    # Metrics
    poster_figure_count: int = 0
    paper_figure_count: int = 0
    matched_figure_count: int = 0

    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
```

**Usage:**

```python
tracker = ProcessingTracker(output_dir)

# Update status after each stage
tracker.update_stage(
    paper_id="12345",
    stage="poster_parsed",
    success=True,
    poster_markdown_path="poster_parsed/12345/markdown/12345.md",
    poster_figure_count=5
)

# Get pending items for a stage
pending = tracker.get_pending("paper_parsed")

# Save to CSV
tracker.save()  # Creates processing_status.csv
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone Dolphin repository
git clone https://github.com/bytedance/Dolphin.git
cd Dolphin

# Install Dolphin dependencies
pip install -r requirements.txt

# Download Dolphin weights
# Follow instructions in Dolphin README

# Install pipeline dependencies
pip install torch transformers accelerate
pip install opencv-python-headless imagehash scikit-image
pip install pandas numpy pillow pymupdf tqdm
pip install flash-attn --no-build-isolation  # Optional, for faster attention
```

### 2. Prepare Data

Create a CSV file with your paper-poster pairs:

```csv
paper_id,local_image_path,local_pdf_path,title,abstract
12345,images/12345.png,pdfs/12345.pdf,"Paper Title","Paper abstract..."
```

### 3. Run Pipeline

**Option A: Single-threaded (simple)**

```python
from pipeline import create_pipeline

pipeline = create_pipeline(
    data_dir="./data",
    output_dir="./output",
    dolphin_model_path="./Dolphin/hf_model"
)

# Process single item
result = pipeline.process_single(
    paper_id="12345",
    poster_path="./data/images/12345.png",
    paper_path="./data/pdfs/12345.pdf"
)

# Process from DataFrame
import pandas as pd
df = pd.read_csv("train.csv")
results = pipeline.process_dataframe(df, limit=100)

# Export training data
pipeline.export_training_data("training_data.jsonl")
```

**Option B: Staged processing (memory efficient)**

```python
from pipeline.parallel import StagedPipeline
from pipeline.config import get_default_config

config = get_default_config("./data")
config.dolphin.model_path = "./Dolphin/hf_model"

pipeline = StagedPipeline(config)
stats = pipeline.run(df, num_workers=4, batch_size=8, resume=True)
```

**Option C: Multi-GPU (fastest)**

```python
from pipeline.parallel import MultiGPUPipeline

pipeline = MultiGPUPipeline(config, gpu_ids=[0, 1, 2, 3])
pipeline.run(df)
```

---

## Output Structure

```
output/
├── checkpoint.json              # Processing checkpoint
├── processing_status.csv        # Detailed status tracker
├── poster_parsed/
│   └── {paper_id}/
│       ├── markdown/
│       │   ├── {paper_id}.md   # Poster markdown
│       │   └── figures/
│       │       ├── fig_001.png
│       │       └── fig_002.png
│       └── layout.json          # Raw recognition results
├── paper_parsed/
│   └── {paper_id}/
│       └── markdown/
│           ├── {paper_id}.md   # Paper markdown
│           └── figures/
├── figure_matches/
│   └── {paper_id}/
│       └── matches.json         # Figure match results
├── poster_layouts/
│   └── {paper_id}_layout.json   # Layout JSON
└── training_data/
    ├── {paper_id}.json          # Individual training examples
    └── training_data.jsonl      # Combined export
```

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
config.dolphin.batch_size = 1

# Use staged pipeline (unloads models between stages)
pipeline = StagedPipeline(config)

# Disable VLM figure matching
config.vlm_matcher.use_vlm = False
```

### Slow Processing

```python
# Enable flash attention
config.poster_descriptor.use_flash_attn = True

# Increase batch size (if VRAM allows)
config.dolphin.batch_size = 8

# Use multi-GPU
pipeline = MultiGPUPipeline(config, gpu_ids=[0, 1, 2, 3])
```

### Resume After Crash

```python
# Staged pipeline auto-resumes
pipeline = StagedPipeline(config)
pipeline.run(df, resume=True)  # Continues from checkpoint

# Check status
tracker = ProcessingTracker(config.output_dir)
pending = tracker.get_pending("paper_parsed")
print(f"Pending: {len(pending)} items")
```

---

## Comparison with Simple Pipeline

| Aspect | Original Pipeline | Simple Pipeline |
|--------|-------------------|-----------------|
| Paper parsing | Dolphin (Qwen2.5-VL) | marker-pdf (CPU) |
| Poster parsing | Dolphin (Qwen2.5-VL) | Claude Haiku API |
| Layout extraction | Qwen3-VL | Claude Haiku API |
| Figure matching | SIFT + Qwen3-VL | SIFT only |
| Hardware | 4-6 GPUs | CPU + internet |
| Cost (7k items) | ~$500 (GPU rental) | ~$22 (API) |
| Time (7k items) | 7-15 days | 12 hours |
| Quality | Highest | Good |

**When to use Original Pipeline:**
- Maximum quality needed
- Have access to multiple GPUs
- Processing very large datasets (cost amortization)
- Need offline processing

**When to use Simple Pipeline:**
- Budget-conscious
- Quick turnaround needed
- No GPU access
- Most use cases
