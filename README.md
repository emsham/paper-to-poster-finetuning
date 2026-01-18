# Paper-to-Poster Finetuning Pipeline

A pipeline for creating training data from academic paper-poster pairs. The goal is to finetune an LLM that can generate academic posters from research papers.

## Results

We processed **7,351 paper-poster pairs** and successfully generated **7,307 training examples** (99.4% success rate).

| Metric | Value |
|--------|-------|
| Training examples | 7,307 |
| Success rate | 99.4% |
| Total API cost | ~$22 |
| Processing time | ~12 hours |
| Output file | `training_data.jsonl` (291 MB) |

## Overview

This pipeline processes paper-poster pairs to create training data where:
- **Input**: Research paper (markdown + metadata)
- **Output**: Poster layout (JSON) + poster content (markdown)

### Two Pipeline Options

| Pipeline | Cost (7k items) | Time | Hardware | Best For |
|----------|-----------------|------|----------|----------|
| **Simple** (Recommended) | ~$22 | ~12 hrs | CPU + API | Most users, budget-conscious |
| **Original** | ~$500+ | 10+ days | 6+ GPUs | Maximum quality, research |

---

## Simple Pipeline (Recommended)

The simple pipeline uses **marker-pdf** (CPU) for paper parsing and **Claude Haiku API** (~$0.003/poster) for poster processing. It's 20x cheaper and 10x faster than the original GPU-based pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIMPLE PIPELINE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐     ┌──────────────┐
                    │   Poster     │     │    Paper     │
                    │   (PNG)      │     │    (PDF)     │
                    └──────┬───────┘     └──────┬───────┘
                           │                    │
                           ▼                    ▼
              ┌─────────────────────┐  ┌─────────────────────┐
              │   CLAUDE HAIKU API  │  │    MARKER-PDF       │
              │   (~$0.003/poster)  │  │    (Free, CPU)      │
              │                     │  │                     │
              │  • Transcribes text │  │  • Extracts text    │
              │  • Extracts layout  │  │  • Extracts figures │
              │  • Identifies figs  │  │  • Preserves format │
              └──────────┬──────────┘  └──────────┬──────────┘
                         │                        │
                         ▼                        ▼
                    ┌─────────┐              ┌─────────┐
                    │ Poster  │              │  Paper  │
                    │Markdown │              │Markdown │
                    │+ Layout │              │+ Figures│
                    └────┬────┘              └────┬────┘
                         │                        │
                         └──────────┬─────────────┘
                                    ▼
                      ┌──────────────────────────┐
                      │   FIGURE MATCHER         │
                      │   (Feature-based only)   │
                      │                          │
                      │  • SIFT keypoints        │
                      │  • Perceptual hash       │
                      │  • SSIM similarity       │
                      │  • NO VLM (fast & free)  │
                      └────────────┬─────────────┘
                                   ▼
                      ┌──────────────────────────┐
                      │    TRAINING EXAMPLE      │
                      │                          │
                      │  {                       │
                      │    input: paper_md,      │
                      │    output: {             │
                      │      layout: {...},      │
                      │      content: poster_md  │
                      │    }                     │
                      │  }                       │
                      └──────────────────────────┘
```

### Quick Start (Simple Pipeline)

#### 1. Install Dependencies

```bash
pip install marker-pdf anthropic httpx
pip install opencv-python-headless imagehash scikit-image
pip install pandas numpy pillow pymupdf tqdm
```

#### 2. Set API Key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Get your API key at: https://console.anthropic.com/

#### 3. Run the Pipeline

```bash
# Test on 10 items first
python run_simple_pipeline.py --data-dir . --limit 10

# Run full dataset
python run_simple_pipeline.py --data-dir . --concurrency 30

# Export results
python run_simple_pipeline.py --data-dir . --export-only
```

#### 4. Programmatic Usage

```python
from pipeline import create_simple_pipeline
import pandas as pd

# Create pipeline
pipe = create_simple_pipeline(
    data_dir="./",
    api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
)

# Process from DataFrame
df = pd.read_csv("train.csv")
results, stats = pipe.process_dataframe(df, limit=100, concurrency=20)

# Export training data
pipe.export_training_data("training_data.jsonl")

print(f"Processed: {stats.completed}, Failed: {stats.failed}")
print(f"Cost: ${stats.api_cost_estimate:.2f}")
```

---

## Simple Pipeline: Code Walkthrough

This section explains every component of the simple pipeline in detail.

### File Structure

```
pipeline/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclasses
├── models.py                # Data models (ParsedDocument, PosterLayout, etc.)
├── marker_parser.py         # Paper parsing with marker-pdf
├── claude_poster_processor.py  # Poster processing with Claude API
├── figure_matcher.py        # Figure matching (feature-based)
├── simple_pipeline.py       # Main orchestrator
└── ...                      # Original pipeline files
```

### 1. Configuration (`config.py`)

Defines settings for each component:

```python
@dataclass
class MarkerConfig:
    """Settings for marker-pdf paper parser."""
    extract_figures: bool = True      # Extract images from PDF
    min_figure_size: int = 100        # Skip images smaller than 100px
    output_format: str = "markdown"   # Output format

@dataclass
class ClaudeConfig:
    """Settings for Claude API poster processor."""
    api_key: str = ""                 # From ANTHROPIC_API_KEY env var
    model: str = "claude-3-haiku-20240307"  # Cheapest vision model
    max_tokens: int = 4096            # Max response length
    temperature: float = 0.1          # Low = more deterministic
    timeout: int = 60                 # API timeout in seconds
    max_retries: int = 3              # Retry on failure

@dataclass
class VLMMatcherConfig:
    """Settings for figure matching."""
    use_vlm: bool = False             # False = feature-only (fast)
    feature_threshold: float = 0.10   # Min score to consider a match
    feature_match_threshold: float = 0.25  # Score for "similar" verdict
```

**How to customize:**
```python
from pipeline import PipelineConfig, ClaudeConfig

config = PipelineConfig()
config.claude.model = "claude-3-haiku-20240307"
config.claude.max_tokens = 8192
config.vlm_matcher.use_vlm = False  # Keep this False for simple pipeline
```

### 2. Data Models (`models.py`)

Defines the data structures passed between components:

#### `ParsedDocument`
Represents a parsed paper or poster:
```python
@dataclass
class ParsedDocument:
    doc_id: str              # Unique identifier (e.g., "19205")
    doc_type: str            # "poster" or "paper"
    source_path: str         # Path to original file
    markdown_content: str    # Extracted text as markdown
    figures: List[ExtractedFigure]  # List of extracted images
    tables: List[Dict]       # Extracted tables (if any)
    equations: List[str]     # Extracted equations (if any)
    metadata: Dict           # Additional info (parser used, paths, etc.)
```

#### `PosterLayout`
Represents the visual structure of a poster:
```python
@dataclass
class PosterLayout:
    paper_id: str            # Links to the paper
    poster_path: str         # Path to poster image
    orientation: str         # "landscape" or "portrait"
    aspect_ratio: str        # e.g., "16:9", "4:3"
    background: str          # Hex color, e.g., "#FFFFFF"
    header: Dict             # Header section details
    footer: Dict             # Footer section details
    body: Dict               # Body layout (columns, widths)
    sections: List[Dict]     # List of content sections
    figures: List[Dict]      # Figure placements
    color_scheme: Dict       # Primary, secondary, accent colors
    reading_order: str       # "columns-left-to-right", etc.
```

#### `TrainingExample`
The final output combining everything:
```python
@dataclass
class TrainingExample:
    paper_id: str
    paper_markdown: str      # Full paper content
    paper_abstract: str      # Paper abstract
    paper_title: str         # Paper title
    poster_layout: PosterLayout  # Visual structure
    poster_markdown: str     # Poster text content
    figure_matches: List[FigureMatch]  # Matched figures
    conference: str          # e.g., "ICLR"
    year: int                # e.g., 2024
    topics: List[str]        # Topic tags
```

### 3. Paper Parser (`marker_parser.py`)

Parses research paper PDFs using the `marker-pdf` library.

#### Class: `MarkerParser`

```python
class MarkerParser:
    def __init__(self, config: MarkerConfig):
        self.config = config
        self._marker_available = None  # Lazy check for marker-pdf
```

#### Method: `parse_paper(pdf_path, output_dir, doc_id) -> ParsedDocument`

**What it does:**
1. Checks if `marker-pdf` is installed (falls back to PyMuPDF if not)
2. Converts PDF to markdown text
3. Extracts embedded images as figure files
4. Returns a `ParsedDocument` with content and figures

**Step-by-step:**
```python
def parse_paper(self, pdf_path, output_dir, doc_id):
    # 1. Create output directories
    markdown_dir = output_dir / "markdown"
    figures_dir = markdown_dir / "figures"

    # 2. Parse PDF to markdown
    if self._check_marker():
        # Use marker-pdf (better quality)
        markdown_content = self._parse_with_marker(pdf_path)
    else:
        # Fallback to PyMuPDF (basic text extraction)
        markdown_content = self._parse_with_pymupdf(pdf_path)

    # 3. Extract figures from PDF
    figures = self._extract_figures(pdf_path, figures_dir, doc_id)

    # 4. Save markdown to file
    with open(markdown_dir / f"{doc_id}.md", 'w') as f:
        f.write(markdown_content)

    # 5. Return structured result
    return ParsedDocument(
        doc_id=doc_id,
        doc_type="paper",
        source_path=str(pdf_path),
        markdown_content=markdown_content,
        figures=figures,
        metadata={"parser": "marker" if self._marker_available else "pymupdf"}
    )
```

#### Method: `_extract_figures(pdf_path, output_dir, doc_id) -> List[ExtractedFigure]`

Extracts images embedded in the PDF:
```python
def _extract_figures(self, pdf_path, output_dir, doc_id):
    figures = []
    doc = fitz.open(pdf_path)  # PyMuPDF

    for page_num, page in enumerate(doc):
        for img_info in page.get_images():
            # Extract image bytes
            base_image = doc.extract_image(img_info[0])

            # Skip small images (icons, artifacts)
            if base_image["width"] < 100 or base_image["height"] < 100:
                continue

            # Save to file
            fig_path = output_dir / f"{doc_id}_fig_{fig_count}.png"
            with open(fig_path, 'wb') as f:
                f.write(base_image["image"])

            figures.append(ExtractedFigure(
                figure_id=f"{doc_id}_fig_{fig_count}",
                source_doc="paper",
                file_path=str(fig_path),
                page_number=page_num + 1
            ))

    return figures
```

### 4. Poster Processor (`claude_poster_processor.py`)

Processes poster images using Claude Haiku API to extract both content and layout.

#### Class: `ClaudePosterProcessor`

```python
class ClaudePosterProcessor:
    COST_PER_POSTER = 0.003  # Estimated cost in USD

    def __init__(self, config: ClaudeConfig):
        self.config = config
        if not config.api_key:
            raise ValueError("Claude API key required")
```

#### Method: `process_poster(image_path, paper_id) -> Tuple[ParsedDocument, PosterLayout]`

**What it does:**
1. Encodes the poster image as base64
2. Sends it to Claude with a detailed prompt
3. Parses the response into content (markdown) and layout (JSON)
4. Returns both as structured objects

**The prompt sent to Claude:**
```
Analyze this academic poster image and extract both its content and layout.

## PART 1: CONTENT EXTRACTION
Transcribe ALL text content in markdown format:
- Title (as # heading)
- Authors and affiliations
- All section headers (as ## headings)
- Body text, equations, figure captions

## PART 2: LAYOUT STRUCTURE
Describe the visual layout as JSON:
{
  "poster": {"orientation": "...", "aspect_ratio": "...", "background": "#..."},
  "header": {"height_pct": ..., "title_alignment": "..."},
  "body": {"columns": N, "column_widths": [...]},
  "sections": [...],
  "figures": [...],
  "color_scheme": {...}
}

Return as:
---CONTENT---
[markdown]
---LAYOUT---
[json]
```

**API call implementation:**
```python
async def process_poster_async(self, image_path, paper_id):
    # 1. Encode image
    image_data, media_type = self._encode_image(image_path)

    # 2. Prepare API request
    payload = {
        "model": self.config.model,
        "max_tokens": self.config.max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }},
                {"type": "text", "text": POSTER_EXTRACTION_PROMPT}
            ]
        }]
    }

    # 3. Call API with retries
    async with httpx.AsyncClient() as client:
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": self.config.api_key, ...},
                    json=payload
                )
                response_text = response.json()["content"][0]["text"]
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    # 4. Parse response
    markdown_content, layout_json = self._parse_response(response_text)

    # 5. Return structured objects
    return (
        ParsedDocument(doc_id=paper_id, doc_type="poster", ...),
        PosterLayout.from_qwen_output(paper_id, image_path, layout_json)
    )
```

#### Method: `process_batch_async(items, concurrency=20)`

Processes multiple posters concurrently:
```python
async def process_batch_async(self, items, concurrency=20):
    semaphore = asyncio.Semaphore(concurrency)  # Limit concurrent calls

    async def process_one(image_path, paper_id):
        async with semaphore:
            return await self.process_poster_async(image_path, paper_id)

    tasks = [process_one(img, pid) for img, pid in items]
    return await asyncio.gather(*tasks)
```

### 5. Figure Matcher (`figure_matcher.py`)

Matches figures between poster and paper using image similarity features.

#### Class: `VLMFigureMatcher`

In simple pipeline mode (`use_vlm=False`), only uses fast feature-based matching.

#### Method: `compute_feature_score(img1_path, img2_path) -> Tuple[float, float]`

Computes similarity using three techniques:
```python
def compute_feature_score(self, img1_path, img2_path):
    # Load images
    pil1, pil2 = Image.open(img1_path), Image.open(img2_path)

    # 1. Perceptual Hash (pHash)
    # Creates a fingerprint of the image's structure
    phash1 = imagehash.phash(pil1, hash_size=16)
    phash2 = imagehash.phash(pil2, hash_size=16)
    phash_score = max(0, 1 - (phash1 - phash2) / 64)  # 0-1, higher=more similar

    # 2. SSIM (Structural Similarity Index)
    # Compares luminance, contrast, and structure
    ssim_score = ssim(grayscale1, grayscale2)  # 0-1

    # 3. SIFT Features
    # Finds and matches keypoints between images
    sift = cv2.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    feature_ratio = len(good_matches) / min(len(kp1), len(kp2))

    # Combined score (weighted average)
    combined = 0.3 * phash_score + 0.3 * ssim_score + 0.4 * feature_ratio

    return feature_ratio, combined
```

#### Method: `match_figures(poster_figures, paper_figures) -> List[FigureMatch]`

**With `use_vlm=False` (simple pipeline):**
```python
def match_figures(self, poster_figures, paper_figures):
    # PASS 1: Compute scores for all pairs
    candidates = []
    for pf in poster_figures:
        for rf in paper_figures:
            ratio, score = self.compute_feature_score(pf.file_path, rf.file_path)
            if ratio >= self.config.feature_threshold:  # 0.10 default
                candidates.append(MatchCandidate(pf, rf, ratio, score))

    # PASS 2 (feature-only): Convert to matches based on score
    matches = []
    for candidate in candidates:
        if candidate.combined_score >= 0.5:
            confidence = "high"
        elif candidate.combined_score >= 0.25:
            confidence = "medium"
        else:
            continue

        matches.append(FigureMatch(
            poster_figure=candidate.poster_figure,
            paper_figure=candidate.paper_figure,
            match_confidence=confidence,
            feature_score=candidate.feature_ratio,
            vlm_verdict="same" if confidence == "high" else "similar",
            vlm_reasoning="Feature-based matching (VLM disabled)"
        ))

    return matches
```

### 6. Simple Pipeline Orchestrator (`simple_pipeline.py`)

Ties everything together.

#### Class: `SimplePipeline`

```python
class SimplePipeline:
    COST_PER_POSTER = 0.003

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.use_simple_pipeline = True
        self.config.vlm_matcher.use_vlm = False

        # Lazy-loaded components
        self._marker = None
        self._claude = None
        self._matcher = None
```

#### Method: `process_single(paper_id, poster_path, paper_path, metadata)`

Processes one paper-poster pair:
```python
def process_single(self, paper_id, poster_path, paper_path, metadata):
    # Step 1: Parse paper with marker-pdf (~10 seconds)
    parsed_paper = self.marker.parse_paper(paper_path, output_dir, paper_id)

    # Step 2: Process poster with Claude (~10 seconds)
    parsed_poster, poster_layout = self.claude.process_poster(poster_path, paper_id)

    # Step 3: Match figures (~1 second)
    figure_matches = self.matcher.match_figures(
        parsed_poster.figures,
        parsed_paper.figures
    )

    # Step 4: Create training example
    training_example = TrainingExample(
        paper_id=paper_id,
        paper_markdown=parsed_paper.markdown_content,
        paper_title=metadata.get("title", ""),
        paper_abstract=metadata.get("abstract", ""),
        poster_layout=poster_layout,
        poster_markdown=parsed_poster.markdown_content,
        figure_matches=figure_matches
    )

    # Step 5: Save to disk
    training_example.save(output_path / f"{paper_id}.json")

    return PipelineResult(success=True, ...)
```

#### Method: `process_dataframe(df, limit, concurrency)`

Batch processes a DataFrame:
```python
def process_dataframe(self, df, limit=None, concurrency=20):
    # 1. Validate and prepare items
    items = []
    for _, row in df.iterrows():
        if Path(row['local_image_path']).exists() and \
           Path(row['local_pdf_path']).exists():
            items.append((row['paper_id'], row['local_image_path'], ...))

    # 2. Run async batch processing
    return asyncio.run(
        self.process_batch_async(items, concurrency)
    )
```

---

## Output Format

### Training Data Structure

Each processed item produces `training_data/{paper_id}.json`:

```json
{
  "paper_id": "19205",
  "paper_title": "A Fast and Provable Algorithm for Sparse Phase Retrieval",
  "paper_abstract": "We study the sparse phase retrieval problem...",
  "paper_markdown": "## Page 1\n\nPublished as a conference paper...",
  "poster_markdown": "# A Fast and Provable Algorithm...\n\n## Introduction\n...",
  "poster_layout": {
    "poster": {
      "orientation": "landscape",
      "aspect_ratio": "16:9",
      "background": "#ffffff"
    },
    "header": {
      "height_pct": 20,
      "background": "linear-gradient(to right, #0072C6, #00B0F0)",
      "title_alignment": "left",
      "logo_positions": ["left", "right"]
    },
    "body": {
      "columns": 3,
      "column_widths": ["30%", "40%", "30%"],
      "gutter_pct": 5
    },
    "sections": [
      {
        "id": 1,
        "title": "Introduction",
        "column": 1,
        "row_in_column": 1,
        "content_type": "text",
        "has_figures": false
      },
      {
        "id": 2,
        "title": "Main Results",
        "column": 2,
        "content_type": "mixed",
        "has_figures": true
      }
    ],
    "figures": [
      {"id": 1, "section_id": 2, "type": "chart"}
    ],
    "color_scheme": {
      "primary": "#0072C6",
      "secondary": "#00B0F0",
      "text": "#000000",
      "background": "#ffffff"
    },
    "reading_order": "columns-left-to-right"
  },
  "figure_matches": [],
  "conference": "ICLR",
  "year": 2024,
  "topics": ["Signal Processing", "Optimization"]
}
```

### Exported Training Format (JSONL)

For LLM finetuning:

```json
{
  "instruction": "Generate an academic poster for the following research paper.\n\nTitle: A Fast and Provable Algorithm...\n\nAbstract: We study...",
  "input": "## Page 1\n\nPublished as a conference paper...",
  "output": "{\"layout\": {...}, \"content\": \"# A Fast and Provable...\"}"
}
```

---

## CLI Reference

### Simple Pipeline CLI

```bash
python run_simple_pipeline.py [OPTIONS]

Required:
  --data-dir, -d PATH     Directory containing train.csv and data files

Optional:
  --api-key KEY           Anthropic API key (or set ANTHROPIC_API_KEY)
  --dataset SPLIT         train, test, or validation (default: train)
  --limit N               Max items to process
  --start N               Start index (default: 0)
  --concurrency N         Max concurrent API calls (default: 20)
  --output-dir PATH       Output directory
  --export-only           Only export existing results
  --export-format FMT     jsonl, json, or parquet (default: jsonl)
```

### Examples

```bash
# Test on 10 items
python run_simple_pipeline.py -d . --limit 10

# Full dataset with high concurrency
python run_simple_pipeline.py -d . --concurrency 50

# Export only (after processing)
python run_simple_pipeline.py -d . --export-only --export-format jsonl

# Process specific range
python run_simple_pipeline.py -d . --start 1000 --limit 500
```

---

## Cost Estimation

| Items | API Cost | Time (est.) |
|-------|----------|-------------|
| 10 | $0.03 | ~3 min |
| 100 | $0.30 | ~30 min |
| 1,000 | $3.00 | ~5 hrs |
| 10,000 | $30.00 | ~2 days |
| 16,000 | ~$48.00 | ~3 days |

Actual costs may vary based on poster image sizes and API pricing changes.

---

## Original Pipeline (GPU-Based)

For users with access to multiple GPUs who want maximum extraction quality.

### Requirements

- 1-6+ GPUs with 24GB+ VRAM each
- ~$1,100+ in GPU rental costs for full dataset
- Models: Dolphin (ByteDance), Qwen3-VL-8B

### Quick Start

```bash
# Download models
huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model

# Run staged (single GPU)
python run_pipeline.py --parallel staged --dolphin-model ./hf_model

# Run multi-GPU
python run_pipeline.py --parallel multi_gpu --gpu-ids 0,1,2
```

See the architecture diagram at the top for the full original pipeline flow.

---

## Troubleshooting

### Simple Pipeline

**API Key Error:**
```
ValueError: Claude API key required
```
Solution: Set `ANTHROPIC_API_KEY` environment variable or pass `--api-key`

**Rate Limiting (429 errors):**
- Reduce `--concurrency` to 10 (default)
- The pipeline automatically retries with exponential backoff (up to 5 retries)
- Respects `Retry-After` headers from the API

**Large Images (400 errors):**
- The pipeline automatically resizes images that exceed Claude's 5MB base64 limit
- Images are progressively compressed (JPEG quality reduction, then scaling)
- No manual intervention needed

### Original Pipeline

**CUDA Out of Memory:**
- Use `--parallel staged` for single GPU
- Enable `--use-flash-attn`

**Missing Transformers Classes:**
- Update transformers: `pip install -U transformers`

---

## Finetuning Guide

Once you have the training data (`training_data.jsonl`), here's how to finetune a model.

### Training Data Format

Each line in `training_data.jsonl` is a JSON object:

```json
{
  "instruction": "Generate an academic poster for the following research paper.\n\nTitle: ...\n\nAbstract: ...",
  "input": "[Full paper markdown content]",
  "output": "{\"layout\": {...}, \"content\": \"[Poster markdown]\"}"
}
```

### Option 1: Finetune with Hugging Face + LoRA (Recommended)

Best for: Limited GPU resources, quick iteration

#### 1. Install Dependencies

```bash
pip install transformers datasets peft accelerate bitsandbytes
pip install trl  # For SFTTrainer
```

#### 2. Prepare Dataset

```python
from datasets import load_dataset

# Load the JSONL file
dataset = load_dataset("json", data_files="training_data.jsonl")

# Split into train/val
dataset = dataset["train"].train_test_split(test_size=0.1)
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
```

#### 3. Finetune Script

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Or Llama, Qwen, etc.
OUTPUT_DIR = "./poster-model"
MAX_LENGTH = 8192  # Adjust based on your GPU memory

# Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)

# Format for instruction tuning
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input'][:4000]}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)

# Quantization config (for fitting on consumer GPUs)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Training config
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    max_seq_length=MAX_LENGTH,
    dataset_text_field="text",
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
```

#### 4. Run Training

```bash
# Single GPU
python train.py

# Multi-GPU with accelerate
accelerate launch --num_processes 4 train.py
```

### Option 2: Full Finetuning with DeepSpeed

Best for: Multiple GPUs, maximum quality

```bash
# Install DeepSpeed
pip install deepspeed

# Create deepspeed config (ds_config.json)
```

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "gradient_accumulation_steps": 8,
  "train_batch_size": 32
}
```

```bash
# Run with DeepSpeed
deepspeed --num_gpus 4 train.py --deepspeed ds_config.json
```

### Option 3: Cloud Finetuning Services

For those without local GPU access:

#### Together AI
```bash
pip install together

# Upload dataset
together files upload training_data.jsonl

# Start finetuning job
together fine-tuning create \
  --training-file file-xxx \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --n-epochs 3
```

#### Modal
```python
# modal_train.py
import modal

app = modal.App("poster-finetune")

@app.function(gpu="A100", timeout=3600*6)
def train():
    # Your training code here
    pass
```

#### RunPod
1. Launch a GPU pod (A100 recommended)
2. Upload `training_data.jsonl`
3. Run the training script above

### Inference

After training, generate posters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model + LoRA weights
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "./poster-model")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def generate_poster(paper_title, paper_abstract, paper_content):
    prompt = f"""### Instruction:
Generate an academic poster for the following research paper.

Title: {paper_title}

Abstract: {paper_abstract}

### Input:
{paper_content[:4000]}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Example usage
result = generate_poster(
    "My Paper Title",
    "This paper presents...",
    "## Introduction\n..."
)
print(result)
```

### Recommended Models for Finetuning

| Model | Size | VRAM Needed | Notes |
|-------|------|-------------|-------|
| Mistral-7B-Instruct | 7B | 16GB (4-bit) | Good balance of quality/speed |
| Llama-3-8B-Instruct | 8B | 20GB (4-bit) | Strong instruction following |
| Qwen2-7B-Instruct | 7B | 16GB (4-bit) | Good multilingual support |
| Mixtral-8x7B | 47B | 48GB (4-bit) | Best quality, needs more VRAM |

### Tips for Better Results

1. **Truncate long papers**: The full paper content can be very long. Consider using just the abstract + introduction + conclusion.

2. **Focus on layout first**: Train a model to predict just the layout JSON, then a separate model for content.

3. **Data augmentation**:
   - Shuffle section order in some examples
   - Add noise to layout percentages
   - Include examples with different number of columns

4. **Evaluation metrics**:
   - JSON validity rate (can the output be parsed?)
   - Layout similarity (do sections match expected structure?)
   - Human evaluation for content quality

---

## Finetuning Scripts Reference

This repository includes ready-to-use scripts for the complete finetuning workflow.

### Script Overview

```
├── prepare_training_data.py   # Preprocess data for finetuning
├── train_mistral_lora.py      # Train Mistral-7B with LoRA on RunPod
├── generate_poster.py         # Generate posters with finetuned model
├── requirements_training.txt  # Training dependencies
└── RUNPOD_TRAINING.md         # RunPod quick start guide
```

---

### 1. Data Preprocessing (`prepare_training_data.py`)

Preprocesses raw training data for LLM finetuning by truncating long papers and formatting for training frameworks.

#### Features

- **Smart truncation**: Extracts key sections (abstract, intro, methods, results, conclusion) to fit context windows
- **Multiple formats**: Outputs in Alpaca, ChatML, Llama, or Mistral format
- **LLM-based extraction**: Optional Claude Haiku for intelligent section categorization
- **Extraction tracking**: Records whether regex or LLM was used per item

#### Usage

```bash
# Basic usage (regex-based, free)
python prepare_training_data.py \
    --input pipeline_output/training_data.jsonl \
    --output prepared_data/ \
    --max-tokens 8192

# With LLM section extraction (~$7 for 7k papers)
python prepare_training_data.py \
    --use-llm \
    --input pipeline_output/training_data.jsonl \
    --output prepared_data/

# Mistral format for training
python prepare_training_data.py \
    --format mistral \
    --max-tokens 8192 \
    --output prepared_data/
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | `pipeline_output/training_data.jsonl` | Input JSONL file |
| `--output`, `-o` | `prepared_data` | Output directory |
| `--max-tokens` | `8192` | Max context tokens |
| `--truncation` | `smart` | Strategy: `smart`, `head`, or `none` |
| `--format` | `alpaca` | Output format: `alpaca`, `chatml`, `llama`, `mistral` |
| `--val-split` | `0.1` | Validation split ratio |
| `--use-llm` | `False` | Use Claude Haiku for section extraction |
| `--api-key` | env var | Anthropic API key (for `--use-llm`) |

#### Truncation Strategies

**`smart` (recommended)**: Extracts and preserves key sections:
```
Original paper (50k chars) → Truncated (24k chars)
├── Abstract (2k chars max)
├── Introduction (4k chars max)
├── Methods (2k chars max)
├── Results (2k chars max)
├── Conclusion (2k chars max)
└── Additional content (remaining space)
```

**`head`**: Simple first-N-characters truncation

**`none`**: No truncation (may exceed context window)

#### Section Extraction Methods

**Regex (default)**: Pattern-based extraction handling multiple formats:
- Markdown headers: `## Introduction`
- Numbered sections: `1 Introduction`, `1. Introduction`
- All caps: `INTRODUCTION`
- Various section names: Introduction/Background/Overview/Motivation

**LLM (`--use-llm`)**: Claude Haiku intelligently categorizes sections regardless of naming:
- "Our Contribution" → introduction
- "Experimental Setup" → methods
- "Findings" → results

#### Output Files

```
prepared_data/
├── train.jsonl      # Training examples (90%)
├── val.jsonl        # Validation examples (10%)
└── combined.jsonl   # All examples
```

#### Output Format (Alpaca)

```json
{
  "instruction": "Generate an academic poster for the following research paper.\n\nTitle: ...\n\nAbstract: ...",
  "input": "## Abstract\n...\n\n## Introduction\n...",
  "output": "{\"layout\": {...}, \"content\": \"...\"}"
}
```

---

### 2. Training Script (`train_mistral_lora.py`)

Finetunes Mistral-7B with LoRA for poster generation. Optimized for RunPod A100/RTX 4090.

#### Features

- **4-bit quantization**: Fits in 16GB VRAM
- **LoRA adapters**: Only trains ~0.5% of parameters
- **Flash Attention 2**: Faster training
- **Gradient checkpointing**: Memory efficient

#### Quick Start

```bash
# On RunPod with A100 40GB
pip install -r requirements_training.txt
pip install flash-attn --no-build-isolation

# Run training
python train_mistral_lora.py
```

#### Configuration

Edit constants at top of script:

```python
# Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./poster-mistral-lora"
MAX_SEQ_LENGTH = 8192

# Training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Effective batch = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Data paths
TRAIN_FILE = "prepared_data/train.jsonl"
VAL_FILE = "prepared_data/val.jsonl"
```

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Mistral-7B-Instruct-v0.2 (4-bit quantized)                     │
│                                                                 │
│  Frozen base weights + LoRA adapters on:                        │
│  • q_proj, k_proj, v_proj, o_proj (attention)                   │
│  • gate_proj, up_proj, down_proj (MLP)                          │
│                                                                 │
│  Trainable: ~27M params (0.4% of total)                         │
└─────────────────────────────────────────────────────────────────┘
```

#### Training Output

```
poster-mistral-lora/
├── adapter_config.json    # LoRA configuration
├── adapter_model.safetensors  # LoRA weights (~200MB)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

#### Expected Results

| Metric | Value |
|--------|-------|
| Training time | 2-4 hours (A100 40GB) |
| Final loss | ~0.5-0.8 |
| Model size | ~200MB (LoRA only) |
| Cost | ~$4-6 on RunPod |

#### Troubleshooting

**Out of Memory:**
```python
MAX_SEQ_LENGTH = 4096  # Reduce from 8192
GRADIENT_ACCUMULATION = 4  # Reduce from 8
```

**Slow Training:**
```bash
# Ensure flash attention is installed
pip install flash-attn --no-build-isolation
```

**Poor Results:**
```python
NUM_EPOCHS = 5  # Increase from 3
LEARNING_RATE = 1e-4  # Decrease from 2e-4
```

---

### 3. Inference Script (`generate_poster.py`)

Generates posters from papers using the finetuned model.

#### Features

- **Accepts markdown or PDF**: Auto-extracts text from PDFs
- **Auto-extracts metadata**: Title and abstract from paper
- **4-bit inference**: Runs on consumer GPUs
- **JSON output**: Structured poster layout + content

#### Usage

```bash
# Generate from markdown
python generate_poster.py --paper paper.md --output poster.json

# Generate from PDF
python generate_poster.py --paper paper.pdf --output poster.json

# Custom model path
python generate_poster.py \
    --model ./poster-mistral-lora \
    --paper paper.pdf \
    --output poster.json

# Adjust creativity
python generate_poster.py \
    --paper paper.md \
    --temperature 0.5 \
    --output poster.json
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `./poster-mistral-lora` | Path to finetuned LoRA model |
| `--paper` | required | Path to paper (markdown or PDF) |
| `--title` | auto-extracted | Paper title |
| `--abstract` | auto-extracted | Paper abstract |
| `--output` | stdout | Output JSON file |
| `--temperature` | `0.7` | Generation temperature |

#### Output Format

```json
{
  "layout": {
    "poster": {
      "orientation": "landscape",
      "aspect_ratio": "16:9",
      "background": "#FFFFFF"
    },
    "header": {
      "height_pct": 15,
      "title_alignment": "center"
    },
    "body": {
      "columns": 3,
      "column_widths": ["equal"]
    },
    "sections": [
      {
        "id": 1,
        "title": "Introduction",
        "column": 1,
        "content_type": "text"
      }
    ],
    "color_scheme": {
      "primary": "#0066CC",
      "secondary": "#003366"
    }
  },
  "content": "# Paper Title\n\n## Introduction\n..."
}
```

#### PDF Support

Requires PyMuPDF:
```bash
pip install pymupdf
```

The script automatically:
1. Extracts text from all PDF pages
2. Identifies title (first heading or line)
3. Extracts abstract (if present)
4. Passes full content to model

---

### Complete Workflow Example

```bash
# 1. Run the simple pipeline to create training data
python run_simple_pipeline.py --data-dir ./data --limit 1000

# 2. Preprocess for training
python prepare_training_data.py \
    --input pipeline_output/training_data.jsonl \
    --output prepared_data/ \
    --format mistral \
    --max-tokens 8192

# 3. Train on RunPod (upload files first)
# SSH into RunPod instance
pip install -r requirements_training.txt
python train_mistral_lora.py

# 4. Download model and generate posters locally
python generate_poster.py \
    --model ./poster-mistral-lora \
    --paper my_paper.pdf \
    --output my_poster.json
```

---

## License

This pipeline is for research purposes. Check licenses for:
- marker-pdf: Apache 2.0
- Claude API: Anthropic Terms of Service
- Dolphin: ByteDance License
- Qwen3-VL: Qwen License
