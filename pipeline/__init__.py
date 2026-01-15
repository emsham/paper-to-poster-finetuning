"""
Paper-to-Poster Finetuning Pipeline
===================================

A pipeline for creating training data from academic paper-poster pairs.

Two pipeline modes available:

1. ORIGINAL PIPELINE (GPU-intensive, high quality):
   - DolphinParser: Parse documents using Dolphin VLM
   - VLMFigureMatcher: Match figures with Qwen3-VL
   - PosterDescriptor: Extract layout with Qwen3-VL
   - Cost: ~$1,100+ for 16k items on RunPod

2. SIMPLE PIPELINE (cost-effective, ~$48 for 16k items):
   - MarkerParser: Parse papers with marker-pdf (CPU)
   - ClaudePosterProcessor: Process posters with Claude Haiku API
   - Feature-only figure matching (no VLM)

Simple Pipeline Usage:
    from pipeline import create_simple_pipeline

    # Create pipeline with API key
    pipe = create_simple_pipeline(
        data_dir="./",
        api_key="your-anthropic-api-key"  # or set ANTHROPIC_API_KEY env var
    )

    # Process from DataFrame
    import pandas as pd
    df = pd.read_csv("train.csv")
    results, stats = pipe.process_dataframe(df, concurrency=20)

    # Export training data
    pipe.export_training_data("training_data.jsonl")

Original Pipeline Usage:
    from pipeline import create_pipeline

    pipe = create_pipeline(
        data_dir="./",
        dolphin_model_path="./hf_model",
        vlm_model_name="Qwen/Qwen3-VL-8B-Instruct"
    )

    result = pipe.process_single(
        paper_id="12345",
        poster_path="./images/12345.png",
        paper_path="./papers/paper.pdf",
        metadata={"title": "...", "abstract": "..."}
    )
"""

from .config import (
    PipelineConfig,
    DolphinConfig,
    VLMMatcherConfig,
    PosterDescriptorConfig,
    MarkerConfig,
    ClaudeConfig,
    get_default_config
)

from .models import (
    ExtractedFigure,
    FigureMatch,
    ParsedDocument,
    PosterLayout,
    TrainingExample,
    PipelineResult
)

from .dolphin_parser import DolphinParser, create_dolphin_parser
from .figure_matcher import VLMFigureMatcher, create_figure_matcher
from .poster_descriptor import PosterDescriptor, create_poster_descriptor
from .pipeline import PaperToPosterPipeline, create_pipeline
from .parallel import StagedPipeline, MultiGPUPipeline, create_parallel_pipeline
from .tracking import ProcessingStatus, ProcessingTracker, create_tracker

# Simple pipeline (cost-effective alternative)
from .marker_parser import MarkerParser
from .claude_poster_processor import ClaudePosterProcessor
from .simple_pipeline import SimplePipeline, create_simple_pipeline

__version__ = "0.1.0"

__all__ = [
    # Config
    "PipelineConfig",
    "DolphinConfig",
    "VLMMatcherConfig",
    "PosterDescriptorConfig",
    "MarkerConfig",
    "ClaudeConfig",
    "get_default_config",
    # Models
    "ExtractedFigure",
    "FigureMatch",
    "ParsedDocument",
    "PosterLayout",
    "TrainingExample",
    "PipelineResult",
    # Original Components (GPU-based)
    "DolphinParser",
    "create_dolphin_parser",
    "VLMFigureMatcher",
    "create_figure_matcher",
    "PosterDescriptor",
    "create_poster_descriptor",
    # Original Pipeline
    "PaperToPosterPipeline",
    "create_pipeline",
    # Parallel Processing
    "StagedPipeline",
    "MultiGPUPipeline",
    "create_parallel_pipeline",
    # Tracking
    "ProcessingStatus",
    "ProcessingTracker",
    "create_tracker",
    # Simple Pipeline (cost-effective, API-based)
    "MarkerParser",
    "ClaudePosterProcessor",
    "SimplePipeline",
    "create_simple_pipeline",
]
