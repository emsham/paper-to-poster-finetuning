"""
Paper-to-Poster Finetuning Pipeline
===================================

A pipeline for creating training data from academic paper-poster pairs.

Components:
- DolphinParser: Parse documents (posters/papers) using Dolphin
- VLMFigureMatcher: Match figures between poster and paper
- PosterDescriptor: Generate JSON layout description of posters
- PaperToPosterPipeline: Orchestrate the full processing pipeline

Usage:
    from pipeline import create_pipeline

    # Create pipeline
    pipe = create_pipeline(
        data_dir="./",
        dolphin_model_path="./hf_model",
        vlm_model_name="Qwen/Qwen3-VL-8B-Instruct"
    )

    # Process single paper-poster pair
    result = pipe.process_single(
        paper_id="12345",
        poster_path="./images/12345.png",
        paper_path="./papers/paper.pdf",
        metadata={"title": "...", "abstract": "..."}
    )

    # Or process from DataFrame
    import pandas as pd
    df = pd.read_csv("train.csv")
    results = pipe.process_dataframe(df, limit=100)

    # Export training data
    pipe.export_training_data("training_data.jsonl", format="jsonl")
"""

from .config import (
    PipelineConfig,
    DolphinConfig,
    VLMMatcherConfig,
    PosterDescriptorConfig,
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

__version__ = "0.1.0"

__all__ = [
    # Config
    "PipelineConfig",
    "DolphinConfig",
    "VLMMatcherConfig",
    "PosterDescriptorConfig",
    "get_default_config",
    # Models
    "ExtractedFigure",
    "FigureMatch",
    "ParsedDocument",
    "PosterLayout",
    "TrainingExample",
    "PipelineResult",
    # Components
    "DolphinParser",
    "create_dolphin_parser",
    "VLMFigureMatcher",
    "create_figure_matcher",
    "PosterDescriptor",
    "create_poster_descriptor",
    # Pipeline
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
]
