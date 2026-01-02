"""
Pipeline Configuration
======================
Central configuration for the paper-to-poster finetuning pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import os


@dataclass
class DolphinConfig:
    """Configuration for Dolphin document parser."""
    model_path: str = "/home/majd/Documents/Projects/Dolphin/hf_model"  # Path to Dolphin model
    device: str = "cuda"
    batch_size: int = 1
    # Output settings
    output_format: str = "markdown"  # "markdown" or "json"
    extract_figures: bool = True
    extract_tables: bool = True
    extract_equations: bool = True


@dataclass
class VLMMatcherConfig:
    """Configuration for VLM-based figure matching."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_flash_attn: bool = False
    feature_threshold: float = 0.10  # Feature ratio threshold for candidates
    device: str = "cuda"
    # Matching thresholds
    high_confidence_threshold: float = 0.6
    medium_confidence_threshold: float = 0.5


@dataclass
class PosterDescriptorConfig:
    """Configuration for Qwen poster layout descriptor."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_flash_attn: bool = False
    device: str = "cuda"
    max_new_tokens: int = 4096
    temperature: float = 0.1
    do_sample: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("./"))
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    validation_csv: str = "validation.csv"
    images_dir: str = "images"
    papers_dir: str = "papers"

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("./pipeline_output"))
    poster_parsed_dir: str = "poster_parsed"
    paper_parsed_dir: str = "paper_parsed"
    figure_matches_dir: str = "figure_matches"
    poster_descriptions_dir: str = "poster_descriptions"
    training_data_dir: str = "training_data"

    # Component configs
    dolphin: DolphinConfig = field(default_factory=DolphinConfig)
    vlm_matcher: VLMMatcherConfig = field(default_factory=VLMMatcherConfig)
    poster_descriptor: PosterDescriptorConfig = field(default_factory=PosterDescriptorConfig)

    # Processing options
    num_workers: int = 1
    skip_existing: bool = True  # Skip already processed items
    save_intermediate: bool = True  # Save intermediate outputs

    def __post_init__(self):
        """Ensure paths are Path objects and create directories."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def setup_directories(self):
        """Create all necessary output directories."""
        dirs = [
            self.output_dir,
            self.output_dir / self.poster_parsed_dir,
            self.output_dir / self.paper_parsed_dir,
            self.output_dir / self.figure_matches_dir,
            self.output_dir / self.poster_descriptions_dir,
            self.output_dir / self.training_data_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_csv

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_csv

    @property
    def validation_path(self) -> Path:
        return self.data_dir / self.validation_csv

    @property
    def images_path(self) -> Path:
        return self.data_dir / self.images_dir

    @property
    def papers_path(self) -> Path:
        return self.data_dir / self.papers_dir


# Default configuration
def get_default_config(data_dir: str = "./") -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig(
        data_dir=Path(data_dir),
        output_dir=Path(data_dir) / "pipeline_output"
    )
