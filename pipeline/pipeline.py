"""
Paper-to-Poster Pipeline Orchestrator
=====================================
Main orchestrator that coordinates all pipeline components:
1. Dolphin parsing of posters and papers
2. VLM figure matching
3. Poster layout description
4. Training data generation
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from tqdm import tqdm

from .config import PipelineConfig, get_default_config
from .models import (
    ParsedDocument,
    PosterLayout,
    FigureMatch,
    TrainingExample,
    PipelineResult,
    ExtractedFigure
)
from .dolphin_parser import DolphinParser, create_dolphin_parser
from .figure_matcher import VLMFigureMatcher, create_figure_matcher
from .poster_descriptor import PosterDescriptor, create_poster_descriptor


class PaperToPosterPipeline:
    """
    Main pipeline for processing paper-poster pairs into training data.

    Workflow:
    1. Parse poster with Dolphin -> markdown + figures
    2. Parse paper with Dolphin -> markdown + figures
    3. Match figures between poster and paper using VLM
    4. Extract poster layout description using Qwen VLM
    5. Generate training examples (paper -> poster)
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: PipelineConfig with all settings
        """
        self.config = config
        self.config.setup_directories()

        # Initialize components (lazy loading)
        self._dolphin_parser: Optional[DolphinParser] = None
        self._figure_matcher: Optional[VLMFigureMatcher] = None
        self._poster_descriptor: Optional[PosterDescriptor] = None

        # Track processing stats
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }

    @property
    def dolphin_parser(self) -> DolphinParser:
        """Lazy load Dolphin parser."""
        if self._dolphin_parser is None:
            self._dolphin_parser = create_dolphin_parser(self.config.dolphin)
        return self._dolphin_parser

    @property
    def figure_matcher(self) -> VLMFigureMatcher:
        """Lazy load figure matcher."""
        if self._figure_matcher is None:
            self._figure_matcher = create_figure_matcher(self.config.vlm_matcher)
        return self._figure_matcher

    @property
    def poster_descriptor(self) -> PosterDescriptor:
        """Lazy load poster descriptor."""
        if self._poster_descriptor is None:
            self._poster_descriptor = create_poster_descriptor(self.config.poster_descriptor)
        return self._poster_descriptor

    def process_single(
        self,
        paper_id: str,
        poster_path: str,
        paper_path: str,
        metadata: Dict[str, Any] = None
    ) -> PipelineResult:
        """
        Process a single paper-poster pair.

        Args:
            paper_id: Unique identifier
            poster_path: Path to poster image
            paper_path: Path to paper PDF
            metadata: Optional metadata (title, abstract, etc.)

        Returns:
            PipelineResult with all outputs
        """
        metadata = metadata or {}
        result = PipelineResult(paper_id=paper_id, success=False)

        try:
            print(f"\n{'='*70}")
            print(f"Processing: {paper_id}")
            print(f"{'='*70}")

            # Setup output directories for this paper
            poster_output_dir = self.config.output_dir / self.config.poster_parsed_dir / paper_id
            paper_output_dir = self.config.output_dir / self.config.paper_parsed_dir / paper_id
            match_output_dir = self.config.output_dir / self.config.figure_matches_dir / paper_id

            # Check if already processed
            if self.config.skip_existing:
                training_data_path = (
                    self.config.output_dir /
                    self.config.training_data_dir /
                    f"{paper_id}.json"
                )
                if training_data_path.exists():
                    print(f"Skipping {paper_id} - already processed")
                    result.success = True
                    return result

            # Step 1: Parse poster with Dolphin
            print("\n[1/4] Parsing poster...")
            parsed_poster = self.dolphin_parser.parse_poster(
                poster_path,
                str(poster_output_dir),
                paper_id
            )
            # Update source_doc for figures
            for fig in parsed_poster.figures:
                fig.source_doc = "poster"
            result.parsed_poster = parsed_poster
            print(f"  Found {len(parsed_poster.figures)} figures in poster")

            # Step 2: Parse paper with Dolphin
            print("\n[2/4] Parsing paper...")
            parsed_paper = self.dolphin_parser.parse_paper(
                paper_path,
                str(paper_output_dir),
                paper_id
            )
            # Update source_doc for figures
            for fig in parsed_paper.figures:
                fig.source_doc = "paper"
            result.parsed_paper = parsed_paper
            print(f"  Found {len(parsed_paper.figures)} figures in paper")

            # Step 3: Match figures using VLM
            print("\n[3/4] Matching figures...")
            if parsed_poster.figures and parsed_paper.figures:
                matches = self.figure_matcher.match_figures(
                    parsed_poster.figures,
                    parsed_paper.figures,
                    str(match_output_dir)
                )
                result.figure_matches = matches
                print(f"  Found {len(matches)} figure matches")
            else:
                print("  No figures to match")
                result.figure_matches = []

            # Step 4: Extract poster layout description
            print("\n[4/4] Extracting poster layout...")
            poster_layout = self.poster_descriptor.describe_poster(
                poster_path,
                paper_id
            )
            result.poster_layout = poster_layout
            print(f"  Layout extracted: {poster_layout.orientation}, {len(poster_layout.sections)} sections")

            # Generate training example
            training_example = self._create_training_example(
                paper_id=paper_id,
                parsed_paper=parsed_paper,
                parsed_poster=parsed_poster,
                poster_layout=poster_layout,
                figure_matches=result.figure_matches,
                metadata=metadata
            )
            result.training_example = training_example

            # Save training example
            if self.config.save_intermediate:
                training_data_path = (
                    self.config.output_dir /
                    self.config.training_data_dir /
                    f"{paper_id}.json"
                )
                training_example.save(training_data_path)
                print(f"\n  Training data saved to: {training_data_path}")

            result.success = True
            print(f"\n{paper_id} processed successfully!")

        except Exception as e:
            result.error = str(e)
            print(f"\nError processing {paper_id}: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _create_training_example(
        self,
        paper_id: str,
        parsed_paper: ParsedDocument,
        parsed_poster: ParsedDocument,
        poster_layout: PosterLayout,
        figure_matches: List[FigureMatch],
        metadata: Dict[str, Any]
    ) -> TrainingExample:
        """Create a training example from processed data."""
        return TrainingExample(
            paper_id=paper_id,
            paper_markdown=parsed_paper.markdown_content,
            paper_abstract=metadata.get("abstract", ""),
            paper_title=metadata.get("title", ""),
            poster_layout=poster_layout,
            poster_markdown=parsed_poster.markdown_content,
            figure_matches=figure_matches,
            conference=metadata.get("conference", ""),
            year=metadata.get("year", 0),
            topics=metadata.get("topics", [])
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
        start_idx: int = 0
    ) -> List[PipelineResult]:
        """
        Process multiple paper-poster pairs from a DataFrame.

        Expected columns:
        - paper_id: Unique identifier
        - local_image_path: Path to poster image
        - local_pdf_path: Path to paper PDF
        - title, abstract, conference, year, topics (optional metadata)

        Args:
            df: DataFrame with paper-poster pairs
            limit: Optional limit on number to process
            start_idx: Starting index in DataFrame

        Returns:
            List of PipelineResult objects
        """
        results = []

        # Filter to valid rows
        valid_df = df[
            df['local_image_path'].notna() &
            df['local_pdf_path'].notna() &
            (df['error'].isna() | (df['error'] == ''))
        ].copy()

        # Apply limit
        if limit:
            valid_df = valid_df.iloc[start_idx:start_idx + limit]
        else:
            valid_df = valid_df.iloc[start_idx:]

        print(f"\nProcessing {len(valid_df)} paper-poster pairs...")
        print(f"Output directory: {self.config.output_dir}")

        self.stats["total"] = len(valid_df)

        for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Processing"):
            paper_id = str(row['paper_id'])

            # Build paths
            poster_path = self.config.data_dir / row['local_image_path']
            paper_path = self.config.data_dir / row['local_pdf_path']

            # Check if files exist
            if not poster_path.exists():
                print(f"Poster not found: {poster_path}")
                self.stats["skipped"] += 1
                continue

            if not paper_path.exists():
                print(f"Paper not found: {paper_path}")
                self.stats["skipped"] += 1
                continue

            # Prepare metadata
            metadata = {
                "title": row.get('title', ''),
                "abstract": row.get('abstract', ''),
                "conference": row.get('conference', ''),
                "year": int(row.get('year', 0)) if pd.notna(row.get('year')) else 0,
                "topics": row.get('topics', '').split(',') if pd.notna(row.get('topics')) else []
            }

            # Process
            result = self.process_single(
                paper_id=paper_id,
                poster_path=str(poster_path),
                paper_path=str(paper_path),
                metadata=metadata
            )

            results.append(result)

            if result.success:
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1

        # Print summary
        self._print_summary()

        return results

    def _print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total:      {self.stats['total']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed:     {self.stats['failed']}")
        print(f"Skipped:    {self.stats['skipped']}")
        print(f"{'='*70}")

    def export_training_data(
        self,
        output_path: str,
        format: str = "jsonl"
    ):
        """
        Export all training data to a single file.

        Args:
            output_path: Path to output file
            format: Output format ("jsonl", "json", or "parquet")
        """
        training_dir = self.config.output_dir / self.config.training_data_dir
        training_files = list(training_dir.glob("*.json"))

        print(f"Exporting {len(training_files)} training examples to {output_path}")

        examples = []
        for f in training_files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    examples.append(data)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, 'w') as f:
                for ex in examples:
                    # Convert to training format
                    training_format = {
                        "instruction": f"Generate an academic poster for the following research paper.\n\nTitle: {ex.get('paper_title', '')}\n\nAbstract: {ex.get('paper_abstract', '')}",
                        "input": ex.get('paper_markdown', ''),
                        "output": json.dumps({
                            "layout": ex.get('poster_layout', {}),
                            "content": ex.get('poster_markdown', '')
                        })
                    }
                    f.write(json.dumps(training_format) + '\n')

        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(examples, f, indent=2)

        elif format == "parquet":
            df = pd.DataFrame(examples)
            df.to_parquet(output_path, index=False)

        print(f"Exported {len(examples)} examples to {output_path}")


def create_pipeline(
    data_dir: str = "./",
    output_dir: str = None,
    dolphin_model_path: str = "./hf_model",
    vlm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
) -> PaperToPosterPipeline:
    """
    Factory function to create a pipeline with common settings.

    Args:
        data_dir: Root data directory
        output_dir: Output directory (defaults to data_dir/pipeline_output)
        dolphin_model_path: Path to Dolphin model
        vlm_model_name: Name of VLM model for figure matching and poster description

    Returns:
        Configured PaperToPosterPipeline
    """
    config = get_default_config(data_dir)

    if output_dir:
        config.output_dir = Path(output_dir)

    config.dolphin.model_path = dolphin_model_path
    config.vlm_matcher.model_name = vlm_model_name
    config.poster_descriptor.model_name = vlm_model_name

    return PaperToPosterPipeline(config)
