"""
Simple Pipeline
===============
Simplified, cost-effective pipeline using marker-pdf + Claude API.

Estimated cost: ~$48 for 16k items (vs ~$1,100+ with original pipeline)
Estimated time: 1-2 days (vs ~18 days with original pipeline on 6 GPUs)
"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig, VLMMatcherConfig
from .models import (
    ParsedDocument, PosterLayout, FigureMatch,
    TrainingExample, PipelineResult
)
from .marker_parser import MarkerParser, MarkerConfig
from .claude_poster_processor import ClaudePosterProcessor, ClaudeConfig
from .figure_matcher import VLMFigureMatcher


@dataclass
class SimplePipelineStats:
    """Statistics for pipeline run."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    api_cost_estimate: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "api_cost_estimate": self.api_cost_estimate,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time else None
        }


class SimplePipeline:
    """
    Simplified pipeline using marker-pdf (CPU) + Claude Haiku (API).

    This pipeline is ~20x cheaper and ~10x faster than the original
    Dolphin + Qwen pipeline while maintaining good quality.

    Usage:
        config = PipelineConfig(use_simple_pipeline=True)
        config.claude.api_key = "your-api-key"

        pipeline = SimplePipeline(config)
        results = pipeline.process_dataframe(df)
    """

    # Cost estimate per poster (Claude Haiku 3)
    COST_PER_POSTER = 0.003

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Ensure simple pipeline mode
        self.config.use_simple_pipeline = True
        self.config.vlm_matcher.use_vlm = False  # Feature-only matching

        # Initialize components (lazy)
        self._marker = None
        self._claude = None
        self._matcher = None

        # Setup directories
        self.config.setup_directories()

    @property
    def marker(self) -> MarkerParser:
        """Lazy-load marker parser."""
        if self._marker is None:
            self._marker = MarkerParser(self.config.marker)
        return self._marker

    @property
    def claude(self) -> ClaudePosterProcessor:
        """Lazy-load Claude processor."""
        if self._claude is None:
            if not self.config.claude.api_key:
                raise ValueError(
                    "Claude API key required. Set ANTHROPIC_API_KEY environment variable "
                    "or config.claude.api_key"
                )
            self._claude = ClaudePosterProcessor(self.config.claude)
        return self._claude

    @property
    def matcher(self) -> VLMFigureMatcher:
        """Lazy-load figure matcher (feature-only mode)."""
        if self._matcher is None:
            self._matcher = VLMFigureMatcher(self.config.vlm_matcher)
        return self._matcher

    def process_single(
        self,
        paper_id: str,
        poster_path: str,
        paper_path: str,
        metadata: Optional[dict] = None
    ) -> PipelineResult:
        """
        Process a single paper-poster pair.

        Args:
            paper_id: Unique identifier
            poster_path: Path to poster image
            paper_path: Path to paper PDF
            metadata: Optional metadata (title, abstract, etc.)

        Returns:
            PipelineResult with all processed data
        """
        metadata = metadata or {}
        output_base = self.config.output_dir

        try:
            # Step 1: Parse paper with marker-pdf
            paper_output_dir = output_base / self.config.paper_parsed_dir / paper_id
            parsed_paper = self.marker.parse_paper(
                paper_path,
                str(paper_output_dir),
                paper_id
            )

            # Step 2: Process poster with Claude (content + layout)
            poster_output_dir = output_base / self.config.poster_parsed_dir / paper_id
            parsed_poster, poster_layout = self.claude.process_poster(
                poster_path,
                paper_id,
                str(poster_output_dir)
            )

            # Save layout
            layout_path = output_base / self.config.poster_descriptions_dir / f"{paper_id}_layout.json"
            poster_layout.save(layout_path)

            # Step 3: Match figures (feature-only)
            figure_matches = []
            if parsed_paper.figures and parsed_poster.figures:
                match_output = output_base / self.config.figure_matches_dir / paper_id
                figure_matches = self.matcher.match_figures(
                    parsed_poster.figures,
                    parsed_paper.figures,
                    str(match_output)
                )

            # Step 4: Create training example
            training_example = TrainingExample(
                paper_id=paper_id,
                paper_markdown=parsed_paper.markdown_content,
                paper_abstract=metadata.get("abstract", ""),
                paper_title=metadata.get("title", ""),
                poster_layout=poster_layout,
                poster_markdown=parsed_poster.markdown_content,
                figure_matches=figure_matches,
                conference=metadata.get("conference", ""),
                year=int(metadata.get("year", 0)),
                topics=metadata.get("topics", [])
            )

            # Save training example
            training_path = output_base / self.config.training_data_dir / f"{paper_id}.json"
            training_example.save(training_path)

            return PipelineResult(
                paper_id=paper_id,
                success=True,
                parsed_poster=parsed_poster,
                parsed_paper=parsed_paper,
                figure_matches=figure_matches,
                poster_layout=poster_layout,
                training_example=training_example
            )

        except Exception as e:
            return PipelineResult(
                paper_id=paper_id,
                success=False,
                error=str(e)
            )

    async def process_batch_async(
        self,
        items: List[Tuple[str, str, str, dict]],
        concurrency: int = 20,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[PipelineResult], SimplePipelineStats]:
        """
        Process multiple items with async Claude calls and parallel PDF parsing.

        Args:
            items: List of (paper_id, poster_path, paper_path, metadata)
            concurrency: Max concurrent Claude API calls
            progress_callback: Optional callback(completed, total, paper_id, success)

        Returns:
            Tuple of (results list, statistics)
        """
        stats = SimplePipelineStats(
            total=len(items),
            start_time=datetime.now()
        )

        results = []
        output_base = self.config.output_dir

        # Step 1: Parse all papers with marker-pdf (CPU, can parallelize)
        print(f"\n{'='*70}")
        print("STEP 1: Parsing papers with marker-pdf")
        print(f"{'='*70}")

        parsed_papers = {}
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {}
            for paper_id, poster_path, paper_path, metadata in items:
                paper_output_dir = output_base / self.config.paper_parsed_dir / paper_id
                future = executor.submit(
                    self._parse_paper_worker,
                    paper_path, str(paper_output_dir), paper_id
                )
                futures[future] = paper_id

            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing papers"):
                paper_id = futures[future]
                try:
                    parsed_papers[paper_id] = future.result()
                except Exception as e:
                    parsed_papers[paper_id] = None
                    print(f"Error parsing {paper_id}: {e}")

        # Step 2: Process posters with Claude API (async, high concurrency)
        print(f"\n{'='*70}")
        print("STEP 2: Processing posters with Claude Haiku")
        print(f"{'='*70}")

        poster_items = [
            (poster_path, paper_id)
            for paper_id, poster_path, _, _ in items
        ]

        poster_results = await self.claude.process_batch_async(
            poster_items,
            output_dir=str(output_base / self.config.poster_parsed_dir),
            concurrency=concurrency,
            progress_callback=lambda c, t: print(f"  Processed {c}/{t} posters", end='\r')
        )

        # Index by paper_id
        poster_data = {r[0]: (r[1], r[2], r[3]) for r in poster_results}

        # Step 3: Match figures and create training examples
        print(f"\n{'='*70}")
        print("STEP 3: Matching figures and creating training data")
        print(f"{'='*70}")

        for paper_id, poster_path, paper_path, metadata in tqdm(items, desc="Creating examples"):
            try:
                parsed_paper = parsed_papers.get(paper_id)
                poster_doc, poster_layout, poster_error = poster_data.get(paper_id, (None, None, "Not processed"))

                if not parsed_paper or not poster_doc:
                    error = poster_error or "Paper parsing failed"
                    results.append(PipelineResult(paper_id=paper_id, success=False, error=error))
                    stats.failed += 1
                    continue

                # Save layout
                layout_path = output_base / self.config.poster_descriptions_dir / f"{paper_id}_layout.json"
                poster_layout.save(layout_path)

                # Match figures
                figure_matches = []
                if parsed_paper.figures:
                    # Note: poster figures come from paper figures in simple mode
                    # since Claude extracts layout but not figure files
                    match_output = output_base / self.config.figure_matches_dir / paper_id
                    figure_matches = self.matcher.match_figures(
                        [],  # No extracted poster figures in simple mode
                        parsed_paper.figures,
                        str(match_output)
                    )

                # Create training example
                training_example = TrainingExample(
                    paper_id=paper_id,
                    paper_markdown=parsed_paper.markdown_content,
                    paper_abstract=metadata.get("abstract", ""),
                    paper_title=metadata.get("title", ""),
                    poster_layout=poster_layout,
                    poster_markdown=poster_doc.markdown_content,
                    figure_matches=figure_matches,
                    conference=metadata.get("conference", ""),
                    year=int(metadata.get("year", 0)) if metadata.get("year") else 0,
                    topics=metadata.get("topics", [])
                )

                # Save
                training_path = output_base / self.config.training_data_dir / f"{paper_id}.json"
                training_example.save(training_path)

                results.append(PipelineResult(
                    paper_id=paper_id,
                    success=True,
                    parsed_poster=poster_doc,
                    parsed_paper=parsed_paper,
                    figure_matches=figure_matches,
                    poster_layout=poster_layout,
                    training_example=training_example
                ))
                stats.completed += 1
                stats.api_cost_estimate += self.COST_PER_POSTER

            except Exception as e:
                results.append(PipelineResult(paper_id=paper_id, success=False, error=str(e)))
                stats.failed += 1

            if progress_callback:
                progress_callback(stats.completed + stats.failed, stats.total, paper_id, results[-1].success)

        stats.end_time = datetime.now()
        return results, stats

    def _parse_paper_worker(self, pdf_path: str, output_dir: str, doc_id: str) -> ParsedDocument:
        """Worker function for parallel paper parsing."""
        parser = MarkerParser(self.config.marker)
        return parser.parse_paper(pdf_path, output_dir, doc_id)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
        start_idx: int = 0,
        concurrency: int = 20
    ) -> Tuple[List[PipelineResult], SimplePipelineStats]:
        """
        Process a dataframe of paper-poster pairs.

        Expected columns:
            - paper_id: Unique identifier
            - local_image_path: Path to poster image
            - local_pdf_path: Path to paper PDF
            - title (optional): Paper title
            - abstract (optional): Paper abstract
            - conference (optional): Conference name
            - year (optional): Publication year
            - topics (optional): Topic tags

        Args:
            df: DataFrame with paper-poster data
            limit: Max items to process
            start_idx: Start from this index
            concurrency: Max concurrent API calls

        Returns:
            Tuple of (results list, statistics)
        """
        # Validate columns
        required_cols = ['paper_id', 'local_image_path', 'local_pdf_path']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Slice dataframe
        if limit:
            df = df.iloc[start_idx:start_idx + limit]
        else:
            df = df.iloc[start_idx:]

        # Build items list
        items = []
        skipped_missing_data = 0
        for _, row in df.iterrows():
            paper_id = str(row['paper_id'])
            poster_path = row['local_image_path']
            paper_path = row['local_pdf_path']

            # Skip rows with missing paths (NaN values)
            if pd.isna(poster_path) or pd.isna(paper_path):
                skipped_missing_data += 1
                continue

            # Validate files exist
            if not Path(poster_path).exists():
                print(f"Warning: Poster not found: {poster_path}")
                continue
            if not Path(paper_path).exists():
                print(f"Warning: Paper not found: {paper_path}")
                continue

            metadata = {
                'title': row.get('title', ''),
                'abstract': row.get('abstract', ''),
                'conference': row.get('conference', ''),
                'year': row.get('year', 0),
                'topics': row.get('topics', [])
            }

            items.append((paper_id, poster_path, paper_path, metadata))

        if skipped_missing_data > 0:
            print(f"Skipped {skipped_missing_data} items with missing PDF/poster paths")

        print(f"\n{'='*70}")
        print(f"SIMPLE PIPELINE - Processing {len(items)} items")
        print(f"{'='*70}")
        print(f"Estimated API cost: ${len(items) * self.COST_PER_POSTER:.2f}")
        print(f"Concurrency: {concurrency}")
        print(f"{'='*70}\n")

        # Run async processing
        return asyncio.run(
            self.process_batch_async(items, concurrency=concurrency)
        )

    def export_training_data(
        self,
        output_path: str,
        format: str = "jsonl"
    ) -> int:
        """
        Export all training examples to a single file.

        Args:
            output_path: Output file path
            format: "jsonl", "json", or "parquet"

        Returns:
            Number of examples exported
        """
        training_dir = self.config.output_dir / self.config.training_data_dir
        examples = []

        for json_file in training_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Convert to training format
                    example = {
                        "instruction": f"Generate an academic poster for the following research paper.\n\nTitle: {data.get('paper_title', '')}\n\nAbstract: {data.get('paper_abstract', '')}",
                        "input": data.get('paper_markdown', ''),
                        "output": json.dumps({
                            "layout": data.get('poster_layout', {}),
                            "content": data.get('poster_markdown', '')
                        })
                    }
                    examples.append(example)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')

        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(examples, f, indent=2)

        elif format == "parquet":
            df = pd.DataFrame(examples)
            df.to_parquet(output_path)

        print(f"Exported {len(examples)} examples to {output_path}")
        return len(examples)


def create_simple_pipeline(
    data_dir: str,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None
) -> SimplePipeline:
    """
    Factory function to create a simple pipeline.

    Args:
        data_dir: Directory containing train.csv, images/, papers/
        api_key: Claude API key (or set ANTHROPIC_API_KEY env var)
        output_dir: Output directory (default: {data_dir}/pipeline_output)

    Returns:
        Configured SimplePipeline instance
    """
    config = PipelineConfig(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir) if output_dir else Path(data_dir) / "pipeline_output",
        use_simple_pipeline=True
    )

    if api_key:
        config.claude.api_key = api_key

    config.vlm_matcher.use_vlm = False

    return SimplePipeline(config)
