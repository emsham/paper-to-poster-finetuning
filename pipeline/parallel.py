"""
Parallel Processing Support
===========================
Efficient parallel processing for the paper-to-poster pipeline.

Strategies:
1. Staged Processing - Process all items through one stage before next
   (memory efficient, single GPU)
2. Multi-GPU - Distribute models across multiple GPUs
3. Worker Pool - Parallel I/O and preprocessing
4. Checkpointing - Resume interrupted processing

Usage:
    from pipeline.parallel import StagedPipeline, MultiGPUPipeline

    # Staged (single GPU, memory efficient)
    pipeline = StagedPipeline(config)
    pipeline.run(df, num_workers=4)

    # Multi-GPU
    pipeline = MultiGPUPipeline(config, gpu_ids=[0, 1, 2])
    pipeline.run(df)
"""

import json
import os
import gc
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading

import pandas as pd
import torch
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
from .tracking import ProcessingTracker, create_tracker


@dataclass
class CheckpointState:
    """Tracks processing progress for resumption."""
    stage: str  # "dolphin_poster", "dolphin_paper", "figure_match", "poster_desc", "combine"
    completed_ids: set = field(default_factory=set)
    failed_ids: set = field(default_factory=set)
    timestamp: str = ""

    def save(self, path: Path):
        """Save checkpoint to disk."""
        data = {
            "stage": self.stage,
            "completed_ids": list(self.completed_ids),
            "failed_ids": list(self.failed_ids),
            "timestamp": datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        """Load checkpoint from disk."""
        if not path.exists():
            return cls(stage="dolphin_poster")
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            stage=data["stage"],
            completed_ids=set(data["completed_ids"]),
            failed_ids=set(data["failed_ids"]),
            timestamp=data["timestamp"]
        )


class StagedPipeline:
    """
    Memory-efficient staged pipeline processing.

    Processes all items through each stage before moving to the next:
    1. Parse all posters with Dolphin
    2. Parse all papers with Dolphin
    3. Match figures for all pairs
    4. Extract layout for all posters
    5. Combine into training examples

    Benefits:
    - Only one model loaded at a time
    - Can checkpoint between stages
    - Efficient GPU memory usage
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.setup_directories()
        self.checkpoint_path = self.config.output_dir / "checkpoint.json"
        self.checkpoint = CheckpointState.load(self.checkpoint_path)
        self.tracker = create_tracker(self.config.output_dir)

    def run(
        self,
        df: pd.DataFrame,
        num_workers: int = 4,
        batch_size: int = 1,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Run the staged pipeline.

        Args:
            df: DataFrame with paper-poster pairs
            num_workers: Number of parallel workers for I/O
            batch_size: Batch size for model inference
            resume: Whether to resume from checkpoint

        Returns:
            Processing statistics
        """
        # Filter valid rows
        valid_df = df[
            df['local_image_path'].notna() &
            df['local_pdf_path'].notna() &
            (df['error'].isna() | (df['error'] == ''))
        ].copy()

        paper_ids = valid_df['paper_id'].astype(str).tolist()

        # Initialize tracker with all items
        self.tracker.initialize_from_dataframe(valid_df)

        print(f"\n{'='*70}")
        print(f"STAGED PIPELINE - {len(paper_ids)} items")
        print(f"{'='*70}")
        print(f"Workers: {num_workers}, Batch size: {batch_size}")
        print(f"Checkpoint: {self.checkpoint_path}")
        if resume and self.checkpoint.completed_ids:
            print(f"Resuming from stage: {self.checkpoint.stage}")
            print(f"Already completed: {len(self.checkpoint.completed_ids)} items")
        print(f"{'='*70}\n")

        stats = {
            "total": len(paper_ids),
            "stages": {}
        }

        stages = [
            ("dolphin_poster", self._stage_dolphin_poster),
            ("dolphin_paper", self._stage_dolphin_paper),
            ("figure_match", self._stage_figure_match),
            ("poster_desc", self._stage_poster_desc),
            ("combine", self._stage_combine)
        ]

        # Find starting stage
        stage_names = [s[0] for s in stages]
        if resume and self.checkpoint.stage in stage_names:
            start_idx = stage_names.index(self.checkpoint.stage)
        else:
            start_idx = 0
            self.checkpoint = CheckpointState(stage="dolphin_poster")

        # Run stages
        for stage_name, stage_func in stages[start_idx:]:
            print(f"\n{'='*70}")
            print(f"STAGE: {stage_name.upper()}")
            print(f"{'='*70}\n")

            self.checkpoint.stage = stage_name
            stage_stats = stage_func(valid_df, num_workers, batch_size)
            stats["stages"][stage_name] = stage_stats

            # Save checkpoint and tracking data
            self.checkpoint.save(self.checkpoint_path)
            self.tracker.save()

            # Free GPU memory between stages
            self._free_gpu_memory()

        # Print tracking summary
        summary = self.tracker.get_summary()
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")

        return stats

    def _free_gpu_memory(self):
        """Free GPU memory between stages."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _stage_dolphin_poster(
        self,
        df: pd.DataFrame,
        num_workers: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Stage 1: Parse all posters with Dolphin."""
        from .dolphin_parser import create_dolphin_parser

        parser = create_dolphin_parser(self.config.dolphin)
        parser.load_model()

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing posters"):
            paper_id = str(row['paper_id'])

            # Check if already done
            output_dir = self.config.output_dir / self.config.poster_parsed_dir / paper_id
            md_path = output_dir / "markdown" / f"{paper_id}.md"
            figures_dir = output_dir / "markdown" / "figures"

            if md_path.exists():
                stats["skipped"] += 1
                self.checkpoint.completed_ids.add(f"poster_{paper_id}")
                # Update tracker with existing data
                figure_count = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0
                self.tracker.update_stage(
                    paper_id, "poster_parsed", success=True,
                    poster_markdown_path=str(md_path.relative_to(self.config.output_dir)),
                    poster_figures_dir=str(figures_dir.relative_to(self.config.output_dir)) if figures_dir.exists() else None,
                    poster_figure_count=figure_count
                )
                continue

            try:
                poster_path = self.config.data_dir / row['local_image_path']
                if not poster_path.exists():
                    stats["failed"] += 1
                    self.tracker.update_stage(paper_id, "poster_parsed", success=False, error="Poster file not found")
                    continue

                parser.parse_poster(
                    str(poster_path),
                    str(output_dir),
                    paper_id
                )

                # Count extracted figures
                figure_count = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0

                # Update tracker
                self.tracker.update_stage(
                    paper_id, "poster_parsed", success=True,
                    poster_markdown_path=str(md_path.relative_to(self.config.output_dir)),
                    poster_figures_dir=str(figures_dir.relative_to(self.config.output_dir)) if figures_dir.exists() else None,
                    poster_figure_count=figure_count
                )

                stats["processed"] += 1
                self.checkpoint.completed_ids.add(f"poster_{paper_id}")

            except Exception as e:
                print(f"Error parsing poster {paper_id}: {e}")
                stats["failed"] += 1
                self.checkpoint.failed_ids.add(f"poster_{paper_id}")
                self.tracker.update_stage(paper_id, "poster_parsed", success=False, error=str(e))

        # Unload model
        del parser
        self._free_gpu_memory()

        return stats

    def _stage_dolphin_paper(
        self,
        df: pd.DataFrame,
        num_workers: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Stage 2: Parse all papers with Dolphin."""
        from .dolphin_parser import create_dolphin_parser

        parser = create_dolphin_parser(self.config.dolphin)
        parser.load_model()

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing papers"):
            paper_id = str(row['paper_id'])

            # Check if already done
            output_dir = self.config.output_dir / self.config.paper_parsed_dir / paper_id
            md_path = output_dir / "markdown" / f"{paper_id}.md"
            figures_dir = output_dir / "markdown" / "figures"

            if md_path.exists():
                stats["skipped"] += 1
                self.checkpoint.completed_ids.add(f"paper_{paper_id}")
                # Update tracker with existing data
                figure_count = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0
                self.tracker.update_stage(
                    paper_id, "paper_parsed", success=True,
                    paper_markdown_path=str(md_path.relative_to(self.config.output_dir)),
                    paper_figures_dir=str(figures_dir.relative_to(self.config.output_dir)) if figures_dir.exists() else None,
                    paper_figure_count=figure_count
                )
                continue

            try:
                paper_path = self.config.data_dir / row['local_pdf_path']
                if not paper_path.exists():
                    stats["failed"] += 1
                    self.tracker.update_stage(paper_id, "paper_parsed", success=False, error="Paper file not found")
                    continue

                parser.parse_paper(
                    str(paper_path),
                    str(output_dir),
                    paper_id
                )

                # Count extracted figures
                figure_count = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0

                # Update tracker
                self.tracker.update_stage(
                    paper_id, "paper_parsed", success=True,
                    paper_markdown_path=str(md_path.relative_to(self.config.output_dir)),
                    paper_figures_dir=str(figures_dir.relative_to(self.config.output_dir)) if figures_dir.exists() else None,
                    paper_figure_count=figure_count
                )

                stats["processed"] += 1
                self.checkpoint.completed_ids.add(f"paper_{paper_id}")

            except Exception as e:
                print(f"Error parsing paper {paper_id}: {e}")
                stats["failed"] += 1
                self.checkpoint.failed_ids.add(f"paper_{paper_id}")
                self.tracker.update_stage(paper_id, "paper_parsed", success=False, error=str(e))

        del parser
        self._free_gpu_memory()

        return stats

    def _stage_figure_match(
        self,
        df: pd.DataFrame,
        num_workers: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Stage 3: Match figures between posters and papers."""
        from .figure_matcher import create_figure_matcher

        matcher = create_figure_matcher(self.config.vlm_matcher)

        stats = {"processed": 0, "failed": 0, "skipped": 0, "no_figures": 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching figures"):
            paper_id = str(row['paper_id'])

            # Check if already done
            match_dir = self.config.output_dir / self.config.figure_matches_dir / paper_id
            report_path = match_dir / "report.json"

            if report_path.exists():
                stats["skipped"] += 1
                # Update tracker with existing data
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                match_count = len(report_data.get("matches", []))
                self.tracker.update_stage(
                    paper_id, "figures_matched", success=True,
                    figure_matches_path=str(report_path.relative_to(self.config.output_dir)),
                    matched_figure_count=match_count
                )
                continue

            try:
                # Load figures from parsed outputs
                poster_figures = self._load_figures(
                    self.config.output_dir / self.config.poster_parsed_dir / paper_id,
                    "poster"
                )
                paper_figures = self._load_figures(
                    self.config.output_dir / self.config.paper_parsed_dir / paper_id,
                    "paper"
                )

                if not poster_figures or not paper_figures:
                    stats["no_figures"] += 1
                    # Create empty report
                    match_dir.mkdir(parents=True, exist_ok=True)
                    with open(report_path, 'w') as f:
                        json.dump({"matches": [], "reason": "no_figures"}, f)
                    self.tracker.update_stage(
                        paper_id, "figures_matched", success=True,
                        figure_matches_path=str(report_path.relative_to(self.config.output_dir)),
                        matched_figure_count=0
                    )
                    continue

                matcher.match_figures(
                    poster_figures,
                    paper_figures,
                    str(match_dir)
                )

                # Read match count from generated report
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        report_data = json.load(f)
                    match_count = len(report_data.get("matches", []))
                else:
                    match_count = 0

                # Update tracker
                self.tracker.update_stage(
                    paper_id, "figures_matched", success=True,
                    figure_matches_path=str(report_path.relative_to(self.config.output_dir)),
                    matched_figure_count=match_count
                )

                stats["processed"] += 1

            except Exception as e:
                print(f"Error matching figures for {paper_id}: {e}")
                stats["failed"] += 1
                self.tracker.update_stage(paper_id, "figures_matched", success=False, error=str(e))

        del matcher
        self._free_gpu_memory()

        return stats

    def _stage_poster_desc(
        self,
        df: pd.DataFrame,
        num_workers: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Stage 4: Extract poster layout descriptions."""
        from .poster_descriptor import create_poster_descriptor

        descriptor = create_poster_descriptor(self.config.poster_descriptor)
        descriptor.load_model()

        stats = {"processed": 0, "failed": 0, "skipped": 0}
        desc_dir = self.config.output_dir / self.config.poster_descriptions_dir

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Describing posters"):
            paper_id = str(row['paper_id'])
            layout_path = desc_dir / f"{paper_id}_layout.json"

            # Check if already done
            if layout_path.exists():
                stats["skipped"] += 1
                # Update tracker with existing data
                with open(layout_path, 'r') as f:
                    layout_data = json.load(f)
                section_count = len(layout_data.get("sections", []))
                self.tracker.update_stage(
                    paper_id, "layout_extracted", success=True,
                    layout_json_path=str(layout_path.relative_to(self.config.output_dir)),
                    section_count=section_count
                )
                continue

            try:
                poster_path = self.config.data_dir / row['local_image_path']
                if not poster_path.exists():
                    stats["failed"] += 1
                    self.tracker.update_stage(paper_id, "layout_extracted", success=False, error="Poster file not found")
                    continue

                layout = descriptor.describe_poster(
                    str(poster_path),
                    paper_id
                )
                layout.save(layout_path)

                # Count sections
                section_count = len(layout.sections) if layout.sections else 0

                # Update tracker
                self.tracker.update_stage(
                    paper_id, "layout_extracted", success=True,
                    layout_json_path=str(layout_path.relative_to(self.config.output_dir)),
                    section_count=section_count
                )

                stats["processed"] += 1

            except Exception as e:
                print(f"Error describing poster {paper_id}: {e}")
                stats["failed"] += 1
                self.tracker.update_stage(paper_id, "layout_extracted", success=False, error=str(e))

        del descriptor
        self._free_gpu_memory()

        return stats

    def _stage_combine(
        self,
        df: pd.DataFrame,
        num_workers: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Stage 5: Combine all outputs into training examples."""
        stats = {"processed": 0, "failed": 0, "skipped": 0}
        training_dir = self.config.output_dir / self.config.training_data_dir

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Combining outputs"):
            paper_id = str(row['paper_id'])
            training_path = training_dir / f"{paper_id}.json"

            # Check if already done
            if training_path.exists():
                stats["skipped"] += 1
                # Update tracker
                self.tracker.update_stage(
                    paper_id, "training_data_created", success=True,
                    training_data_path=str(training_path.relative_to(self.config.output_dir))
                )
                self.tracker.mark_complete(paper_id)
                continue

            try:
                start_time = time.time()

                # Load all components
                poster_md = self._load_markdown(
                    self.config.output_dir / self.config.poster_parsed_dir / paper_id
                )
                paper_md = self._load_markdown(
                    self.config.output_dir / self.config.paper_parsed_dir / paper_id
                )
                layout = self._load_layout(
                    self.config.output_dir / self.config.poster_descriptions_dir / f"{paper_id}_layout.json"
                )
                matches = self._load_matches(
                    self.config.output_dir / self.config.figure_matches_dir / paper_id / "report.json"
                )

                if not all([poster_md, paper_md, layout]):
                    stats["failed"] += 1
                    self.tracker.update_stage(
                        paper_id, "training_data_created", success=False,
                        error="Missing required components"
                    )
                    continue

                # Create training example
                example = {
                    "paper_id": paper_id,
                    "paper_title": row.get('title', ''),
                    "paper_abstract": row.get('abstract', ''),
                    "paper_markdown": paper_md,
                    "poster_markdown": poster_md,
                    "poster_layout": layout,
                    "figure_matches": matches,
                    "conference": row.get('conference', ''),
                    "year": int(row.get('year', 0)) if pd.notna(row.get('year')) else 0,
                }

                with open(training_path, 'w') as f:
                    json.dump(example, f, indent=2)

                processing_time = time.time() - start_time

                # Update tracker
                self.tracker.update_stage(
                    paper_id, "training_data_created", success=True,
                    training_data_path=str(training_path.relative_to(self.config.output_dir))
                )
                self.tracker.mark_complete(paper_id, processing_time)

                stats["processed"] += 1

            except Exception as e:
                print(f"Error combining {paper_id}: {e}")
                stats["failed"] += 1
                self.tracker.update_stage(paper_id, "training_data_created", success=False, error=str(e))

        return stats

    def _load_figures(self, parsed_dir: Path, source: str) -> List[ExtractedFigure]:
        """Load extracted figures from parsed output."""
        figures = []
        figures_dir = parsed_dir / "markdown" / "figures"

        if not figures_dir.exists():
            return figures

        for fig_path in sorted(figures_dir.glob("*.png")):
            figures.append(ExtractedFigure(
                figure_id=fig_path.stem,
                source_doc=source,
                file_path=str(fig_path)
            ))

        return figures

    def _load_markdown(self, parsed_dir: Path) -> Optional[str]:
        """Load markdown content from parsed output."""
        md_dir = parsed_dir / "markdown"
        if not md_dir.exists():
            return None

        md_files = list(md_dir.glob("*.md"))
        if not md_files:
            return None

        with open(md_files[0], 'r') as f:
            return f.read()

    def _load_layout(self, path: Path) -> Optional[Dict]:
        """Load poster layout from JSON."""
        if not path.exists():
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def _load_matches(self, path: Path) -> List[Dict]:
        """Load figure matches from report."""
        if not path.exists():
            return []
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get("matches", [])

    def export_tracking(
        self,
        source_df: pd.DataFrame,
        output_path: str = None,
        format: str = "csv"
    ) -> pd.DataFrame:
        """
        Export tracking data merged with source dataframe.

        Args:
            source_df: Original dataframe (train.csv, etc.)
            output_path: Where to save (optional)
            format: "csv" or "parquet"

        Returns:
            Merged dataframe with tracking columns
        """
        merged = self.tracker.merge_with_source(source_df)

        if output_path:
            if format == "parquet":
                merged.to_parquet(output_path, index=False)
            else:
                merged.to_csv(output_path, index=False)
            print(f"Exported tracking data to {output_path}")

        return merged

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get processing summary from tracker."""
        return self.tracker.get_summary()

    def get_pending_items(self, stage: str = None) -> List[str]:
        """Get paper IDs that haven't completed a stage."""
        return self.tracker.get_pending(stage)

    def get_failed_items(self) -> List[str]:
        """Get paper IDs that have errors."""
        return self.tracker.get_failed()


class MultiGPUPipeline:
    """
    Multi-GPU pipeline for maximum throughput.

    Distributes different models across available GPUs:
    - GPU 0: Dolphin (poster parsing)
    - GPU 1: Dolphin (paper parsing)
    - GPU 2: VLM (figure matching + poster description)

    Or with 2 GPUs:
    - GPU 0: Dolphin
    - GPU 1: VLM

    Uses queues to pipeline data between stages.
    """

    def __init__(
        self,
        config: PipelineConfig,
        gpu_ids: List[int] = None
    ):
        self.config = config
        self.config.setup_directories()

        # Detect available GPUs
        if gpu_ids is None:
            gpu_count = torch.cuda.device_count()
            gpu_ids = list(range(gpu_count)) if gpu_count > 0 else []

        self.gpu_ids = gpu_ids
        print(f"Multi-GPU Pipeline initialized with GPUs: {gpu_ids}")

    def run(
        self,
        df: pd.DataFrame,
        queue_size: int = 10
    ) -> Dict[str, Any]:
        """
        Run multi-GPU pipeline.

        Args:
            df: DataFrame with paper-poster pairs
            queue_size: Size of inter-stage queues

        Returns:
            Processing statistics
        """
        if len(self.gpu_ids) < 2:
            print("Warning: Less than 2 GPUs available, falling back to staged pipeline")
            staged = StagedPipeline(self.config)
            return staged.run(df)

        # Filter valid rows
        valid_df = df[
            df['local_image_path'].notna() &
            df['local_pdf_path'].notna() &
            (df['error'].isna() | (df['error'] == ''))
        ].copy()

        print(f"\n{'='*70}")
        print(f"MULTI-GPU PIPELINE - {len(valid_df)} items")
        print(f"GPUs: {self.gpu_ids}")
        print(f"{'='*70}\n")

        # Create queues for pipelining
        poster_queue = Queue(maxsize=queue_size)
        paper_queue = Queue(maxsize=queue_size)
        match_queue = Queue(maxsize=queue_size)
        done_queue = Queue()

        # Assign GPUs based on count
        if len(self.gpu_ids) >= 3:
            dolphin_poster_gpu = self.gpu_ids[0]
            dolphin_paper_gpu = self.gpu_ids[1]
            vlm_gpu = self.gpu_ids[2]
        else:
            dolphin_poster_gpu = self.gpu_ids[0]
            dolphin_paper_gpu = self.gpu_ids[0]
            vlm_gpu = self.gpu_ids[1]

        # Start worker threads
        workers = []

        # Dolphin poster worker
        w1 = threading.Thread(
            target=self._dolphin_worker,
            args=(valid_df, poster_queue, "poster", dolphin_poster_gpu),
            name="dolphin_poster"
        )
        workers.append(w1)

        # Dolphin paper worker
        w2 = threading.Thread(
            target=self._dolphin_worker,
            args=(valid_df, paper_queue, "paper", dolphin_paper_gpu),
            name="dolphin_paper"
        )
        workers.append(w2)

        # Combiner worker (waits for both Dolphin outputs)
        w3 = threading.Thread(
            target=self._combiner_worker,
            args=(valid_df, poster_queue, paper_queue, match_queue),
            name="combiner"
        )
        workers.append(w3)

        # VLM worker (figure matching + poster description)
        w4 = threading.Thread(
            target=self._vlm_worker,
            args=(valid_df, match_queue, done_queue, vlm_gpu),
            name="vlm"
        )
        workers.append(w4)

        # Start all workers
        for w in workers:
            w.start()

        # Wait for completion
        for w in workers:
            w.join()

        # Collect results
        stats = {
            "total": len(valid_df),
            "processed": done_queue.qsize()
        }

        return stats

    def _dolphin_worker(
        self,
        df: pd.DataFrame,
        output_queue: Queue,
        doc_type: str,
        gpu_id: int
    ):
        """Worker for Dolphin parsing."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from .dolphin_parser import create_dolphin_parser

        config = self.config.dolphin
        config.device = f"cuda:0"  # Since we set CUDA_VISIBLE_DEVICES

        parser = create_dolphin_parser(config)
        parser.load_model()

        for _, row in df.iterrows():
            paper_id = str(row['paper_id'])

            try:
                if doc_type == "poster":
                    path = self.config.data_dir / row['local_image_path']
                    output_dir = self.config.output_dir / self.config.poster_parsed_dir / paper_id
                    if path.exists():
                        parser.parse_poster(str(path), str(output_dir), paper_id)
                else:
                    path = self.config.data_dir / row['local_pdf_path']
                    output_dir = self.config.output_dir / self.config.paper_parsed_dir / paper_id
                    if path.exists():
                        parser.parse_paper(str(path), str(output_dir), paper_id)

                output_queue.put((paper_id, True))

            except Exception as e:
                print(f"Dolphin {doc_type} error for {paper_id}: {e}")
                output_queue.put((paper_id, False))

        # Signal completion
        output_queue.put(None)

    def _combiner_worker(
        self,
        df: pd.DataFrame,
        poster_queue: Queue,
        paper_queue: Queue,
        output_queue: Queue
    ):
        """Worker that waits for both poster and paper to be ready."""
        poster_done = {}
        paper_done = {}
        poster_finished = False
        paper_finished = False

        while not (poster_finished and paper_finished):
            # Check poster queue
            if not poster_finished:
                try:
                    item = poster_queue.get(timeout=0.1)
                    if item is None:
                        poster_finished = True
                    else:
                        paper_id, success = item
                        poster_done[paper_id] = success
                except:
                    pass

            # Check paper queue
            if not paper_finished:
                try:
                    item = paper_queue.get(timeout=0.1)
                    if item is None:
                        paper_finished = True
                    else:
                        paper_id, success = item
                        paper_done[paper_id] = success
                except:
                    pass

            # Send ready items to output
            ready_ids = set(poster_done.keys()) & set(paper_done.keys())
            for paper_id in ready_ids:
                if poster_done[paper_id] and paper_done[paper_id]:
                    output_queue.put(paper_id)
                del poster_done[paper_id]
                del paper_done[paper_id]

        # Send remaining ready items
        ready_ids = set(poster_done.keys()) & set(paper_done.keys())
        for paper_id in ready_ids:
            if poster_done[paper_id] and paper_done[paper_id]:
                output_queue.put(paper_id)

        output_queue.put(None)

    def _vlm_worker(
        self,
        df: pd.DataFrame,
        input_queue: Queue,
        output_queue: Queue,
        gpu_id: int
    ):
        """Worker for VLM tasks (figure matching + poster description)."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from .figure_matcher import create_figure_matcher
        from .poster_descriptor import create_poster_descriptor

        # Create lookup for metadata
        df_lookup = df.set_index(df['paper_id'].astype(str))

        matcher_config = self.config.vlm_matcher
        matcher = create_figure_matcher(matcher_config)

        desc_config = self.config.poster_descriptor
        descriptor = create_poster_descriptor(desc_config)

        while True:
            item = input_queue.get()
            if item is None:
                break

            paper_id = item

            try:
                row = df_lookup.loc[paper_id]

                # Load figures
                poster_figures = self._load_figures_for_vlm(
                    self.config.output_dir / self.config.poster_parsed_dir / paper_id,
                    "poster"
                )
                paper_figures = self._load_figures_for_vlm(
                    self.config.output_dir / self.config.paper_parsed_dir / paper_id,
                    "paper"
                )

                # Match figures
                match_dir = self.config.output_dir / self.config.figure_matches_dir / paper_id
                if poster_figures and paper_figures:
                    matcher.match_figures(poster_figures, paper_figures, str(match_dir))
                else:
                    match_dir.mkdir(parents=True, exist_ok=True)
                    with open(match_dir / "report.json", 'w') as f:
                        json.dump({"matches": []}, f)

                # Describe poster
                poster_path = self.config.data_dir / row['local_image_path']
                layout = descriptor.describe_poster(str(poster_path), paper_id)
                layout.save(
                    self.config.output_dir / self.config.poster_descriptions_dir / f"{paper_id}_layout.json"
                )

                output_queue.put((paper_id, True))

            except Exception as e:
                print(f"VLM error for {paper_id}: {e}")
                output_queue.put((paper_id, False))

    def _load_figures_for_vlm(self, parsed_dir: Path, source: str) -> List[ExtractedFigure]:
        """Load figures for VLM worker."""
        figures = []
        figures_dir = parsed_dir / "markdown" / "figures"

        if not figures_dir.exists():
            return figures

        for fig_path in sorted(figures_dir.glob("*.png")):
            figures.append(ExtractedFigure(
                figure_id=fig_path.stem,
                source_doc=source,
                file_path=str(fig_path)
            ))

        return figures


def create_parallel_pipeline(
    config: PipelineConfig,
    strategy: str = "auto"
) -> StagedPipeline | MultiGPUPipeline:
    """
    Factory function to create parallel pipeline.

    Args:
        config: Pipeline configuration
        strategy: "staged", "multi_gpu", or "auto" (detects available GPUs)

    Returns:
        Pipeline instance
    """
    if strategy == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            return MultiGPUPipeline(config)
        else:
            return StagedPipeline(config)
    elif strategy == "multi_gpu":
        return MultiGPUPipeline(config)
    else:
        return StagedPipeline(config)
