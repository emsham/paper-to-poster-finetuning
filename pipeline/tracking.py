"""
Processing Tracker
==================
Tracks processing status and output paths for each paper-poster pair.

Creates/updates a tracking CSV with:
- Processing status for each stage
- Paths to generated outputs
- Timestamps and error messages
- Easy filtering of processed/pending items
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
import pandas as pd


@dataclass
class ProcessingStatus:
    """Status for a single paper-poster pair."""
    paper_id: str

    # Stage completion status
    poster_parsed: bool = False
    paper_parsed: bool = False
    figures_matched: bool = False
    layout_extracted: bool = False
    training_data_created: bool = False

    # Output paths (relative to output_dir)
    poster_markdown_path: Optional[str] = None
    poster_figures_dir: Optional[str] = None
    paper_markdown_path: Optional[str] = None
    paper_figures_dir: Optional[str] = None
    figure_matches_path: Optional[str] = None
    layout_json_path: Optional[str] = None
    training_data_path: Optional[str] = None

    # Counts
    poster_figure_count: int = 0
    paper_figure_count: int = 0
    matched_figure_count: int = 0
    section_count: int = 0

    # Metadata
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_updated: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingStatus":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ProcessingTracker:
    """
    Tracks processing status for all paper-poster pairs.

    Usage:
        tracker = ProcessingTracker(output_dir)

        # Update status
        tracker.update_stage(paper_id, "poster_parsed",
                            poster_markdown_path="poster_parsed/123/markdown/123.md",
                            poster_figure_count=5)

        # Get pending items
        pending = tracker.get_pending("poster_parsed")

        # Export to CSV
        tracker.save()
    """

    def __init__(self, output_dir: Path, tracking_file: str = "processing_status.csv"):
        self.output_dir = Path(output_dir)
        self.tracking_file = self.output_dir / tracking_file
        self.status: Dict[str, ProcessingStatus] = {}

        # Load existing tracking data
        self._load()

    def _load(self):
        """Load existing tracking data from CSV."""
        if self.tracking_file.exists():
            try:
                df = pd.read_csv(self.tracking_file)
                for _, row in df.iterrows():
                    paper_id = str(row['paper_id'])
                    self.status[paper_id] = ProcessingStatus.from_dict(row.to_dict())
                print(f"Loaded tracking data for {len(self.status)} items")
            except Exception as e:
                print(f"Warning: Could not load tracking file: {e}")

    def save(self):
        """Save tracking data to CSV."""
        if not self.status:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        records = [s.to_dict() for s in self.status.values()]
        df = pd.DataFrame(records)
        df.to_csv(self.tracking_file, index=False)
        print(f"Saved tracking data for {len(self.status)} items to {self.tracking_file}")

    def initialize_from_dataframe(self, df: pd.DataFrame):
        """Initialize tracking for all items in a dataframe."""
        for _, row in df.iterrows():
            paper_id = str(row['paper_id'])
            if paper_id not in self.status:
                self.status[paper_id] = ProcessingStatus(paper_id=paper_id)

    def get_status(self, paper_id: str) -> ProcessingStatus:
        """Get or create status for a paper."""
        paper_id = str(paper_id)
        if paper_id not in self.status:
            self.status[paper_id] = ProcessingStatus(paper_id=paper_id)
        return self.status[paper_id]

    def update_stage(
        self,
        paper_id: str,
        stage: str,
        success: bool = True,
        error: str = None,
        **kwargs
    ):
        """
        Update processing status for a stage.

        Args:
            paper_id: Paper identifier
            stage: One of: poster_parsed, paper_parsed, figures_matched,
                   layout_extracted, training_data_created
            success: Whether the stage completed successfully
            error: Error message if failed
            **kwargs: Additional fields to update (paths, counts, etc.)
        """
        status = self.get_status(paper_id)

        # Update stage completion
        if hasattr(status, stage):
            setattr(status, stage, success)

        # Update error
        if error:
            status.error_message = error

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(status, key):
                setattr(status, key, value)

        # Update timestamp
        status.last_updated = datetime.now().isoformat()

        # Mark started if first update
        if status.started_at is None:
            status.started_at = status.last_updated

    def mark_complete(self, paper_id: str, processing_time: float = None):
        """Mark a paper as fully processed."""
        status = self.get_status(paper_id)
        status.completed_at = datetime.now().isoformat()
        if processing_time:
            status.processing_time_seconds = processing_time

    def get_pending(self, stage: str = None) -> List[str]:
        """
        Get paper IDs that haven't completed a stage.

        Args:
            stage: Stage to check, or None for fully incomplete items

        Returns:
            List of paper IDs
        """
        pending = []
        for paper_id, status in self.status.items():
            if stage:
                if not getattr(status, stage, False):
                    pending.append(paper_id)
            else:
                # Check if fully complete
                if not status.training_data_created:
                    pending.append(paper_id)
        return pending

    def get_completed(self, stage: str = None) -> List[str]:
        """Get paper IDs that have completed a stage."""
        completed = []
        for paper_id, status in self.status.items():
            if stage:
                if getattr(status, stage, False):
                    completed.append(paper_id)
            else:
                if status.training_data_created:
                    completed.append(paper_id)
        return completed

    def get_failed(self) -> List[str]:
        """Get paper IDs that have errors."""
        return [
            paper_id for paper_id, status in self.status.items()
            if status.error_message
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics."""
        total = len(self.status)
        if total == 0:
            return {"total": 0}

        return {
            "total": total,
            "poster_parsed": sum(1 for s in self.status.values() if s.poster_parsed),
            "paper_parsed": sum(1 for s in self.status.values() if s.paper_parsed),
            "figures_matched": sum(1 for s in self.status.values() if s.figures_matched),
            "layout_extracted": sum(1 for s in self.status.values() if s.layout_extracted),
            "training_data_created": sum(1 for s in self.status.values() if s.training_data_created),
            "failed": len(self.get_failed()),
            "avg_poster_figures": sum(s.poster_figure_count for s in self.status.values()) / total,
            "avg_paper_figures": sum(s.paper_figure_count for s in self.status.values()) / total,
            "avg_matched_figures": sum(s.matched_figure_count for s in self.status.values()) / total,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export tracking data as a DataFrame."""
        if not self.status:
            return pd.DataFrame()
        records = [s.to_dict() for s in self.status.values()]
        return pd.DataFrame(records)

    def merge_with_source(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge tracking data with source dataframe.

        Returns a new dataframe with all original columns plus tracking columns.
        """
        tracking_df = self.to_dataframe()
        if tracking_df.empty:
            return source_df

        # Ensure paper_id is string in both
        source_df = source_df.copy()
        source_df['paper_id'] = source_df['paper_id'].astype(str)
        tracking_df['paper_id'] = tracking_df['paper_id'].astype(str)

        # Merge
        merged = source_df.merge(
            tracking_df,
            on='paper_id',
            how='left',
            suffixes=('', '_tracking')
        )

        return merged

    def export_with_source(
        self,
        source_df: pd.DataFrame,
        output_path: str,
        format: str = "csv"
    ):
        """
        Export source dataframe merged with tracking data.

        Args:
            source_df: Original dataframe (train.csv, etc.)
            output_path: Where to save
            format: "csv" or "parquet"
        """
        merged = self.merge_with_source(source_df)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            merged.to_parquet(output_path, index=False)
        else:
            merged.to_csv(output_path, index=False)

        print(f"Exported {len(merged)} rows to {output_path}")


def create_tracker(output_dir: str) -> ProcessingTracker:
    """Factory function to create a tracker."""
    return ProcessingTracker(Path(output_dir))
