"""
Data Models
===========
Pydantic/dataclass models for the pipeline data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


@dataclass
class ExtractedFigure:
    """Represents a figure extracted from a document."""
    figure_id: str
    source_doc: str  # "poster" or "paper"
    file_path: str
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2)
    caption: Optional[str] = None
    figure_type: Optional[str] = None  # "chart", "diagram", "image", etc.
    page_number: int = 1

    def to_dict(self) -> dict:
        return {
            "figure_id": self.figure_id,
            "source_doc": self.source_doc,
            "file_path": self.file_path,
            "bbox": self.bbox,
            "caption": self.caption,
            "figure_type": self.figure_type,
            "page_number": self.page_number
        }


@dataclass
class FigureMatch:
    """Represents a matched pair of figures between poster and paper."""
    poster_figure: ExtractedFigure
    paper_figure: ExtractedFigure
    match_confidence: str  # "high", "medium", "low"
    feature_score: float
    vlm_verdict: str  # "same", "similar", "different"
    vlm_confidence: float
    vlm_reasoning: str

    def to_dict(self) -> dict:
        return {
            "poster_figure": self.poster_figure.to_dict(),
            "paper_figure": self.paper_figure.to_dict(),
            "match_confidence": self.match_confidence,
            "feature_score": self.feature_score,
            "vlm_verdict": self.vlm_verdict,
            "vlm_confidence": self.vlm_confidence,
            "vlm_reasoning": self.vlm_reasoning
        }


@dataclass
class ParsedDocument:
    """Represents a parsed document (poster or paper)."""
    doc_id: str
    doc_type: str  # "poster" or "paper"
    source_path: str
    markdown_content: str
    figures: List[ExtractedFigure] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "source_path": self.source_path,
            "markdown_content": self.markdown_content,
            "figures": [f.to_dict() for f in self.figures],
            "tables": self.tables,
            "equations": self.equations,
            "metadata": self.metadata
        }

    def save(self, output_path: Path):
        """Save parsed document to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, json_path: Path) -> "ParsedDocument":
        """Load parsed document from JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        figures = [ExtractedFigure(**fig) for fig in data.get("figures", [])]
        return cls(
            doc_id=data["doc_id"],
            doc_type=data["doc_type"],
            source_path=data["source_path"],
            markdown_content=data["markdown_content"],
            figures=figures,
            tables=data.get("tables", []),
            equations=data.get("equations", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class PosterLayoutSection:
    """Represents a section in the poster layout."""
    id: int
    title: str
    column: int
    column_span: int
    row_in_column: int
    height_pct: float
    style: Dict[str, Any]
    content_type: str  # "text", "bullets", "figure", "table", "equation", "mixed"
    content_layout: Dict[str, Any]


@dataclass
class PosterLayout:
    """Represents the full poster layout structure."""
    paper_id: str
    poster_path: str
    orientation: str  # "landscape" or "portrait"
    aspect_ratio: str
    background: str
    header: Dict[str, Any]
    footer: Dict[str, Any]
    body: Dict[str, Any]
    sections: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    flowcharts: List[Dict[str, Any]]
    special_elements: List[Dict[str, Any]]
    color_scheme: Dict[str, Any]
    reading_order: str
    raw_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "poster_path": self.poster_path,
            "poster": {
                "orientation": self.orientation,
                "aspect_ratio": self.aspect_ratio,
                "background": self.background
            },
            "header": self.header,
            "footer": self.footer,
            "body": self.body,
            "sections": self.sections,
            "figures": self.figures,
            "flowcharts": self.flowcharts,
            "special_elements": self.special_elements,
            "color_scheme": self.color_scheme,
            "reading_order": self.reading_order
        }

    def save(self, output_path: Path):
        """Save poster layout to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_qwen_output(cls, paper_id: str, poster_path: str, json_output: dict) -> "PosterLayout":
        """Create PosterLayout from Qwen VLM output."""
        poster_info = json_output.get("poster", {})
        return cls(
            paper_id=paper_id,
            poster_path=poster_path,
            orientation=poster_info.get("orientation", "landscape"),
            aspect_ratio=poster_info.get("aspect_ratio", "16:9"),
            background=poster_info.get("background", "#FFFFFF"),
            header=json_output.get("header", {}),
            footer=json_output.get("footer", {}),
            body=json_output.get("body", {}),
            sections=json_output.get("sections", []),
            figures=json_output.get("figures", []),
            flowcharts=json_output.get("flowcharts", []),
            special_elements=json_output.get("special_elements", []),
            color_scheme=json_output.get("color_scheme", {}),
            reading_order=json_output.get("reading_order", "columns-left-to-right"),
            raw_json=json_output
        )


@dataclass
class TrainingExample:
    """Represents a single training example for the LLM."""
    paper_id: str
    # Input (from paper)
    paper_markdown: str
    paper_abstract: str
    paper_title: str
    # Output (for poster generation)
    poster_layout: PosterLayout
    poster_markdown: str  # Content that fills the poster sections
    # Matched figures mapping
    figure_matches: List[FigureMatch] = field(default_factory=list)
    # Metadata
    conference: str = ""
    year: int = 0
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_markdown": self.paper_markdown,
            "paper_abstract": self.paper_abstract,
            "paper_title": self.paper_title,
            "poster_layout": self.poster_layout.to_dict(),
            "poster_markdown": self.poster_markdown,
            "figure_matches": [m.to_dict() for m in self.figure_matches],
            "conference": self.conference,
            "year": self.year,
            "topics": self.topics
        }

    def to_training_format(self) -> dict:
        """Convert to format suitable for LLM finetuning."""
        return {
            "instruction": f"Generate an academic poster for the following research paper.\n\nTitle: {self.paper_title}\n\nAbstract: {self.paper_abstract}",
            "input": self.paper_markdown,
            "output": json.dumps({
                "layout": self.poster_layout.to_dict(),
                "content": self.poster_markdown,
                "figure_mapping": [
                    {
                        "paper_figure": m.paper_figure.figure_id,
                        "poster_position": m.poster_figure.figure_id
                    }
                    for m in self.figure_matches
                ]
            }, indent=2)
        }

    def save(self, output_path: Path):
        """Save training example to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class PipelineResult:
    """Result from processing a single paper-poster pair."""
    paper_id: str
    success: bool
    error: Optional[str] = None
    parsed_poster: Optional[ParsedDocument] = None
    parsed_paper: Optional[ParsedDocument] = None
    figure_matches: List[FigureMatch] = field(default_factory=list)
    poster_layout: Optional[PosterLayout] = None
    training_example: Optional[TrainingExample] = None

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "success": self.success,
            "error": self.error,
            "parsed_poster": self.parsed_poster.to_dict() if self.parsed_poster else None,
            "parsed_paper": self.parsed_paper.to_dict() if self.parsed_paper else None,
            "figure_matches": [m.to_dict() for m in self.figure_matches],
            "poster_layout": self.poster_layout.to_dict() if self.poster_layout else None
        }
