"""
Marker Parser
=============
Fast paper parser using marker-pdf library.
Replaces DolphinParser for paper parsing at a fraction of the cost.
"""

import os
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from dataclasses import dataclass

from .models import ParsedDocument, ExtractedFigure


@dataclass
class MarkerConfig:
    """Configuration for marker-pdf parser."""
    extract_figures: bool = True
    min_figure_size: int = 100  # Minimum width/height in pixels
    output_format: str = "markdown"


class MarkerParser:
    """
    Fast paper parser using marker-pdf.

    This is a drop-in replacement for DolphinParser that runs on CPU
    and is significantly faster (~10 sec vs ~7 min per paper).
    """

    def __init__(self, config: Optional[MarkerConfig] = None):
        self.config = config or MarkerConfig()
        self._marker_available = None

    def _check_marker(self) -> bool:
        """Check if marker-pdf is available."""
        if self._marker_available is None:
            try:
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict
                self._marker_available = True
            except ImportError:
                self._marker_available = False
        return self._marker_available

    def parse_paper(
        self,
        pdf_path: str,
        output_dir: str,
        doc_id: str
    ) -> ParsedDocument:
        """
        Parse a research paper PDF to markdown.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save outputs
            doc_id: Unique identifier for the document

        Returns:
            ParsedDocument with markdown content and extracted figures
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create output directories
        markdown_dir = output_dir / "markdown"
        figures_dir = markdown_dir / "figures"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Try marker-pdf first, fallback to PyMuPDF
        if self._check_marker():
            markdown_content = self._parse_with_marker(pdf_path)
        else:
            markdown_content = self._parse_with_pymupdf(pdf_path)

        # Extract figures
        figures = []
        if self.config.extract_figures:
            figures = self._extract_figures(pdf_path, figures_dir, doc_id)

        # Save markdown
        markdown_path = markdown_dir / f"{doc_id}.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return ParsedDocument(
            doc_id=doc_id,
            doc_type="paper",
            source_path=str(pdf_path),
            markdown_content=markdown_content,
            figures=figures,
            metadata={
                "output_dir": str(output_dir),
                "markdown_path": str(markdown_path),
                "parser": "marker" if self._marker_available else "pymupdf"
            }
        )

    def _parse_with_marker(self, pdf_path: Path) -> str:
        """Parse PDF using marker-pdf library."""
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        # Create model dict (uses CPU by default)
        model_dict = create_model_dict()

        # Convert PDF to markdown
        converter = PdfConverter(artifact_dict=model_dict)
        rendered = converter(str(pdf_path))

        # Extract text content
        text, _, _ = text_from_rendered(rendered)
        return text

    def _parse_with_pymupdf(self, pdf_path: Path) -> str:
        """Fallback: Parse PDF using PyMuPDF text extraction."""
        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            # Get text with layout preservation
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"## Page {page_num + 1}\n\n{text}")

        doc.close()
        return "\n\n".join(text_parts)

    def _extract_figures(
        self,
        pdf_path: Path,
        output_dir: Path,
        doc_id: str
    ) -> list:
        """Extract figures/images from PDF."""
        figures = []
        doc = fitz.open(pdf_path)

        fig_count = 0
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip small images (likely icons or artifacts)
                    if width < self.config.min_figure_size or height < self.config.min_figure_size:
                        continue

                    # Save image
                    fig_count += 1
                    fig_filename = f"{doc_id}_fig_{fig_count}.{image_ext}"
                    fig_path = output_dir / fig_filename

                    with open(fig_path, 'wb') as f:
                        f.write(image_bytes)

                    figures.append(ExtractedFigure(
                        figure_id=f"{doc_id}_fig_{fig_count}",
                        source_doc="paper",
                        file_path=str(fig_path),
                        bbox=None,  # PyMuPDF doesn't give bbox easily
                        caption=None,
                        figure_type="figure",
                        page_number=page_num + 1
                    ))

                except Exception as e:
                    # Skip problematic images
                    continue

        doc.close()
        return figures

    def parse_poster(
        self,
        image_path: str,
        output_dir: str,
        doc_id: str
    ) -> ParsedDocument:
        """
        Parse a poster image using OCR.

        Note: For the simplified pipeline, poster content is extracted
        via Claude API instead. This method exists for compatibility.

        Args:
            image_path: Path to the poster image
            output_dir: Directory to save outputs
            doc_id: Unique identifier for the document

        Returns:
            ParsedDocument with OCR text content
        """
        import pytesseract
        from PIL import Image

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create output directories
        markdown_dir = output_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)

        # Run OCR
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)

        # Basic markdown formatting
        markdown_content = f"# Poster Content\n\n{text}"

        # Save markdown
        markdown_path = markdown_dir / f"{doc_id}.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return ParsedDocument(
            doc_id=doc_id,
            doc_type="poster",
            source_path=str(image_path),
            markdown_content=markdown_content,
            figures=[],  # Poster figures extracted by Claude
            metadata={
                "output_dir": str(output_dir),
                "markdown_path": str(markdown_path),
                "parser": "tesseract"
            }
        )
