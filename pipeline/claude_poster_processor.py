"""
Claude Poster Processor
=======================
Extracts poster content and layout using Claude Haiku API.
Combines two expensive VLM operations into one cheap API call.
"""

import os
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import httpx

from .models import ParsedDocument, PosterLayout, ExtractedFigure


@dataclass
class ClaudeConfig:
    """Configuration for Claude API."""
    api_key: str = ""  # From env var ANTHROPIC_API_KEY
    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3
    base_url: str = "https://api.anthropic.com/v1"

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")


POSTER_EXTRACTION_PROMPT = """Analyze this academic poster image and extract both its content and visual layout structure.

## PART 1: CONTENT EXTRACTION
Transcribe ALL text content from the poster in markdown format:
- Title (as # heading)
- Authors and affiliations
- All section headers (as ## headings)
- Body text in each section
- Figure captions (format: [Figure N: caption text])
- Table captions if present
- Any equations or formulas
- References/citations if visible

## PART 2: LAYOUT STRUCTURE
Describe the visual layout as JSON with this exact structure:
{
  "poster": {
    "orientation": "landscape" or "portrait",
    "aspect_ratio": "approximate ratio like 16:9, 4:3, etc",
    "background": "#hexcolor"
  },
  "header": {
    "height_pct": number (0-100),
    "background": "#hexcolor or gradient or none",
    "title_alignment": "left" or "center",
    "logo_positions": ["left", "right", "none"]
  },
  "footer": {
    "present": true/false,
    "height_pct": number,
    "content": "description"
  },
  "body": {
    "columns": number,
    "column_widths": ["equal"] or ["30%", "40%", "30%"],
    "gutter_pct": number
  },
  "sections": [
    {
      "id": number,
      "title": "Section Title",
      "column": number,
      "column_span": 1,
      "row_in_column": number,
      "height_pct": number,
      "style": {
        "header_bg": "#hex or none",
        "body_bg": "#hex or transparent",
        "border": "#hex or none"
      },
      "content_type": "text/bullets/figure/table/mixed",
      "has_figures": true/false
    }
  ],
  "figures": [
    {
      "id": number,
      "section_id": number,
      "type": "chart/diagram/photo/graph",
      "caption_visible": true/false
    }
  ],
  "color_scheme": {
    "primary": "#hex",
    "secondary": "#hex",
    "accent": "#hex",
    "text": "#hex",
    "background": "#hex"
  },
  "reading_order": "columns-left-to-right" or "rows-top-to-bottom"
}

## OUTPUT FORMAT
Return your response in exactly this format:

---CONTENT---
[Your markdown transcription here]

---LAYOUT---
[Your JSON layout here]
"""


class ClaudePosterProcessor:
    """
    Extracts poster content and layout using Claude Haiku API.

    Cost: ~$0.003 per poster (vs ~$2+ with local VLMs on GPU)
    Speed: ~5 seconds per poster (vs ~3+ minutes with local VLMs)
    """

    def __init__(self, config: Optional[ClaudeConfig] = None):
        self.config = config or ClaudeConfig()
        if not self.config.api_key:
            raise ValueError(
                "Claude API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in ClaudeConfig."
            )

    def _encode_image(self, image_path: Path) -> Tuple[str, str]:
        """Encode image to base64 and determine media type."""
        suffix = image_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_types.get(suffix, 'image/png')

        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        return image_data, media_type

    def _parse_response(
        self,
        response_text: str,
        paper_id: str,
        poster_path: str
    ) -> Tuple[str, dict]:
        """Parse Claude response into content and layout."""
        content = ""
        layout = {}

        # Split by markers
        if "---CONTENT---" in response_text and "---LAYOUT---" in response_text:
            parts = response_text.split("---LAYOUT---")
            content_part = parts[0].replace("---CONTENT---", "").strip()
            layout_part = parts[1].strip() if len(parts) > 1 else "{}"

            content = content_part

            # Parse JSON layout
            try:
                # Handle markdown code blocks
                if "```json" in layout_part:
                    layout_part = layout_part.split("```json")[1].split("```")[0]
                elif "```" in layout_part:
                    layout_part = layout_part.split("```")[1].split("```")[0]

                layout = json.loads(layout_part)
            except json.JSONDecodeError:
                # Try to extract JSON object
                try:
                    start = layout_part.find('{')
                    end = layout_part.rfind('}') + 1
                    if start >= 0 and end > start:
                        layout = json.loads(layout_part[start:end])
                except:
                    layout = self._get_default_layout()
        else:
            # Fallback: treat entire response as content
            content = response_text
            layout = self._get_default_layout()

        return content, layout

    def _get_default_layout(self) -> dict:
        """Return default layout structure when parsing fails."""
        return {
            "poster": {"orientation": "landscape", "aspect_ratio": "16:9", "background": "#FFFFFF"},
            "header": {"height_pct": 15, "background": "none", "title_alignment": "center", "logo_positions": []},
            "footer": {"present": False, "height_pct": 0, "content": ""},
            "body": {"columns": 3, "column_widths": ["equal"], "gutter_pct": 2},
            "sections": [],
            "figures": [],
            "color_scheme": {"primary": "#000000", "secondary": "#333333", "accent": "#0066CC", "text": "#000000", "background": "#FFFFFF"},
            "reading_order": "columns-left-to-right"
        }

    async def process_poster_async(
        self,
        image_path: str,
        paper_id: str,
        output_dir: Optional[str] = None
    ) -> Tuple[ParsedDocument, PosterLayout]:
        """
        Process a poster image asynchronously.

        Args:
            image_path: Path to the poster image
            paper_id: Unique identifier
            output_dir: Optional directory to save outputs

        Returns:
            Tuple of (ParsedDocument, PosterLayout)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode image
        image_data, media_type = self._encode_image(image_path)

        # Prepare API request
        headers = {
            "x-api-key": self.config.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": POSTER_EXTRACTION_PROMPT
                        }
                    ]
                }
            ]
        }

        # Make API request with retries
        response_text = ""
        last_error = None

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            for attempt in range(self.config.max_retries):
                try:
                    response = await client.post(
                        f"{self.config.base_url}/messages",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result["content"][0]["text"]
                    break

                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    elif e.response.status_code >= 500:
                        # Server error - retry
                        await asyncio.sleep(1)
                    else:
                        raise

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_error = e
                    await asyncio.sleep(1)

            else:
                raise last_error or Exception("Max retries exceeded")

        # Parse response
        markdown_content, layout_json = self._parse_response(
            response_text, paper_id, str(image_path)
        )

        # Save outputs if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            markdown_dir = output_dir / "markdown"
            markdown_dir.mkdir(parents=True, exist_ok=True)

            markdown_path = markdown_dir / f"{paper_id}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

        # Create ParsedDocument
        parsed_doc = ParsedDocument(
            doc_id=paper_id,
            doc_type="poster",
            source_path=str(image_path),
            markdown_content=markdown_content,
            figures=[],  # Figures identified in layout JSON
            metadata={
                "parser": "claude-haiku",
                "model": self.config.model
            }
        )

        # Create PosterLayout
        poster_layout = PosterLayout.from_qwen_output(
            paper_id=paper_id,
            poster_path=str(image_path),
            json_output=layout_json
        )

        return parsed_doc, poster_layout

    def process_poster(
        self,
        image_path: str,
        paper_id: str,
        output_dir: Optional[str] = None
    ) -> Tuple[ParsedDocument, PosterLayout]:
        """
        Process a poster image (synchronous wrapper).

        Args:
            image_path: Path to the poster image
            paper_id: Unique identifier
            output_dir: Optional directory to save outputs

        Returns:
            Tuple of (ParsedDocument, PosterLayout)
        """
        return asyncio.run(
            self.process_poster_async(image_path, paper_id, output_dir)
        )

    async def process_batch_async(
        self,
        items: List[Tuple[str, str]],  # List of (image_path, paper_id)
        output_dir: Optional[str] = None,
        concurrency: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[str, Optional[ParsedDocument], Optional[PosterLayout], Optional[str]]]:
        """
        Process multiple posters concurrently.

        Args:
            items: List of (image_path, paper_id) tuples
            output_dir: Optional directory to save outputs
            concurrency: Max concurrent API calls
            progress_callback: Optional callback(completed, total)

        Returns:
            List of (paper_id, parsed_doc, layout, error) tuples
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        completed = 0

        async def process_one(image_path: str, paper_id: str):
            nonlocal completed
            async with semaphore:
                try:
                    doc, layout = await self.process_poster_async(
                        image_path, paper_id, output_dir
                    )
                    result = (paper_id, doc, layout, None)
                except Exception as e:
                    result = (paper_id, None, None, str(e))

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

                return result

        tasks = [process_one(img, pid) for img, pid in items]
        results = await asyncio.gather(*tasks)

        return list(results)

    def process_batch(
        self,
        items: List[Tuple[str, str]],
        output_dir: Optional[str] = None,
        concurrency: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[str, Optional[ParsedDocument], Optional[PosterLayout], Optional[str]]]:
        """
        Process multiple posters (synchronous wrapper).
        """
        return asyncio.run(
            self.process_batch_async(items, output_dir, concurrency, progress_callback)
        )
