"""
Poster Layout Descriptor
========================
Module for extracting structured JSON layout descriptions from academic posters
using Qwen3-VL vision-language model.

The output JSON can be used to:
1. Generate LaTeX poster templates
2. Train LLMs to produce poster layouts from paper content
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image

from .config import PosterDescriptorConfig
from .models import PosterLayout


# Prompt for poster layout extraction
LAYOUT_EXTRACTION_PROMPT = """You are a precise layout analysis system for academic posters. Your task is to extract structural and spatial information that enables exact LaTeX reconstruction.

Focus on:
- Grid structure and column layout
- Section positions and dimensions (as percentages)
- Section styling (colors, borders, headers)
- Figure types and arrangements within sections
- Visual hierarchy and reading flow

Do NOT transcribe text content - only capture structure.

Output valid JSON following the schema exactly.

Analyze this academic poster's layout structure. Output JSON following this schema:

{
  "poster": {
    "orientation": "landscape|portrait",
    "aspect_ratio": "e.g., 16:9, 4:3, 3:2",
    "background": "#hex"
  },
  "header": {
    "height_pct": number,
    "background": "#hex|gradient|none",
    "gradient_colors": ["#hex1", "#hex2"] or null,
    "title_alignment": "left|center",
    "logo_positions": ["left", "right", "none"],
    "has_author_affiliation_superscripts": boolean
  },
  "footer": {
    "present": boolean,
    "height_pct": number or null,
    "background": "#hex|gradient|none",
    "content": "contact|references|qr_code|none"
  },
  "body": {
    "columns": number,
    "column_widths": ["equal"] or ["30%", "40%", "30%"],
    "gutter_pct": number
  },
  "sections": [
    {
      "id": number,
      "title": "Section Title Text",
      "column": number,
      "column_span": number,
      "row_in_column": number,
      "height_pct": number,
      "style": {
        "header_bg": "#hex|none",
        "header_text_color": "#hex",
        "body_bg": "#hex|transparent",
        "border": "#hex|none",
        "border_radius": "none|small|medium|large"
      },
      "content_type": "text|bullets|figure|table|equation|flowchart|mixed",
      "content_layout": {
        "arrangement": "vertical|horizontal|grid",
        "split": "description if mixed, e.g., 'text-left figure-right'",
        "figure_count": number or null,
        "has_equations": boolean,
        "has_tables": boolean
      }
    }
  ],
  "figures": [
    {
      "id": number,
      "section_id": number,
      "caption": "Figure caption or description",
      "position": "description of position in section",
      "type": "line chart|bar chart|scatter plot|diagram|flowchart|image|table|other",
      "description": "Brief visual description"
    }
  ],
  "flowcharts": [
    {
      "id": number,
      "section_id": number,
      "node_count": number,
      "direction": "horizontal|vertical|mixed",
      "description": "Brief description of the flowchart"
    }
  ],
  "special_elements": [
    {
      "type": "qr_code|logo|icon|highlight_box|callout",
      "position": "description",
      "description": "what it represents"
    }
  ],
  "color_scheme": {
    "primary": "#hex",
    "secondary": "#hex",
    "accent": "#hex",
    "text": "#hex",
    "background": "#hex"
  },
  "reading_order": "columns-left-to-right|rows-top-to-bottom|numbered|arrows"
}

Respond ONLY with valid JSON. No explanations before or after."""


class PosterDescriptor:
    """
    Extracts structured layout descriptions from academic posters.

    Uses Qwen3-VL to analyze poster images and output JSON
    describing the layout structure for LaTeX reconstruction.
    """

    def __init__(self, config: PosterDescriptorConfig):
        """
        Initialize the poster descriptor.

        Args:
            config: PosterDescriptorConfig with model settings
        """
        self.config = config
        self.model = None
        self.processor = None
        self._model_loaded = False

    def load_model(self):
        """Load the Qwen3-VL model."""
        if self._model_loaded:
            return

        print(f"Loading poster descriptor model: {self.config.model_name}...")

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        if self.config.use_flash_attn:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self._model_loaded = True
        print("Poster descriptor model loaded!")

    def describe_poster(
        self,
        image_path: str,
        paper_id: str,
        custom_prompt: Optional[str] = None
    ) -> PosterLayout:
        """
        Extract layout description from a poster image.

        Args:
            image_path: Path to the poster image
            paper_id: Identifier for this paper/poster
            custom_prompt: Optional custom prompt (uses default if None)

        Returns:
            PosterLayout object with structured layout info
        """
        if not self._model_loaded:
            self.load_model()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare prompt
        prompt = custom_prompt or LAYOUT_EXTRACTION_PROMPT

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON from output
        json_output = self._parse_json_output(output_text)

        # Create PosterLayout object
        return PosterLayout.from_qwen_output(paper_id, image_path, json_output)

    def _parse_json_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse JSON from model output.

        Handles various formats including markdown code blocks.
        """
        # Try direct JSON parse first
        try:
            return json.loads(output_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', output_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty structure if parsing fails
        print(f"Warning: Failed to parse JSON from output. Raw output:\n{output_text[:500]}...")
        return self._get_default_layout()

    def _get_default_layout(self) -> Dict[str, Any]:
        """Return a default layout structure when parsing fails."""
        return {
            "poster": {
                "orientation": "landscape",
                "aspect_ratio": "16:9",
                "background": "#FFFFFF"
            },
            "header": {
                "height_pct": 10,
                "background": "#FFFFFF",
                "gradient_colors": None,
                "title_alignment": "center",
                "logo_positions": [],
                "has_author_affiliation_superscripts": False
            },
            "footer": {
                "present": False,
                "height_pct": None,
                "background": "none",
                "content": "none"
            },
            "body": {
                "columns": 3,
                "column_widths": ["equal"],
                "gutter_pct": 2
            },
            "sections": [],
            "figures": [],
            "flowcharts": [],
            "special_elements": [],
            "color_scheme": {
                "primary": "#000000",
                "secondary": "#FFFFFF",
                "accent": "#0000FF"
            },
            "reading_order": "columns-left-to-right"
        }

    def batch_describe(
        self,
        image_paths: list,
        paper_ids: list,
        output_dir: Optional[str] = None
    ) -> list:
        """
        Process multiple posters in batch.

        Args:
            image_paths: List of paths to poster images
            paper_ids: List of paper identifiers
            output_dir: Optional directory to save results

        Returns:
            List of PosterLayout objects
        """
        if not self._model_loaded:
            self.load_model()

        results = []

        for i, (image_path, paper_id) in enumerate(zip(image_paths, paper_ids)):
            print(f"\nProcessing poster {i+1}/{len(image_paths)}: {paper_id}")

            try:
                layout = self.describe_poster(image_path, paper_id)
                results.append(layout)

                # Save if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    layout.save(output_path / f"{paper_id}_layout.json")

            except Exception as e:
                print(f"Error processing {paper_id}: {e}")
                results.append(None)

        return results


def create_poster_descriptor(config: PosterDescriptorConfig) -> PosterDescriptor:
    """Factory function to create a poster descriptor."""
    return PosterDescriptor(config)
