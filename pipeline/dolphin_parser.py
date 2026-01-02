"""
Dolphin Document Parser
=======================
Wrapper for ByteDance's Dolphin model for parsing posters and papers.

This module provides a unified interface for:
- Parsing poster images to extract markdown content + figures
- Parsing paper PDFs to extract markdown content + figures
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from PIL import Image

# Add Dolphin to path
DOLPHIN_PATH = Path(__file__).parent.parent / "Dolphin"
if str(DOLPHIN_PATH) not in sys.path:
    sys.path.insert(0, str(DOLPHIN_PATH))

from .config import DolphinConfig
from .models import ParsedDocument, ExtractedFigure


class DolphinParser:
    """
    Document parser using ByteDance's Dolphin model.

    Supports:
    - Poster images (PNG, JPG)
    - Paper PDFs (multi-page)

    Outputs:
    - Markdown content with figure references
    - Extracted figure images
    - Structured JSON with layout information
    """

    def __init__(self, config: DolphinConfig):
        """
        Initialize the Dolphin parser.

        Args:
            config: DolphinConfig with model path and settings
        """
        self.config = config
        self.model = None
        self._model_loaded = False

    def load_model(self):
        """Load the Dolphin model."""
        if self._model_loaded:
            return

        print(f"Loading Dolphin model from {self.config.model_path}...")

        # Import Dolphin components
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path
        )
        self.model.eval()

        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.device == "cuda":
            self.model = self.model.bfloat16()
        else:
            self.model = self.model.float()

        # Set tokenizer
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        self._model_loaded = True
        print(f"Dolphin model loaded on {self.device}!")

    def chat(self, prompt: str, image) -> str:
        """
        Run inference with the Dolphin model.

        Args:
            prompt: Text prompt for the model
            image: PIL Image or list of PIL Images

        Returns:
            Model response string (or list of strings for batch)
        """
        from qwen_vl_utils import process_vision_info
        from utils.utils import resize_img

        is_batch = isinstance(image, list)

        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        assert len(images) == len(prompts)

        # Preprocess all images
        processed_images = [resize_img(img) for img in images]

        # Generate all messages
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question}
                    ],
                }
            ]
            all_messages.append(messages)

        # Prepare all texts
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        # Collect all image inputs
        all_image_inputs = []
        for msgs in all_messages:
            image_inputs, video_inputs = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)

        # Prepare model inputs
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=None,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        results = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if not is_batch:
            return results[0]
        return results

    def parse_poster(
        self,
        image_path: str,
        output_dir: str,
        doc_id: str
    ) -> ParsedDocument:
        """
        Parse a poster image.

        Args:
            image_path: Path to the poster image
            output_dir: Directory to save outputs
            doc_id: Unique identifier for this document

        Returns:
            ParsedDocument with markdown and figures
        """
        return self._parse_document(image_path, output_dir, doc_id, "poster")

    def parse_paper(
        self,
        pdf_path: str,
        output_dir: str,
        doc_id: str
    ) -> ParsedDocument:
        """
        Parse a research paper PDF.

        Args:
            pdf_path: Path to the paper PDF
            output_dir: Directory to save outputs
            doc_id: Unique identifier for this document

        Returns:
            ParsedDocument with markdown and figures
        """
        return self._parse_document(pdf_path, output_dir, doc_id, "paper")

    def _parse_document(
        self,
        input_path: str,
        output_dir: str,
        doc_id: str,
        doc_type: str
    ) -> ParsedDocument:
        """
        Parse a document (poster or paper).

        Args:
            input_path: Path to input file (image or PDF)
            output_dir: Directory to save outputs
            doc_id: Unique identifier
            doc_type: "poster" or "paper"

        Returns:
            ParsedDocument with all extracted content
        """
        if not self._model_loaded:
            self.load_model()

        from utils.utils import (
            convert_pdf_to_images,
            setup_output_dirs,
            save_outputs,
            save_combined_pdf_results,
            parse_layout_string,
            process_coordinates,
            save_figure_to_local,
            check_bbox_overlap
        )
        from utils.markdown_utils import MarkdownConverter

        input_path = Path(input_path)
        output_dir = Path(output_dir)

        # Setup output directories
        setup_output_dirs(str(output_dir))

        file_ext = input_path.suffix.lower()

        if file_ext == '.pdf':
            # Multi-page PDF
            recognition_results, figures = self._process_pdf(
                input_path, output_dir, doc_id
            )
        else:
            # Single image
            pil_image = Image.open(input_path).convert("RGB")
            recognition_results, figures = self._process_single_image(
                pil_image, output_dir, doc_id
            )

        # Generate markdown
        markdown_converter = MarkdownConverter()
        markdown_content = markdown_converter.convert(recognition_results)

        # Save markdown
        markdown_path = output_dir / "markdown" / f"{doc_id}.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return ParsedDocument(
            doc_id=doc_id,
            doc_type=doc_type,
            source_path=str(input_path),
            markdown_content=markdown_content,
            figures=figures,
            metadata={
                "output_dir": str(output_dir),
                "markdown_path": str(markdown_path),
                "recognition_results": recognition_results
            }
        )

    def _process_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        doc_id: str
    ) -> Tuple[List[Dict], List[ExtractedFigure]]:
        """Process a multi-page PDF."""
        from utils.utils import convert_pdf_to_images

        images = convert_pdf_to_images(str(pdf_path))
        if not images:
            raise RuntimeError(f"Failed to convert PDF: {pdf_path}")

        all_results = []
        all_figures = []

        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")
            page_name = f"{doc_id}_page_{page_idx + 1:03d}"

            results, figures = self._process_single_image(
                pil_image, output_dir, page_name
            )

            # Add page number to results
            for r in results:
                r["page_number"] = page_idx + 1

            all_results.extend(results)
            all_figures.extend(figures)

        return all_results, all_figures

    def _process_single_image(
        self,
        pil_image: Image.Image,
        output_dir: Path,
        image_name: str
    ) -> Tuple[List[Dict], List[ExtractedFigure]]:
        """
        Process a single image using two-stage parsing.

        Stage 1: Layout and reading order detection
        Stage 2: Element-level content extraction
        """
        from utils.utils import (
            parse_layout_string,
            process_coordinates,
            save_figure_to_local,
            check_bbox_overlap
        )

        # Stage 1: Parse layout
        layout_output = self.chat("Parse the reading order of this document.", pil_image)

        # Parse layout string
        layout_results_list = parse_layout_string(layout_output)
        if not layout_results_list or not (layout_output.startswith("[") and layout_output.endswith("]")):
            layout_results_list = [([0, 0, *pil_image.size], 'distorted_page', [])]
        elif len(layout_results_list) > 1 and check_bbox_overlap(layout_results_list, pil_image):
            print("Falling back to distorted_page mode due to high bbox overlap")
            layout_results_list = [([0, 0, *pil_image.size], 'distorted_page', [])]

        # Stage 2: Process elements
        recognition_results = []
        figures = []

        # Group elements by type for batch processing
        tab_elements = []
        equ_elements = []
        code_elements = []
        text_elements = []

        reading_order = 0
        for bbox, label, tags in layout_results_list:
            try:
                if label == "distorted_page":
                    x1, y1, x2, y2 = 0, 0, *pil_image.size
                    pil_crop = pil_image
                else:
                    x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
                    pil_crop = pil_image.crop((x1, y1, x2, y2))

                if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                    if label == "fig":
                        # Save figure
                        figure_filename = save_figure_to_local(
                            pil_crop, str(output_dir), image_name, reading_order
                        )
                        figure_path = output_dir / "markdown" / "figures" / figure_filename

                        recognition_results.append({
                            "label": label,
                            "text": f"![Figure](figures/{figure_filename})",
                            "figure_path": f"figures/{figure_filename}",
                            "bbox": [x1, y1, x2, y2],
                            "reading_order": reading_order,
                            "tags": tags,
                        })

                        figures.append(ExtractedFigure(
                            figure_id=f"{image_name}_fig_{reading_order}",
                            source_doc="unknown",  # Will be set by caller
                            file_path=str(figure_path),
                            bbox=(x1, y1, x2, y2),
                            caption=None,
                            figure_type="figure",
                            page_number=1
                        ))
                    else:
                        element_info = {
                            "crop": pil_crop,
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "reading_order": reading_order,
                            "tags": tags,
                        }

                        if label == "tab":
                            tab_elements.append(element_info)
                        elif label == "equ":
                            equ_elements.append(element_info)
                        elif label == "code":
                            code_elements.append(element_info)
                        else:
                            text_elements.append(element_info)

                reading_order += 1

            except Exception as e:
                print(f"Error processing bbox with label {label}: {str(e)}")
                continue

        # Batch process elements by type
        if tab_elements:
            results = self._process_element_batch(tab_elements, "Parse the table in the image.")
            recognition_results.extend(results)

        if equ_elements:
            results = self._process_element_batch(equ_elements, "Read formula in the image.")
            recognition_results.extend(results)

        if code_elements:
            results = self._process_element_batch(code_elements, "Read code in the image.")
            recognition_results.extend(results)

        if text_elements:
            results = self._process_element_batch(text_elements, "Read text in the image.")
            recognition_results.extend(results)

        # Sort by reading order
        recognition_results.sort(key=lambda x: x.get("reading_order", 0))

        return recognition_results, figures

    def _process_element_batch(
        self,
        elements: List[Dict],
        prompt: str
    ) -> List[Dict]:
        """Process elements of the same type in batches."""
        results = []
        batch_size = self.config.batch_size or len(elements)

        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i + batch_size]
            crops_list = [elem["crop"] for elem in batch_elements]
            prompts_list = [prompt] * len(crops_list)

            # Batch inference
            batch_results = self.chat(prompts_list, crops_list)

            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                results.append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                    "tags": elem["tags"],
                })

        return results


def create_dolphin_parser(config: DolphinConfig) -> DolphinParser:
    """Factory function to create a Dolphin parser."""
    return DolphinParser(config)
