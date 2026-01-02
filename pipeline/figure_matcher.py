"""
VLM Figure Matcher
==================
Module for matching figures between posters and papers using VLM (Qwen3-VL).

Uses a two-pass approach:
1. Fast feature matching (SIFT, perceptual hash, SSIM) to filter candidates
2. Qwen3-VL semantic comparison for final matching
"""

import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
import cv2
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from .config import VLMMatcherConfig
from .models import ExtractedFigure, FigureMatch


@dataclass
class MatchCandidate:
    """Candidate pair for VLM comparison."""
    poster_figure: ExtractedFigure
    paper_figure: ExtractedFigure
    feature_ratio: float
    combined_score: float


class VLMFigureMatcher:
    """
    VLM-based figure matcher using Qwen3-VL.

    Matches figures between poster and paper using:
    1. Feature-based pre-filtering (SIFT, hash, SSIM)
    2. VLM semantic comparison for candidates
    """

    def __init__(self, config: VLMMatcherConfig):
        """
        Initialize the figure matcher.

        Args:
            config: VLMMatcherConfig with model settings
        """
        self.config = config
        self.model = None
        self.processor = None
        self._model_loaded = False

    def load_model(self):
        """Load the Qwen3-VL model for semantic comparison."""
        if self._model_loaded:
            return

        print(f"Loading VLM model: {self.config.model_name}...")

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
        print("VLM model loaded!")

    def compute_feature_score(
        self,
        img1_path: str,
        img2_path: str
    ) -> Tuple[float, float]:
        """
        Compute feature-based similarity score for quick filtering.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image

        Returns:
            Tuple of (feature_ratio, combined_score)
        """
        pil1 = Image.open(img1_path).convert('RGB')
        pil2 = Image.open(img2_path).convert('RGB')

        arr1 = np.array(pil1)
        arr2 = np.array(pil2)

        # Perceptual hash
        phash1 = imagehash.phash(pil1, hash_size=16)
        phash2 = imagehash.phash(pil2, hash_size=16)
        phash_score = max(0, 1 - (phash1 - phash2) / 64)

        # SSIM
        size = (256, 256)
        ssim_arr1 = np.array(pil1.convert('L').resize(size))
        ssim_arr2 = np.array(pil2.convert('L').resize(size))
        ssim_score, _ = ssim(ssim_arr1, ssim_arr2, full=True)

        # Feature matching (SIFT)
        gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.resize(gray1, (512, 512))
        gray2 = cv2.resize(gray2, (512, 512))

        feature_ratio = 0.0
        try:
            sift = cv2.SIFT_create(nfeatures=500)
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)

            if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.7 * n.distance]
                feature_ratio = len(good) / min(len(kp1), len(kp2))
        except Exception:
            pass

        # Combined score
        combined = 0.3 * phash_score + 0.3 * ssim_score + 0.4 * feature_ratio

        return feature_ratio, combined

    def vlm_compare(self, img1_path: str, img2_path: str) -> dict:
        """
        Compare two figures using Qwen3-VL.

        Args:
            img1_path: Path to poster figure
            img2_path: Path to paper figure

        Returns:
            Dict with verdict, confidence, reasoning
        """
        import re

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1_path},
                    {"type": "image", "image": img2_path},
                    {
                        "type": "text",
                        "text": """Compare these two figures from academic documents (a poster and a research paper).

Determine if they represent the SAME figure (possibly resized, recolored, or reformatted).

Consider:
- Do they show the same data/visualization?
- Same chart type (scatter, line, bar, etc.)?
- Same axes labels and ranges?
- Same overall pattern/trend?

Respond in this exact JSON format:
{
    "verdict": "same" or "similar" or "different",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}

Only respond with the JSON, nothing else."""
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON response
        try:
            json_match = re.search(r'\{[^}]+\}', output_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "verdict": result.get("verdict", "unknown"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", "")
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse text response
        output_lower = output_text.lower()
        if "same" in output_lower and "different" not in output_lower:
            verdict = "same"
        elif "similar" in output_lower:
            verdict = "similar"
        else:
            verdict = "different"

        return {
            "verdict": verdict,
            "confidence": 0.5,
            "reasoning": output_text[:200]
        }

    def match_figures(
        self,
        poster_figures: List[ExtractedFigure],
        paper_figures: List[ExtractedFigure],
        output_dir: Optional[str] = None
    ) -> List[FigureMatch]:
        """
        Match figures between poster and paper.

        Args:
            poster_figures: List of figures extracted from poster
            paper_figures: List of figures extracted from paper
            output_dir: Optional directory to save match results

        Returns:
            List of FigureMatch objects
        """
        if not poster_figures or not paper_figures:
            print("No figures to match")
            return []

        print(f"\n{'='*70}")
        print(f"VLM FIGURE MATCHING")
        print(f"{'='*70}")
        print(f"Poster figures: {len(poster_figures)}")
        print(f"Paper figures:  {len(paper_figures)}")
        print(f"Total possible pairs: {len(poster_figures) * len(paper_figures)}")
        print(f"Feature threshold: {self.config.feature_threshold:.0%}")
        print(f"{'='*70}\n")

        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "same").mkdir(exist_ok=True)
            (output_path / "similar").mkdir(exist_ok=True)
            (output_path / "uncertain").mkdir(exist_ok=True)

        # PASS 1: Feature-based filtering
        print("PASS 1: Feature-based filtering...")
        candidates = []

        for pf in poster_figures:
            for rf in paper_figures:
                try:
                    feature_ratio, combined = self.compute_feature_score(
                        pf.file_path, rf.file_path
                    )
                    if feature_ratio >= self.config.feature_threshold:
                        candidates.append(MatchCandidate(
                            poster_figure=pf,
                            paper_figure=rf,
                            feature_ratio=feature_ratio,
                            combined_score=combined
                        ))
                except Exception as e:
                    print(f"Error computing score: {e}")
                    continue

        # Sort by feature ratio
        candidates.sort(key=lambda x: x.feature_ratio, reverse=True)
        print(f"Found {len(candidates)} candidate pairs")

        if not candidates:
            print("No candidates found. Try lowering feature_threshold")
            return []

        # PASS 2: VLM comparison
        print("\nPASS 2: VLM semantic comparison...")
        self.load_model()

        matches = []
        for i, candidate in enumerate(candidates):
            print(f"\n[{i+1}/{len(candidates)}] Comparing:")
            print(f"  Poster: {Path(candidate.poster_figure.file_path).name[:50]}")
            print(f"  Paper:  {Path(candidate.paper_figure.file_path).name[:50]}")
            print(f"  Feature ratio: {candidate.feature_ratio:.1%}")

            # Run VLM comparison
            vlm_result = self.vlm_compare(
                candidate.poster_figure.file_path,
                candidate.paper_figure.file_path
            )

            verdict = vlm_result["verdict"]
            confidence = vlm_result["confidence"]
            reasoning = vlm_result["reasoning"]

            print(f"  VLM verdict: {verdict.upper()} (confidence: {confidence:.2f})")

            # Determine final match status
            if verdict == "same" and confidence >= self.config.high_confidence_threshold:
                is_match = True
                final_confidence = "high"
            elif verdict == "same" or (verdict == "similar" and confidence >= 0.7):
                is_match = True
                final_confidence = "medium"
            elif verdict == "similar":
                is_match = True
                final_confidence = "low"
            else:
                is_match = False
                final_confidence = "none"

            if is_match:
                match = FigureMatch(
                    poster_figure=candidate.poster_figure,
                    paper_figure=candidate.paper_figure,
                    match_confidence=final_confidence,
                    feature_score=candidate.feature_ratio,
                    vlm_verdict=verdict,
                    vlm_confidence=confidence,
                    vlm_reasoning=reasoning
                )
                matches.append(match)

                # Save to output directory
                if output_dir:
                    if verdict == "same":
                        save_dir = output_path / "same"
                    elif verdict == "similar":
                        save_dir = output_path / "similar"
                    else:
                        save_dir = output_path / "uncertain"

                    match_idx = len(list(save_dir.glob("match_*_poster*"))) + 1
                    poster_ext = Path(candidate.poster_figure.file_path).suffix
                    paper_ext = Path(candidate.paper_figure.file_path).suffix

                    shutil.copy(
                        candidate.poster_figure.file_path,
                        save_dir / f"match_{match_idx:02d}_poster{poster_ext}"
                    )
                    shutil.copy(
                        candidate.paper_figure.file_path,
                        save_dir / f"match_{match_idx:02d}_paper{paper_ext}"
                    )

                    # Save reasoning
                    with open(save_dir / f"match_{match_idx:02d}_reasoning.txt", 'w') as f:
                        f.write(f"Poster: {candidate.poster_figure.file_path}\n")
                        f.write(f"Paper: {candidate.paper_figure.file_path}\n")
                        f.write(f"Feature ratio: {candidate.feature_ratio:.3f}\n")
                        f.write(f"VLM verdict: {verdict}\n")
                        f.write(f"VLM confidence: {confidence:.2f}\n")
                        f.write(f"Reasoning: {reasoning}\n")

        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Candidates analyzed: {len(candidates)}")
        print(f"Matches found: {len(matches)}")

        same_count = sum(1 for m in matches if m.vlm_verdict == "same")
        similar_count = sum(1 for m in matches if m.vlm_verdict == "similar")

        print(f"  Same figures: {same_count}")
        print(f"  Similar figures: {similar_count}")

        # Save report
        if output_dir:
            report = {
                "timestamp": datetime.now().isoformat(),
                "poster_figures_count": len(poster_figures),
                "paper_figures_count": len(paper_figures),
                "candidates_found": len(candidates),
                "matches": [m.to_dict() for m in matches]
            }

            with open(output_path / "report.json", 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\nResults saved to: {output_path}/")

        return matches


def create_figure_matcher(config: VLMMatcherConfig) -> VLMFigureMatcher:
    """Factory function to create a figure matcher."""
    return VLMFigureMatcher(config)
