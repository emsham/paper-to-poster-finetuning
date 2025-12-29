"""
Figure Extraction and Comparison Pipeline
=========================================
Uses Dolphin for figure extraction from documents (poster + paper),
then compares figures to find matches.

Requirements:
    pip install torch torchvision transformers pillow imagehash scikit-image numpy opencv-python

Usage:
    python figure_comparison.py --model_path ./hf_model --doc1 poster.pdf --doc2 paper.pdf
"""

import argparse
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
import cv2


@dataclass
class ExtractedFigure:
    """Represents an extracted figure from a document."""
    source_doc: str
    figure_id: str
    bbox: tuple  # (x1, y1, x2, y2)
    page_num: int
    image: Image.Image
    phash: Optional[str] = None
    
    def compute_hash(self):
        """Compute perceptual hash for the figure."""
        self.phash = str(imagehash.phash(self.image))
        return self.phash


@dataclass 
class FigureMatch:
    """Represents a match between two figures."""
    fig1: ExtractedFigure
    fig2: ExtractedFigure
    hash_distance: int
    ssim_score: float
    is_match: bool
    confidence: str  # "exact", "high", "medium", "low"


class FigureComparator:
    """Compare figures using multiple similarity metrics."""
    
    def __init__(self, hash_threshold: int = 10, ssim_threshold: float = 0.85):
        """
        Args:
            hash_threshold: Max Hamming distance for perceptual hash (0 = identical, <10 = very similar)
            ssim_threshold: Min SSIM score to consider a match (1.0 = identical)
        """
        self.hash_threshold = hash_threshold
        self.ssim_threshold = ssim_threshold
    
    def compute_hash_distance(self, fig1: ExtractedFigure, fig2: ExtractedFigure) -> int:
        """Compute Hamming distance between perceptual hashes."""
        if fig1.phash is None:
            fig1.compute_hash()
        if fig2.phash is None:
            fig2.compute_hash()
        
        hash1 = imagehash.hex_to_hash(fig1.phash)
        hash2 = imagehash.hex_to_hash(fig2.phash)
        return hash1 - hash2
    
    def compute_ssim(self, fig1: ExtractedFigure, fig2: ExtractedFigure) -> float:
        """Compute Structural Similarity Index between two figures."""
        # Resize to same dimensions for comparison
        size = (256, 256)
        img1 = fig1.image.convert('L').resize(size)
        img2 = fig2.image.convert('L').resize(size)
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        score, _ = ssim(arr1, arr2, full=True)
        return score
    
    def compute_feature_match(self, fig1: ExtractedFigure, fig2: ExtractedFigure) -> float:
        """Use ORB feature matching for more robust comparison."""
        # Convert to OpenCV format
        img1 = cv2.cvtColor(np.array(fig1.image), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(fig2.image), cv2.COLOR_RGB2GRAY)
        
        # Resize for consistency
        img1 = cv2.resize(img1, (512, 512))
        img2 = cv2.resize(img2, (512, 512))
        
        # ORB detector
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Calculate match ratio
        good_matches = [m for m in matches if m.distance < 50]
        match_ratio = len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        
        return match_ratio
    
    def compare(self, fig1: ExtractedFigure, fig2: ExtractedFigure) -> FigureMatch:
        """Compare two figures and determine if they match."""
        hash_dist = self.compute_hash_distance(fig1, fig2)
        ssim_score = self.compute_ssim(fig1, fig2)
        
        # Determine match confidence
        if hash_dist == 0 and ssim_score > 0.99:
            confidence = "exact"
            is_match = True
        elif hash_dist <= 5 and ssim_score > 0.95:
            confidence = "high"
            is_match = True
        elif hash_dist <= self.hash_threshold and ssim_score > self.ssim_threshold:
            confidence = "medium"
            is_match = True
        elif hash_dist <= self.hash_threshold or ssim_score > self.ssim_threshold:
            confidence = "low"
            is_match = True
        else:
            confidence = "none"
            is_match = False
        
        return FigureMatch(
            fig1=fig1,
            fig2=fig2,
            hash_distance=hash_dist,
            ssim_score=ssim_score,
            is_match=is_match,
            confidence=confidence
        )
    
    def find_all_matches(self, figures1: list[ExtractedFigure], 
                         figures2: list[ExtractedFigure]) -> list[FigureMatch]:
        """Find all matching figures between two sets."""
        matches = []
        
        for fig1 in figures1:
            for fig2 in figures2:
                match = self.compare(fig1, fig2)
                if match.is_match:
                    matches.append(match)
        
        # Sort by confidence (exact > high > medium > low)
        confidence_order = {"exact": 0, "high": 1, "medium": 2, "low": 3}
        matches.sort(key=lambda m: (confidence_order[m.confidence], m.hash_distance))
        
        return matches


class DolphinFigureExtractor:
    """Extract figures from documents using Dolphin."""
    
    def __init__(self, model_path: str):
        """
        Initialize Dolphin model.
        
        Args:
            model_path: Path to Dolphin model weights
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Lazy load the Dolphin model."""
        if self.model is not None:
            return
            
        from transformers import AutoModel, AutoProcessor
        
        print("Loading Dolphin model...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        ).eval()
        
        # Move to GPU if available
        import torch
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("Model loaded!")
    
    def extract_figures_from_image(self, image: Image.Image, 
                                   source_doc: str, 
                                   page_num: int = 0) -> list[ExtractedFigure]:
        """
        Extract figures from a single page image using Dolphin's layout analysis.
        
        Args:
            image: PIL Image of the document page
            source_doc: Name/path of source document
            page_num: Page number in the document
            
        Returns:
            List of ExtractedFigure objects
        """
        self.load_model()
        
        import torch
        
        # Use Dolphin's layout analysis to find figures
        # The model outputs elements with their bounding boxes and types
        inputs = self.processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False
            )
        
        # Decode the output
        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the layout results to find figures
        figures = []
        
        try:
            # Dolphin outputs JSON with element types and bboxes
            # Look for elements with type "figure" or "image"
            parsed = json.loads(result) if result.strip().startswith('{') else {"elements": []}
            
            for idx, elem in enumerate(parsed.get("elements", [])):
                elem_type = elem.get("type", "").lower()
                
                if elem_type in ["figure", "image", "chart", "diagram", "graph"]:
                    bbox = elem.get("bbox", [0, 0, image.width, image.height])
                    
                    # Crop the figure from the image
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    cropped = image.crop((x1, y1, x2, y2))
                    
                    fig = ExtractedFigure(
                        source_doc=source_doc,
                        figure_id=f"fig_{page_num}_{idx}",
                        bbox=(x1, y1, x2, y2),
                        page_num=page_num,
                        image=cropped
                    )
                    fig.compute_hash()
                    figures.append(fig)
                    
        except json.JSONDecodeError:
            # If output isn't JSON, try to parse as structured text
            warnings.warn("Could not parse Dolphin output as JSON, using fallback")
        
        return figures
    
    def extract_figures_from_pdf(self, pdf_path: str) -> list[ExtractedFigure]:
        """Extract figures from all pages of a PDF."""
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        all_figures = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            figures = self.extract_figures_from_image(
                img, 
                source_doc=pdf_path,
                page_num=page_num
            )
            all_figures.extend(figures)
        
        doc.close()
        return all_figures


def extract_figures_simple(image_path: str, source_name: str) -> list[ExtractedFigure]:
    """
    Simple figure extraction without Dolphin - treats entire image as a figure.
    Useful for testing or when figures are already cropped.
    """
    img = Image.open(image_path).convert('RGB')
    fig = ExtractedFigure(
        source_doc=source_name,
        figure_id="fig_0",
        bbox=(0, 0, img.width, img.height),
        page_num=0,
        image=img
    )
    fig.compute_hash()
    return [fig]


def compare_figure_images(img1_path: str, img2_path: str) -> FigureMatch:
    """
    Quick comparison of two figure images.
    
    Args:
        img1_path: Path to first figure image
        img2_path: Path to second figure image
        
    Returns:
        FigureMatch object with comparison results
    """
    figs1 = extract_figures_simple(img1_path, "doc1")
    figs2 = extract_figures_simple(img2_path, "doc2")
    
    comparator = FigureComparator()
    return comparator.compare(figs1[0], figs2[0])


def main():
    parser = argparse.ArgumentParser(description="Extract and compare figures from documents")
    parser.add_argument("--model_path", type=str, help="Path to Dolphin model")
    parser.add_argument("--doc1", type=str, required=True, help="Path to first document (poster)")
    parser.add_argument("--doc2", type=str, required=True, help="Path to second document (paper)")
    parser.add_argument("--output_dir", type=str, default="./comparison_results", 
                        help="Output directory for results")
    parser.add_argument("--hash_threshold", type=int, default=10,
                        help="Maximum hash distance for match (0=exact, <10=similar)")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="Minimum SSIM score for match (1.0=identical)")
    parser.add_argument("--simple_mode", action="store_true",
                        help="Use simple comparison (treat whole images as figures)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract figures
    if args.simple_mode:
        print("Using simple mode (no Dolphin)")
        figures1 = extract_figures_simple(args.doc1, "poster")
        figures2 = extract_figures_simple(args.doc2, "paper")
    else:
        if not args.model_path:
            raise ValueError("--model_path required when not using --simple_mode")
        
        extractor = DolphinFigureExtractor(args.model_path)
        
        print(f"Extracting figures from {args.doc1}...")
        figures1 = extractor.extract_figures_from_pdf(args.doc1)
        print(f"  Found {len(figures1)} figures")
        
        print(f"Extracting figures from {args.doc2}...")
        figures2 = extractor.extract_figures_from_pdf(args.doc2)
        print(f"  Found {len(figures2)} figures")
    
    # Compare figures
    comparator = FigureComparator(
        hash_threshold=args.hash_threshold,
        ssim_threshold=args.ssim_threshold
    )
    
    print("\nComparing figures...")
    matches = comparator.find_all_matches(figures1, figures2)
    
    # Report results
    print(f"\n{'='*60}")
    print(f"RESULTS: Found {len(matches)} matching figure pairs")
    print(f"{'='*60}\n")
    
    results = []
    for i, match in enumerate(matches):
        result = {
            "match_id": i + 1,
            "confidence": match.confidence,
            "hash_distance": match.hash_distance,
            "ssim_score": round(match.ssim_score, 4),
            "figure1": {
                "source": match.fig1.source_doc,
                "id": match.fig1.figure_id,
                "page": match.fig1.page_num
            },
            "figure2": {
                "source": match.fig2.source_doc,
                "id": match.fig2.figure_id,
                "page": match.fig2.page_num
            }
        }
        results.append(result)
        
        print(f"Match {i+1}: {match.confidence.upper()} confidence")
        print(f"  - Poster: {match.fig1.figure_id} (page {match.fig1.page_num})")
        print(f"  - Paper:  {match.fig2.figure_id} (page {match.fig2.page_num})")
        print(f"  - Hash distance: {match.hash_distance}")
        print(f"  - SSIM score: {match.ssim_score:.4f}")
        
        # Save matched figure pair
        pair_dir = Path(args.output_dir) / f"match_{i+1}"
        pair_dir.mkdir(exist_ok=True)
        match.fig1.image.save(pair_dir / "figure1_poster.png")
        match.fig2.image.save(pair_dir / "figure2_paper.png")
        print(f"  - Saved to: {pair_dir}")
        print()
    
    # Save results JSON
    results_file = Path(args.output_dir) / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
