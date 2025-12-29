"""
Robust Figure Comparison Script v2
==================================
Compares figures between poster and paper with:
- Lowered thresholds to catch more true matches
- POSSIBLE tier for borderline cases
- Auto-saves all matched pairs for easy review

Usage:
    python robust_compare_v2.py --folder1 ./poster_figs --folder2 ./paper_figs
    python robust_compare_v2.py --folder1 ./poster_figs --folder2 ./paper_figs --output ./matches
"""

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import imagehash
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass, asdict
from typing import List, Tuple
from datetime import datetime


@dataclass
class ComparisonResult:
    file1: str
    file2: str
    phash_distance: int
    dhash_distance: int
    ssim_score: float
    feature_matches: int
    feature_ratio: float
    histogram_correlation: float
    combined_score: float
    is_match: bool
    confidence: str  # "high", "medium", "low", "possible", "none"


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compare color histograms - robust to resizing."""
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def compute_feature_matches(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, float]:
    """Feature-based matching using SIFT with fallback to ORB."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    size = (512, 512)
    gray1 = cv2.resize(gray1, size)
    gray2 = cv2.resize(gray2, size)
    
    # Try SIFT first
    try:
        sift = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            total_features = min(len(kp1), len(kp2))
            ratio = len(good_matches) / total_features if total_features > 0 else 0
            return len(good_matches), ratio
    except:
        pass
    
    # Fallback to ORB
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0, 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        total_features = min(len(kp1), len(kp2))
        ratio = len(good_matches) / total_features if total_features > 0 else 0
        return len(good_matches), ratio
    except:
        return 0, 0.0


def compare_figures(img1_path: str, img2_path: str) -> ComparisonResult:
    """Compare two figures using multiple robust methods."""
    pil1 = Image.open(img1_path).convert('RGB')
    pil2 = Image.open(img2_path).convert('RGB')
    
    arr1 = np.array(pil1)
    arr2 = np.array(pil2)
    
    # Perceptual hashes
    phash1 = imagehash.phash(pil1, hash_size=16)
    phash2 = imagehash.phash(pil2, hash_size=16)
    phash_dist = phash1 - phash2
    
    dhash1 = imagehash.dhash(pil1)
    dhash2 = imagehash.dhash(pil2)
    dhash_dist = dhash1 - dhash2
    
    # SSIM
    size = (256, 256)
    ssim_arr1 = np.array(pil1.convert('L').resize(size))
    ssim_arr2 = np.array(pil2.convert('L').resize(size))
    ssim_score, _ = ssim(ssim_arr1, ssim_arr2, full=True)
    
    # Histogram similarity
    hist_corr = compute_histogram_similarity(arr1, arr2)
    
    # Feature matching
    feature_matches, feature_ratio = compute_feature_matches(arr1, arr2)
    
    # Combined score
    phash_score = max(0, 1 - phash_dist / 64)
    dhash_score = max(0, 1 - dhash_dist / 64)
    
    combined_score = (
        0.15 * phash_score +
        0.10 * dhash_score +
        0.25 * ssim_score +
        0.20 * max(0, hist_corr) +
        0.30 * feature_ratio
    )
    
    # Determine match with LOWERED thresholds and new POSSIBLE tier
    if combined_score > 0.70 or feature_ratio > 0.35:
        confidence = "high"
        is_match = True
    elif combined_score > 0.55 or feature_ratio > 0.20:
        confidence = "medium"
        is_match = True
    elif combined_score > 0.40 or feature_ratio > 0.12:
        confidence = "low"
        is_match = True
    elif combined_score > 0.30 or feature_ratio > 0.08 or (phash_dist < 25 and ssim_score > 0.4):
        confidence = "possible"
        is_match = True
    else:
        confidence = "none"
        is_match = False
    
    return ComparisonResult(
        file1=str(img1_path),
        file2=str(img2_path),
        phash_distance=phash_dist,
        dhash_distance=dhash_dist,
        ssim_score=round(ssim_score, 4),
        feature_matches=feature_matches,
        feature_ratio=round(feature_ratio, 4),
        histogram_correlation=round(hist_corr, 4),
        combined_score=round(combined_score, 4),
        is_match=is_match,
        confidence=confidence
    )


def create_side_by_side(img1_path: str, img2_path: str, output_path: str, result: ComparisonResult):
    """Create a side-by-side comparison image with labels."""
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # Resize to same height
    target_height = 400
    
    w1 = int(img1.width * target_height / img1.height)
    w2 = int(img2.width * target_height / img2.height)
    
    img1 = img1.resize((w1, target_height), Image.Resampling.LANCZOS)
    img2 = img2.resize((w2, target_height), Image.Resampling.LANCZOS)
    
    # Create combined image with padding and labels
    padding = 20
    label_height = 60
    total_width = w1 + w2 + padding * 3
    total_height = target_height + label_height + padding * 2
    
    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Paste images
    combined.paste(img1, (padding, label_height + padding))
    combined.paste(img2, (w1 + padding * 2, label_height + padding))
    
    # Add text labels using PIL
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Labels
    draw.text((padding, 10), f"POSTER: {Path(img1_path).name[:40]}", fill=(0, 0, 0), font=font)
    draw.text((w1 + padding * 2, 10), f"PAPER: {Path(img2_path).name[:40]}", fill=(0, 0, 0), font=font)
    
    # Confidence badge
    conf_colors = {
        "high": (0, 150, 0),
        "medium": (0, 100, 200),
        "low": (200, 150, 0),
        "possible": (150, 100, 150)
    }
    conf_color = conf_colors.get(result.confidence, (100, 100, 100))
    
    stats_text = f"[{result.confidence.upper()}] Score: {result.combined_score:.3f} | Features: {result.feature_ratio:.1%} | SSIM: {result.ssim_score:.3f}"
    draw.text((padding, 35), stats_text, fill=conf_color, font=small_font)
    
    combined.save(output_path)


def compare_folders(folder1: str, folder2: str, output_dir: str = None) -> List[ComparisonResult]:
    """Compare all figures between two folders."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    files1 = sorted([f for f in Path(folder1).iterdir() if f.suffix.lower() in extensions])
    files2 = sorted([f for f in Path(folder2).iterdir() if f.suffix.lower() in extensions])
    
    print(f"\n{'='*70}")
    print(f"FIGURE COMPARISON")
    print(f"{'='*70}")
    print(f"Poster figures: {len(files1)}")
    print(f"Paper figures:  {len(files2)}")
    print(f"Total comparisons: {len(files1) * len(files2)}")
    print(f"{'='*70}\n")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(f"./figure_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    output_path.mkdir(exist_ok=True)
    (output_path / "high").mkdir(exist_ok=True)
    (output_path / "medium").mkdir(exist_ok=True)
    (output_path / "low").mkdir(exist_ok=True)
    (output_path / "possible").mkdir(exist_ok=True)
    
    matches = []
    all_results = []
    
    for f1 in files1:
        best_match = None
        best_score = 0
        
        for f2 in files2:
            result = compare_figures(str(f1), str(f2))
            all_results.append(result)
            
            if result.combined_score > best_score:
                best_score = result.combined_score
                best_match = result
        
        if best_match and best_match.is_match:
            matches.append(best_match)
            
            # Print result
            emoji_map = {"high": "✅", "medium": "✅", "low": "⚠️", "possible": "❓"}
            emoji = emoji_map.get(best_match.confidence, "  ")
            
            print(f"{emoji} [{best_match.confidence.upper():8}] {Path(best_match.file1).name[:35]}")
            print(f"   └─> {Path(best_match.file2).name[:35]}")
            print(f"       Score: {best_match.combined_score:.3f} | Features: {best_match.feature_ratio:.1%} | SSIM: {best_match.ssim_score:.3f}")
            print()
            
            # Save matched pair
            conf_dir = output_path / best_match.confidence
            match_idx = len(list(conf_dir.glob("match_*"))) // 2 + 1
            
            # Copy original files
            shutil.copy(best_match.file1, conf_dir / f"match_{match_idx:02d}_poster{Path(best_match.file1).suffix}")
            shutil.copy(best_match.file2, conf_dir / f"match_{match_idx:02d}_paper{Path(best_match.file2).suffix}")
            
            # Create side-by-side comparison
            try:
                create_side_by_side(
                    best_match.file1, 
                    best_match.file2, 
                    str(conf_dir / f"match_{match_idx:02d}_comparison.png"),
                    best_match
                )
            except Exception as e:
                print(f"       (Could not create side-by-side: {e})")
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    by_confidence = {}
    for m in matches:
        by_confidence[m.confidence] = by_confidence.get(m.confidence, 0) + 1
    
    print(f"Total matches found: {len(matches)}")
    for conf in ["high", "medium", "low", "possible"]:
        count = by_confidence.get(conf, 0)
        if count > 0:
            emoji_map = {"high": "✅", "medium": "✅", "low": "⚠️", "possible": "❓"}
            print(f"  {emoji_map[conf]} {conf.upper():8}: {count}")
    
    print(f"\nResults saved to: {output_path}/")
    print(f"  ├── high/      - High confidence matches")
    print(f"  ├── medium/    - Medium confidence matches")
    print(f"  ├── low/       - Low confidence matches")
    print(f"  └── possible/  - Possible matches (review recommended)")
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "folder1": str(folder1),
        "folder2": str(folder2),
        "total_poster_figures": len(files1),
        "total_paper_figures": len(files2),
        "matches": [
            {
                "poster": Path(m.file1).name,
                "paper": Path(m.file2).name,
                "confidence": m.confidence,
                "combined_score": m.combined_score,
                "feature_ratio": m.feature_ratio,
                "ssim_score": m.ssim_score,
                "phash_distance": m.phash_distance
            }
            for m in matches
        ]
    }
    
    with open(output_path / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  └── report.json - Detailed results")
    print(f"{'='*70}\n")
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Robust figure comparison v2")
    parser.add_argument("--folder1", required=True, help="Folder with poster figures")
    parser.add_argument("--folder2", required=True, help="Folder with paper figures")
    parser.add_argument("--output", help="Output directory for matched pairs (default: auto-generated)")
    
    args = parser.parse_args()
    compare_folders(args.folder1, args.folder2, args.output)


if __name__ == "__main__":
    main()
