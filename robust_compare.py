"""
Robust Figure Comparison Script
===============================
Uses multiple methods to catch figures that are the same but have been:
- Resized/rescaled
- Re-rendered (vector to raster)
- Cropped differently
- Color adjusted
- Compressed differently

Usage:
    python robust_compare.py --folder1 ./poster_figs --folder2 ./paper_figs
    python robust_compare.py --img1 figure1.png --img2 figure2.png
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import imagehash
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ComparisonResult:
    file1: str
    file2: str
    phash_distance: int
    dhash_distance: int
    ssim_score: float
    orb_matches: int
    orb_ratio: float
    histogram_correlation: float
    combined_score: float
    is_match: bool
    confidence: str


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compare color histograms - robust to resizing."""
    # Convert to HSV for better color comparison
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    
    # Compute histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare using correlation
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def compute_orb_matches(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, float]:
    """Feature-based matching using ORB - robust to scale/rotation."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Resize to consistent size for fair comparison
    size = (512, 512)
    gray1 = cv2.resize(gray1, size)
    gray2 = cv2.resize(gray2, size)
    
    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0, 0.0
    
    # Match features using BFMatcher with ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except:
        return 0, 0.0
    
    # Apply Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    # Calculate match ratio
    total_features = min(len(kp1), len(kp2))
    ratio = len(good_matches) / total_features if total_features > 0 else 0
    
    return len(good_matches), ratio


def compute_sift_matches(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, float]:
    """Feature-based matching using SIFT - more robust than ORB."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Resize
    size = (512, 512)
    gray1 = cv2.resize(gray1, size)
    gray2 = cv2.resize(gray2, size)
    
    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0, 0.0
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0, 0.0
    
    # Ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    total_features = min(len(kp1), len(kp2))
    ratio = len(good_matches) / total_features if total_features > 0 else 0
    
    return len(good_matches), ratio


def compare_figures(img1_path: str, img2_path: str, verbose: bool = False) -> ComparisonResult:
    """
    Compare two figures using multiple robust methods.
    """
    # Load images
    pil1 = Image.open(img1_path).convert('RGB')
    pil2 = Image.open(img2_path).convert('RGB')
    
    arr1 = np.array(pil1)
    arr2 = np.array(pil2)
    
    # 1. Perceptual hashes (multiple types for robustness)
    phash1 = imagehash.phash(pil1, hash_size=16)  # Larger hash for more detail
    phash2 = imagehash.phash(pil2, hash_size=16)
    phash_dist = phash1 - phash2
    
    dhash1 = imagehash.dhash(pil1)
    dhash2 = imagehash.dhash(pil2)
    dhash_dist = dhash1 - dhash2
    
    # 2. SSIM on normalized size
    size = (256, 256)
    ssim_arr1 = np.array(pil1.convert('L').resize(size))
    ssim_arr2 = np.array(pil2.convert('L').resize(size))
    ssim_score, _ = ssim(ssim_arr1, ssim_arr2, full=True)
    
    # 3. Histogram similarity
    hist_corr = compute_histogram_similarity(arr1, arr2)
    
    # 4. Feature matching (ORB)
    orb_matches, orb_ratio = compute_orb_matches(arr1, arr2)
    
    # 5. Try SIFT if available (usually more robust)
    try:
        sift_matches, sift_ratio = compute_sift_matches(arr1, arr2)
        feature_ratio = max(orb_ratio, sift_ratio)
        feature_matches = max(orb_matches, sift_matches)
    except:
        feature_ratio = orb_ratio
        feature_matches = orb_matches
    
    # Combined score (weighted average of normalized metrics)
    # Normalize each metric to 0-1 range where 1 = identical
    phash_score = max(0, 1 - phash_dist / 64)  # 16x16 hash has max dist ~64
    dhash_score = max(0, 1 - dhash_dist / 64)
    
    combined_score = (
        0.15 * phash_score +
        0.10 * dhash_score +
        0.25 * ssim_score +
        0.20 * max(0, hist_corr) +
        0.30 * feature_ratio  # Feature matching weighted highest
    )
    
    # Determine match
    if combined_score > 0.70 or feature_ratio > 0.25 or (phash_dist < 15 and ssim_score > 0.5):
        if combined_score > 0.85 or feature_ratio > 0.40:
            confidence = "high"
        elif combined_score > 0.70 or feature_ratio > 0.25:
            confidence = "medium"
        else:
            confidence = "low"
        is_match = True
    elif combined_score > 0.50 or feature_ratio > 0.15:
        confidence = "low"
        is_match = True
    else:
        confidence = "none"
        is_match = False
    
    result = ComparisonResult(
        file1=str(img1_path),
        file2=str(img2_path),
        phash_distance=phash_dist,
        dhash_distance=dhash_dist,
        ssim_score=round(ssim_score, 4),
        orb_matches=feature_matches,
        orb_ratio=round(feature_ratio, 4),
        histogram_correlation=round(hist_corr, 4),
        combined_score=round(combined_score, 4),
        is_match=is_match,
        confidence=confidence
    )
    
    if verbose:
        print(f"\nComparing:")
        print(f"  {Path(img1_path).name}")
        print(f"  {Path(img2_path).name}")
        print(f"  ├─ pHash distance:    {phash_dist}")
        print(f"  ├─ dHash distance:    {dhash_dist}")
        print(f"  ├─ SSIM score:        {ssim_score:.4f}")
        print(f"  ├─ Histogram corr:    {hist_corr:.4f}")
        print(f"  ├─ Feature matches:   {feature_matches} ({feature_ratio:.2%})")
        print(f"  ├─ Combined score:    {combined_score:.4f}")
        print(f"  └─ Result:            {'✅ MATCH' if is_match else '❌ NO MATCH'} ({confidence})")
    
    return result


def compare_folders(folder1: str, folder2: str) -> List[ComparisonResult]:
    """Compare all figures between two folders."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    files1 = [f for f in Path(folder1).iterdir() if f.suffix.lower() in extensions]
    files2 = [f for f in Path(folder2).iterdir() if f.suffix.lower() in extensions]
    
    print(f"Comparing {len(files1)} figures from folder1 with {len(files2)} figures from folder2")
    print("=" * 60)
    
    all_results = []
    matches = []
    
    for f1 in files1:
        best_match = None
        best_score = 0
        
        for f2 in files2:
            result = compare_figures(str(f1), str(f2), verbose=False)
            all_results.append(result)
            
            if result.combined_score > best_score:
                best_score = result.combined_score
                best_match = result
        
        if best_match and best_match.is_match:
            matches.append(best_match)
            emoji = "✅" if best_match.confidence in ["high", "medium"] else "⚠️"
            print(f"{emoji} {Path(best_match.file1).name[:45]}")
            print(f"   └─> {Path(best_match.file2).name[:45]}")
            print(f"       Score: {best_match.combined_score:.3f} | "
                  f"Features: {best_match.orb_ratio:.1%} | "
                  f"SSIM: {best_match.ssim_score:.3f} | "
                  f"Conf: {best_match.confidence.upper()}")
            print()
    
    print("=" * 60)
    print(f"Found {len(matches)} matching figure pairs")
    
    if matches:
        print("\nSummary of matches:")
        for m in sorted(matches, key=lambda x: x.combined_score, reverse=True):
            print(f"  [{m.confidence.upper():6}] {Path(m.file1).name[:30]} <-> {Path(m.file2).name[:30]}")
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Robust figure comparison")
    parser.add_argument("--folder1", help="First folder of figures")
    parser.add_argument("--folder2", help="Second folder of figures")
    parser.add_argument("--img1", help="Single image 1")
    parser.add_argument("--img2", help="Single image 2")
    
    args = parser.parse_args()
    
    if args.img1 and args.img2:
        compare_figures(args.img1, args.img2, verbose=True)
    elif args.folder1 and args.folder2:
        compare_folders(args.folder1, args.folder2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
