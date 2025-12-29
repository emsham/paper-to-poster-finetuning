"""
Quick Figure Comparison Script
==============================
Compares two figure images without needing Dolphin.
Perfect for testing if figures extracted from poster and paper are the same.

Requirements:
    pip install pillow imagehash scikit-image numpy opencv-python

Usage:
    python quick_compare.py figure1.png figure2.png
    python quick_compare.py --folder1 poster_figures/ --folder2 paper_figures/
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def compare_two_figures(img1_path: str, img2_path: str, verbose: bool = True) -> dict:
    """
    Compare two figure images using multiple metrics.
    
    Returns dict with:
        - is_same: bool (likely the same figure)
        - confidence: str (exact/high/medium/low/none)
        - hash_distance: int (0 = identical)
        - ssim_score: float (1.0 = identical)
        - feature_match_ratio: float
    """
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # 1. Perceptual Hash (fast, good for detecting resized/compressed versions)
    phash1 = imagehash.phash(img1)
    phash2 = imagehash.phash(img2)
    hash_distance = phash1 - phash2
    
    # 2. SSIM (Structural Similarity - pixel-level comparison)
    size = (256, 256)
    arr1 = np.array(img1.convert('L').resize(size))
    arr2 = np.array(img2.convert('L').resize(size))
    ssim_score, _ = ssim(arr1, arr2, full=True)
    
    # 3. Feature matching (good for partial matches, different crops)
    cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    cv_img1 = cv2.resize(cv_img1, (512, 512))
    cv_img2 = cv2.resize(cv_img2, (512, 512))
    
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(cv_img1, None)
    kp2, des2 = orb.detectAndCompute(cv_img2, None)
    
    feature_ratio = 0.0
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = [m for m in matches if m.distance < 50]
        feature_ratio = len(good_matches) / max(len(kp1), len(kp2))
    
    # Determine confidence
    if hash_distance == 0 and ssim_score > 0.99:
        confidence = "exact"
        is_same = True
    elif hash_distance <= 5 and ssim_score > 0.95:
        confidence = "high"
        is_same = True
    elif hash_distance <= 10 and ssim_score > 0.85:
        confidence = "medium"
        is_same = True
    elif hash_distance <= 15 or ssim_score > 0.80:
        confidence = "low"
        is_same = True
    else:
        confidence = "none"
        is_same = False
    
    result = {
        "is_same": is_same,
        "confidence": confidence,
        "hash_distance": hash_distance,
        "ssim_score": round(ssim_score, 4),
        "feature_match_ratio": round(feature_ratio, 4)
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Comparing:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        print(f"{'='*50}")
        print(f"  Hash Distance:     {hash_distance:3d}  (0 = identical, <10 = very similar)")
        print(f"  SSIM Score:        {ssim_score:.4f}  (1.0 = identical, >0.85 = similar)")
        print(f"  Feature Match:     {feature_ratio:.4f}  (>0.3 = significant overlap)")
        print(f"{'='*50}")
        
        if is_same:
            emoji = "✅" if confidence in ["exact", "high"] else "⚠️"
            print(f"  {emoji} MATCH DETECTED - Confidence: {confidence.upper()}")
        else:
            print(f"  ❌ NO MATCH - Figures appear to be different")
        print()
    
    return result


def compare_folders(folder1: str, folder2: str) -> list:
    """Compare all figures in folder1 against all figures in folder2."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
    
    files1 = [f for f in Path(folder1).iterdir() if f.suffix.lower() in extensions]
    files2 = [f for f in Path(folder2).iterdir() if f.suffix.lower() in extensions]
    
    print(f"Comparing {len(files1)} figures from folder1 with {len(files2)} figures from folder2")
    print()
    
    matches = []
    
    for f1 in files1:
        for f2 in files2:
            result = compare_two_figures(str(f1), str(f2), verbose=False)
            
            if result["is_same"]:
                result["file1"] = str(f1)
                result["file2"] = str(f2)
                matches.append(result)
                
                emoji = "✅" if result["confidence"] in ["exact", "high"] else "⚠️"
                print(f"{emoji} {f1.name} <-> {f2.name}: {result['confidence'].upper()}")
    
    print(f"\n{'='*50}")
    print(f"Found {len(matches)} matching pairs out of {len(files1) * len(files2)} comparisons")
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Compare figure images")
    parser.add_argument("image1", nargs="?", help="Path to first image")
    parser.add_argument("image2", nargs="?", help="Path to second image")
    parser.add_argument("--folder1", help="Folder with figures from document 1")
    parser.add_argument("--folder2", help="Folder with figures from document 2")
    
    args = parser.parse_args()
    
    if args.folder1 and args.folder2:
        compare_folders(args.folder1, args.folder2)
    elif args.image1 and args.image2:
        compare_two_figures(args.image1, args.image2)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python quick_compare.py figure1.png figure2.png")
        print("  python quick_compare.py --folder1 poster_figs/ --folder2 paper_figs/")


if __name__ == "__main__":
    main()
