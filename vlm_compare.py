"""
VLM-Based Figure Comparison using Qwen3-VL
==========================================
Uses a two-pass approach:
1. Fast feature matching to filter obvious non-matches
2. Qwen3-VL for semantic comparison of candidates

Usage:
    python vlm_compare.py \
        --folder1 ./poster_output/markdown/figures/ \
        --folder2 ./paper_output/markdown/figures/ \
        --output ./vlm_matches
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
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
import torch


@dataclass
class ComparisonResult:
    file1: str
    file2: str
    feature_score: float
    vlm_verdict: Optional[str]  # "same", "similar", "different"
    vlm_confidence: Optional[float]
    vlm_reasoning: Optional[str]
    is_match: bool
    confidence: str


def compute_feature_score(img1_path: str, img2_path: str) -> Tuple[float, float]:
    """
    Quick feature-based score for filtering.
    Returns (feature_ratio, combined_score) - filter primarily on feature_ratio.
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
    
    # Feature matching - this is the key metric for re-rendered figures
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
            
            good = [m for m, n in matches if len([m,n]) == 2 and m.distance < 0.7 * n.distance]
            feature_ratio = len(good) / min(len(kp1), len(kp2))
    except:
        pass
    
    # Combined score (for reporting)
    combined = 0.3 * phash_score + 0.3 * ssim_score + 0.4 * feature_ratio
    
    return feature_ratio, combined


class QwenVLComparator:
    """Qwen3-VL based figure comparator."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", use_flash_attn: bool = False):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading {model_name}...")
        
        if use_flash_attn:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded!")
    
    def compare_figures(self, img1_path: str, img2_path: str) -> dict:
        """
        Compare two figures using Qwen3-VL.
        
        Returns:
            dict with 'verdict', 'confidence', 'reasoning'
        """
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
            # Try to extract JSON from response
            import re
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


def compare_folders_with_vlm(
    folder1: str, 
    folder2: str, 
    output_dir: str = None,
    feature_threshold: float = 0.15,
    use_flash_attn: bool = False
) -> List[ComparisonResult]:
    """
    Compare figures using two-pass approach:
    1. Feature matching to filter candidates
    2. VLM for semantic comparison
    """
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    files1 = sorted([f for f in Path(folder1).iterdir() if f.suffix.lower() in extensions])
    files2 = sorted([f for f in Path(folder2).iterdir() if f.suffix.lower() in extensions])
    
    print(f"\n{'='*70}")
    print(f"VLM FIGURE COMPARISON (Qwen3-VL)")
    print(f"{'='*70}")
    print(f"Poster figures: {len(files1)}")
    print(f"Paper figures:  {len(files2)}")
    print(f"Total possible pairs: {len(files1) * len(files2)}")
    print(f"Feature ratio threshold: {feature_threshold:.0%}")
    print(f"{'='*70}\n")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(f"./vlm_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    output_path.mkdir(exist_ok=True)
    (output_path / "same").mkdir(exist_ok=True)
    (output_path / "similar").mkdir(exist_ok=True)
    (output_path / "uncertain").mkdir(exist_ok=True)
    
    # PASS 1: Feature-based filtering
    print("PASS 1: Feature-based filtering...")
    print("-" * 50)
    
    candidates = []
    for f1 in files1:
        for f2 in files2:
            feature_ratio, combined = compute_feature_score(str(f1), str(f2))
            # Filter on feature_ratio (works better for re-rendered figures)
            if feature_ratio >= feature_threshold:
                candidates.append((f1, f2, feature_ratio, combined))
    
    # Sort by feature_ratio descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Found {len(candidates)} candidate pairs with feature ratio >= {feature_threshold:.0%}")
    print()
    
    if not candidates:
        print("No candidates found. Try lowering --threshold")
        return []
    
    # PASS 2: VLM comparison
    print("PASS 2: VLM semantic comparison...")
    print("-" * 50)
    
    vlm = QwenVLComparator(use_flash_attn=use_flash_attn)
    
    matches = []
    
    for i, (f1, f2, feature_ratio, combined_score) in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] Comparing:")
        print(f"  Poster: {f1.name[:50]}")
        print(f"  Paper:  {f2.name[:50]}")
        print(f"  Feature ratio: {feature_ratio:.1%} (combined: {combined_score:.3f})")
        
        # Run VLM comparison
        vlm_result = vlm.compare_figures(str(f1), str(f2))
        
        verdict = vlm_result["verdict"]
        confidence = vlm_result["confidence"]
        reasoning = vlm_result["reasoning"]
        
        print(f"  VLM verdict: {verdict.upper()} (confidence: {confidence:.2f})")
        print(f"  Reasoning: {reasoning[:100]}...")
        
        # Determine final match status
        if verdict == "same" and confidence >= 0.6:
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
        
        result = ComparisonResult(
            file1=str(f1),
            file2=str(f2),
            feature_score=feature_ratio,
            vlm_verdict=verdict,
            vlm_confidence=confidence,
            vlm_reasoning=reasoning,
            is_match=is_match,
            confidence=final_confidence
        )
        
        if is_match:
            matches.append(result)
            
            # Save to appropriate folder
            if verdict == "same":
                save_dir = output_path / "same"
            elif verdict == "similar":
                save_dir = output_path / "similar"
            else:
                save_dir = output_path / "uncertain"
            
            match_idx = len(list(save_dir.glob("match_*_poster*"))) + 1
            shutil.copy(f1, save_dir / f"match_{match_idx:02d}_poster{f1.suffix}")
            shutil.copy(f2, save_dir / f"match_{match_idx:02d}_paper{f2.suffix}")
            
            # Save reasoning
            with open(save_dir / f"match_{match_idx:02d}_reasoning.txt", 'w') as f:
                f.write(f"Poster: {f1.name}\n")
                f.write(f"Paper: {f2.name}\n")
                f.write(f"Feature ratio: {feature_ratio:.3f}\n")
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
    
    print(f"  ✅ Same figures: {same_count}")
    print(f"  🔄 Similar figures: {similar_count}")
    
    print(f"\nResults saved to: {output_path}/")
    print(f"  ├── same/      - Confirmed same figures")
    print(f"  ├── similar/   - Similar figures")
    print(f"  └── uncertain/ - Uncertain matches")
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "folder1": str(folder1),
        "folder2": str(folder2),
        "feature_threshold": feature_threshold,
        "candidates_found": len(candidates),
        "matches": [
            {
                "poster": Path(m.file1).name,
                "paper": Path(m.file2).name,
                "feature_score": m.feature_score,
                "vlm_verdict": m.vlm_verdict,
                "vlm_confidence": m.vlm_confidence,
                "vlm_reasoning": m.vlm_reasoning
            }
            for m in matches
        ]
    }
    
    with open(output_path / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  └── report.json")
    print(f"{'='*70}\n")
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="VLM-based figure comparison using Qwen3-VL")
    parser.add_argument("--folder1", required=True, help="Folder with poster figures")
    parser.add_argument("--folder2", required=True, help="Folder with paper figures")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Feature ratio threshold for VLM candidates (default: 0.10 = 10%% matching features)")
    parser.add_argument("--flash-attn", action="store_true",
                        help="Use flash attention 2 (faster, less memory)")
    
    args = parser.parse_args()
    
    compare_folders_with_vlm(
        args.folder1, 
        args.folder2, 
        args.output,
        feature_threshold=args.threshold,
        use_flash_attn=args.flash_attn
    )


if __name__ == "__main__":
    main()