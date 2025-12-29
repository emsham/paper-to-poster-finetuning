"""
Dolphin + Figure Comparison Integration
=======================================
This script shows how to:
1. Run Dolphin's layout analysis on documents
2. Extract figure regions from the layout output
3. Compare figures across documents to find matches

Based on Dolphin's actual output format from demo_layout.py
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ============================================================================
# STEP 1: Run Dolphin to get layout (you do this first)
# ============================================================================
"""
Run Dolphin's layout analysis on both documents:

    python demo_layout.py --model_path ./hf_model --save_dir ./poster_layout \
        --input_path ./poster.pdf
    
    python demo_layout.py --model_path ./hf_model --save_dir ./paper_layout \
        --input_path ./paper.pdf

This will create JSON files with layout information including figure bounding boxes.
"""


# ============================================================================
# STEP 2: Parse Dolphin's layout output to extract figures
# ============================================================================

def extract_figures_from_dolphin_output(layout_json_path: str, 
                                        source_image_path: str,
                                        output_dir: str = None) -> list:
    """
    Extract figure images from a document using Dolphin's layout analysis output.
    
    Args:
        layout_json_path: Path to Dolphin's layout JSON output
        source_image_path: Path to the original document image/page
        output_dir: Optional directory to save cropped figures
        
    Returns:
        List of dicts with figure info and PIL images
    """
    # Load the layout JSON
    with open(layout_json_path, 'r') as f:
        layout = json.load(f)
    
    # Load the source image
    source_img = Image.open(source_image_path).convert('RGB')
    img_width, img_height = source_img.size
    
    figures = []
    
    # Dolphin outputs elements with types like "figure", "image", "chart"
    # The exact format may vary, but typically includes bbox coordinates
    elements = layout.get("elements", layout.get("content", []))
    
    if isinstance(elements, str):
        # Sometimes the output is a string that needs parsing
        try:
            elements = json.loads(elements)
        except:
            elements = []
    
    for idx, elem in enumerate(elements):
        elem_type = elem.get("type", elem.get("category", "")).lower()
        
        # Look for figure-like elements
        if elem_type in ["figure", "image", "chart", "diagram", "graph", "picture"]:
            # Get bounding box - format may be [x1, y1, x2, y2] or {"x1":..., "y1":..., etc}
            bbox = elem.get("bbox", elem.get("box", elem.get("position", None)))
            
            if bbox is None:
                continue
            
            # Normalize bbox format
            if isinstance(bbox, dict):
                x1, y1 = bbox.get("x1", bbox.get("left", 0)), bbox.get("y1", bbox.get("top", 0))
                x2, y2 = bbox.get("x2", bbox.get("right", 0)), bbox.get("y2", bbox.get("bottom", 0))
            elif isinstance(bbox, (list, tuple)):
                x1, y1, x2, y2 = bbox[:4]
            else:
                continue
            
            # Handle normalized coordinates (0-1 range)
            if max(x1, y1, x2, y2) <= 1:
                x1 = int(x1 * img_width)
                y1 = int(y1 * img_height)
                x2 = int(x2 * img_width)
                y2 = int(y2 * img_height)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure valid bbox
            x1, x2 = max(0, min(x1, x2)), min(img_width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(img_height, max(y1, y2))
            
            if x2 - x1 < 10 or y2 - y1 < 10:  # Skip tiny regions
                continue
            
            # Crop the figure
            cropped = source_img.crop((x1, y1, x2, y2))
            
            # Compute perceptual hash
            phash = str(imagehash.phash(cropped))
            
            figure_info = {
                "id": f"fig_{idx}",
                "type": elem_type,
                "bbox": (x1, y1, x2, y2),
                "image": cropped,
                "phash": phash,
                "source": source_image_path,
                "caption": elem.get("caption", elem.get("text", ""))
            }
            figures.append(figure_info)
            
            # Optionally save
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cropped.save(f"{output_dir}/figure_{idx}.png")
    
    return figures


def extract_figures_from_page_output(page_json_path: str,
                                     source_image_path: str) -> list:
    """
    Alternative extraction from Dolphin's page-level parsing output.
    This handles the JSON/Markdown structured output format.
    """
    with open(page_json_path, 'r') as f:
        content = f.read()
    
    # Try to parse as JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Might be markdown or other format
        return []
    
    source_img = Image.open(source_image_path).convert('RGB')
    figures = []
    
    # Walk through the parsed structure looking for figures
    def find_figures(obj, path=""):
        if isinstance(obj, dict):
            if obj.get("type") in ["figure", "image"]:
                if "bbox" in obj:
                    bbox = obj["bbox"]
                    cropped = source_img.crop(tuple(bbox))
                    figures.append({
                        "id": f"fig_{len(figures)}",
                        "bbox": bbox,
                        "image": cropped,
                        "phash": str(imagehash.phash(cropped)),
                        "source": source_image_path
                    })
            for k, v in obj.items():
                find_figures(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_figures(item, f"{path}[{i}]")
    
    find_figures(data)
    return figures


# ============================================================================
# STEP 3: Compare figures across documents
# ============================================================================

def compare_figures(fig1: dict, fig2: dict) -> dict:
    """
    Compare two figures using multiple metrics.
    
    Returns similarity metrics and whether they're likely the same.
    """
    # Perceptual hash distance
    hash1 = imagehash.hex_to_hash(fig1["phash"])
    hash2 = imagehash.hex_to_hash(fig2["phash"])
    hash_distance = hash1 - hash2
    
    # SSIM score
    size = (256, 256)
    arr1 = np.array(fig1["image"].convert('L').resize(size))
    arr2 = np.array(fig2["image"].convert('L').resize(size))
    ssim_score, _ = ssim(arr1, arr2, full=True)
    
    # Determine if same
    if hash_distance == 0 and ssim_score > 0.99:
        confidence = "exact"
    elif hash_distance <= 5 and ssim_score > 0.95:
        confidence = "high"
    elif hash_distance <= 10 and ssim_score > 0.85:
        confidence = "medium"
    elif hash_distance <= 15 or ssim_score > 0.80:
        confidence = "low"
    else:
        confidence = "none"
    
    return {
        "is_same": confidence != "none",
        "confidence": confidence,
        "hash_distance": hash_distance,
        "ssim_score": round(ssim_score, 4),
        "fig1_id": fig1["id"],
        "fig2_id": fig2["id"]
    }


def find_matching_figures(poster_figures: list, paper_figures: list) -> list:
    """
    Find all matching figures between poster and paper.
    """
    matches = []
    
    for pf in poster_figures:
        best_match = None
        best_score = -1
        
        for rf in paper_figures:
            result = compare_figures(pf, rf)
            
            if result["is_same"] and result["ssim_score"] > best_score:
                best_score = result["ssim_score"]
                best_match = result
                best_match["poster_figure"] = pf
                best_match["paper_figure"] = rf
        
        if best_match:
            matches.append(best_match)
    
    return matches


# ============================================================================
# STEP 4: Main workflow
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare figures across poster and paper using Dolphin layout output"
    )
    parser.add_argument("--poster_layout", required=True, 
                        help="Path to Dolphin layout JSON for poster")
    parser.add_argument("--poster_image", required=True,
                        help="Path to poster image/PDF page")
    parser.add_argument("--paper_layout", required=True,
                        help="Path to Dolphin layout JSON for paper")
    parser.add_argument("--paper_image", required=True,
                        help="Path to paper image/PDF page")
    parser.add_argument("--output_dir", default="./figure_matches",
                        help="Output directory for results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract figures from both documents
    print("Extracting figures from poster...")
    poster_figs = extract_figures_from_dolphin_output(
        args.poster_layout, 
        args.poster_image,
        f"{args.output_dir}/poster_figures"
    )
    print(f"  Found {len(poster_figs)} figures")
    
    print("Extracting figures from paper...")
    paper_figs = extract_figures_from_dolphin_output(
        args.paper_layout,
        args.paper_image,
        f"{args.output_dir}/paper_figures"
    )
    print(f"  Found {len(paper_figs)} figures")
    
    # Find matches
    print("\nComparing figures...")
    matches = find_matching_figures(poster_figs, paper_figs)
    
    # Report results
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(matches)} matching figure pairs found")
    print(f"{'='*60}\n")
    
    results = []
    for i, match in enumerate(matches):
        print(f"Match {i+1}: {match['confidence'].upper()} confidence")
        print(f"  Poster: {match['fig1_id']}")
        print(f"  Paper:  {match['fig2_id']}")
        print(f"  Hash distance: {match['hash_distance']}, SSIM: {match['ssim_score']}")
        
        # Save matched pair
        match_dir = Path(args.output_dir) / f"match_{i+1}"
        match_dir.mkdir(exist_ok=True)
        match["poster_figure"]["image"].save(match_dir / "poster.png")
        match["paper_figure"]["image"].save(match_dir / "paper.png")
        
        results.append({
            "match_id": i + 1,
            "confidence": match["confidence"],
            "poster_figure": match["fig1_id"],
            "paper_figure": match["fig2_id"],
            "hash_distance": match["hash_distance"],
            "ssim_score": match["ssim_score"]
        })
    
    # Save summary
    with open(f"{args.output_dir}/matches.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
"""
Complete workflow:

1. First, run Dolphin layout analysis on both documents:

   python demo_layout.py --model_path ./hf_model --save_dir ./poster_output \
       --input_path ./poster.pdf
   
   python demo_layout.py --model_path ./hf_model --save_dir ./paper_output \
       --input_path ./paper.pdf

2. Then run this comparison script:

   python dolphin_figure_integration.py \
       --poster_layout ./poster_output/page_1_layout.json \
       --poster_image ./poster_output/page_1.png \
       --paper_layout ./paper_output/page_1_layout.json \
       --paper_image ./paper_output/page_1.png \
       --output_dir ./figure_comparison_results

3. Check the output:
   - figure_comparison_results/matches.json - Summary of all matches
   - figure_comparison_results/match_N/ - Side-by-side images of each match
   - figure_comparison_results/poster_figures/ - All extracted poster figures
   - figure_comparison_results/paper_figures/ - All extracted paper figures
"""
