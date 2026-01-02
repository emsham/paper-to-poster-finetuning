# RunPod Step-by-Step Instructions

## Overview

- **Goal**: Process 5,000 paper-poster pairs
- **Hardware**: 1 pod with 4x RTX 3090 (or 4x RTX 4090)
- **Estimated Time**: 4-5 days
- **Estimated Cost**: ~$185-200

---

## Step 1: Create RunPod Pod

1. Go to [runpod.io](https://runpod.io) → **Pods** → **+ Deploy**

2. Select GPU:
   - Click **Community Cloud** (cheaper)
   - Search for **4x RTX 3090** or **4x RTX 4090**
   - ~$0.88/hr for 4x 3090

3. Configure Pod:
   - **Template**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Container Disk**: 50 GB
   - **Volume Disk**: 100 GB (for your data + outputs)

4. Click **Deploy**

5. Wait for pod to start, then click **Connect** → **Jupyter Lab**

---

## Step 2: Upload Your Data

In Jupyter, open a **Terminal** and run:

```bash
# Create data directory
mkdir -p /workspace/data
cd /workspace/data
```

**Option A: Upload via Jupyter UI**
- Use the Jupyter file browser to upload:
  - `train.csv`
  - `images/` folder (poster PNGs)
  - `papers/` folder (paper PDFs)

**Option B: Download from cloud storage**
```bash
# From Google Drive (using gdown)
pip install gdown
gdown --folder YOUR_GOOGLE_DRIVE_FOLDER_ID

# Or from Dropbox/S3/etc
wget "YOUR_DOWNLOAD_URL" -O data.zip
unzip data.zip
```

---

## Step 3: Setup Pipeline

In Jupyter Terminal:

```bash
# Go to workspace
cd /workspace

# Clone the pipeline repository
git clone https://github.com/YOUR_USERNAME/paper-to-poster-finetuning.git
cd paper-to-poster-finetuning

# Install dependencies
pip install torch transformers accelerate pandas numpy pillow
pip install opencv-python imagehash scikit-image pymupdf
pip install qwen-vl-utils tqdm

# Download Dolphin model (~5-10 minutes)
huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model

# Link your data (adjust paths as needed)
ln -s /workspace/data/train.csv ./train.csv
ln -s /workspace/data/images ./images
ln -s /workspace/data/papers ./papers
```

---

## Step 4: Verify Setup

```bash
# Check GPUs are available
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should print: GPUs: 4

# Check data is linked
ls -la train.csv images/ papers/

# Quick test (optional - processes 1 item)
python run_pipeline.py --parallel staged \
    --limit 1 \
    --dolphin-model ./hf_model \
    --output-dir ./test_output
```

---

## Step 5: Create Run Script

Create a file called `run_all_gpus.sh`:

```bash
cat > run_all_gpus.sh << 'EOF'
#!/bin/bash

# Configuration
TOTAL_ITEMS=5000
NUM_GPUS=4
ITEMS_PER_GPU=$((TOTAL_ITEMS / NUM_GPUS))

echo "=============================================="
echo "Starting pipeline on $NUM_GPUS GPUs"
echo "$ITEMS_PER_GPU items per GPU"
echo "=============================================="

# Launch a process on each GPU
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    START=$((GPU_ID * ITEMS_PER_GPU))

    echo "Starting GPU $GPU_ID: items $START to $((START + ITEMS_PER_GPU))"

    CUDA_VISIBLE_DEVICES=$GPU_ID python run_pipeline.py \
        --parallel staged \
        --dataset train \
        --start $START \
        --limit $ITEMS_PER_GPU \
        --output-dir ./output_gpu$GPU_ID \
        --dolphin-model ./hf_model \
        --export-tracking tracking_gpu$GPU_ID.csv \
        --resume \
        > log_gpu$GPU_ID.txt 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "=============================================="
echo "All GPUs started!"
echo "Monitor with: tail -f log_gpu*.txt"
echo "Check progress with: ./check_progress.sh"
echo "=============================================="
EOF

chmod +x run_all_gpus.sh
```

---

## Step 6: Create Progress Checker

```bash
cat > check_progress.sh << 'EOF'
#!/bin/bash

echo "=============================================="
echo "PROGRESS REPORT - $(date)"
echo "=============================================="

total_done=0
total_items=0

for i in 0 1 2 3; do
    if [ -f "output_gpu$i/processing_status.csv" ]; then
        done=$(python -c "
import pandas as pd
df = pd.read_csv('output_gpu$i/processing_status.csv')
print(int(df['training_data_created'].sum()))
" 2>/dev/null)
        total=$(wc -l < "output_gpu$i/processing_status.csv" 2>/dev/null)
        total=$((total - 1))  # subtract header

        if [ ! -z "$done" ]; then
            pct=$((100 * done / total))
            echo "GPU $i: $done / $total ($pct%)"
            total_done=$((total_done + done))
            total_items=$((total_items + total))
        fi
    else
        echo "GPU $i: Not started yet"
    fi
done

echo "----------------------------------------------"
if [ $total_items -gt 0 ]; then
    overall_pct=$((100 * total_done / total_items))
    echo "TOTAL: $total_done / $total_items ($overall_pct%)"
fi
echo "=============================================="
EOF

chmod +x check_progress.sh
```

---

## Step 7: Start Processing

```bash
# Start all 4 GPUs
./run_all_gpus.sh
```

You should see:
```
==============================================
Starting pipeline on 4 GPUs
1250 items per GPU
==============================================
Starting GPU 0: items 0 to 1250
  PID: 12345
Starting GPU 1: items 1250 to 2500
  PID: 12346
...
```

---

## Step 8: Monitor Progress

**Option A: Check progress script**
```bash
./check_progress.sh
```

**Option B: Watch logs**
```bash
# All logs
tail -f log_gpu*.txt

# Specific GPU
tail -f log_gpu0.txt
```

**Option C: GPU utilization**
```bash
watch -n 5 nvidia-smi
```

---

## Step 9: If Pod Restarts / Disconnects

The `--resume` flag saves progress. Just run again:

```bash
cd /workspace/paper-to-poster-finetuning
./run_all_gpus.sh
```

It will skip already-completed items.

---

## Step 10: Merge Results (After Completion)

```bash
cat > merge_results.sh << 'EOF'
#!/bin/bash

echo "Merging results from all GPUs..."

python << 'PYTHON'
import pandas as pd
import shutil
from pathlib import Path
import json

# Merge tracking CSVs
print("Merging tracking CSVs...")
dfs = []
for i in range(4):
    csv_path = f"output_gpu{i}/processing_status.csv"
    if Path(csv_path).exists():
        dfs.append(pd.read_csv(csv_path))
        print(f"  Loaded {csv_path}")

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv("final_tracking.csv", index=False)
    print(f"Saved final_tracking.csv ({len(merged)} items)")

# Merge training data
print("\nMerging training data...")
output_dir = Path("final_training_data")
output_dir.mkdir(exist_ok=True)

count = 0
for i in range(4):
    src_dir = Path(f"output_gpu{i}/training_data")
    if src_dir.exists():
        for f in src_dir.glob("*.json"):
            shutil.copy(f, output_dir / f.name)
            count += 1

print(f"Merged {count} training examples to final_training_data/")

# Summary
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
df = pd.read_csv("final_tracking.csv")
print(f"Total items: {len(df)}")
print(f"Successfully processed: {df['training_data_created'].sum()}")
print(f"Failed: {df['error_message'].notna().sum()}")
print(f"Avg poster figures: {df['poster_figure_count'].mean():.1f}")
print(f"Avg paper figures: {df['paper_figure_count'].mean():.1f}")
print(f"Avg matched figures: {df['matched_figure_count'].mean():.1f}")
PYTHON

EOF

chmod +x merge_results.sh
./merge_results.sh
```

---

## Step 11: Export for Fine-tuning

```bash
# Create JSONL for fine-tuning
python run_pipeline.py --export \
    --export-output final_training_data.jsonl \
    --export-format jsonl
```

---

## Step 12: Download Results

**Option A: Jupyter UI**
- Right-click on `final_training_data/` → Download as zip
- Download `final_tracking.csv`
- Download `final_training_data.jsonl`

**Option B: Using runpodctl**
```bash
# On your local machine
runpodctl receive final_training_data.jsonl
```

**Option C: Upload to cloud**
```bash
# To Google Drive
pip install gdown
# Use gdown or rclone to upload

# To S3
aws s3 cp final_training_data/ s3://your-bucket/ --recursive
```

---

## Troubleshooting

### GPU Out of Memory
```bash
# Check which GPU is having issues
nvidia-smi

# The staged pipeline should handle this, but if issues persist,
# try processing fewer items per GPU
```

### Process Died
```bash
# Check if processes are running
ps aux | grep python

# Restart with resume
./run_all_gpus.sh
```

### Check for Errors
```bash
# See failed items
python -c "
import pandas as pd
df = pd.read_csv('output_gpu0/processing_status.csv')
failed = df[df['error_message'].notna()]
print(f'Failed items: {len(failed)}')
print(failed[['paper_id', 'error_message']].head(10))
"
```

### Disk Space
```bash
# Check disk usage
df -h

# Training data size
du -sh output_gpu*/training_data/
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `./run_all_gpus.sh` | Start processing on all GPUs |
| `./check_progress.sh` | Check progress |
| `tail -f log_gpu0.txt` | Watch GPU 0 logs |
| `nvidia-smi` | Check GPU usage |
| `./merge_results.sh` | Combine results after done |

---

## Estimated Timeline

| Phase | Time |
|-------|------|
| Setup & data upload | 30 min - 2 hrs |
| Model download | 10 min |
| Processing 5k items | 4-5 days |
| Merge & download | 30 min |

**Total: ~5 days**

---

## Cost Summary

- 4x RTX 3090 @ $0.88/hr
- ~120 hours runtime
- **Total: ~$105-120**

(Less than initial estimate because 4 GPUs process in parallel!)
