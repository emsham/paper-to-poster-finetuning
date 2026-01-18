# RunPod Training Guide

## Quick Start

### 1. Launch Pod

- Go to [RunPod](https://runpod.io)
- Select **A100 40GB** or **RTX 4090**
- Use template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
- Storage: 50GB minimum

### 2. Setup Environment

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/paper-poster-finetuning.git
cd paper-poster-finetuning

# Install dependencies
pip install -r requirements_training.txt

# Install flash attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

### 3. Prepare Data

```bash
# If you have the raw training data
python prepare_training_data.py --max-tokens 8192

# Or upload prepared_data/ directly
```

### 4. Train

```bash
# Start training
python train_mistral_lora.py

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 5. Download Model

After training completes:
```bash
# Zip the model
zip -r poster-mistral-lora.zip poster-mistral-lora/

# Download via RunPod UI or use:
# runpodctl send poster-mistral-lora.zip
```

## Expected Results

| Metric | Value |
|--------|-------|
| Training time | 2-4 hours |
| Final loss | ~0.5-0.8 |
| Model size | ~200MB (LoRA weights only) |

## Inference

```bash
# Generate poster from paper (markdown)
python generate_poster.py --paper paper.md --output poster.json

# Generate poster from paper (PDF)
python generate_poster.py --paper paper.pdf --output poster.json

# With custom model path
python generate_poster.py --model ./poster-mistral-lora --paper paper.pdf
```

## Troubleshooting

**Out of Memory:**
- Reduce `MAX_SEQ_LENGTH` to 4096
- Reduce `GRADIENT_ACCUMULATION` to 4

**Slow Training:**
- Ensure flash attention is installed
- Check GPU utilization with `nvidia-smi`

**Poor Results:**
- Increase `NUM_EPOCHS` to 5
- Try lower learning rate (1e-4)
