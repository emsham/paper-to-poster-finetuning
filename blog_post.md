# The 4096 Token Bug: How a Context Length Mismatch Silently Killed My Finetuning Run

*A cautionary tale about the subtle ways ML projects can fail — and what I learned debugging a paper-to-poster generation model.*

---

## The Vision

Academic posters are a pain to create. You spend weeks on research, write a dense 10-page paper, then somehow need to distill it into a visually appealing poster for a conference. What if an LLM could do this automatically?

That was the idea: finetune a model to take research papers as input and output structured poster layouts as JSON. The JSON would describe sections, columns, figures, and content — everything needed to render a professional academic poster.

```json
{
  "layout": {
    "poster": {"orientation": "landscape", "aspect_ratio": "16:9"},
    "header": {"height_pct": 15, "background": "linear-gradient(...)"},
    "body": {"columns": 3},
    "sections": [
      {"id": 1, "title": "INTRODUCTION", "column": 1, "content_type": "text"},
      {"id": 2, "title": "METHODS", "column": 2, "has_figures": true},
      ...
    ]
  },
  "content": "# Title\n\n## Introduction\nKey findings..."
}
```

## The Architecture

I chose a fairly standard approach for 2024/2025:

**Base Model:** Mistral-7B-Instruct-v0.2
**Finetuning Method:** LoRA (Low-Rank Adaptation)
**Training Framework:** HuggingFace TRL's SFTTrainer
**Hardware:** 2x RTX 4090 on RunPod

LoRA made sense here — I didn't need to modify the entire model, just teach it a new output format. With rank 16 and alpha 32, only ~0.1% of parameters were trainable.

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

## The Data Pipeline

This was the interesting part. I had access to a dataset of paper-poster pairs — actual research papers matched with their corresponding conference posters. The pipeline:

1. **Parse PDFs** using marker-pdf (papers) and extract poster layouts
2. **Generate JSON schemas** describing each poster's structure
3. **Create training pairs**: paper text → poster JSON
4. **Truncate intelligently** to fit context windows

The data preparation script had options for smart truncation — keeping abstract, intro, methods, results, and conclusions while trimming the middle.

```python
# From prepare_training_data.py
def truncate_paper(paper_content: str, max_chars: int, strategy: str = 'smart'):
    """Intelligently truncate papers while preserving key sections."""
    ...
```

## Training

With 6,576 training examples and the standard setup, training looked fine:

```
Train examples: 6576
Val examples: 731
Trainable parameters: 41,943,040 (0.56%)

Training...
[Epoch 1/3] Loss: 1.24
[Epoch 2/3] Loss: 0.89
[Epoch 3/3] Loss: 0.71
```

Loss was decreasing. Validation metrics looked reasonable. The model saved successfully. I pushed it to HuggingFace and moved on to evaluation.

## The Evaluation Framework

I built a proper evaluation setup:

1. Load test papers (PDFs or built-in samples)
2. Generate outputs from three models: my finetuned model, GPT-4o, and Claude
3. Use Claude as a judge to score outputs on JSON validity, layout structure, design, and content

```python
class ClaudeJudge:
    def evaluate(self, title, abstract, output_a, output_b, output_c):
        # Score each model's output on 5 criteria
        # Return structured evaluation with winner
```

## The Bug Reveals Itself

When I ran evaluation, the finetuned model's output was... wrong. Very wrong.

Instead of JSON like this:
```json
{"layout": {"poster": {"orientation": "landscape"}, ...}}
```

I got academic paper text:
```
have been disputed recently (Dinh et al., 2017) due to a
perceived lack of parameterisation invariance, with further
work considering a parameterisation invariant flatness
metric (Tsuzuku et al., 2020)...
```

The model was generating *more research paper*, not poster JSON. It was even making up fake citations and adding page numbers ("## Page 3").

## Debugging

My first hypothesis was input length — maybe the evaluation inputs were too long and the model was confused. I had set `MAX_PAPER_CHARS = 20000` in evaluation.

Quick token count:
```python
# 20000 chars -> ~5400 tokens
# Training MAX_SEQ_LENGTH was 4096
# The model had never seen inputs this long during training!
```

So I reduced `MAX_PAPER_CHARS` to fit in context. Same problem.

I tried the built-in short samples (~300 tokens). Same problem.

I tried a minimal 127-token prompt. Same problem.

The model consistently generated paper-like text, never JSON. Something was fundamentally wrong.

## The Root Cause

I finally checked the training data statistics:

```python
# Check token counts of training examples
for example in training_data[:5]:
    tokens = len(tokenizer.encode(full_example))
    print(f"{tokens} tokens")

# Output:
# 9684 tokens
# 9458 tokens
# 8864 tokens
# 9179 tokens
# 8730 tokens
```

The training examples were **9000+ tokens each**.

But `MAX_SEQ_LENGTH` in training was **4096 tokens**.

I checked how many examples fit in context:

```python
fitting = [ex for ex in examples if ex.tokens <= 4096]
print(f"Examples fitting in context: {len(fitting)}/{len(examples)}")

# Output:
# Examples fitting in context: 0/6576
```

**Zero.** Not a single training example fit in the context window.

## What Actually Happened

Here's the sequence of events:

1. **Data preparation** used `--max-tokens 8192` (the default)
2. **Training** used `MAX_SEQ_LENGTH = 4096`
3. Every training example was truncated at 4096 tokens
4. The JSON output comes *after* the paper input
5. With inputs alone being 5000-6000 tokens, the JSON was *always* cut off
6. The model only ever saw: `[paper text][paper text][paper text]...[TRUNCATED]`
7. It learned to generate more paper text, because that's all it ever saw

The model literally never saw a single complete input→output pair during training. It learned nothing about JSON generation because the JSON was always truncated away.

## The Numbers

| Setting | Value |
|---------|-------|
| Data prep `--max-tokens` | 8192 (default) |
| Training `MAX_SEQ_LENGTH` | 4096 |
| Shortest training example | 4327 tokens |
| Examples fitting in context | **0 / 6,576** |
| JSON outputs seen during training | **0** |

## Lessons Learned

### 1. Validate End-to-End Before Full Training

I should have:
- Checked token counts of prepared data
- Verified at least some examples fit in context
- Run a quick 100-example training and checked outputs

A 5-minute validation would have caught this before wasting GPU hours.

### 2. Config Coupling is Dangerous

The data preparation and training scripts had independent config:
- `prepare_training_data.py`: `--max-tokens 8192`
- `train_mistral_lora.py`: `MAX_SEQ_LENGTH = 4096`

These should have been coupled or at least cross-validated. When you have preparation and training as separate steps, it's easy for assumptions to diverge.

### 3. Loss Decreasing ≠ Learning the Task

Training loss went down nicely (1.24 → 0.71). But the model was just getting better at predicting the next token in truncated paper text — not learning the actual task.

This is a reminder that loss curves can be deceiving. Always check actual outputs during training.

### 4. Causal LM Truncation is Order-Dependent

In a causal language model, truncation happens at the *end*. If your target output comes after a long input, truncation removes the target first. This is the opposite of what you want.

For tasks where `len(input) + len(output) > context_window`, you must either:
- Truncate inputs aggressively
- Use a model with longer context
- Restructure the task (e.g., multi-turn generation)

### 5. Silent Failures are the Worst Failures

The training completed successfully. No errors, no warnings. The model saved and uploaded fine. Everything *looked* correct until I ran evaluation.

ML systems need better invariant checking. Something like:
```python
assert any(len(tokenize(ex)) <= MAX_SEQ_LENGTH for ex in data), \
    "No training examples fit in context window!"
```

## The Fix

The fix is straightforward — just requires re-running:

```bash
# Re-prepare data with correct token limit
python prepare_training_data.py --max-tokens 3500  # Leave room for output

# Verify before training
python -c "
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
with open('prepared_data/train.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        tokens = len(tokenizer.encode(ex['instruction'] + ex['input'] + ex['output']))
        assert tokens <= 4096, f'Example too long: {tokens} tokens'
print('All examples fit in context!')
"

# Then train
accelerate launch --multi_gpu train_mistral_lora.py
```

## What Worked

Despite the bug, the project validated several things:

- **The pipeline architecture is sound**: PDF parsing, data preparation, training, and evaluation all work correctly
- **The evaluation framework is useful**: Comparing against GPT-4 and Claude with automated judging
- **LoRA finetuning on Mistral works**: The mechanics are correct, just the data was wrong
- **The task is feasible**: With correct data, this approach should work

## Conclusion

This bug cost me a training run and several hours of debugging. But it's a good reminder that ML engineering is as much about data and config management as it is about model architecture.

The flashy parts of ML (attention mechanisms, LoRA adapters, multi-GPU training) get all the attention. But the boring parts (token counting, config validation, end-to-end testing) are where projects actually succeed or fail.

Next time: validate the data *before* burning GPU hours.

---

*If you've encountered similar silent failures in ML projects, I'd love to hear about them. The community benefits from sharing these debugging stories.*
