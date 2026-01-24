#!/usr/bin/env python3
"""
Finetune Mistral-7B with LoRA for Poster Generation (Multi-GPU)
================================================================

RunPod Usage:
    1. Launch a pod with 2x RTX 4090 or similar
    2. Upload prepared_data/train.jsonl and prepared_data/val.jsonl
    3. Run: pip install -r requirements_training.txt
    4. Run: accelerate launch --multi_gpu --num_processes=2 train_mistral_lora.py

For single GPU: python train_mistral_lora.py
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./poster-mistral-lora"
MAX_SEQ_LENGTH = 4096

# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch size = 8 per GPU, 16 total with 2 GPUs
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Paths (adjust if needed)
TRAIN_FILE = "prepared_data/train.jsonl"
VAL_FILE = "prepared_data/val.jsonl"


def format_prompt(example):
    """Format example for Mistral instruction format."""
    instruction = example['instruction']
    input_text = example['input']
    output = example['output']

    # Mistral format
    text = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output}</s>"
    return {"text": text}


def main():
    print("=" * 60)
    print("POSTER GENERATION FINETUNING (Multi-GPU)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: No GPU detected! Training will be very slow.")

    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("\nLoading dataset...")

    dataset = load_dataset("json", data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE
    })

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Val examples: {len(dataset['validation'])}")

    # Format for Mistral
    dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)

    # ========================================================================
    # Load Model (bf16, no quantization for multi-GPU compatibility)
    # ========================================================================
    print("\nLoading model in bf16...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ========================================================================
    # Setup LoRA
    # ========================================================================
    print("\nSetting up LoRA...")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # ========================================================================
    # Training Configuration
    # ========================================================================
    print("\nConfiguring training...")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # ========================================================================
    # Train
    # ========================================================================
    print("\nStarting training...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTo use the model:")
    print("""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "./poster-mistral-lora")
tokenizer = AutoTokenizer.from_pretrained("./poster-mistral-lora")
""")


if __name__ == "__main__":
    main()
