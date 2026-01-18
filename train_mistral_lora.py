#!/usr/bin/env python3
"""
Finetune Mistral-7B with LoRA for Poster Generation
====================================================

RunPod Usage:
    1. Launch a pod with A100 40GB or RTX 4090
    2. Upload prepared_data/train.jsonl and prepared_data/val.jsonl
    3. Run: pip install -r requirements_training.txt
    4. Run: python train_mistral_lora.py

Estimated time: 2-4 hours on A100 40GB
Estimated cost: ~$4-6 on RunPod
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./poster-mistral-lora"
MAX_SEQ_LENGTH = 8192

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Effective batch size = 8
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
    print("POSTER GENERATION FINETUNING")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    # Load Model with 4-bit Quantization
    # ========================================================================
    print("\nLoading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Faster attention
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

    model = prepare_model_for_kbit_training(model)
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
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # Set to "wandb" if you want logging
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
        tokenizer=tokenizer,
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
