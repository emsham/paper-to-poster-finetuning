#!/usr/bin/env python3
"""
Generate Poster from Paper
==========================

Usage:
    python generate_poster.py --title "Paper Title" --abstract "Abstract text" --content "Paper content"
    python generate_poster.py --paper paper.md --output poster.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(model_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load the finetuned model."""
    print("Loading model...")

    # Quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def generate_poster(
    model,
    tokenizer,
    title: str,
    abstract: str,
    content: str = "",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict:
    """Generate poster layout and content from paper."""

    # Truncate content if needed
    max_content_chars = 20000
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n\n[Content truncated...]"

    # Format prompt
    instruction = f"""Generate an academic poster for the following research paper.

Title: {title}

Abstract: {abstract}"""

    prompt = f"<s>[INST] {instruction}\n\n{content} [/INST]"

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the generated part (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    # Try to parse as JSON
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # Return raw response if not valid JSON
        result = {"raw_response": response}

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate poster from paper")
    parser.add_argument("--model", default="./poster-mistral-lora", help="Path to finetuned model")
    parser.add_argument("--title", type=str, help="Paper title")
    parser.add_argument("--abstract", type=str, help="Paper abstract")
    parser.add_argument("--content", type=str, default="", help="Paper content (optional)")
    parser.add_argument("--paper", type=str, help="Path to paper markdown file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")

    args = parser.parse_args()

    # Load paper from file if provided
    if args.paper:
        with open(args.paper, 'r') as f:
            paper_content = f.read()

        # Extract title and abstract from markdown if not provided
        if not args.title:
            # Try to find title (first # heading)
            import re
            title_match = re.search(r'^#\s+(.+)$', paper_content, re.MULTILINE)
            args.title = title_match.group(1) if title_match else "Untitled"

        if not args.abstract:
            # Try to find abstract
            abstract_match = re.search(r'(?:abstract|summary)[:\s]*\n(.+?)(?=\n#|\n\n\n)', paper_content, re.IGNORECASE | re.DOTALL)
            args.abstract = abstract_match.group(1).strip()[:1000] if abstract_match else ""

        args.content = paper_content

    if not args.title:
        parser.error("--title is required (or provide --paper)")

    # Load model
    model, tokenizer = load_model(args.model)

    # Generate
    print(f"\nGenerating poster for: {args.title[:50]}...")
    result = generate_poster(
        model,
        tokenizer,
        args.title,
        args.abstract or "",
        args.content,
        temperature=args.temperature,
    )

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("GENERATED POSTER")
        print("=" * 60)
        print(json.dumps(result, indent=2)[:2000])
        if len(json.dumps(result)) > 2000:
            print("\n... [output truncated]")


if __name__ == "__main__":
    main()
