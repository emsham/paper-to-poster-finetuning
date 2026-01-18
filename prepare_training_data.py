#!/usr/bin/env python3
"""
Prepare Training Data for Finetuning
=====================================

This script preprocesses the raw training data for LLM finetuning by:
1. Truncating long paper content to fit context windows
2. Creating separate datasets for different training strategies
3. Formatting data for common training frameworks

Usage:
    python prepare_training_data.py --max-tokens 8192 --output prepared_data/
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import re


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def extract_sections(markdown: str) -> Dict[str, str]:
    """Extract key sections from paper markdown."""
    sections = {
        'abstract': '',
        'introduction': '',
        'conclusion': '',
        'methods': '',
        'results': '',
        'full': markdown
    }

    # Try to find abstract
    abstract_match = re.search(r'(?:^|\n)#+\s*Abstract\s*\n(.*?)(?=\n#+|\Z)', markdown, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        sections['abstract'] = abstract_match.group(1).strip()[:2000]

    # Try to find introduction
    intro_match = re.search(r'(?:^|\n)#+\s*(?:1\.?\s*)?Introduction\s*\n(.*?)(?=\n#+|\Z)', markdown, re.IGNORECASE | re.DOTALL)
    if intro_match:
        sections['introduction'] = intro_match.group(1).strip()[:4000]

    # Try to find conclusion
    concl_match = re.search(r'(?:^|\n)#+\s*(?:\d+\.?\s*)?Conclusion[s]?\s*\n(.*?)(?=\n#+|\Z)', markdown, re.IGNORECASE | re.DOTALL)
    if concl_match:
        sections['conclusion'] = concl_match.group(1).strip()[:2000]

    return sections


def truncate_paper(paper_content: str, max_chars: int, strategy: str = 'smart') -> str:
    """
    Truncate paper content to fit context window.

    Strategies:
    - 'smart': Keep abstract + intro + conclusion
    - 'head': Keep first N characters
    - 'none': No truncation
    """
    if len(paper_content) <= max_chars:
        return paper_content

    if strategy == 'none':
        return paper_content

    if strategy == 'head':
        return paper_content[:max_chars] + "\n\n[Content truncated...]"

    if strategy == 'smart':
        sections = extract_sections(paper_content)

        # Build truncated version
        parts = []

        if sections['abstract']:
            parts.append(f"## Abstract\n{sections['abstract']}")

        if sections['introduction']:
            parts.append(f"## Introduction\n{sections['introduction']}")

        if sections['methods']:
            parts.append(f"## Methods\n{sections['methods'][:2000]}")

        if sections['results']:
            parts.append(f"## Results\n{sections['results'][:2000]}")

        if sections['conclusion']:
            parts.append(f"## Conclusion\n{sections['conclusion']}")

        truncated = "\n\n".join(parts)

        # If still too long, fall back to head truncation
        if len(truncated) > max_chars:
            return truncated[:max_chars] + "\n\n[Content truncated...]"

        # If we have room, add more content
        remaining = max_chars - len(truncated) - 100
        if remaining > 1000 and len(paper_content) > len(truncated):
            truncated += f"\n\n## Additional Content\n{paper_content[len(truncated):len(truncated)+remaining]}"

        return truncated

    return paper_content[:max_chars]


def format_for_training(
    instruction: str,
    input_text: str,
    output_text: str,
    format_type: str = 'alpaca'
) -> Dict:
    """
    Format example for different training frameworks.

    Formats:
    - 'alpaca': instruction/input/output (default)
    - 'chatml': ChatML format for chat models
    - 'llama': Llama 2/3 chat format
    """
    if format_type == 'alpaca':
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output_text
        }

    elif format_type == 'chatml':
        return {
            'messages': [
                {'role': 'system', 'content': 'You are an expert at creating academic posters from research papers.'},
                {'role': 'user', 'content': f"{instruction}\n\n{input_text}"},
                {'role': 'assistant', 'content': output_text}
            ]
        }

    elif format_type == 'llama':
        return {
            'text': f"<s>[INST] <<SYS>>\nYou are an expert at creating academic posters from research papers.\n<</SYS>>\n\n{instruction}\n\n{input_text} [/INST] {output_text} </s>"
        }

    elif format_type == 'mistral':
        return {
            'text': f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output_text}</s>"
        }

    return {'instruction': instruction, 'input': input_text, 'output': output_text}


def prepare_dataset(
    input_file: str,
    output_dir: str,
    max_tokens: int = 8192,
    truncation_strategy: str = 'smart',
    output_format: str = 'alpaca',
    split_ratio: float = 0.1
):
    """
    Prepare training dataset with proper preprocessing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f.readlines()]

    print(f"Total examples: {len(examples)}")

    # Separate full and partial examples
    full_examples = [ex for ex in examples if len(ex['input']) >= 100]
    partial_examples = [ex for ex in examples if len(ex['input']) < 100]

    print(f"Full examples (with paper): {len(full_examples)}")
    print(f"Partial examples (title/abstract only): {len(partial_examples)}")

    # Calculate max chars based on tokens
    # Reserve tokens for instruction (~400) and output (~1500)
    max_input_chars = (max_tokens - 2000) * 4

    processed = []
    stats = {'truncated': 0, 'kept_full': 0, 'partial': 0}

    # Process full examples with truncation
    print(f"\nProcessing full examples (max {max_input_chars:,} chars for input)...")
    for ex in full_examples:
        original_len = len(ex['input'])
        truncated_input = truncate_paper(ex['input'], max_input_chars, truncation_strategy)

        if len(truncated_input) < original_len:
            stats['truncated'] += 1
        else:
            stats['kept_full'] += 1

        formatted = format_for_training(
            ex['instruction'],
            truncated_input,
            ex['output'],
            output_format
        )
        formatted['_meta'] = {'type': 'full', 'original_len': original_len}
        processed.append(formatted)

    # Process partial examples (no truncation needed)
    print("Processing partial examples...")
    for ex in partial_examples:
        formatted = format_for_training(
            ex['instruction'],
            ex['input'],
            ex['output'],
            output_format
        )
        formatted['_meta'] = {'type': 'partial'}
        stats['partial'] += 1
        processed.append(formatted)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(processed)

    # Split into train/val
    split_idx = int(len(processed) * (1 - split_ratio))
    train_data = processed[:split_idx]
    val_data = processed[split_idx:]

    # Remove meta before saving
    for ex in train_data + val_data:
        ex.pop('_meta', None)

    # Save datasets
    train_file = output_dir / 'train.jsonl'
    val_file = output_dir / 'val.jsonl'

    with open(train_file, 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex) + '\n')

    with open(val_file, 'w') as f:
        for ex in val_data:
            f.write(json.dumps(ex) + '\n')

    # Also save combined
    combined_file = output_dir / 'combined.jsonl'
    with open(combined_file, 'w') as f:
        for ex in train_data + val_data:
            f.write(json.dumps(ex) + '\n')

    # Print summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Truncated: {stats['truncated']}")
    print(f"Kept full: {stats['kept_full']}")
    print(f"Partial (no paper): {stats['partial']}")
    print(f"\nTrain examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")
    print(f"\nOutput files:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {combined_file}")

    # Verify token counts
    print(f"\nVerifying token counts...")
    train_tokens = [estimate_tokens(json.dumps(ex)) for ex in train_data[:100]]
    print(f"Sample train tokens: min={min(train_tokens)}, max={max(train_tokens)}, avg={sum(train_tokens)//len(train_tokens)}")

    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for finetuning')
    parser.add_argument('--input', '-i', default='pipeline_output/training_data.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--output', '-o', default='prepared_data',
                        help='Output directory')
    parser.add_argument('--max-tokens', type=int, default=8192,
                        help='Max context tokens (default: 8192)')
    parser.add_argument('--truncation', choices=['smart', 'head', 'none'], default='smart',
                        help='Truncation strategy')
    parser.add_argument('--format', choices=['alpaca', 'chatml', 'llama', 'mistral'],
                        default='alpaca', help='Output format')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')

    args = parser.parse_args()

    prepare_dataset(
        args.input,
        args.output,
        max_tokens=args.max_tokens,
        truncation_strategy=args.truncation,
        output_format=args.format,
        split_ratio=args.val_split
    )


if __name__ == '__main__':
    main()
