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

    # Use LLM for intelligent section extraction (~$7 for 7k papers)
    python prepare_training_data.py --use-llm --max-tokens 8192 --output prepared_data/
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import re

# Optional: Pydantic for structured LLM output
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Optional: Anthropic for LLM-based extraction
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Pydantic models for structured section extraction
if PYDANTIC_AVAILABLE:
    class ExtractedSections(BaseModel):
        """Structured output for paper sections."""
        abstract: str = Field(default="", description="The paper's abstract")
        introduction: str = Field(default="", description="Introduction or background section")
        methods: str = Field(default="", description="Methods, approach, or methodology section")
        results: str = Field(default="", description="Results, experiments, or evaluation section")
        conclusion: str = Field(default="", description="Conclusion, summary, or discussion section")


class LLMSectionExtractor:
    """Extract sections using Claude Haiku for intelligent categorization."""

    def __init__(self, api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-haiku-20240307"
        self.cache = {}

    def extract(self, paper_content: str, max_section_chars: int = 3000) -> Dict[str, str]:
        """Extract sections using LLM with structured output."""
        # Truncate input to avoid excessive cost (first 15k chars usually has all sections)
        content_sample = paper_content[:15000]

        prompt = f"""Analyze this academic paper and extract the key sections.
Papers may use various naming conventions (e.g., "Background" instead of "Introduction",
"Experimental Setup" instead of "Methods", "Findings" instead of "Results").

Identify and extract content for each category regardless of the exact section name used.
Return ONLY valid JSON with these fields (use empty string if section not found):

{{
  "abstract": "The paper's abstract text",
  "introduction": "Introduction/background/motivation section content",
  "methods": "Methods/approach/methodology/framework section content",
  "results": "Results/experiments/evaluation/findings section content",
  "conclusion": "Conclusion/summary/discussion section content"
}}

Truncate each section to ~{max_section_chars} characters if longer.

PAPER CONTENT:
{content_sample}

JSON OUTPUT:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            sections = json.loads(response_text)
            sections['full'] = paper_content
            return sections

        except Exception as e:
            print(f"LLM extraction failed: {e}, falling back to regex")
            return extract_sections_regex(paper_content)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def extract_sections_regex(markdown: str) -> Dict[str, str]:
    """Extract key sections from paper markdown using regex patterns."""
    sections = {
        'abstract': '',
        'introduction': '',
        'conclusion': '',
        'methods': '',
        'results': '',
        'full': markdown
    }

    # Section name variations
    intro_names = r'introduction|background|overview|motivation'
    methods_names = r'method|approach|methodology|model|framework|proposed|our\s+approach|technique'
    results_names = r'result|experiment|evaluation|empirical|finding|analysis'
    conclusion_names = r'conclusion|summary|discussion|future\s+work|closing'

    # Patterns for section headers (handles multiple formats):
    # - "## Abstract" (markdown)
    # - "1 Introduction" or "1. Introduction" (numbered)
    # - "ABSTRACT" (all caps)
    # - "Abstract" at start of line
    def find_section(pattern_names: str, max_chars: int = 3000) -> str:
        patterns = [
            # Markdown headers
            rf'(?:^|\n)#+\s*(?:\d+\.?\s*)?({pattern_names})\s*\n(.*?)(?=\n#+|\n\d+\.?\s+[A-Z]|\Z)',
            # Numbered sections: "1 Introduction" or "1. Introduction"
            rf'(?:^|\n)(\d+\.?\s+)({pattern_names})\s*\n(.*?)(?=\n\d+\.?\s+[A-Z]|\n#+|\Z)',
            # All caps: "INTRODUCTION"
            rf'(?:^|\n)({pattern_names.upper()})\s*\n(.*?)(?=\n[A-Z]{{4,}}|\n\d+\.?\s+[A-Z]|\Z)',
            # Title case at line start
            rf'(?:^|\n)({pattern_names})\s*\n(.*?)(?=\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n|\n\d+|\Z)',
        ]
        for pattern in patterns:
            match = re.search(pattern, markdown, re.IGNORECASE | re.DOTALL)
            if match:
                # Get the content (last group)
                content = match.groups()[-1].strip()
                return content[:max_chars]
        return ''

    # Extract abstract (special handling - often right after title)
    abstract_patterns = [
        r'(?:^|\n)(?:#+\s*)?Abstract\s*\n(.*?)(?=\n(?:#+|\d+\.?\s+[A-Z]|Introduction|INTRODUCTION))',
        r'(?:^|\n)ABSTRACT\s*\n(.*?)(?=\n[A-Z]{4,}|\n\d+)',
        r'Abstract\s*\n(.*?)(?=\n\s*\n\s*\d+\.?\s+[A-Z])',
    ]
    for pattern in abstract_patterns:
        match = re.search(pattern, markdown, re.IGNORECASE | re.DOTALL)
        if match:
            sections['abstract'] = match.group(1).strip()[:2000]
            break

    # Extract other sections
    sections['introduction'] = find_section(intro_names, 4000)
    sections['methods'] = find_section(methods_names, 3000)
    sections['results'] = find_section(results_names, 3000)
    sections['conclusion'] = find_section(conclusion_names, 2000)

    return sections


# Global extractor (set by main when --use-llm is specified)
_llm_extractor: Optional[LLMSectionExtractor] = None


def extract_sections(markdown: str) -> Dict[str, str]:
    """Extract sections using LLM if available, otherwise regex."""
    if _llm_extractor is not None:
        return _llm_extractor.extract(markdown)
    return extract_sections_regex(markdown)


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
    use_llm = _llm_extractor is not None
    desc = "Processing (LLM)" if use_llm else "Processing"

    for ex in tqdm(full_examples, desc=desc, disable=not TQDM_AVAILABLE):
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
    global _llm_extractor

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
    parser.add_argument('--use-llm', action='store_true',
                        help='Use Claude Haiku for intelligent section extraction (~$7 for 7k papers)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')

    args = parser.parse_args()

    # Initialize LLM extractor if requested
    if args.use_llm:
        if not ANTHROPIC_AVAILABLE:
            print("ERROR: --use-llm requires anthropic package. Install with: pip install anthropic")
            return
        api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: --use-llm requires API key via --api-key or ANTHROPIC_API_KEY env var")
            return
        print("Using Claude Haiku for intelligent section extraction...")
        _llm_extractor = LLMSectionExtractor(api_key=api_key)

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
