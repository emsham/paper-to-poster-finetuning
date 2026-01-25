#!/usr/bin/env python3
"""
Compare Finetuned Model vs GPT vs Claude on Poster Generation
==============================================================

Usage:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    export HF_MODEL_ID="your-username/poster-mistral-lora"

    # With PDF files (runs through marker-pdf first)
    python evaluate_models.py --pdf-dir ./test_papers/

    # With built-in samples
    python evaluate_models.py --samples 5
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import tempfile

# API clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Marker PDF
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

SYSTEM_PROMPT = """You are an expert at creating academic posters. Given a research paper's content,
generate a well-structured academic poster with clear sections, key findings, and visual layout descriptions."""

GENERATION_PROMPT = """Generate an academic poster based on the following paper content.
Include: Title, Authors, Introduction/Background, Methods, Key Results, Conclusions, and suggest figure placements.

Paper Content:
{paper_content}"""

JUDGE_PROMPT = """You are evaluating academic poster generations from three different AI models.
Rate each poster on a scale of 1-10 for the following criteria:

1. **Structure** (1-10): Does it have clear sections (Title, Intro, Methods, Results, Conclusions)?
2. **Content Quality** (1-10): Is the information accurate, relevant, and well-summarized?
3. **Visual Layout** (1-10): Does it describe figure placements and visual organization?
4. **Readability** (1-10): Is it concise and suitable for a poster format?
5. **Academic Tone** (1-10): Does it maintain professional academic language?

For each model output, provide scores and brief justification.
Then declare a winner.

---
PAPER CONTENT:
{paper_content}

---
MODEL A (Finetuned Mistral):
{output_a}

---
MODEL B (GPT-4):
{output_b}

---
MODEL C (Claude):
{output_c}

---
Provide your evaluation as JSON:
{{
    "model_a": {{"structure": X, "content": X, "layout": X, "readability": X, "tone": X, "total": X, "notes": "..."}},
    "model_b": {{"structure": X, "content": X, "layout": X, "readability": X, "tone": X, "total": X, "notes": "..."}},
    "model_c": {{"structure": X, "content": X, "layout": X, "readability": X, "tone": X, "total": X, "notes": "..."}},
    "winner": "model_a|model_b|model_c",
    "reasoning": "..."
}}"""


# ============================================================================
# SAMPLE PAPERS (for testing)
# ============================================================================

SAMPLE_PAPERS = [
    {
        "id": "sample_1",
        "title": "Deep Learning for Medical Image Segmentation",
        "content": """
Title: U-Net++: A Nested U-Net Architecture for Medical Image Segmentation
Authors: John Smith, Jane Doe, University of Medical AI

Abstract: We propose U-Net++, a deeply supervised encoder-decoder network that uses nested dense skip connections. Our architecture improves segmentation accuracy by 3.5% on liver CT scans and 2.8% on lung X-rays compared to standard U-Net.

Introduction: Medical image segmentation is crucial for diagnosis and treatment planning. Existing methods struggle with boundary detection and small lesions.

Methods: We redesign skip connections as dense nested blocks. Each decoder level receives features from all preceding encoder levels through a series of convolutions. We use deep supervision with auxiliary losses at each decoder level.

Results: On the CHAOS dataset (liver CT), we achieve Dice score of 0.967. On the Montgomery dataset (lung X-ray), we achieve 0.982 Dice. Training time reduced by 15% due to efficient gradient flow.

Conclusions: Nested skip connections significantly improve segmentation quality. Our method is applicable to various medical imaging modalities.
"""
    },
    {
        "id": "sample_2",
        "title": "Transformer Models for Climate Prediction",
        "content": """
Title: ClimateFormer: Long-Range Weather Prediction using Attention Mechanisms
Authors: Alice Chen, Bob Johnson, Climate AI Lab

Abstract: We present ClimateFormer, a transformer-based model for medium-range weather forecasting. Our model predicts temperature and precipitation up to 14 days ahead with 23% lower error than numerical weather prediction models.

Introduction: Accurate weather prediction is essential for agriculture, disaster preparedness, and energy planning. Traditional numerical models are computationally expensive and struggle beyond 7 days.

Methods: We use a vision transformer backbone adapted for spatiotemporal data. Input features include satellite imagery, ground station measurements, and historical patterns. We train on 40 years of ERA5 reanalysis data covering global measurements.

Results: Our model achieves RMSE of 1.2°C for 7-day temperature forecasts vs 1.8°C for ECMWF. For precipitation, we achieve 0.85 F1-score for rain/no-rain classification at day 10.

Conclusions: Transformer architectures can effectively capture long-range temporal dependencies in climate data, offering faster inference than traditional numerical methods.
"""
    },
    {
        "id": "sample_3",
        "title": "Quantum Machine Learning for Drug Discovery",
        "content": """
Title: Variational Quantum Classifiers for Molecular Property Prediction
Authors: David Lee, Maria Garcia, Quantum Pharma Research

Abstract: We demonstrate that variational quantum circuits can predict molecular toxicity with 91% accuracy using only 8 qubits. This represents a 5x speedup potential over classical methods for screening large chemical libraries.

Introduction: Drug discovery requires predicting molecular properties from structure. Classical ML methods scale poorly with molecular complexity. Quantum computers may offer exponential speedups for certain molecular simulations.

Methods: We encode molecular fingerprints into quantum states using amplitude encoding. Our variational circuit uses 3 layers of parameterized rotations and entangling gates. We train using SPSA optimizer on IBM quantum hardware.

Results: On the Tox21 dataset, our 8-qubit circuit achieves 91% accuracy vs 89% for random forest baseline. On BBBP (blood-brain barrier), we achieve 87% accuracy. Quantum circuits show better generalization with limited training data.

Conclusions: Near-term quantum devices can already contribute to drug discovery pipelines. Hybrid quantum-classical approaches are most promising for practical applications.
"""
    },
    {
        "id": "sample_4",
        "title": "Federated Learning for Privacy-Preserving Healthcare",
        "content": """
Title: FedHealth: Secure Federated Learning for Multi-Hospital Diagnosis
Authors: Sarah Kim, Tom Wilson, HealthAI Consortium

Abstract: FedHealth enables collaborative training of diagnostic models across 12 hospitals without sharing patient data. Our federated approach matches centralized training accuracy (94.2%) while ensuring HIPAA compliance.

Introduction: Healthcare AI requires large diverse datasets, but privacy regulations prevent data sharing. Federated learning allows model training on distributed data without centralization.

Methods: We implement horizontal federated learning with secure aggregation. Each hospital trains locally on their patient records. Model updates are encrypted before aggregation. We use differential privacy with epsilon=1.0.

Results: Across 12 hospitals and 500,000 patient records, our federated model achieves 94.2% accuracy for diabetes prediction, matching a centralized baseline. Communication costs reduced 80% using gradient compression.

Conclusions: Federated learning is practical for healthcare AI. Privacy guarantees enable collaboration between competing institutions.
"""
    },
    {
        "id": "sample_5",
        "title": "Reinforcement Learning for Robotic Manipulation",
        "content": """
Title: DexGrasp: Learning Dexterous Grasping from Human Demonstrations
Authors: Kevin Zhang, Emily Brown, Robotics Institute

Abstract: DexGrasp learns complex manipulation skills for a 24-DOF robotic hand using only 50 human demonstrations. Our imitation learning approach achieves 89% success rate on novel objects, outperforming prior methods by 34%.

Introduction: Dexterous manipulation remains challenging for robots. High-dimensional action spaces and contact dynamics make learning difficult. Human demonstrations provide rich supervision but are expensive to collect.

Methods: We use a transformer policy that attends to point cloud observations and proprioceptive state. Training combines behavioral cloning with online RL fine-tuning. We collect demonstrations using a VR teleoperation system.

Results: On the DexYCB benchmark, we achieve 89% grasp success vs 55% for PPO baseline. Our policy generalizes to unseen objects with 76% success. Training requires only 2 hours on a single GPU.

Conclusions: Combining imitation learning with RL enables sample-efficient learning of dexterous skills. Transformer architectures effectively model complex hand-object interactions.
"""
    },
]


# ============================================================================
# PDF PARSER (using marker-pdf)
# ============================================================================

class PDFParser:
    """Parse PDFs using marker-pdf library."""

    def __init__(self):
        if not MARKER_AVAILABLE:
            raise ImportError("marker-pdf required. pip install marker-pdf")

        print("Initializing marker-pdf...")
        config_parser = ConfigParser({"output_format": "markdown"})
        self.config = config_parser.generate_config_dict()
        self.models = create_model_dict()
        print("Marker-pdf ready.")

    def parse(self, pdf_path: str) -> str:
        """Parse a PDF and return markdown content."""
        converter = PdfConverter(
            config=self.config,
            artifact_dict=self.models,
        )
        result = converter(pdf_path)
        return result.markdown


class SimplePDFParser:
    """Fallback PDF parser using PyMuPDF if marker not available."""

    def __init__(self):
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            print("Using PyMuPDF for PDF parsing (simpler extraction)")
        except ImportError:
            raise ImportError("Either marker-pdf or PyMuPDF required. pip install marker-pdf or pip install PyMuPDF")

    def parse(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        doc = self.fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)


def get_pdf_parser():
    """Get the best available PDF parser."""
    if MARKER_AVAILABLE:
        return PDFParser()
    else:
        return SimplePDFParser()


def load_pdfs_from_directory(pdf_dir: str, parser) -> list:
    """Load and parse all PDFs from a directory."""
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF files")

    samples = []
    for pdf_path in pdf_files:
        print(f"  Parsing: {pdf_path.name}...")
        try:
            content = parser.parse(str(pdf_path))
            samples.append({
                "id": pdf_path.stem,
                "title": pdf_path.stem.replace("_", " ").replace("-", " "),
                "content": content,
                "source_file": str(pdf_path)
            })
            print(f"    Extracted {len(content)} characters")
        except Exception as e:
            print(f"    Error parsing {pdf_path.name}: {e}")

    return samples


# ============================================================================
# MODEL RUNNERS
# ============================================================================

class FinetunedModel:
    """Run inference with the finetuned Mistral LoRA model."""

    def __init__(self, model_id: str):
        if not HF_AVAILABLE:
            raise ImportError("transformers and peft required. pip install transformers peft torch")

        print(f"Loading finetuned model: {model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Finetuned model loaded.")

    def generate(self, paper_content: str) -> str:
        instruction = "Generate an academic poster based on the following paper content."
        prompt = f"<s>[INST] {instruction}\n\n{paper_content} [/INST]"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("[/INST]")[-1].strip()


class GPTModel:
    """Run inference with OpenAI GPT-4."""

    def __init__(self, model: str = "gpt-4o"):
        if OpenAI is None:
            raise ImportError("openai required. pip install openai")
        self.client = OpenAI()
        self.model = model
        print(f"GPT model ready: {model}")

    def generate(self, paper_content: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_PROMPT.format(paper_content=paper_content)}
            ],
            max_tokens=2048,
            temperature=0.7
        )
        return response.choices[0].message.content


class ClaudeModel:
    """Run inference with Anthropic Claude."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        if Anthropic is None:
            raise ImportError("anthropic required. pip install anthropic")
        self.client = Anthropic()
        self.model = model
        print(f"Claude model ready: {model}")

    def generate(self, paper_content: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": GENERATION_PROMPT.format(paper_content=paper_content)}
            ]
        )
        return response.content[0].text


class ClaudeJudge:
    """Use Claude to evaluate and compare outputs."""

    def __init__(self):
        if Anthropic is None:
            raise ImportError("anthropic required. pip install anthropic")
        self.client = Anthropic()

    def evaluate(self, paper_content: str, output_a: str, output_b: str, output_c: str) -> dict:
        prompt = JUDGE_PROMPT.format(
            paper_content=paper_content,
            output_a=output_a,
            output_b=output_b,
            output_c=output_c
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON from response
        text = response.content[0].text
        # Find JSON block
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {"error": "Failed to parse evaluation", "raw": text}


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"evaluation_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Select samples based on input source
    if args.pdf_dir:
        # Parse PDFs from directory
        print(f"Loading PDFs from: {args.pdf_dir}")
        parser = get_pdf_parser()
        samples = load_pdfs_from_directory(args.pdf_dir, parser)
        if args.samples and args.samples < len(samples):
            samples = samples[:args.samples]
    elif args.use_real_data and Path("prepared_data/val.jsonl").exists():
        print("Loading real validation data...")
        with open("prepared_data/val.jsonl") as f:
            real_data = [json.loads(line) for line in f]
        random.shuffle(real_data)
        samples = [{"id": f"real_{i}", "content": d["input"], "title": d["input"][:50]}
                   for i, d in enumerate(real_data[:args.samples])]
    else:
        # Use built-in samples
        samples = SAMPLE_PAPERS[:args.samples]

    print(f"\nEvaluating {len(samples)} papers")
    print(f"Output directory: {output_dir}\n")

    # Initialize models
    models = {}

    if args.finetuned_model and not args.skip_finetuned:
        try:
            models["finetuned"] = FinetunedModel(args.finetuned_model)
        except Exception as e:
            print(f"Warning: Could not load finetuned model: {e}")
            models["finetuned"] = None
    elif args.skip_finetuned:
        print("Skipping finetuned model (--skip-finetuned)")
        models["finetuned"] = None

    if os.environ.get("OPENAI_API_KEY"):
        try:
            models["gpt"] = GPTModel(args.gpt_model)
        except Exception as e:
            print(f"Warning: Could not initialize GPT: {e}")
            models["gpt"] = None
    else:
        print("Warning: OPENAI_API_KEY not set, skipping GPT")
        models["gpt"] = None

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            models["claude"] = ClaudeModel(args.claude_model)
            judge = ClaudeJudge()
        except Exception as e:
            print(f"Warning: Could not initialize Claude: {e}")
            models["claude"] = None
            judge = None
    else:
        print("Warning: ANTHROPIC_API_KEY not set, skipping Claude")
        models["claude"] = None
        judge = None

    # Run evaluation
    results = []

    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(samples)}: {sample.get('title', sample['id'])[:50]}...")
        print("="*60)

        paper_content = sample.get("content", sample.get("input", ""))
        outputs = {}

        # Generate from each model
        for name, model in models.items():
            if model is None:
                outputs[name] = "[Model not available]"
                continue

            print(f"  Generating with {name}...")
            try:
                outputs[name] = model.generate(paper_content)
                print(f"    Done ({len(outputs[name])} chars)")
            except Exception as e:
                print(f"    Error: {e}")
                outputs[name] = f"[Error: {e}]"

        # Save individual outputs
        sample_dir = output_dir / f"sample_{i+1}"
        sample_dir.mkdir(exist_ok=True)

        with open(sample_dir / "paper.txt", "w") as f:
            f.write(paper_content)

        for name, output in outputs.items():
            with open(sample_dir / f"output_{name}.txt", "w") as f:
                f.write(output)

        # Judge outputs
        evaluation = None
        if judge and all(v != "[Model not available]" for v in outputs.values()):
            print("  Evaluating outputs with Claude judge...")
            try:
                evaluation = judge.evaluate(
                    paper_content,
                    outputs.get("finetuned", "N/A"),
                    outputs.get("gpt", "N/A"),
                    outputs.get("claude", "N/A")
                )
                with open(sample_dir / "evaluation.json", "w") as f:
                    json.dump(evaluation, f, indent=2)
                print(f"    Winner: {evaluation.get('winner', 'N/A')}")
            except Exception as e:
                print(f"    Evaluation error: {e}")

        results.append({
            "sample_id": sample["id"],
            "outputs": {k: len(v) for k, v in outputs.items()},
            "evaluation": evaluation
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Count wins
    wins = {"model_a": 0, "model_b": 0, "model_c": 0}
    total_scores = {"model_a": [], "model_b": [], "model_c": []}

    for r in results:
        if r["evaluation"] and "winner" in r["evaluation"]:
            winner = r["evaluation"]["winner"]
            if winner in wins:
                wins[winner] += 1

            for model in ["model_a", "model_b", "model_c"]:
                if model in r["evaluation"] and "total" in r["evaluation"][model]:
                    total_scores[model].append(r["evaluation"][model]["total"])

    print(f"\nWins:")
    print(f"  Finetuned (Model A): {wins['model_a']}")
    print(f"  GPT-4 (Model B):     {wins['model_b']}")
    print(f"  Claude (Model C):    {wins['model_c']}")

    print(f"\nAverage Scores:")
    for model, scores in total_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            name = {"model_a": "Finetuned", "model_b": "GPT-4", "model_c": "Claude"}[model]
            print(f"  {name}: {avg:.1f}/50")

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_samples": len(samples),
            "wins": wins,
            "avg_scores": {k: sum(v)/len(v) if v else 0 for k, v in total_scores.items()},
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare poster generation models")
    parser.add_argument("--pdf-dir", type=str, help="Directory containing PDF files to evaluate")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--finetuned-model", type=str, default=os.environ.get("HF_MODEL_ID"),
                        help="HuggingFace model ID for finetuned model")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--claude-model", type=str, default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--use-real-data", action="store_true", help="Use real validation data if available")
    parser.add_argument("--skip-finetuned", action="store_true", help="Skip finetuned model (for quick API-only comparison)")

    args = parser.parse_args()
    run_evaluation(args)
