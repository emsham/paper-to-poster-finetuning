#!/usr/bin/env python3
"""
Pipeline Test Script
====================

Tests each component of the paper-to-poster pipeline to verify everything works.

Usage:
    # Run all tests
    python test_pipeline.py

    # Test specific components
    python test_pipeline.py --component dolphin
    python test_pipeline.py --component vlm
    python test_pipeline.py --component descriptor
    python test_pipeline.py --component pipeline

    # Test with specific sample
    python test_pipeline.py --paper-id 19205

    # Quick test (skip heavy VLM tests)
    python test_pipeline.py --quick

    # Dry run (check files exist, don't run models)
    python test_pipeline.py --dry-run
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


class PipelineTester:
    """Test harness for the paper-to-poster pipeline."""

    def __init__(
        self,
        data_dir: str = "./",
        dolphin_model: str = "./hf_model",
        vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        output_dir: str = "./test_output"
    ):
        self.data_dir = Path(data_dir)
        self.dolphin_model = dolphin_model
        self.vlm_model = vlm_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }

    def find_sample(self, paper_id: str = None) -> dict:
        """Find a sample paper-poster pair for testing."""
        train_path = self.data_dir / "train.csv"

        if not train_path.exists():
            return None

        df = pd.read_csv(train_path)

        # Filter valid rows
        valid_df = df[
            df['local_image_path'].notna() &
            df['local_pdf_path'].notna() &
            (df['error'].isna() | (df['error'] == ''))
        ]

        if paper_id:
            sample = valid_df[valid_df['paper_id'].astype(str) == str(paper_id)]
            if len(sample) == 0:
                print_error(f"Paper ID {paper_id} not found in valid samples")
                return None
            row = sample.iloc[0]
        else:
            row = valid_df.iloc[0]

        sample = {
            "paper_id": str(row['paper_id']),
            "poster_path": self.data_dir / row['local_image_path'],
            "paper_path": self.data_dir / row['local_pdf_path'],
            "title": row.get('title', ''),
            "abstract": row.get('abstract', '')[:200] + "..." if pd.notna(row.get('abstract')) else '',
            "conference": row.get('conference', ''),
            "year": row.get('year', '')
        }

        return sample

    def test_imports(self) -> bool:
        """Test that all pipeline modules can be imported."""
        print_header("TEST: Import Pipeline Modules")

        try:
            from pipeline import (
                PipelineConfig,
                DolphinConfig,
                VLMMatcherConfig,
                PosterDescriptorConfig,
                get_default_config,
                DolphinParser,
                VLMFigureMatcher,
                PosterDescriptor,
                PaperToPosterPipeline,
                create_pipeline,
                StagedPipeline,
                MultiGPUPipeline,
            )
            print_success("All pipeline modules imported successfully")
            self.results["tests"]["imports"] = {"passed": True}
            self.results["passed"] += 1
            return True

        except ImportError as e:
            print_error(f"Import failed: {e}")
            self.results["tests"]["imports"] = {"passed": False, "error": str(e)}
            self.results["failed"] += 1
            return False

    def test_data_files(self) -> bool:
        """Test that data files exist."""
        print_header("TEST: Data Files")

        files_to_check = [
            ("train.csv", self.data_dir / "train.csv"),
            ("test.csv", self.data_dir / "test.csv"),
            ("validation.csv", self.data_dir / "validation.csv"),
            ("images/", self.data_dir / "images"),
            ("papers/", self.data_dir / "papers"),
            ("Dolphin/", self.data_dir / "Dolphin"),
        ]

        all_found = True
        for name, path in files_to_check:
            if path.exists():
                if path.is_dir():
                    count = len(list(path.iterdir()))
                    print_success(f"{name} found ({count} items)")
                else:
                    size = path.stat().st_size / 1024 / 1024
                    print_success(f"{name} found ({size:.1f} MB)")
            else:
                print_warning(f"{name} not found at {path}")
                all_found = False

        self.results["tests"]["data_files"] = {"passed": all_found}
        if all_found:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        return all_found

    def test_sample_files(self, sample: dict) -> bool:
        """Test that sample files exist."""
        print_header("TEST: Sample Files")

        print_info(f"Paper ID: {sample['paper_id']}")
        print_info(f"Title: {sample['title'][:60]}...")
        print_info(f"Conference: {sample['conference']} {sample['year']}")

        poster_exists = sample['poster_path'].exists()
        paper_exists = sample['paper_path'].exists()

        if poster_exists:
            size = sample['poster_path'].stat().st_size / 1024 / 1024
            print_success(f"Poster found: {sample['poster_path'].name} ({size:.1f} MB)")
        else:
            print_error(f"Poster not found: {sample['poster_path']}")

        if paper_exists:
            size = sample['paper_path'].stat().st_size / 1024 / 1024
            print_success(f"Paper found: {sample['paper_path'].name} ({size:.1f} MB)")
        else:
            print_error(f"Paper not found: {sample['paper_path']}")

        passed = poster_exists and paper_exists
        self.results["tests"]["sample_files"] = {"passed": passed}
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        return passed

    def test_dolphin_model(self) -> bool:
        """Test that Dolphin model can be found."""
        print_header("TEST: Dolphin Model")

        model_path = Path(self.dolphin_model)

        if not model_path.exists():
            print_warning(f"Dolphin model not found at {model_path}")
            print_info("Download with: huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model")
            self.results["tests"]["dolphin_model"] = {"passed": False, "error": "Model not found"}
            self.results["failed"] += 1
            return False

        # Check for key files
        config_file = model_path / "config.json"
        if config_file.exists():
            print_success(f"Model config found at {model_path}")
            with open(config_file, 'r') as f:
                config = json.load(f)
                print_info(f"Model type: {config.get('model_type', 'unknown')}")
        else:
            print_warning("config.json not found in model directory")

        self.results["tests"]["dolphin_model"] = {"passed": True}
        self.results["passed"] += 1
        return True

    def test_dolphin_parser(self, sample: dict, dry_run: bool = False) -> bool:
        """Test Dolphin parser on a sample."""
        print_header("TEST: Dolphin Parser")

        if dry_run:
            print_info("Dry run - skipping model loading")
            self.results["tests"]["dolphin_parser"] = {"passed": True, "skipped": True}
            self.results["skipped"] += 1
            return True

        try:
            from pipeline import DolphinConfig, create_dolphin_parser

            config = DolphinConfig(model_path=self.dolphin_model)
            parser = create_dolphin_parser(config)

            print_info("Loading Dolphin model...")
            start_time = time.time()
            parser.load_model()
            load_time = time.time() - start_time
            print_success(f"Model loaded in {load_time:.1f}s")

            # Test poster parsing
            print_info("Parsing poster...")
            output_dir = self.output_dir / "dolphin_test" / "poster"
            start_time = time.time()
            result = parser.parse_poster(
                str(sample['poster_path']),
                str(output_dir),
                sample['paper_id']
            )
            parse_time = time.time() - start_time

            print_success(f"Poster parsed in {parse_time:.1f}s")
            print_info(f"Markdown length: {len(result.markdown_content)} chars")
            print_info(f"Figures extracted: {len(result.figures)}")

            # Test paper parsing (first page only for speed)
            print_info("Parsing paper (this may take a while for multi-page PDFs)...")
            output_dir = self.output_dir / "dolphin_test" / "paper"
            start_time = time.time()
            result = parser.parse_paper(
                str(sample['paper_path']),
                str(output_dir),
                sample['paper_id']
            )
            parse_time = time.time() - start_time

            print_success(f"Paper parsed in {parse_time:.1f}s")
            print_info(f"Markdown length: {len(result.markdown_content)} chars")
            print_info(f"Figures extracted: {len(result.figures)}")

            self.results["tests"]["dolphin_parser"] = {"passed": True}
            self.results["passed"] += 1
            return True

        except Exception as e:
            print_error(f"Dolphin parser test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["tests"]["dolphin_parser"] = {"passed": False, "error": str(e)}
            self.results["failed"] += 1
            return False

    def test_vlm_matcher(self, dry_run: bool = False) -> bool:
        """Test VLM figure matcher."""
        print_header("TEST: VLM Figure Matcher")

        if dry_run:
            print_info("Dry run - skipping model loading")
            self.results["tests"]["vlm_matcher"] = {"passed": True, "skipped": True}
            self.results["skipped"] += 1
            return True

        # Check if we have figures from Dolphin test
        poster_figures_dir = self.output_dir / "dolphin_test" / "poster" / "markdown" / "figures"
        paper_figures_dir = self.output_dir / "dolphin_test" / "paper" / "markdown" / "figures"

        if not poster_figures_dir.exists() or not paper_figures_dir.exists():
            print_warning("No figures from Dolphin test - run dolphin test first")
            self.results["tests"]["vlm_matcher"] = {"passed": False, "error": "No figures available"}
            self.results["failed"] += 1
            return False

        poster_figures = list(poster_figures_dir.glob("*.png"))
        paper_figures = list(paper_figures_dir.glob("*.png"))

        print_info(f"Poster figures: {len(poster_figures)}")
        print_info(f"Paper figures: {len(paper_figures)}")

        if not poster_figures or not paper_figures:
            print_warning("Not enough figures to test matching")
            self.results["tests"]["vlm_matcher"] = {"passed": True, "skipped": True}
            self.results["skipped"] += 1
            return True

        try:
            from pipeline import VLMMatcherConfig, create_figure_matcher
            from pipeline.models import ExtractedFigure

            config = VLMMatcherConfig(model_name=self.vlm_model)
            matcher = create_figure_matcher(config)

            # Create ExtractedFigure objects
            poster_figs = [
                ExtractedFigure(
                    figure_id=f.stem,
                    source_doc="poster",
                    file_path=str(f)
                )
                for f in poster_figures[:3]  # Limit for testing
            ]
            paper_figs = [
                ExtractedFigure(
                    figure_id=f.stem,
                    source_doc="paper",
                    file_path=str(f)
                )
                for f in paper_figures[:3]  # Limit for testing
            ]

            print_info("Running figure matching...")
            output_dir = self.output_dir / "vlm_test"
            start_time = time.time()
            matches = matcher.match_figures(poster_figs, paper_figs, str(output_dir))
            match_time = time.time() - start_time

            print_success(f"Matching completed in {match_time:.1f}s")
            print_info(f"Matches found: {len(matches)}")

            self.results["tests"]["vlm_matcher"] = {"passed": True, "matches": len(matches)}
            self.results["passed"] += 1
            return True

        except Exception as e:
            print_error(f"VLM matcher test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["tests"]["vlm_matcher"] = {"passed": False, "error": str(e)}
            self.results["failed"] += 1
            return False

    def test_poster_descriptor(self, sample: dict, dry_run: bool = False) -> bool:
        """Test poster layout descriptor."""
        print_header("TEST: Poster Descriptor")

        if dry_run:
            print_info("Dry run - skipping model loading")
            self.results["tests"]["poster_descriptor"] = {"passed": True, "skipped": True}
            self.results["skipped"] += 1
            return True

        try:
            from pipeline import PosterDescriptorConfig, create_poster_descriptor

            config = PosterDescriptorConfig(model_name=self.vlm_model)
            descriptor = create_poster_descriptor(config)

            print_info("Extracting poster layout...")
            start_time = time.time()
            layout = descriptor.describe_poster(
                str(sample['poster_path']),
                sample['paper_id']
            )
            desc_time = time.time() - start_time

            print_success(f"Layout extracted in {desc_time:.1f}s")
            print_info(f"Orientation: {layout.orientation}")
            print_info(f"Columns: {layout.body.get('columns', 'unknown')}")
            print_info(f"Sections: {len(layout.sections)}")
            print_info(f"Figures: {len(layout.figures)}")

            # Save layout
            output_path = self.output_dir / "descriptor_test" / f"{sample['paper_id']}_layout.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            layout.save(output_path)
            print_info(f"Layout saved to: {output_path}")

            self.results["tests"]["poster_descriptor"] = {"passed": True}
            self.results["passed"] += 1
            return True

        except Exception as e:
            print_error(f"Poster descriptor test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["tests"]["poster_descriptor"] = {"passed": False, "error": str(e)}
            self.results["failed"] += 1
            return False

    def test_full_pipeline(self, sample: dict, dry_run: bool = False) -> bool:
        """Test full pipeline on a single sample."""
        print_header("TEST: Full Pipeline")

        if dry_run:
            print_info("Dry run - skipping full pipeline")
            self.results["tests"]["full_pipeline"] = {"passed": True, "skipped": True}
            self.results["skipped"] += 1
            return True

        try:
            from pipeline import create_pipeline

            output_dir = self.output_dir / "pipeline_test"

            pipeline = create_pipeline(
                data_dir=str(self.data_dir),
                output_dir=str(output_dir),
                dolphin_model_path=self.dolphin_model,
                vlm_model_name=self.vlm_model
            )

            print_info(f"Processing paper: {sample['paper_id']}")
            start_time = time.time()

            result = pipeline.process_single(
                paper_id=sample['paper_id'],
                poster_path=str(sample['poster_path']),
                paper_path=str(sample['paper_path']),
                metadata={
                    "title": sample['title'],
                    "abstract": sample['abstract'],
                    "conference": sample['conference'],
                    "year": sample['year']
                }
            )

            total_time = time.time() - start_time

            if result.success:
                print_success(f"Pipeline completed in {total_time:.1f}s")
                print_info(f"Poster figures: {len(result.parsed_poster.figures) if result.parsed_poster else 0}")
                print_info(f"Paper figures: {len(result.parsed_paper.figures) if result.parsed_paper else 0}")
                print_info(f"Figure matches: {len(result.figure_matches)}")
                print_info(f"Training data saved to: {output_dir / 'training_data'}")

                self.results["tests"]["full_pipeline"] = {"passed": True, "time": total_time}
                self.results["passed"] += 1
                return True
            else:
                print_error(f"Pipeline failed: {result.error}")
                self.results["tests"]["full_pipeline"] = {"passed": False, "error": result.error}
                self.results["failed"] += 1
                return False

        except Exception as e:
            print_error(f"Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["tests"]["full_pipeline"] = {"passed": False, "error": str(e)}
            self.results["failed"] += 1
            return False

    def test_staged_pipeline(self, dry_run: bool = False) -> bool:
        """Test staged parallel pipeline."""
        print_header("TEST: Staged Pipeline")

        if dry_run:
            print_info("Dry run - skipping staged pipeline test")
            self.results["tests"]["staged_pipeline"] = {"passed": True, "skipped": True}
            self.results["skipped"] += 1
            return True

        try:
            from pipeline import get_default_config, StagedPipeline

            config = get_default_config(str(self.data_dir))
            config.output_dir = self.output_dir / "staged_test"
            config.dolphin.model_path = self.dolphin_model
            config.vlm_matcher.model_name = self.vlm_model
            config.poster_descriptor.model_name = self.vlm_model

            pipeline = StagedPipeline(config)

            # Load small subset
            df = pd.read_csv(self.data_dir / "train.csv")
            df = df.head(2)  # Just 2 items for testing

            print_info("Running staged pipeline on 2 items...")
            start_time = time.time()
            stats = pipeline.run(df, num_workers=2, resume=False)
            total_time = time.time() - start_time

            print_success(f"Staged pipeline completed in {total_time:.1f}s")
            print_info(f"Stats: {stats}")

            self.results["tests"]["staged_pipeline"] = {"passed": True, "time": total_time}
            self.results["passed"] += 1
            return True

        except Exception as e:
            print_error(f"Staged pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["tests"]["staged_pipeline"] = {"passed": False, "error": str(e)}
            self.results["failed"] += 1
            return False

    def print_summary(self):
        """Print test summary."""
        print_header("TEST SUMMARY")

        total = self.results["passed"] + self.results["failed"] + self.results["skipped"]

        print(f"Total tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.results['passed']}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.results['failed']}{Colors.END}")
        print(f"{Colors.YELLOW}Skipped: {self.results['skipped']}{Colors.END}")

        if self.results["failed"] > 0:
            print(f"\n{Colors.RED}Some tests failed!{Colors.END}")
            for test_name, result in self.results["tests"].items():
                if not result.get("passed", False) and not result.get("skipped", False):
                    print(f"  - {test_name}: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n{Colors.GREEN}All tests passed!{Colors.END}")

        # Save results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    def run_all(
        self,
        sample: dict,
        quick: bool = False,
        dry_run: bool = False,
        component: str = None
    ):
        """Run all tests or specific component."""

        if component:
            # Run specific component
            if component == "dolphin":
                self.test_dolphin_model()
                self.test_dolphin_parser(sample, dry_run)
            elif component == "vlm":
                self.test_vlm_matcher(dry_run)
            elif component == "descriptor":
                self.test_poster_descriptor(sample, dry_run)
            elif component == "pipeline":
                self.test_full_pipeline(sample, dry_run)
            elif component == "staged":
                self.test_staged_pipeline(dry_run)
            else:
                print_error(f"Unknown component: {component}")
        else:
            # Run all tests
            self.test_imports()
            self.test_data_files()
            self.test_sample_files(sample)
            self.test_dolphin_model()

            if not dry_run and not quick:
                self.test_dolphin_parser(sample, dry_run)
                self.test_poster_descriptor(sample, dry_run)
                self.test_vlm_matcher(dry_run)
                self.test_full_pipeline(sample, dry_run)
            elif quick:
                print_info("Quick mode - skipping heavy model tests")
                self.results["skipped"] += 4

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Test the paper-to-poster pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./",
        help="Root data directory"
    )
    parser.add_argument(
        "--dolphin-model",
        type=str,
        default="./hf_model",
        help="Path to Dolphin model"
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="VLM model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_output",
        help="Output directory for test results"
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help="Specific paper ID to test with"
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["dolphin", "vlm", "descriptor", "pipeline", "staged"],
        default=None,
        help="Test specific component only"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (skip heavy model tests)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (check files, don't run models)"
    )

    args = parser.parse_args()

    # Create tester
    tester = PipelineTester(
        data_dir=args.data_dir,
        dolphin_model=args.dolphin_model,
        vlm_model=args.vlm_model,
        output_dir=args.output_dir
    )

    # Find sample
    sample = tester.find_sample(args.paper_id)
    if not sample:
        print_error("Could not find a valid sample to test with")
        sys.exit(1)

    # Run tests
    tester.run_all(
        sample=sample,
        quick=args.quick,
        dry_run=args.dry_run,
        component=args.component
    )

    # Exit with appropriate code
    sys.exit(0 if tester.results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
