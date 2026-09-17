"""
Master pipeline script — runs the full pipeline end-to-end.
Designed to reproduce all results in under 15 minutes.
"""
import sys
import io
import os
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent))
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

from src.config import *


def run_step(step_name, module_path):
    """Run a pipeline step and report timing."""
    print(f"\n{'='*70}")
    print(f"  {step_name}")
    print(f"{'='*70}")
    start = time.time()
    exit_code = os.system(f'python "{module_path}"')
    elapsed = time.time() - start
    status = "OK" if exit_code == 0 else "FAILED"
    print(f"\n  [{status}] {step_name} completed in {elapsed:.1f}s")
    return exit_code == 0


def main():
    overall_start = time.time()
    
    print("=" * 70)
    print("  AMAZON HELP AI SUPPORT AGENT — FULL PIPELINE")
    print("=" * 70)
    print(f"  Brand: {BRAND}")
    print(f"  Subsample: {MAX_CONVERSATIONS} conversations")
    print(f"  Golden set: {GOLDEN_SET_SIZE} examples")
    print(f"  LLM: {GROQ_MODEL_GENERATION}")
    print(f"  Embeddings: {EMBEDDING_MODEL}")
    
    steps = [
        ("Step 1: Data Preprocessing", "src/pipeline/preprocess.py"),
        ("Step 2: Build Vector Store", "src/pipeline/build_vector_store.py"),
        ("Step 3: Build Golden Evaluation Set", "src/evaluation/golden_set_builder.py"),
        ("Step 4: Evaluate Intent Classification", "src/evaluation/eval_classification.py"),
        ("Step 5: Evaluate Escalation Decisions", "src/evaluation/eval_escalation.py"),
        ("Step 6: Evaluate Reply Quality", "src/evaluation/eval_replies.py"),
    ]
    
    results = {}
    for step_name, module_path in steps:
        # Check if prerequisite files exist
        if "vector_store" in module_path and not PROCESSED_DATA.exists():
            print(f"\n  SKIPPING {step_name}: prerequisite {PROCESSED_DATA} not found")
            results[step_name] = False
            continue
        if "golden_set" in module_path and not PROCESSED_DATA.exists():
            print(f"\n  SKIPPING {step_name}: prerequisite {PROCESSED_DATA} not found")
            results[step_name] = False
            continue
        if "eval_" in module_path and not GOLDEN_SET_LABELLED.exists():
            print(f"\n  SKIPPING {step_name}: prerequisite {GOLDEN_SET_LABELLED} not found")
            results[step_name] = False
            continue
            
        success = run_step(step_name, str(PROJECT_ROOT / module_path))
        results[step_name] = success
        if not success:
            print(f"\n  WARNING: {step_name} failed. Continuing with remaining steps...")
    
    # Summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for step, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {step}")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Golden set saved to: {GOLDEN_SET_LABELLED}")
    print("\n  To launch the demo:")
    print(f"  python src/demo/app.py")


if __name__ == "__main__":
    main()
