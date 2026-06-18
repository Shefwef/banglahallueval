#!/usr/bin/env python3
"""
Wrapper script to run TigerLLM evaluation with better error handling and resumption.
"""
import os
import sys
import subprocess
import csv
from pathlib import Path

# Check the current state of the output file
output_file = "QA/Results/qa_cot_hallu_tigerllm_9b.csv"

if Path(output_file).exists():
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        completed = sum(1 for r in rows if r.get("is_hallucinated", "").lower() in ("yes", "no"))
        total = len(rows)
        print(f"Current progress: {completed}/{total} rows labeled")
        if total > 0:
            pct = (completed / total) * 100
            print(f"Completion: {pct:.1f}%")
else:
    print(f"Output file not found: {output_file}")
    print("Starting fresh...")

# Set environment variables for GPU memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Run the evaluation script
print("\nStarting TigerLLM evaluation...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "scripts/evaluate_cot_tigerllm.py", "--task", "qa_hallu"],
    cwd=Path(__file__).parent,
)

sys.exit(result.returncode)
