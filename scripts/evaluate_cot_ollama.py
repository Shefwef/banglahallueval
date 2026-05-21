#!/usr/bin/env python3
"""CoT (Chain-of-Thought) hallucination evaluation using Ollama models.

Instead of asking for a bare yes/no, CoT prompts guide the model to reason
step-by-step before giving its final verdict. Results are saved to separate
_cot_ files so they can be compared against the baseline.

Tasks:
  summ_hallu   → Summarization/Evaluation_Results/summ_3000_cot_{model}.csv
  summ_gt      → Summarization/Evaluation_Results/summ_1000_cot_{model}.csv
  reason_hallu → Reasoning/Results/reasoning_cot_{model}.csv
  reason_gt    → Reasoning/Results/reasoning_gt_cot_{model}.csv

Models (default: all four):
  qwen2.5:32b-instruct, gemma2:27b, mistral-nemo:latest, llama3.1:8b

Usage:
    python scripts/evaluate_cot_ollama.py --task all
    python scripts/evaluate_cot_ollama.py --task reason_hallu --models qwen2.5:32b-instruct gemma2:27b
    python scripts/evaluate_cot_ollama.py --task summ_hallu --models mistral-nemo:latest
"""

import csv
import os
import re
import json
import argparse
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Models ─────────────────────────────────────────────────────────────────────

ALL_MODELS = [
    ("qwen2.5:32b-instruct", "qwen2_5_32b"),
    ("gemma2:27b",           "gemma2_27b"),
    ("mistral-nemo:latest",  "mistral_nemo"),
    ("llama3.1:8b",          "llama3_1_8b"),
]

# ── CoT Prompts ────────────────────────────────────────────────────────────────

QA_COT_PROMPT = (
    "You are an evaluator checking whether a model answer is hallucinated.\n\n"
    "Question: {question}\n"
    "Model Answer: {answer}\n\n"
    "Analyze step by step:\n"
    "Step 1: What factual claims does the answer make?\n"
    "Step 2: Are these claims supported by or inferable from the question context?\n"
    "Step 3: Based on steps 1-2, is the answer hallucinated?\n\n"
    "Final answer (write only this word on the last line): yes or no\n"
    "(yes = answer is hallucinated, no = answer is not hallucinated)"
)

SUMM_COT_PROMPT = (
    "You are an evaluator checking whether a summary is hallucinated relative to a document.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Analyze step by step:\n"
    "Step 1: List the key claims made in the summary.\n"
    "Step 2: For each claim, check whether it is directly supported by the document.\n"
    "Step 3: Based on steps 1-2, decide your final answer.\n\n"
    "Final answer (write only this word on the last line): yes or no\n"
    "(yes = summary is hallucinated, no = summary is not hallucinated)"
)

REASONING_COT_PROMPT = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n\n"
    "Question: {question}\n"
    "Reasoning Chain: {chain}\n"
    "Answer: {answer}\n\n"
    "Analyze step by step:\n"
    "Step 1: Is each calculation or logical step in the reasoning chain mathematically correct?\n"
    "Step 2: Does the final answer follow logically from the reasoning chain?\n"
    "Step 3: Based on steps 1-2, is this reasoning chain hallucinated (incorrect or fabricated)?\n\n"
    'Respond ONLY with a JSON object on the last line: {{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}'
)

# ── Ollama call ────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, base_url: str, num_predict: int = 512) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""

# ── Label parsers ──────────────────────────────────────────────────────────────

def parse_summ_label(raw: str) -> str:
    """
    Extract yes/no from a CoT response.
    CoT adds reasoning before the answer, so look at the LAST yes/no occurrence,
    which is the final verdict rather than an intermediate analysis mention.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Look for explicit final-answer line patterns
    for line in reversed(text.splitlines()):
        line = line.strip().lower()
        if line in ("yes", "no"):
            return line
        if line.startswith("yes"):
            return "yes"
        if line.startswith("no"):
            return "no"
        if "final answer" in line or "step 3" in line:
            if "yes" in line:
                return "yes"
            if "no" in line:
                return "no"

    # Last resort: last word-boundary yes/no in the whole response
    matches = list(re.finditer(r'\b(yes|no)\b', text.lower()))
    if matches:
        return matches[-1].group(1)

    # Search inside think tags if present
    think = " ".join(re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)).lower()
    matches = list(re.finditer(r'\b(yes|no)\b', think))
    if matches:
        return matches[-1].group(1)

    print(f"  [WARN] Could not parse label, defaulting to yes")
    return "yes"


def parse_reasoning_label(raw: str) -> str:
    """
    Extract Yes/No from a CoT reasoning response.
    Look for the JSON verdict at the end of the response.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try the last JSON object in the response (CoT outputs reasoning first)
    json_matches = list(re.finditer(r'\{[^{}]*"is_hallucinated"[^{}]*\}', text, re.IGNORECASE))
    if json_matches:
        last_json = json_matches[-1].group(0)
        try:
            obj = json.loads(last_json)
            val = obj.get("is_hallucinated", "").strip().lower()
            if val in ("yes", "no"):
                return val.capitalize()
        except Exception:
            pass
        m = re.search(r'"is_hallucinated"\s*:\s*"(Yes|No)"', last_json, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # Try full JSON parse on cleaned text
    try:
        obj = json.loads(text)
        val = obj.get("is_hallucinated", "").strip().lower()
        if val in ("yes", "no"):
            return val.capitalize()
    except Exception:
        pass

    # Look for conclusion in last few lines
    for line in reversed(text.splitlines()):
        line_l = line.strip().lower()
        if "step 3" in line_l or "final" in line_l or "hallucinated" in line_l:
            if "yes" in line_l and "not" not in line_l:
                return "Yes"
            if "no" in line_l or "not hallucinated" in line_l:
                return "No"

    # Last word-boundary yes/no
    matches = list(re.finditer(r'\b(yes|no)\b', text.lower()))
    if matches:
        return matches[-1].group(1).capitalize()

    print(f"  [WARN] Could not parse label, defaulting to Yes")
    return "Yes"

# ── Generic CSV task runner ────────────────────────────────────────────────────

def run_csv_task(
    input_file: str,
    output_file: str,
    build_prompt_fn,
    parse_fn,
    model: str,
    base_url: str,
    num_predict: int = 512,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed: set = set()
    if Path(output_file).exists():
        with open(output_file, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id")
                lbl = r.get("is_hallucinated", "")
                if sid and lbl in ("yes", "no", "Yes", "No"):
                    completed.add(sid)
        print(f"  Resuming — {len(completed)} rows already labeled.")

    pending = [
        (i, r) for i, r in enumerate(rows)
        if (r.get("id") or r.get("source_id") or str(i)) not in completed
    ]
    print(f"  Pending: {len(pending)}")

    write_header = not Path(output_file).exists() or Path(output_file).stat().st_size == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in pending:
            sid = r.get("id") or r.get("source_id") or str(i)
            prompt = build_prompt_fn(r)
            raw = call_ollama(prompt, model, base_url, num_predict)
            label = parse_fn(raw)
            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            print(f"  {i}: {sid} -> {label}")

    print(f"  Saved → {output_file}\n")


def run_reasoning_task(
    input_file: str,
    output_file: str,
    is_groundtruth: bool,
    model: str,
    base_url: str,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["chain"] = df["answer"]
        df["ans_col"] = ""
    else:
        df["chain"] = df["hallucinated_chain"]
        df["ans_col"] = df["hallucinated_answer"]

    if os.path.exists(output_file):
        print(f"  Resuming from {output_file}...")
        ckpt = pd.read_csv(output_file)
        df = ckpt
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].index.tolist()
    print(f"  Total: {len(df)} | Pending: {len(pending)}")

    chain_col = "chain" if "chain" in df.columns else "hallucinated_chain"
    ans_col = "ans_col" if "ans_col" in df.columns else "hallucinated_answer"

    for idx in pending:
        prompt = REASONING_COT_PROMPT.format(
            question=df.at[idx, "question"],
            chain=df.at[idx, chain_col],
            answer=df.at[idx, ans_col],
        )
        raw = call_ollama(prompt, model, base_url, num_predict=512)
        label = parse_reasoning_label(raw)
        df.at[idx, "is_hallucinated"] = label
        print(f"  {idx}: {df.at[idx, 'id']} -> {label}")
        df.to_csv(output_file, index=False)

    print(f"  Saved → {output_file}\n")

# ── Task definitions ───────────────────────────────────────────────────────────

def get_tasks(model: str, slug: str, base_url: str) -> dict:
    return {
        "qa_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            f"QA/Results/qa_cot_hallu_{slug}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("hallucinated_answer", ""),
            ),
            parse_summ_label,
            model, base_url, num_predict=512,
        ),
        "qa_gt": lambda: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            f"QA/Results/qa_cot_gt_{slug}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("right_answer", ""),
            ),
            parse_summ_label,
            model, base_url, num_predict=512,
        ),
        "summ_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/summarization_3000_corrected.csv",
            f"Summarization/Evaluation_Results/summ_3000_cot_{slug}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("document", "") or r.get("question", ""),
                summary=r.get("hallucinated_summary", "") or r.get("summary", ""),
            ),
            parse_summ_label,
            model, base_url, num_predict=512,
        ),
        "summ_gt": lambda: run_csv_task(
            "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
            f"Summarization/Evaluation_Results/summ_1000_cot_{slug}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("question", ""),
                summary=r.get("summary", ""),
            ),
            parse_summ_label,
            model, base_url, num_predict=512,
        ),
        "reason_hallu": lambda: run_reasoning_task(
            "Reasoning/1000_hallucinated Samples/somadhan_1000_hallucinated.csv",
            f"Reasoning/Results/reasoning_cot_{slug}.csv",
            is_groundtruth=False,
            model=model, base_url=base_url,
        ),
        "reason_gt": lambda: run_reasoning_task(
            "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
            f"Reasoning/Results/reasoning_gt_cot_{slug}.csv",
            is_groundtruth=True,
            model=model, base_url=base_url,
        ),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["all", "qa_hallu", "qa_gt", "summ_hallu", "summ_gt", "reason_hallu", "reason_gt"],
        default="all",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run specific models e.g. --models qwen2.5:32b-instruct gemma2:27b",
    )
    p.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL. Falls back to OLLAMA_BASE_URL env var, then http://localhost:11434",
    )
    args = p.parse_args()

    base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    print(f"Ollama at: {base_url}\n")

    models_to_run = ALL_MODELS
    if args.models:
        models_to_run = [(m, s) for m, s in ALL_MODELS if m in args.models]

    for model, slug in models_to_run:
        tasks = get_tasks(model, slug, base_url)
        to_run = list(tasks.keys()) if args.task == "all" else [args.task]

        for task in to_run:
            print(f"\n{'='*60}")
            print(f"Model: {model} | Task: {task}")
            print(f"{'='*60}")
            tasks[task]()

    print("All done.")


if __name__ == "__main__":
    main()
