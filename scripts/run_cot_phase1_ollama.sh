#!/usr/bin/env bash
# Phase 1: CoT eval for the Ollama models. Writes logs/phase1.done ONLY on a
# clean exit, which is the signal Phase 2 (TigerLLM) waits for. Running this
# again after a crash/reboot just resumes from the CSVs.
set -u
cd "/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval"
mkdir -p logs
LOG="logs/cot_ollama_run.log"
rm -f logs/phase1.done

echo "[phase1] $(date) starting (concurrency=2)" | tee -a "$LOG"
python -u scripts/evaluate_cot_ollama.py \
    --task all \
    --models qwen2.5:32b-instruct gemma2:27b deepseek-r1:14b \
    --concurrency 2 >> "$LOG" 2>&1
rc=$?
echo "[phase1] $(date) finished rc=$rc" | tee -a "$LOG"
[ $rc -eq 0 ] && touch logs/phase1.done && echo "[phase1] wrote logs/phase1.done" | tee -a "$LOG"
