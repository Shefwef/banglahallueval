#!/usr/bin/env bash
# Phase 2 launcher: waits for the Phase 1 Ollama CoT run to finish (so TigerLLM
# gets the 24 GB GPU to itself), then runs the TigerLLM CoT evaluation.
set -u

cd "/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval"
PY="$HOME/anaconda3/envs/attention/bin/python"
LOG="logs/cot_tigerllm_run.log"
mkdir -p logs

echo "[phase2] $(date) waiting for Phase 1 (evaluate_cot_ollama.py) to finish..." | tee -a "$LOG"
while pgrep -f "evaluate_cot_ollama.py" >/dev/null 2>&1; do
    sleep 60
done
echo "[phase2] $(date) Phase 1 done. Starting TigerLLM CoT." | tee -a "$LOG"

"$PY" -u scripts/evaluate_cot_tigerllm.py --task all >> "$LOG" 2>&1
echo "[phase2] $(date) TigerLLM CoT finished (rc=$?)." | tee -a "$LOG"
