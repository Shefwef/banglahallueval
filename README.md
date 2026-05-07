Label 400 samples with yes/no hallucination using local Ollama Qwen model

This repository script helps you label model outputs as hallucinated or not using a locally hosted Ollama model (for example `qwen-2.5-32b`).

Files added:
- scripts/label_hallucinations_ollama.py  : main script
- requirements.txt                       : minimal Python deps

Quick steps:
1. Install Python deps: pip install -r requirements.txt
2. Ensure Ollama is running locally with the desired model (default HTTP at http://localhost:11434 and model name `qwen-2.5-32b`).
3. Run the script:
   python scripts/label_hallucinations_ollama.py --input Results/hallucinated_answers_generation_qa.csv \
       --output Results/hallucinated_answers_generation_qa_with_labels.csv --model qwen-2.5-32b --start 0 --end 400

If you don't have the HTTP API, the script will try to call the `ollama` CLI as a fallback.

Notes:
- The model is instructed to answer with a single token: `yes` or `no`.
- The script maps common Bengali yes/no tokens to English `yes`/`no`.
- Use --dry-run to validate parsing without calling the model.
