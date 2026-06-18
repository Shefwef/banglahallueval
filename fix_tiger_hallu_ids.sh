#!/bin/bash

TIGER="/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval/QA/Results/qa_cot_hallu_tigerllm_9b.csv"
LLAMA="/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval/QA/Results/qa_cot_hallu_llama3_1_8b.csv"
OUT="/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval/QA/Results/qa_cot_hallu_tigerllm_9b_fixed.csv"
BACKUP="/home/bio/Desktop/Thesis-401/cot phase/BanglaHalluEval/QA/Results/qa_cot_hallu_tigerllm_9b_backup.csv"

echo "=== Step 1: Count lines ==="
echo "Tiger lines (incl header): $(wc -l < "$TIGER")"
echo "Llama lines (incl header): $(wc -l < "$LLAMA")"

echo ""
echo "=== Step 2: Extract reference IDs from Llama file (col 1, skip header) ==="
# Save llama IDs to a temp file
awk -F',' 'NR>1 {print $1}' "$LLAMA" | sort -u > /tmp/ref_ids.txt
echo "Unique reference IDs from Llama: $(wc -l < /tmp/ref_ids.txt)"

echo ""
echo "=== Step 3: Count unique IDs in Tiger file ==="
awk -F',' 'NR>1 {print $1}' "$TIGER" | sort -u > /tmp/tiger_ids_all.txt
echo "Unique IDs in Tiger: $(wc -l < /tmp/tiger_ids_all.txt)"

echo ""
echo "=== Step 4: IDs in Tiger that are in Llama reference ==="
comm -12 /tmp/tiger_ids_all.txt /tmp/ref_ids.txt | wc -l | xargs echo "Matched IDs:"

echo ""
echo "=== Step 5: IDs in Tiger NOT in Llama reference (extras to be removed) ==="
comm -23 /tmp/tiger_ids_all.txt /tmp/ref_ids.txt | wc -l | xargs echo "Extra IDs not in reference:"

echo ""
echo "=== Step 6: IDs in Llama reference NOT in Tiger (still missing) ==="
comm -13 /tmp/tiger_ids_all.txt /tmp/ref_ids.txt | wc -l | xargs echo "Missing IDs from tiger:"
comm -13 /tmp/tiger_ids_all.txt /tmp/ref_ids.txt > /tmp/missing_ids.txt

echo ""
echo "=== Step 7: Filtering Tiger file - keep only ref IDs, first occurrence only ==="
# Use awk: load ref IDs, then filter tiger keeping first occurrence per ID
awk -F',' '
    NR==FNR { ref[$1]=1; next }
    FNR==1  { print; next }
    ($1 in ref) && !seen[$1]++ { print }
' /tmp/ref_ids.txt "$TIGER" > "$OUT"

OUT_LINES=$(wc -l < "$OUT")
echo "Output file rows (incl header): $OUT_LINES"
echo "Output data rows: $((OUT_LINES - 1))"

echo ""
echo "=== Step 8: Backup original and replace ==="
cp "$TIGER" "$BACKUP"
echo "Backup saved to: $BACKUP"
mv "$OUT" "$TIGER"
echo "Tiger file updated: $TIGER"

echo ""
echo "=== DONE ==="
echo "Final tiger file rows (incl header): $(wc -l < "$TIGER")"
