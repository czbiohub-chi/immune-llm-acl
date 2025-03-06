API_URL="https://hierophant.prod-ml-platform.prod.czi.team/v1/api/llm/databricks"
API_KEY=""
MODEL="databricks-meta-llama-3-1-405b-instruct"
BENCHMARK_PATH="../benchmarks/benchmark-difficult"
OUTPUT_PATH="runs-benchmark-difficult"

for PROMPT in \
0-shot \
0-shot-cot \
1-shot-pos \
1-shot-pos-cot \
1-shot-neg \
1-shot-neg-cot \
2-shot-pos-neg \
2-shot-pos-neg-cot \
2-shot-neg-pos \
2-shot-neg-pos-cot \
; do
    if [[ -e "runs/$MODEL-$PROMPT" ]]; then
        echo "Skipping existing run: $MODEL-$PROMPT"
        echo
        continue
    fi
    echo $MODEL
    echo $PROMPT
    echo
    python src/run.py \
    --prompt_file prompts/$PROMPT.json \
    --benchmark_tsv $BENCHMARK_PATH/screens.tsv \
    --screens_dir $BENCHMARK_PATH \
    --outputs_dir $OUTPUT_PATH/$MODEL-$PROMPT \
    --api_url "$API_URL" \
    --api_key "$API_KEY" \
    --model "$MODEL" \
    --num_parallel_requests 0

    if [[ $? -ne 0 ]]; then
        echo "Run failed"
        exit 1
    fi
done
