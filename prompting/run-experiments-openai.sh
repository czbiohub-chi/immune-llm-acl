BENCHMARK_PATH="../benchmarks/benchmark-difficult"
OUTPUT_PATH="runs-benchmark-difficult"

for MODEL in \
o1-2024-12-17 \
o1-mini-2024-09-12 \
gpt-4o-2024-11-20 \
gpt-4o-mini-2024-07-18 \
gpt-4-turbo-2024-04-09 \
gpt-4-0125-preview \
gpt-3.5-turbo-0125 \
; do
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
        if [[ -e "$OUTPUT_PATH/$MODEL-$PROMPT" ]]; then
            echo "Skipping existing run: $MODEL-$PROMPT"
            echo
            continue
        elif [[ "$MODEL" == o1-* && "$PROMPT" == *-cot ]]; then
            echo "Skipping CoT run for o1: $MODEL-$PROMPT"
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
        --api_url 'https://api.openai.com/v1' \
        --api_key "$OPENAI_API_KEY" \
        --num_parallel_requests 4 \
        --model $MODEL

        if [[ $? -ne 0 ]]; then
            echo "Run failed"
            exit 1
        fi
    done
done

