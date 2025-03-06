API_URL="http://localhost:8000/v1"
BENCHMARK_PATH="../benchmarks/benchmark-difficult"
OUTPUT_PATH="runs-benchmark-difficult"

for i in \
"llama-2-7B-chat meta-llama/Llama-2-7B-chat-hf" \
"llama-2-13B-chat meta-llama/Llama-2-13B-chat-hf" \
"llama-3-8B-instruct meta-llama/Meta-Llama-3-8B-Instruct" \
"llama-3.1-8B-instruct meta-llama/Meta-Llama-3.1-8B-Instruct" \
"llama-3.2-1B-instruct meta-llama/Llama-3.2-1B-Instruct" \
"llama-3.2-3B-instruct meta-llama/Llama-3.2-3B-Instruct" \
"llama-2-70B-chat meta-llama/Llama-2-70B-chat-hf" \
"llama-3-70B-instruct meta-llama/Meta-Llama-3-70B-Instruct" \
"llama-3.1-70B-instruct meta-llama/Meta-Llama-3.1-70B-Instruct" \
"llama-3.3-70B-instruct meta-llama/Llama-3.3-70B-Instruct" \
; do
    set -- $i # Convert the "tuple" into the param args $1 $2...
    MODEL_SHORT="$1"
    MODEL="$2"
    IMAGE="vllm/vllm-openai:v0.6.1.post1"

    VLLM_ARGS="--model $MODEL --enforce-eager"
    if [[ "$MODEL" == meta-llama/Meta-Llama-3.1-70B-Instruct || "$MODEL" == meta-llama/Llama-3.3-70B-Instruct ]]; then
        if [[ "$MODEL" == meta-llama/Llama-3.3-70B-Instruct ]]; then
            IMAGE="vllm/vllm-openai:v0.6.2"
        fi
        # For llama-3.1-70B, llama-3.3-70B
        TIMEOUT=150
        VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 40960"
    elif [[ "$MODEL" == meta-llama/Meta-Llama-3-70B-Instruct || "$MODEL" == meta-llama/Llama-2-70B-chat-hf ]]; then
        # For llama-2-70B, llama-3-70B
        TIMEOUT=150
        VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size 2 --gpu-memory-utilization 0.95"
    else
        # For llama-2-7B, llama-2-13B, llama-3-8B, llama-3.1-8B, llama-3.2-1B, llama-3.2-3B
        TIMEOUT=60
    fi
    echo "Starting container:"
    docker run --detach --rm --name vllm --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN="$HF_TOKEN" -p 8000:8000 --ipc=host $IMAGE $VLLM_ARGS

    for i in $(seq 1 $TIMEOUT); do
        sleep 1
        printf "\rWaiting $i/$TIMEOUT seconds for server to start"
    done
    echo

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
        if [[ -e "$OUTPUT_PATH/$MODEL_SHORT-$PROMPT" ]]; then
            echo "Skipping existing run: $MODEL_SHORT-$PROMPT"
            echo
            continue
        fi
        echo $MODEL_SHORT
        echo $MODEL
        echo $PROMPT
        echo
        python src/run.py \
        --prompt_file prompts/$PROMPT.json \
        --benchmark_tsv $BENCHMARK_PATH/screens.tsv \
        --screens_dir $BENCHMARK_PATH \
        --outputs_dir $OUTPUT_PATH/$MODEL_SHORT-$PROMPT \
        --api_url "$API_URL" \
        --api_key EMPTY \
        --model $MODEL \
        --num_parallel_requests 32

        if [[ $? -ne 0 ]]; then
            echo "Run failed"
            docker stop vllm
            exit 1
        fi
    done

    echo "Stopping container:"
    docker stop vllm
done
