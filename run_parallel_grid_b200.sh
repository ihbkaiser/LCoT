#!/usr/bin/env bash
set -u

# Launch the six ProsQA QAT configurations concurrently, round-robin across
# the visible GPUs. With two GPUs this starts three one-process jobs per GPU.

CONFIG_SPEC="${CONFIG_SPEC:-args/prosqa_finite_state_grid_b200.yaml}"
CONFIG_DIR="${CONFIG_DIR:-results/prosqa_grid_b200/configs}"
LOG_DIR="${LOG_DIR:-results/prosqa_grid_b200/logs}"
GPU_IDS="${GPU_IDS:-0,1}"
BASE_PORT="${BASE_PORT:-29600}"

mkdir -p "$LOG_DIR"

# Generate deterministic per-run YAML files. This does not execute training.
python experiments/run_prosqa_grid.py "$CONFIG_SPEC" --dry-run

mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
if [ "${#configs[@]}" -ne 6 ]; then
    echo "Expected 6 generated configs in $CONFIG_DIR, found ${#configs[@]}" >&2
    exit 1
fi

IFS=',' read -r -a gpus <<< "$GPU_IDS"
if [ "${#gpus[@]}" -lt 1 ]; then
    echo "GPU_IDS must contain at least one GPU id" >&2
    exit 1
fi

run_one() {
    local gpu="$1"
    local config="$2"
    local index="$3"
    local stem
    local port
    stem="$(basename "$config" .yaml)"
    port=$((BASE_PORT + index))

    echo "Launching $stem on physical GPU $gpu (port $port)"

    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    python -m torch.distributed.run \
        --rdzv_backend=c10d \
        --rdzv_endpoint="127.0.0.1:$port" \
        --rdzv_id="prosqa-$index" \
        --nnodes=1 \
        --nproc_per_node=1 \
        run.py "$config" \
        > "$LOG_DIR/${stem}.gpu${gpu}.log" 2>&1
}

pids=()
for index in "${!configs[@]}"; do
    gpu="${gpus[$((index % ${#gpus[@]}))]}"
    run_one "$gpu" "${configs[$index]}" "$index" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "At least one grid configuration failed. See logs in $LOG_DIR." >&2
fi
exit "$status"
