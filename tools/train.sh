# Preload datasets

set -e  # Exit on error
set -o pipefail  # Fail on first command failure in a pipeline

log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $1"
}

error_exit() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') [ERROR] $1" >&2
    exit 1
}

# Determine which datasets to preload based on environment variables
log "Preloading datasets into memory ..."
if [[ "$PRELOAD_COCO" == "1" ]]; then
    SPLIT="$COCO_SPLIT" /home/skhan/bin/preload_coco.sh
fi
if [[ "$PRELOAD_CROWDPOSE" == "1" ]]; then
    /home/skhan/bin/preload_crowdpose.sh
fi
if [[ "$PRELOAD_MPII" == "1" ]]; then
    /home/skhan/bin/preload_mpii.sh
fi
if [[ "$PRELOAD_DSV" == "1" ]]; then
    /home/skhan/bin/preload_dsv.sh
fi
if [[ "$PRELOAD_EXLPOSE" == "1" ]]; then
    /home/skhan/bin/preload_exlpose.sh
fi

# Train
log "Training ..."
args=("$@")
configs=()
optional_args=()

# Separate configs and optional arguments
for arg in "${args[@]}"; do
    if [[ "$arg" == --launcher* ]]; then
        log "Distributed training is disabled. Using a single GPU with multiple processes due to port binding issues caused by the SLURM launcher."
        continue
    elif [[ "$arg" == --* ]]; then
        optional_args+=("$arg")
    else
        configs+=("$arg")
    fi
done

timestamp=$(date +'%Y%m%d_%H%M%S')
mkdir -p "work_dirs/$timestamp/logs"

if [[ "${#configs[@]}" -eq 1 ]]; then
    config="${configs[0]}"
    log_file="work_dirs/$timestamp/logs/$(basename "$config").log"
    log "Training started for $config with optional arguments: ${optional_args[*]}. Logs: $log_file"
    python -u tools/train.py "$config" --work-dir "$timestamp" "${optional_args[@]}" > "$log_file" 2>&1
    log "Training for $config completed."
else
    for config in "${configs[@]}"; do
        log_file="work_dirs/$timestamp/logs/$(basename "$config").log"
        python -u tools/train.py "$config" --work-dir "$timestamp" "${optional_args[@]}" > "$log_file" 2>&1 &
        log "Training started for $config with optional arguments: ${optional_args[*]}. Logs: $log_file"
    done
    log "Started all training processes. Waiting for them to complete..."
    wait
    log "All training processes completed."
fi