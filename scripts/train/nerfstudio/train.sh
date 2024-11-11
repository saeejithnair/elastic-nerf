#!/bin/bash

# Function to display an error message and exit
function error_exit {
    echo "Error: $1" >&2
    exit 1
}

# Function to forward SIGINT to all subprocesses
function forward_signal {
    local signal=$1
    echo "Forwarding signal $signal to subprocesses..."
    kill -s $signal -- -$$
}

# Function to get GPU memory size in GB
get_gpu_memory_size() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print int($1/1024)}'
}

# Trap SIGINT and forward it to all subprocesses
trap 'forward_signal SIGINT' SIGINT

# Initialize variable to hold the value of --max-num-iterations
max_num_iterations=0

# Parse mandatory arguments and the new --max-num-iterations argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method=*) method="${1#*=}" ;;
        --scene=*) scene="${1#*=}" ;;
        --dataset=*) dataset="${1#*=}" ;;
        --max-num-iterations=*) max_num_iterations="${1#*=}" ;;
        *) extra_args+=("$1") ;;  # Collect variable number of arguments
    esac
    shift  # Shift to the next argument
done

if [[ -f .env ]]; then
    source .env
else
    echo "Error: .env file not found" >&2
    exit 1
fi


# Check if mandatory arguments are provided
[[ -z $method || -z $scene || -z $dataset ]] && \
    error_exit "Missing mandatory arguments."

# Check if mandatory environment variables have been set
[[ -z $NERFSTUDIO_CACHE_DIR ]] && \
    error_exit "Environment variable NERFSTUDIO_CACHE_DIR has not been set."

# If NERFSTUDIO_CACHE_DIR does not exist, throw an error
[[ ! -d $NERFSTUDIO_CACHE_DIR ]] && \
    error_exit "Directory $NERFSTUDIO_CACHE_DIR does not exist."

# Compose the dataset directory path
export PATH_TO_SCENE_DIR="$NERFSTUDIO_CACHE_DIR/data/$dataset/$scene"

# Ensure save directory exists before downloading data
PATH_TO_DATASET_DIR="$NERFSTUDIO_CACHE_DIR/data"
[[ ! -d $PATH_TO_SCENE_DIR ]] && {
    mkdir -p "$PATH_TO_DATASET_DIR"
    echo "Downloading data for $dataset/$scene"
    # Conditional check for dataset type to decide on passing --capture-name
    if [[ $dataset == "nerfstudio" ]]; then
        ns-download-data $dataset --capture-name=$scene --save-dir="$PATH_TO_DATASET_DIR"
    elif [[ $dataset == "blender" ]]; then
        ns-download-data $dataset --save-dir="$PATH_TO_DATASET_DIR"
    else
        error_exit "Unknown dataset type for ns-download-data command."
    fi
    if [[ $? -ne 0 ]]; then
        echo "ns-download-data command failed, removing $PATH_TO_SCENE_DIR"
        rm -rf "$PATH_TO_SCENE_DIR"
        exit 1
    fi
}

# Compose the output directory path
EXPERIMENT_NAME="${method}_${scene}"
export PATH_TO_OUTPUT_DIR="$NERFSTUDIO_CACHE_DIR/output/$dataset/$EXPERIMENT_NAME"

# Set default max_num_iterations based on the method
case $method in
    nerfacto) default_max_num_iterations=30000 ;;
    nerfacto-big) default_max_num_iterations=100000 ;;
    nerfacto-huge) default_max_num_iterations=100000 ;;
    *) default_max_num_iterations=0 ;;
esac

# If max_num_iterations is zero (was not passed as argument), set it to the default value based on the method
if [[ $max_num_iterations -eq 0 ]]; then
    max_num_iterations=$default_max_num_iterations
fi

# Check if --max-num-iterations was provided and calculate
# --steps-per-eval-all-images value

# Check if --steps-per-eval-all-images is passed in as an extra arg
steps_per_eval_all_images_passed=false
for arg in "${extra_args[@]}"; do
    if [[ $arg == --steps-per-eval-all-images=* ]]; then
        steps_per_eval_all_images_passed=true
        break
    fi
done

# If --max-num-iterations was provided and
# --steps-per-eval-all-images was not passed as an extra arg
# then calculate --steps-per-eval-all-images value and add to extra_args
if [[ $max_num_iterations -ne 0 && $steps_per_eval_all_images_passed == false ]]; then
    max_num_iterations_updated=$((max_num_iterations + 1))
    extra_args+=(
         \
        "--steps-per-eval-all-images=$max_num_iterations"
    )
    max_num_iterations=$((max_num_iterations_updated))
fi


# Get GPU memory size
gpu_memory_size=$(get_gpu_memory_size)
gpu_memory_size=${gpu_memory_size//[!0-9]/}  # Remove non-numeric characters
# Check if the GPU memory size is more than 32G and append additional arguments
if [[ $gpu_memory_size -gt 32 ]]; then
    extra_args+=("--pipeline.datamanager.images-on-gpu=True")
    extra_args+=("--pipeline.datamanager.masks-on-gpu=True")
else
    extra_args+=("--pipeline.datamanager.images-on-gpu=False")
    extra_args+=("--pipeline.datamanager.masks-on-gpu=False")
fi

# Build the argument list for train.py
args=(
    $method
    --data="$PATH_TO_SCENE_DIR"
    --output-dir="$PATH_TO_OUTPUT_DIR"
    --experiment-name="$EXPERIMENT_NAME"
    --logging.profiler=none
    --vis=wandb
    --logging.local-writer.enable=False
    --max-num-iterations=$max_num_iterations
    "${extra_args[@]}"
    "$dataset"-data
)

# Build the command string for train.py
cmd="python src/gen-nerf/scripts/train.py ${args[@]}"

# Echo the command string
echo "Running command: $cmd"

# Run the train.py script with the argument list
# Use "exec" to replace the shell with the python process
exec $cmd
