#!/bin/bash

# Parse arguments: -p project root, -r real images directory
while getopts "p:r:" opt; do
    case $opt in
        p) PROJECT_ROOT="$OPTARG" ;;
        r) REAL_IMG_DIR="$OPTARG" ;;
        *) echo "Usage: $0 -p <project_root> -r <real_images_dir>"
           exit 1 ;;
    esac
done

if [ -z "$PROJECT_ROOT" ]; then
    echo "Error: Project root directory (-p) is required"
    echo "Usage: $0 -p <project_root> -r <real_images_dir>"
    exit 1
fi

if [ -z "$REAL_IMG_DIR" ]; then
    echo "Error: Real images directory (-r) is required"
    echo "Usage: $0 -p <project_root> -r <real_images_dir>"
    exit 1
fi

# === Synthetic Images Configuration ===
export SYNTHETIC_CSV="${PROJECT_ROOT}/eval/data/generation/generations_with_metadata.csv"
export SYNTHETIC_IMG_DIR="${PROJECT_ROOT}/results/generation_images_c2.0"
export SYNTHETIC_PROMPT_COL="annotated_prompt"

# === Prompt Configuration ===
export TRAINING_PROMPT="Llavarad"
export EVAL_PROMPT="Llavarad"

# === Real Images Configuration ===
export REAL_CSV="${PROJECT_ROOT}/eval/gen_metrics/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="$REAL_IMG_DIR"

# === Experiment Configuration ===
export EXTRA_INFO="UniX"
export EXPERIMENT_TYPE="regular"

# === Results & Shards Directory ===
export RESULTS_SAVEDIR="${PROJECT_ROOT}/eval/gen_metrics/Results"
mkdir -p "$RESULTS_SAVEDIR"
export SHARDS_DIR="${RESULTS_SAVEDIR}/saved_shards"
mkdir -p "$SHARDS_DIR"

# === Compute Settings ===
export NUM_SHARDS=4
export SHARD=0
export BATCH_SIZE=128
export NUM_WORKERS=4

echo "Calculating regular metrics for $EXTRA_INFO"

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
    export SHARD=$shard
    echo "Calculating FID, KID, IS for shard $SHARD ..."
    scripts/fid.sh
done

# Combine the shards here
python tools/combine_shards.py --shards_dir=$SHARDS_DIR --extra_info=$EXTRA_INFO --output_dir=$RESULTS_SAVEDIR --delete_after_combining 

# Calculate img-text alignment scores
echo "Calculating Image Text Alignment Scores ..."
scripts/img_text_alignment.sh
