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

# === Results Directory ===
export RESULTS_SAVEDIR="${PROJECT_ROOT}/eval/gen_metrics/Results"
mkdir -p "$RESULTS_SAVEDIR"

# === Real Images Configuration ===
export REAL_CSV="${PROJECT_ROOT}/eval/gen_metrics/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="$REAL_IMG_DIR"

# === Experiment Configuration ===
export EXTRA_INFO="UniX"
export EXPERIMENT_TYPE="conditional"

# === Compute Settings ===
export NUM_SHARDS=-1
export SHARD=-1
export BATCH_SIZE=128
export NUM_WORKERS=4

MIMIC_PATHOLOGIES=("Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Enlarged Cardiomediastinum" "Fracture" "Lung Lesion" "Lung Opacity" "No Finding" "Pleural Effusion" "Pleural Other" "Pneumonia" "Pneumothorax" "Support Devices")

for pathology in "${MIMIC_PATHOLOGIES[@]}"; do
    export EXPERIMENT_TYPE="conditional"
    export PATHOLOGY=$pathology
    echo "Conditional Experiment for: '$PATHOLOGY'"
    echo "Calculating FID, KID, IS ..."
    scripts/fid.sh

    echo "Calculating Image Text Alignment Scores ..."
    scripts/img_text_alignment.sh
done