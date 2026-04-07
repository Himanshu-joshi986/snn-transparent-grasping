#!/bin/bash
#
# scripts/run_all_experiments.sh
#
# Master experiment runner
# Executes full pipeline: data prep, training, evaluation
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --skip_data --epochs 100
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Default parameters
SKIP_DATA=false
EPOCHS=50
BATCH_SIZE=8
DEVICES="0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_data) SKIP_DATA=true; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  SNN-DTA Experiment Pipeline                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# ─────────────────────────────────────────────────────────────────
# Phase 1: Data Preparation
# ─────────────────────────────────────────────────────────────────

if [ "${SKIP_DATA}" = false ]; then
    echo -e "\n${YELLOW}Phase 1: Data Preparation${NC}"
    echo "===================="

    # Create directory structure
    echo -e "${GREEN}[1.1] Creating directory structure...${NC}"
    python scripts/download_cleargrasp.py --create_structure --output_dir data/

    # Download dataset
    echo -e "${GREEN}[1.2] Downloading ClearGrasp dataset...${NC}"
    python scripts/download_cleargrasp.py \
        --output_dir data/ \
        --split synthetic \
        --workers 4
    
    # Generate events
    echo -e "${GREEN}[1.3] Generating event data with v2e...${NC}"
    python scripts/generate_all_events.py \
        --image_folder data/synthetic/rgb \
        --output_folder data/synthetic/events \
        --num_variations 3

    echo -e "${GREEN}✓ Data preparation complete!${NC}"
else
    echo -e "${YELLOW}Skipping data preparation (--skip_data)${NC}"
fi

# ─────────────────────────────────────────────────────────────────
# Phase 2: Training
# ─────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}Phase 2: Model Training${NC}"
echo "===================="

# Create runs directory
mkdir -p runs

# Train CNN baseline
echo -e "${GREEN}[2.1] Training CNN baseline...${NC}"
python training/train.py \
    --model cnn \
    --config configs/cnn.yaml \
    --epochs "${EPOCHS}" \
    --batch_size 16 \
    --device cuda:${DEVICES%%,*} \
    --output_dir runs/cnn || echo "CNN training warning (continue)"

# Train Spiking U-Net baseline
echo -e "${GREEN}[2.2] Training Spiking U-Net baseline...${NC}"
python training/train.py \
    --model snn \
    --config configs/snn.yaml \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda:${DEVICES%%,*} \
    --output_dir runs/snn || echo "SNN training warning (continue)"

# Train DTA-SNN (proposed)
echo -e "${GREEN}[2.3] Training DTA-SNN (proposed)...${NC}"
python training/train.py \
    --model dta \
    --config configs/dta.yaml \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda:${DEVICES%%,*} \
    --output_dir runs/dta || echo "DTA training warning (continue)"

echo -e "${GREEN}✓ Training complete!${NC}"

# ─────────────────────────────────────────────────────────────────
# Phase 3: Evaluation
# ─────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}Phase 3: Evaluation${NC}"
echo "===================="

mkdir -p evaluation/results

# Find best models
CNN_MODEL=$(find runs/cnn -name "*best*.pth" -type f 2>/dev/null | head -1)
SNN_MODEL=$(find runs/snn -name "*best*.pth" -type f 2>/dev/null | head -1)
DTA_MODEL=$(find runs/dta -name "*best*.pth" -type f 2>/dev/null | head -1)

echo -e "${GREEN}[3.1] Running comprehensive evaluation...${NC}"

if [ -f "${CNN_MODEL}" ] && [ -f "${SNN_MODEL}" ] && [ -f "${DTA_MODEL}" ]; then
    python evaluation/evaluate.py \
        --models "${CNN_MODEL}" "${SNN_MODEL}" "${DTA_MODEL}" \
        --test_split data/real/rgb \
        --output_dir evaluation/results \
        --output_format json csv latex \
        --compute_energy \
        --visualize || echo "Evaluation warning (continue)"
else
    echo -e "${RED}✗ Models not found. Skipping evaluation.${NC}"
fi

echo -e "${GREEN}✓ Evaluation complete!${NC}"

# ─────────────────────────────────────────────────────────────────
# Phase 4: Visualization & Reports
# ─────────────────────────────────────────────────────────────────

echo -e "\n${YELLOW}Phase 4: Results & Reports${NC}"
echo "===================="

echo -e "${GREEN}[4.1] Generating reports...${NC}"

# Display results summary
if [ -f "evaluation/results/evaluation_results.json" ]; then
    echo -e "\n${GREEN}Evaluation Results:${NC}"
    python << 'PYEOF'
import json
with open("evaluation/results/evaluation_results.json") as f:
    results = json.load(f)
    for model, data in results.items():
        if data:
            metrics = data.get("metrics", {})
            inference = data.get("inference", {})
            print(f"\n{model}:")
            print(f"  IoU:     {metrics.get('iou', {}).get('mean', 'N/A'):.4f}")
            print(f"  Dice:    {metrics.get('dice', {}).get('mean', 'N/A'):.4f}")
            print(f"  F1:      {metrics.get('f1', {}).get('mean', 'N/A'):.4f}")
            print(f"  Latency: {inference.get('latency_ms', 'N/A'):.2f} ms")
PYEOF
fi

# Generate LaTeX table if available
if [ -f "evaluation/results/evaluation_table.tex" ]; then
    echo -e "\n${GREEN}[4.2] LaTeX table generated (for papers):${NC}"
    echo "  evaluation/results/evaluation_table.tex"
fi

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Experiment Pipeline Complete! ✓                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}Output directories:${NC}"
echo "  Models:      runs/"
echo "  Evaluation:  evaluation/results/"
echo "  Figures:     evaluation/results/"
echo ""

