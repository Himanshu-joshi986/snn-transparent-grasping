#!/bin/bash
#
# scripts/setup_environment.sh
#
# Complete environment setup for SNN-DTA project
# Installs conda environment, PyTorch, dependencies, and v2e
#
# Usage:
#   bash scripts/setup_environment.sh
#   bash scripts/setup_environment.sh --gpu cuda11.8
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  SNN-DTA Environment Setup Script                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# Parse arguments
GPU_VERSION="11.8"
if [ "$1" == "--gpu" ]; then
    GPU_VERSION="$2"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo -e "${YELLOW}Project Root: ${PROJECT_ROOT}${NC}"
echo -e "${YELLOW}CUDA Version: ${GPU_VERSION}${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 1: Create conda environment
# ─────────────────────────────────────────────────────────────────

echo -e "\n${GREEN}[1/5] Creating conda environment...${NC}"
if conda env list | grep -q "snn-grasp"; then
    echo "  Environment 'snn-grasp' already exists. Removing..."
    conda env remove -n snn-grasp -y
fi

conda create -n snn-grasp python=3.10 -y
echo -e "${GREEN}✓ Conda environment created${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 2: Install PyTorch
# ─────────────────────────────────────────────────────────────────

echo -e "\n${GREEN}[2/5] Installing PyTorch...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snn-grasp

case "${GPU_VERSION}" in
    "11.8")
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        ;;
    "12.1")
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        ;;
    "cpu")
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ;;
    *)
        echo -e "${RED}Unknown CUDA version: ${GPU_VERSION}${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✓ PyTorch installed${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 3: Install core dependencies
# ─────────────────────────────────────────────────────────────────

echo -e "\n${GREEN}[3/5] Installing core dependencies...${NC}"
cd "${PROJECT_ROOT}"
pip install -q -r requirements.txt

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 4: Install v2e from source
# ─────────────────────────────────────────────────────────────────

echo -e "\n${GREEN}[4/5] Installing v2e (event simulator)...${NC}"

V2E_DIR="/tmp/v2e_install"
if [ -d "${V2E_DIR}" ]; then
    rm -rf "${V2E_DIR}"
fi

git clone -q https://github.com/SensorsINI/v2e.git "${V2E_DIR}"
cd "${V2E_DIR}"
pip install -q -e .
cd "${PROJECT_ROOT}"

echo -e "${GREEN}✓ v2e installed${NC}"

# ─────────────────────────────────────────────────────────────────
# Step 5: Verification
# ─────────────────────────────────────────────────────────────────

echo -e "\n${GREEN}[5/5] Verifying installation...${NC}"

python << 'PYEOF'
import sys
import importlib

packages = ['torch', 'spikingjelly', 'cv2', 'pybullet', 'v2e', 'yaml']
failures = []

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} FAILED")
        failures.append(pkg)

if failures:
    print(f"\n{len(failures)} package(s) failed to import!")
    sys.exit(1)
else:
    print("\n✓ All packages verified!")
PYEOF

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete! ✓                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "  1. Activate environment: conda activate snn-grasp"
echo "  2. Download dataset: python scripts/download_cleargrasp.py"
echo "  3. Generate events: python utils/generate_events.py"
echo "  4. Train model: python training/train.py --model dta"
echo ""
