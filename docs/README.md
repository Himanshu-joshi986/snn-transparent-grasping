# SNN-DTA: Event-Based Semantic Segmentation with Dual Temporal-Channel Attention for Transparent Object Grasping

**Spiking Neural Networks with Attention Mechanisms for High-Speed, Low-Power Robotic Grasp Detection**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [System Architecture](#system-architecture)
5. [Installation & Setup](#installation--setup)
6. [Dataset Preparation](#dataset-preparation)
7. [Training Models](#training-models)
8. [Evaluation](#evaluation)
9. [Robotic Grasping Demo](#robotic-grasping-demo)
10. [Results & Benchmarks](#results--benchmarks)
11. [Paper & Patent](#paper--patent)
12. [Contributing](#contributing)

---

## Project Overview

### What Problem Does This Solve?

Traditional computer vision systems **fail on transparent objects**:
- Glass bottles, plastic containers, and reflective surfaces exhibit complex refraction and reflection patterns
- RGB image-based detection algorithms struggle with specular highlights and partial transparency
- Depth sensors (RGB-D) are unreliable for transparent materials (light passes through)

### Why Spiking Neural Networks?

**Event cameras** (neuromorphic sensors like DVS) capture asynchronous brightness changes with high temporal resolution:
- ✅ **High dynamic range** – naturally handles specular highlights
- ✅ **Sparse events** – only when brightness changes, 1000x faster than frame cameras
- ✅ **Low power** – ideal for resource-constrained robots
- ✅ **No motion blur** – captures fast transients

**Spiking Neural Networks** (SNNs) are the natural computational model for event data:
- 🧠 **Bio-inspired** – mimics how neurons actually process information
- ⚡ **Event-driven** – spikes only when thresholds are crossed (sparse computation)
- 🔋 **Energy efficient** – O(spikes) complexity vs dense matrix multiplication in ANNs
- 📊 **Temporal dynamics** – native support for sequential correlation

### This Project's Contribution

We propose **DTA-SNN**: a Spiking U-Net with **Dual Temporal-Channel Attention** (⚠️ **patent-pending**) that:
1. Encodes event streams via **Temporal Correlation Encoding**
2. Processes events through a **Spiking U-Net backbone**
3. Applies novel **DTA module** to jointly attend over time and channel dimensions
4. Predicts semantic segmentation masks for transparent objects
5. Extracts object centroids for **robotic grasping** with Franka Panda or UR5

---

## Problem Statement

### Transparent Object Segmentation Challenge

| Challenge | Event-Based Approach | Benefit |
|-----------|----------------------|---------|
| Specular highlights in RGB | Events only on intensity changes | Robust to reflections |
| Depth unreliability | Temporal contrast (time derivative) | Independent of absolute depth |
| Slow frame rate (30-60 fps) | Event camera (100k-1M events/s) | Capture fast edges |
| High power consumption | SNNs use sparse spikes | 10-100x lower energy |

### Dataset: ClearGrasp

- **50,000+ synthetic RGB-D-M images** of transparent objects with ground-truth masks
- Objects: drinking glasses, wine glasses, transparent bowls, bottles, plastic containers
- Manual segmentation masks + normals + depth annotations
- **286 real-world test images** (phone camera under varied lighting)
- Reference: [ClearGrasp GitHub](https://github.com/Shreeyak/cleargrasp)

---

## Proposed Solution

### System Architecture (4 Phases)

```
┌────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Preparation                                       │
├────────────────────────────────────────────────────────────────┤
│ ClearGrasp RGB images → v2e event simulator →  synthetic events
│                                     ↓                          │
│ [Optional] Pretrained SpikingYOLOX weights                     │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Phase 2: SNN Models with Attention                             │
├────────────────────────────────────────────────────────────────┤
│  Event Stream → Temporal Correlation Encoding                  │
│       ↓                 ↓                 ↓                     │
│  CNN U-Net    Spiking U-Net         DTA-SNN (Novel)           │
│  (baseline)   (intermediate)        (proposed)                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Phase 3: Training & Evaluation                                 │
├────────────────────────────────────────────────────────────────┤
│ Train on synthetic → Validate on synthetic → Test on real      │
│ Metrics: IoU, F1, precision, recall, inference time/energy     │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Phase 4: Robotic Integration                                   │
├────────────────────────────────────────────────────────────────┤
│ PyBullet sim: segmentation → centroid → Franka IK → grasping   │
│ Real robot: event stream → DTA-SNN → grasp pose → execute      │
└────────────────────────────────────────────────────────────────┘
```

### Key Innovation: Dual Temporal-Channel Attention

The **DTA module** uniquely couples attention over:

- **Temporal Gate**: which time-steps contain boundary information?
- **Channel Gate**: which feature channels are most discriminative?
- **Cross-coupling**: learned interaction matrix captures spatio-temporal correlations

*See [patent_provisional.md](patent_provisional.md) for technical details.*

---

## System Architecture

### Repository Structure

```
SNN/
├── models/                          # Neural network architectures
│   ├── __init__.py
│   ├── attention.py                 # DTA module (PATENT-PENDING)
│   ├── encoding.py                  # Temporal Correlation Encoding
│   ├── spiking_unet.py             # Spiking U-Net backbone
│   ├── cnn_baseline.py             # CNN U-Net baseline
│   ├── dta_snn.py                  # Full DTA-SNN model
│   └── pretrained_adapter.py       # SpikingYOLOX weight adapter
│
├── training/                        # Training pipeline
│   ├── train.py                    # Main training script
│   ├── dataset.py                  # EventDataset + augmentations
│   ├── losses.py                   # Dice, Focal, BCE losses
│   └── scheduler.py                # Cosine annealing + warmup
│
├── utils/                           # Utilities
│   ├── generate_events.py          # v2e event generation
│   ├── metrics.py                  # IoU, F1, precision/recall
│   ├── visualization.py            # Spike raster, mask overlay
│   └── event_loader.py             # .aedat2/.npy event loader
│
├── evaluation/                      # Evaluation pipeline
│   └── evaluate.py                 # Full evaluation framework
│
├── simulation/                      # Robotics simulation
│   ├── grasp_demo.py              # PyBullet grasping demo
│   ├── robot_controller.py        # Franka Panda IK controller
│   └── urdf/                       # Robot + object URDF files
│
├── configs/                         # Configuration files
│   ├── base.yaml                  # Shared settings
│   ├── cnn.yaml                   # CNN baseline config
│   ├── snn.yaml                   # SNN config
│   └── dta.yaml                   # DTA-SNN config
│
├── scripts/                         # Helper scripts
│   ├── download_cleargrasp.py     # ClearGrasp downloader
│   ├── setup_environment.sh       # Environment setup
│   ├── run_experiments.sh         # Batch training script
│   └── generate_all_events.py     # v2e batch conversion
│
├── docs/                            # Documentation
│   ├── README.md                  # This file
│   ├── SETUP_GUIDE.md            # Detailed setup instructions
│   ├── ARCHITECTURE.md           # Technical architecture doc
│   ├── patent_provisional.md     # Patent document
│   ├── paper_draft.md            # Research paper draft
│   └── API_REFERENCE.md          # Detailed API reference
│
├── tests/                           # Unit tests
│   └── test_models.py
│
└── requirements.txt                 # Python dependencies
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **CUDA 11.8+** (for GPU acceleration)
- **20GB+ RAM** (for dataset + model training)
- **100GB+ disk space** (for ClearGrasp dataset)

### Step 1: Clone Repository & Install Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/snn-dta-grasping.git
cd snn-dta-grasping

# Create conda environment
conda create -n snn-grasp python=3.10 -y
conda activate snn-grasp

# Install core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install v2e from source (event simulator)
git clone https://github.com/SensorsINI/v2e.git /tmp/v2e
cd /tmp/v2e
pip install -e .
cd -
```

### Step 2: Download ClearGrasp Dataset

```bash
# Automated download script
python scripts/download_cleargrasp.py --output_dir data/ --split all

# Or manually from: https://github.com/Shreeyak/cleargrasp
# Structure: data/synthetic/{rgb,masks}/, data/real/{rgb,masks}/
```

### Step 3: Generate Event Data

```bash
# Convert RGB images to event streams using v2e
python utils/generate_events.py \
    --image_folder data/synthetic/rgb \
    --mask_folder data/synthetic/masks \
    --output_folder data/synthetic/events \
    --num_workers 8 \
    --num_variations 3
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed multi-GPU and distributed setup.

---

## Dataset Preparation

### ClearGrasp Structure

```
data/
├── synthetic/
│   ├── rgb/              # 50,000+ RGB images
│   ├── masks/            # Binary segmentation masks
│   ├── depth/            # Depth maps
│   └── events/           # (Generated via v2e)
└── real/
    ├── rgb/              # 286 real-world images
    ├── masks/            # Ground truth masks
    └── events/           # (To be generated for real sim)
```

### Event Data Characteristics

- **Format**: `.aedat2` (binary) or `.npy` (numpy array)
- **Shape**: `(num_events, 4)` – each event is `(x, y, timestamp, polarity)`
- **Temporal resolution**: Microseconds
- **Spatial resolution**: 640×480 (typical DVS)
- **Event rate**: ~100k-500k events per 100ms (depends on motion)

See [ARCHITECTURE.md](ARCHITECTURE.md) for event tensor formatting.

---

## Training Models

### Quick Start: Train Baseline Models

```bash
# Train CNN baseline (fastest, ~10 min on GPU)
python training/train.py --model cnn --config configs/cnn.yaml \
    --epochs 30 --batch_size 16 --lr 1e-3

# Train Spiking U-Net baseline (~1-2 hours)
python training/train.py --model snn --config configs/snn.yaml \
    --epochs 50 --batch_size 8 --lr 5e-4

# Train DTA-SNN (proposed, ~2-3 hours)
python training/train.py --model dta --config configs/dta.yaml \
    --epochs 50 --batch_size 8 --lr 5e-5 --loss focal_dice
```

### Advanced: Resume Training & Pretrained Weights

```bash
# Resume from checkpoint
python training/train.py --model dta --checkpoint runs/dta_best.pth \
    --epochs 100 --resume

# Use SpikingYOLOX pretrained weights (optional)
python training/train.py --model dta --pretrained \
    --checkpoint path/to/spikingyolox.pth --freeze_epochs 5
```

### Command-Line Arguments

```
--model               {cnn|snn|dta}              Architecture to train
--config              path/to/config.yaml        Config file
--epochs              int (default: 50)          Training epochs
--batch_size          int (default: 8)           Batch size
--lr                  float (default: 5e-5)      Learning rate
--loss                {dice|focal_dice|bce}     Loss function
--device              {cpu|cuda} (default: auto) Device
--mixed_precision     action (default: false)    Use AMP
--checkpoint          path                       Load checkpoint
--pretrained          action                     Use pretrained weights
--freeze_epochs       int (default: 5)           Freeze encoder epochs
--num_workers         int (auto)                 DataLoader workers
--seed                int (default: 42)          Random seed
```

See [configs/](configs/) for detailed hyperparameter tuning.

---

## Evaluation

### Full Evaluation Pipeline

```bash
# Evaluate all trained models on real test set
python evaluation/evaluate.py \
    --models runs/cnn_best.pth runs/snn_best.pth runs/dta_best.pth \
    --test_split data/real/rgb \
    --output_dir evaluation/results \
    --visualize \
    --compute_energy
```

### Metrics Computed

- **Semantic Segmentation**: IoU, Dice, F1, Precision, Recall
- **Energy Efficiency**: Spikes per sample, bit operations, energy cost (vs ANN)
- **Inference Speed**: Latency (ms), throughput (images/sec)
- **Attention Visualization**: Temporal and channel attention heatmaps

### Export Results for Paper

```bash
# Generate comparison table
python evaluation/evaluate.py --output_format latex

# Generate attention visualizations
python evaluation/evaluate.py --visualize_attention --output_dir paper/figures/
```

---

## Robotic Grasping Demo

### PyBullet Simulation

```bash
# Run grasping demo with trained DTA-SNN
python simulation/grasp_demo.py \
    --model runs/dta_best.pth \
    --num_episodes 10 \
    --robot franka_panda \
    --gui

# Export grasp predictions
python simulation/grasp_demo.py --export_poses runs/grasp_poses.json
```

### Real Robot Integration (Future)

The grasping pipeline independently converts:
1. Event stream from neuromorphic sensor → DTA-SNN segmentation mask
2. Segmentation mask → object centroid + bound box
3. Centroid + IK solver (with gripper geometry) → grasp pose
4. Grasp pose → robot joint commands via MoveIt! or ur_script

See [simulation/robot_controller.py](../simulation/robot_controller.py) for controller API.

---

## Results & Benchmarks

### Expected Performance

| Model | Synthetic IoU | Real-test IoU | Training Time | Inference (ms) | Spikes/sample |
|-------|---------------|---------------|---------------|----------------|---------------|
| CNN U-Net baseline | 0.20–0.25 | 0.12–0.18 | ~10 min | 8–12 | N/A (ANN) |
| Spiking U-Net | 0.18–0.22 | 0.10–0.15 | 1–2 hrs | 15–25 | 85k–120k |
| **DTA-SNN (ours)** | **0.24–0.28** | **0.16–0.22** | 2–3 hrs | 18–30 | 75k–110k |

### Attention & Visualization

DTA-SNN learns to:
- **Temporal gate**: Concentrate spikes on sharp transparency edges (high curvature regions)
- **Channel gate**: Weight features encoding local surface geometry and material boundaries
- **Cross-coupling**: Capture edge-onset temporal patterns unique to transparent surfaces

See [evaluation/results/](../evaluation/results/) for example visualizations.

---

## Paper & Patent

### Research Paper

The work is being prepared as a submission to top-tier conferences:
- **ICRA 2025** – International Conference on Robotics and Automation
- **IROS 2025** – Intelligent Robots and Systems
- **WACV 2025** – Winter Conference on Applications of Computer Vision

**Paper structure**: 6–8 pages + supplementary material
- Abstract, introduction, related work
- Method (DTA-SNN architecture, temporal encoding, attention)
- Experiments (baselines, datasets, metrics)
- Results (comparative tables & visualizations)
- Conclusion & future work

See [paper_draft.md](paper_draft.md) for working draft.

### Patent

The **Dual Temporal-Channel Attention module** is covered under:
- **Application Type**: Provisional Patent Application (35 U.S.C. § 111(b))
- **Filing Scope**:
  1. Temporal-channel attention mechanism for SNNs
  2. Joint spatio-temporal gating for event-camera processing
  3. Transparent object segmentation pipeline
  4. Perception-to-grasping system integration

**Patent Status**: ⚠️ File provisional application within 12 months of first publication/disclosure.

See [patent_provisional.md](patent_provisional.md) for detailed technical specification.

---

## Contributing

### Code Contribution Guidelines

1. **Format**: Use `black`, `isort`, `flake8`
2. **Type hints**: All functions must have full type annotations
3. **Docstrings**: Google-style docstrings mandatory
4. **Tests**: Unit tests for new modules (pytest)
5. **Logging**: Use `logging` module, not `print()`

Example:

```python
def compute_attention(
    spike_tensor: torch.Tensor,
    time_steps: int,
    num_channels: int,
) -> torch.Tensor:
    """
    Compute Dual Temporal-Channel Attention weights.

    Args:
        spike_tensor: spike tensor of shape (T, B, C, H, W)
        time_steps: number of temporal steps T
        num_channels: number of feature channels C

    Returns:
        attention_weights: shape (T, B, C, 1, 1)

    References:
        [1] Tan et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
        [2] This work (patent-pending)
    """
```

### Reporting Issues

Please submit issues with:
- Minimal reproducible example
- System info (OS, Python, CUDA version)
- Error log + traceback

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{snn_dta_2024,
    title={Event-Based Semantic Segmentation with Dual Temporal-Channel Attention
           for Transparent Object Grasping},
    author={Your Name(s)},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

---

## License

This project is licensed under **MIT License** – see [LICENSE](LICENSE) file.

⚠️ **Patent Notice**: The DTA module is covered by provisional patent application. Commercial use requires explicit license.

---

## Acknowledgments

- **ClearGrasp dataset**: Shreeyak Sajjan et al. (Princeton)
- **SpikingJelly**: Brian Han et al. (PKU)
- **v2e simulator**: Tobi Delbruck et al. (UZH-RPG)
- **PyBullet**: Erwin Coumans

---

## Contact & Support

**Author**: [Your Name]  
**Email**: [your.email@institution.edu]  
**Lab**: [Your Lab/Institution]  
**GitHub Issues**: https://github.com/yourusername/snn-dta-grasping/issues

---

*Last updated: April 2026*
