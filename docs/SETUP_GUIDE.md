# SETUP_GUIDE.md - Complete Installation & Configuration

## Prerequisites

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.10 or 3.11 (3.12 may have compatibility issues)
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU**: NVIDIA RTX 3090, A100, or equivalent (11GB+ VRAM)
- **Disk**: 100GB+ (50GB for ClearGrasp, 20GB for models/logs)
- **RAM**: 32GB+ (for dataloading)

---

## Step-by-Step Installation

### 1. Clone and Create Environment

```bash
# Clone repository
git clone https://github.com/yourusername/snn-dta-grasping.git
cd snn-dta-grasping

# Create conda environment
conda create -n snn-grasp python=3.10 -y
conda activate snn-grasp

# Verify Python version
python --version  # Should be 3.10.x
```

### 2. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Verify CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **SpikingJelly** (SNN framework)
- **PyBullet** (robotics simulation)
- **OpenCV**, **Pillow**, **scikit-image** (image processing)
- **TensorBoard**, **wandb** (logging)
- **h5py** (HDF5 files)
- **pytest** (testing)

### 4. Install v2e (Event Simulator) from Source

```bash
# Clone v2e repository
git clone https://github.com/SensorsINI/v2e.git /tmp/v2e
cd /tmp/v2e

# Install in development mode
pip install -e .

# Verify installation
python -c "import v2e; print(v2e.__version__)"

cd -  # Return to project directory
```

### 5. Verify Installation

```bash
python -c "
import torch
import spikingjelly
import cv2
import pybullet
import v2e
print('✓ All packages installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

---

## Dataset Setup

### Download ClearGrasp

**Automated Download** (Recommended):

```bash
python scripts/download_cleargrasp.py \
    --output_dir data/ \
    --split all \
    --workers 4
```

**Manual Download**:

```bash
# Visit: https://github.com/Shreeyak/cleargrasp
# Follow README instructions
# Directory structure should be:
mkdir -p data/synthetic/rgb
mkdir -p data/synthetic/masks
mkdir -p data/synthetic/depth
mkdir -p data/real/rgb
mkdir -p data/real/masks
```

**Verify Dataset**:

```bash
ls -lh data/synthetic/rgb | head -5
# Should show 50,000+ images

du -sh data/
# Should be ~50GB total
```

### Generate Event Data

```bash
# Single GPU (on CUDA device 0)
python utils/generate_events.py \
    --image_folder data/synthetic/rgb \
    --mask_folder data/synthetic/masks \
    --output_folder data/synthetic/events \
    --num_workers 8 \
    --device cuda:0 \
    --num_variations 3

# Multi-GPU (distributed)
python utils/generate_events.py \
    --image_folder data/synthetic/rgb \
    --mask_folder data/synthetic/masks \
    --output_folder data/synthetic/events \
    --num_workers 32 \
    --device cuda \
    --num_variations 3 \
    --distributed
```

**Monitoring Progress**:

```bash
# In another terminal, watch file growth
watch -n 10 'ls data/synthetic/events | wc -l'
```

**Event Data Format**:
- Output files: `.aedat2` (binary) or `.npy` (numpy)
- Directory: `data/synthetic/events/`
- Size per event file: ~50-200 KB
- Total storage: ~10-15 GB

---

## Configuration

### Create Local Config Override

```bash
mkdir -p configs_local
cp configs/dta.yaml configs_local/dta_custom.yaml
```

**Edit `configs_local/dta_custom.yaml`**:

```yaml
# Override defaults for your hardware
model:
  type: dta_snn
  depth: 4
  channels: [64, 128, 256, 512]
  timesteps: 4
  
training:
  batch_size: 16                # Increase if GPU memory > 24GB
  num_workers: 8                # Match CPU cores
  mixed_precision: true         # FP16 training
  gradient_accumulation: 1      # For gradient smoothing
  pin_memory: true              # FastDataLoader
```

### Multi-GPU Training Config

```yaml
# For multi-GPU (8x RTX 3090)
distributed:
  enabled: true
  backend: nccl
  num_processes: 8

training:
  batch_size: 8                 # Per GPU
  num_workers: 8                # Per GPU
```

---

## Training Setup

### Training a Single Model

```bash
# CNN baseline
python training/train.py \
    --model cnn \
    --config configs/cnn.yaml \
    --epochs 30 \
    --batch_size 16 \
    --device cuda:0

# Spiking U-Net
python training/train.py \
    --model snn \
    --config configs/snn.yaml \
    --epochs 50 \
    --batch_size 8 \
    --device cuda:0

# DTA-SNN (proposed)
python training/train.py \
    --model dta \
    --config configs/dta.yaml \
    --epochs 50 \
    --batch_size 8 \
    --device cuda:0
```

### Distributed Training (Multi-GPU)

```bash
# Using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    training/train.py \
    --model dta \
    --config configs/dta.yaml \
    --distributed
```

### Resume Training

```bash
python training/train.py \
    --model dta \
    --config configs/dta.yaml \
    --checkpoint runs/dta_epoch_25.pth \
    --resume \
    --epochs 50
```

### Using Pretrained Weights (Optional)

```bash
# Download SpikingYOLOX checkpoint manually
# Then:
python training/train.py \
    --model dta \
    --config configs/dta.yaml \
    --pretrained \
    --checkpoint path/to/spikingyolox_weights.pth \
    --freeze_encoder_epochs 5
```

---

## Monitoring Training

### TensorBoard

```bash
# Live monitoring during training
tensorboard --logdir runs/ --port 6006

# Visit: http://localhost:6006
```

### Weights & Biases (Optional)

```bash
pip install wandb
wandb login

# Training will auto-log to W&B
python training/train.py --model dta --config configs/dta.yaml --use_wandb
```

### Training Logs

```bash
# View training info
tail -f runs/dta_snn/train.log

# Typical output:
# [Epoch 1/50] Loss: 0.342 | Val IoU: 0.145 | Time: 45s
# [Epoch 2/50] Loss: 0.298 | Val IoU: 0.156 | Time: 44s
```

---

## Evaluation Setup

### Run Full Evaluation

```bash
python evaluation/evaluate.py \
    --models runs/cnn_best.pth runs/snn_best.pth runs/dta_best.pth \
    --test_split data/real/rgb \
    --output_dir evaluation/results \
    --visualize \
    --compute_energy
```

### Export Metrics for Paper

```bash
# LaTeX table format
python evaluation/evaluate.py \
    --models runs/dta_best.pth \
    --output_format latex \
    --save_csv evaluation/results/metrics.csv
```

---

## Robotics Simulation Setup

### Install Additional Packages

```bash
pip install ikpy open3d trimesh  # For Franka IK/visualization
```

### Run Grasping Demo

```bash
# Simulation with visualization
python simulation/grasp_demo.py \
    --model runs/dta_best.pth \
    --num_episodes 10 \
    --robot franka_panda \
    --gui \
    --render_width 1280 \
    --render_height 720

# Headless mode (for servers)
python simulation/grasp_demo.py \
    --model runs/dta_best.pth \
    --num_episodes 100 \
    --robot franka_panda \
    --headless \
    --export_poses runs/grasp_poses.json
```

---

## Docker Setup (Optional)

### Build Docker Image

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git

WORKDIR /workspace
COPY . .

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
```

### Build & Run

```bash
docker build -f Dockerfile.gpu -t snn-dta:latest .

docker run --gpus all -it --rm \
    -v ${PWD}/data:/workspace/data \
    -v ${PWD}/runs:/workspace/runs \
    snn-dta:latest
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```yaml
# configs/dta.yaml
training:
  batch_size: 4              # Reduce from 8
  gradient_accumulation: 2   # Accumulate over 2 steps
  mixed_precision: true      # Use FP16
```

### Issue: v2e Installation Fails

**Solution**:
```bash
# Ensure dependencies first
pip install opencv-python numpy scipy

# Then retry
git clone https://github.com/SensorsINI/v2e.git /tmp/v2e
cd /tmp/v2e && pip install -e .
```

### Issue: SNN Training Produces NaN Loss

**Solution**:
```python
# In configs/dta.yaml:
training:
  learning_rate: 1e-5      # Reduce from 5e-5
  gradient_clip: 1.0       # Add gradient clipping
  weight_decay: 0.0001
```

### Issue: Low Validation IoU After Training

**Solution**:
1. Verify data is loaded correctly: `pytest tests/test_models.py -v`
2. Check data augmentations: reduce if too aggressive
3. Increase training epochs to 100+
4. Use `--pretrained` flag with SpikingYOLOX

---

## Performance Optimization

### Mixed Precision Training

```python
# Automatic in train.py if enabled
training:
  mixed_precision: true
  # ~2x speedup + ~50% memory savings
```

### DataLoader Optimization

```yaml
training:
  num_workers: 8           # Match CPU cores
  pin_memory: true         # GPU memory pinning
  prefetch_factor: 2       # Prefetch batches
```

### Gradient Checkpointing

```python
# In models/dta_snn.py (if memory very limited)
model.gradient_checkpointing = True  # ~60% memory reduction, 20% slowdown
```

---

##  Production Deployment

### Export Model for Inference

```python
# Save model with metadata
torch.save({
    'model_state': model.state_dict(),
    'config': config,
    'version': '1.0',
    'training_metrics': {
        'best_iou': 0.198,
        'epochs_trained': 50,
    }
}, 'models/dta_snn_v1.pth')
```

### ONNX Export (for edge devices)

```python
dummy_input = torch.randn(1, 1, 480, 640).cuda()
torch.onnx.export(
    model, dummy_input, 'models/dta_snn.onnx',
    input_names=['events'],
    output_names=['segmentation_mask'],
    opset_version=12
)
```

---

*For issues or questions, please refer to GitHub Issues or contact: your.email@institution.edu*
