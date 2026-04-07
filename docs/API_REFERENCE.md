# SNN-DTA API Reference

Complete documentation of all public APIs in the SNN-DTA framework.

---

## Table of Contents

1. [Models API](#models-api)
2. [Training API](#training-api)
3. [Evaluation API](#evaluation-api)
4. [Utils API](#utils-api)
5. [Simulation API](#simulation-api)

---

## Models API

### Module: `models`

#### `build_model(model_type: str, config: Dict[str, Any]) → nn.Module`

Factory function to instantiate models.

**Parameters:**
- `model_type` (str): One of `"cnn"`, `"snn"`, `"dta"`
- `config` (Dict): Model configuration dict (typically loaded from YAML)

**Returns:**
- PyTorch `nn.Module` instance

**Example:**
```python
from models import build_model
from omegaconf import OmegaConf

config = OmegaConf.load("configs/dta.yaml")
model = build_model("dta", config.model)
model = model.cuda()
```

---

### Class: `DTA_SNN`

Novel Spiking Neural Network with Dual Temporal-Channel Attention.

```python
from models.dta_snn import DTA_SNN

model = DTA_SNN(
    in_channels=4,        # 4 temporal bins
    num_classes=2,        # binary segmentation
    timesteps=4,
    vth=1.0,              # LIF neuron threshold
    tau_m=0.5, tau_s=1.0, # membrane/synapse decay
    attention_heads=8,
    reduction=16,         # channel attention reduction
)
```

**Key Methods:**

##### `forward(events: Tensor) → Tensor`
- **Input**: `(B, T, H, W)` event tensor
- **Output**: `(B, 2, H, W)` segmentation logits
- **Processing**: Encodes events → U-Net backbone → DTA → decoder

**Example:**
```python
# Synthetic event frames: (batch_size, 4 timesteps, 480 height, 640 width)
events = torch.randn(2, 4, 480, 640).cuda()
logits = model(events)  # Output: (2, 2, 480, 640)
probs = torch.softmax(logits, dim=1)
```

---

### Class: `DualTemporalChannelAttention` (Patent-Pending)

Located in `models/attention.py`. Core contribution: joint temporal-channel attention.

```python
from models.attention import DualTemporalChannelAttention

dta = DualTemporalChannelAttention(
    in_channels=64,
    timesteps=4,
    num_heads=8,
    reduction=16,
)

# Input: (B, C, T, H, W) with temporal dimension
x = torch.randn(2, 64, 4, 480, 640)
output = dta(x)  # Same shape output
```

**Key Components:**
- `TemporalGate`: Multi-head attention over time dimension
- `ChannelGate`: Squeeze-excitation over channels
- `CrossCoupling`: Learned temporal-channel interaction matrix

**Patent Claims:**
- Claim 1: TemporalGate with multi-head self-attention
- Claim 2: Adaptive ChannelGate (SE-Net variant)
- Claim 3: CrossCoupling learned interaction matrix
- Claims 4–5: Integration with SNN and image encoding

---

### Class: `SpikingUNet`

Spiking neural network backbone using LIF neurons.

```python
from models.spiking_unet import SpikingUNet

unet = SpikingUNet(
    in_channels=4,
    out_channels=2,
    timesteps=4,
    vth=1.0,
    tau_m=0.5, tau_s=1.0,
)

events = torch.randn(2, 4, 480, 640)
output_spikes = unet(events)  # (B, 2, H, W)
```

**Architecture:**
- 4-level encoder with LIF neurons (double conv → pool)
- Channel progression: 4 → 64 → 128 → 256 → 512
- 4-level decoder with skip connections
- Output: binary segmentation mask

---

### Class: `TemporalCorrelationEncoding`

Custom event-to-tensor encoder.

```python
from models.encoding import TemporalCorrelationEncoding

encoder = TemporalCorrelationEncoding(
    height=480,
    width=640,
    num_bins=4,
    normalize=True,
)

# Events: list of (x, y, t, p) tuples or numpy array
events = np.array([[100, 200, 0.1, 1], [101, 201, 0.15, -1], ...])
tensor = encoder(events)  # Output: (4, 480, 640)
```

**Processing:**
1. Temporal binning: Discretize continuous timestamps into T bins
2. Polarity-aware accumulation: Positive/negative events separately
3. Normalization: Per-bin min-max or L2 normalization
4. Output: Dense tensor ready for network input

---

## Training API

### Module: `training`

#### Function: `build_dataloader(...) → DataLoader`

```python
from training.dataset import build_dataloader

train_loader = build_dataloader(
    data_dir="data/synthetic",
    split="train",
    batch_size=8,
    num_workers=4,
    augment=True,
)

for batch_idx, (events, masks, metadata) in enumerate(train_loader):
    # events: (B, T=4, H, W) tensor
    # masks: (B, H, W) binary segmentation ground truth
    # metadata: dict with 'image_id', 'dataset' keys
    print(f"Batch {batch_idx}: events shape {events.shape}")
```

**Parameters:**
- `data_dir` (str): Path to dataset root
- `split` (str): "train", "val", or "test"
- `batch_size` (int): Batch size
- `num_workers` (int): Parallel data loading workers
- `augment` (bool): Apply data augmentation (rotation, flip, etc.)

**Returns:**
- PyTorch `DataLoader` with (events, masks, metadata) tuples

---

### Class: `EventDataset`

Dataset loader for event data with augmentation.

```python
from training.dataset import EventDataset

dataset = EventDataset(
    data_dir="data/synthetic",
    split="train",
    augment=True,
    augment_config={
        "rotate": (-15, 15),      # degrees
        "flip_h": True,
        "flip_v": False,
        "scale": (0.9, 1.1),
        "brightness": (0.8, 1.2),
    }
)

# Single sample access
events, mask, metadata = dataset[0]
print(f"Sample 0: events {events.shape}, mask {mask.shape}")
```

---

### Class: `Trainer`

High-level training orchestrator.

```python
from training.train import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=cfg,
)

# Train for N epochs
history = trainer.train(epochs=50)

# Access results
print(f"Best IoU: {history['best_iou']:.4f}")
print(f"Best epoch: {history['best_epoch']}")
```

**Key Methods:**

##### `train_one_epoch() → Dict[str, float]`
Executes single training epoch.

**Returns:**
```python
{
    "loss": 0.245,           # Average batch loss
    "iou": 0.68,             # Validation IoU
    "f1": 0.71,              # Validation F1
    "lr": 5e-5,              # Current learning rate
    "timestamp": "2024-12-15 14:23:45"
}
```

##### `save_checkpoint(path: str, is_best: bool = False)`
Save model weights.

**Example:**
```python
trainer.save_checkpoint("runs/epoch_25.pth")
trainer.save_checkpoint("runs/best.pth", is_best=True)
```

---

### Loss Functions

```python
from training.losses import DiceLoss, FocalLoss, DiceFocalLoss

# Binary Dice loss (recommended for segmentation)
dice_loss = DiceLoss(smooth=1.0)

# Focal loss (handles class imbalance)
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Combined (best results)
combined_loss = DiceFocalLoss(alpha=0.25, dice_weight=1.0, focal_weight=0.5)

logits = model(events)  # (B, 2, H, W)
loss = combined_loss(logits, masks)
```

---

### Scheduling

```python
from training.scheduler import CosineWarmupScheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineWarmupScheduler(
    optimizer=optimizer,
    warmup_epochs=5,
    epochs=50,
)

for epoch in range(50):
    train_one_epoch()
    scheduler.step(epoch)
```

---

## Evaluation API

### Class: `EvaluationPipeline`

Comprehensive evaluation framework.

```python
from evaluation.evaluate import EvaluationPipeline

evaluator = EvaluationPipeline(
    model_paths=[
        "runs/cnn_best.pth",
        "runs/snn_best.pth", 
        "runs/dta_best.pth"
    ],
    test_dir="data/real/rgb",
    device="cuda:0",
)

# Run evaluation
results = evaluator.run_all_evaluations()

# Export results
evaluator.save_results(format="json", output_dir="runs/results/")
evaluator.save_results(format="csv", output_dir="runs/results/")
evaluator.save_results(format="latex", output_dir="runs/results/")
```

**Key Methods:**

##### `evaluate_single_model(model_path: str) → Dict`

Returns comprehensive metrics:
```python
{
    "iou": {"mean": 0.68, "std": 0.12, "min": 0.38, "max": 0.92},
    "dice": {"mean": 0.78, "std": 0.10, "min": 0.48, "max": 0.96},
    "f1": {"mean": 0.76, "std": 0.11, "min": 0.46, "max": 0.95},
    "precision": {"mean": 0.81, "std": 0.09},
    "recall": {"mean": 0.72, "std": 0.13},
    "accuracy": {"mean": 0.89, "std": 0.08},
    "inference_time_ms": {"mean": 18.3, "std": 2.1},
    "throughput_fps": {"mean": 54.6, "std": 6.2},
}
```

##### `save_results(format: str, output_dir: str)`

Export in multiple formats:
- `"json"`: Raw results as JSON
- `"csv"`: Spreadsheet-compatible CSV
- `"latex"`: Publication-ready LaTeX table

---

## Utils API

### Event Loading

```python
from utils.event_loader import EventLoader
from pathlib import Path

# Supports: .npy, .aedat2 (DVS binary), .csv
events = EventLoader.load_events(Path("data/synthetic/events/img1.npy"))

# Convert to tensor (standard interface)
tensor = EventLoader.events_to_tensor(
    events,
    sensor_height=480,
    sensor_width=640,
    num_bins=4,  # Temporal binning
)
# Output shape: (4, 480, 640)
```

**Event Format:**
Events are numpy arrays with shape `(N, 4)`:
- Column 0: x-coordinate (0–639)
- Column 1: y-coordinate (0–479)
- Column 2: timestamp (0–1, normalized)
- Column 3: polarity (-1 or +1)

---

### Event Generation

```python
from utils.generate_events import EventGenerator

generator = EventGenerator(
    height=480,
    width=640,
    polarity_threshold=0.1,
    num_variations=3,  # Generate 3 event streams per image
)

# Generate from RGB image sequences
rgb_sequence = [img1, img2, img3, ...]  # PIL/numpy images
events = generator.generate(rgb_sequence)

# Augment with noise
events_augmented = generator.augment(
    events,
    polarity_flip_rate=0.05,     # 5% of events flip polarity
    spatial_jitter_px=1,         # ±1 pixel jitter
    temporal_jitter_ms=1,        # ±1ms temporal jitter
)
```

---

### Metrics

```python
from utils.metrics import IoU, DiceScore, F1Score, PrecisionRecall

# Compute metrics
iou = IoU()(predictions, ground_truth)
dice = DiceScore()(predictions, ground_truth)
f1 = F1Score()(predictions, ground_truth)
precision, recall = PrecisionRecall()(predictions, ground_truth)

print(f"IoU: {iou:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
```

**Inputs:**
- `predictions`: (B, H, W) or (B, 2, H, W) model output (logits or probs)
- `ground_truth`: (B, H, W) binary masks (0 or 1)

---

### Visualization

```python
from utils.visualization import (
    plot_spike_raster,
    overlay_mask,
    visualize_attention_map,
)

# Spike raster plot for paper figures
plot_spike_raster(
    events=events,
    pixel_x=100, pixel_y=200,
    output_path="figures/spike_raster.png",
    title="Single-pixel spike timing",
)

# Overlay segmentation on RGB
overlay_mask(
    rgb_image=rgb,
    mask=pred_mask,
    alpha=0.4,
    output_path="figures/segmentation_overlay.png",
)

# Attention heatmap
visualize_attention_map(
    attention_weights=attn,  # (T, H, W)
    background_image=rgb,
    colormap="hot",
    output_path="figures/attention_heatmap.png",
)
```

---

## Simulation API

### Class: `GraspingSimulator`

Full robotic grasping simulation with DTA-SNN segmentation.

```python
from simulation.grasp_demo import GraspingSimulator

sim = GraspingSimulator(
    model_path="runs/dta_best.pth",
    robot_type="franka_panda",
    gui=True,  # Visual interface
    timestep=1/240,  # 240 Hz simulation
)

# Run single episode
result = sim.run_episode(
    object_id="glass_cup",
    noise_level=0.0,
)

print(f"Grasp success: {result['success']}")
print(f"Grasp pose: {result['grasp_pose']}")

# Batch episodes
results = sim.run_simulation(
    num_episodes=20,
    object_list=["cup", "bottle", "bowl"],
)

print(f"Success rate: {results['success_rate']:.2%}")
```

**Returns:**
```python
{
    "success": True,
    "grasp_pose": np.array([x, y, z, roll, pitch, yaw]),
    "segmentation_mask": (480, 640) binary array,
    "centroid_pixel": (x, y),
    "centroid_world": (x, y, z),
    "inference_time_ms": 18.3,
    "robot_time_ms": 245.5,
}
```

**Key Methods:**

##### `simulate_camera_image(pos, orn) → np.ndarray`
Render scene from virtual RGB-D camera.

##### `simulate_events_from_motion(img1, img2) → np.ndarray`
Generate event stream from frame difference (motion).

##### `events_to_segmentation(events) → np.ndarray`
Run DTA-SNN inference on events.

##### `pixel_to_world_coordinates(centroid_px, depth_map) → tuple`
Project 2D pixel → 3D world coordinates.

---

### Class: `FrankaPandaController`

7-DOF Franka Panda inverse kinematics.

```python
from simulation.robot_controller import FrankaPandaController

controller = FrankaPandaController()

# Compute inverse kinematics
target_pos = np.array([0.5, 0.2, 0.3])  # (x, y, z) in meters
target_orn = np.array([0, 0, 0])        # (roll, pitch, yaw)

joint_angles = controller.inverse_kinematics(
    target_position=target_pos,
    target_orientation=target_orn,
    initial_guess=None,
    max_iterations=1000,
    tolerance=1e-4,
)

print(f"Joint angles: {joint_angles}")

# Check joint limits
within_limits = controller.check_joint_limits(joint_angles)
print(f"Within limits: {within_limits}")
```

**Key Methods:**

##### `forward_kinematics(joint_angles: np.ndarray) → tuple`
Compute end-effector position from joints.

**Returns:** `(position, orientation)` tuples as np.ndarray

##### `inverse_kinematics(...) → np.ndarray`
Solve for joint angles given target EE pose.

##### `plan_trajectory(start_angles, end_angles, num_steps=50) → np.ndarray`
Linear interpolation trajectory.

---

## Configuration API

All models use YAML configuration. Load via OmegaConf:

```python
from omegaconf import OmegaConf

# Load base config
cfg = OmegaConf.load("configs/base.yaml")

# Merge model-specific config
cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/dta.yaml"))

# Override from command line
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

print(OmegaConf.to_yaml(cfg))
```

---

## Command-Line Interfaces

### Training

```bash
python training/train.py \
    --model dta \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-5 \
    --loss focal_dice \
    --device cuda:0 \
    --checkpoint runs/dta_latest.pth \
    --wandb_project snn-dta
```

### Evaluation

```bash
python evaluation/evaluate.py \
    --model_path runs/dta_best.pth \
    --test_dir data/real/rgb \
    --output_format json csv latex \
    --visualize
```

### Simulation

```bash
python simulation/grasp_demo.py \
    --model runs/dta_best.pth \
    --gui \
    --num_episodes 20 \
    --export_poses runs/grasp_results.json
```

---

## Performance Optimization Tips

### GPU Memory Management
```python
# Enable mixed precision
torch.set_float32_matmul_precision('high')

# Gradient accumulation
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Multi-GPU Training
```bash
# Distributed training
python -m torch.distributed.launch \
    --nproc_per_node=8 training/train.py --model dta
```

### Inference Speedup
```python
# Model optimization
model.eval()
model = torch.jit.script(model)  # TorchScript compilation

# Reduced precision
with torch.no_grad():
    with torch.cuda.amp.autocast():
        output = model(events)
```

---

## Error Handling

### Common Issues & Solutions

**NaN Loss During SNN Training**
```python
# Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

**GPU Out of Memory**
```python
# Reduce batch size
batch_size = 4  # down from 8

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()
```

**Event Loader Hangs**
```python
# Reduce num_workers
dataloader = DataLoader(dataset, num_workers=0)
```

---

## References

- **SpikingJelly Framework**: https://spikingjelly.readthedocs.io/
- **PyBullet Docs**: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/
- **v2e Event Simulator**: https://github.com/SensorsINI/v2e
- **OmegaConf Docs**: https://omegaconf.readthedocs.io/

---

*Last updated: December 2024*
