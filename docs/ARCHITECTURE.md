# ARCHITECTURE.md - Technical Architecture Document

## System Overview

```
Input Event Stream (DVS/Event Camera)
           ↓
┌─────────────────────────────────────┐
│ Temporal Correlation Encoding (TCE) │  Convert events → (T, H, W) tensor
│ - 4 temporal bins                   │
│ - Polarity-aware event counting     │
│ - Exponential decay option          │
└─────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│ Model Selection                                          │
├──────────────────────────────────────────────────────────┤
│ Option 1: CNN U-Net        (baseline, fastest)           │
│ Option 2: Spiking U-Net    (SNN baseline)                │
│ Option 3: DTA-SNN          (proposed, best accuracy)     │
└──────────────────────────────────────────────────────────┘
           ↓
    ┌─ Output: Segmentation Mask (B, 1, H, W) ─┐
    ↓                                           ↓
[Training]                            [Inference/Robotics]
    ↓                                           ↓
Dice Loss / Focal Loss          → Extract centroid
Adam / AdamW optimizer          → Compute grasp pose
Cosine annealing + warmup       → Robot IK controller
                                → Execute grasping
```

---

## Component Details

### 1. Temporal Correlation Encoding (TCE)

**Input**: Raw event stream {(x, y, t, p) | p ∈ {-1, +1}}

**Process**:

```python
# Bin events into T=4 temporal windows
window_duration = (event_times[-1] - event_times[0]) / T
for w in range(T):
    t_start = event_times[0] + w * window_duration
    t_end = t_start + window_duration
    
    # Count events per polarity in window w
    pos = (events[t_start:t_end, p] == +1).sum(spatial_dims)  # (H, W)
    neg = (events[t_start:t_end, p] == -1).sum(spatial_dims)  # (H, W)
    
    # Stack: (2, H, W) → concatenate to (T, H, W)
    tensor[w] = [pos / pos.max(), neg / neg.max()]
```

**Output**: `(T, 2, H, W)` → squeeze/concat → `(T, H, W)`

**Intuition**:
- Positive events = brightness increase (object boundary highlights/reflections)
- Negative events = brightness decrease (object boundary shadows)
- Temporal binning captures onset/offset dynamics

**Alternative (not used)**: Exponential decay weighting favors recent events
```python
decay = exp(-lambda * (t_max - t))  # λ typically 0.01-0.1
```

---

### 2. CNN U-Net Baseline

**Architecture**:
```
Input (3, H, W) / (1, H, W)
    ↓
Encoder:
  Block1: Conv(3→64) → Conv(64→64) → MaxPool → (64, H/2, W/2)
  Block2: Conv(64→128) → MaxPool → (128, H/4, W/4)
  Block3: Conv(128→256) → MaxPool → (256, H/8, W/8)
  Block4: Conv(256→512) → MaxPool → (512, H/16, W/16)
    ↓
Bottleneck:
  Conv(512→512) → Conv(512→512)
    ↓
Decoder (symmetric upsampling + skip connections):
  Block4_up: ConvTranspose → Concat(Block4) → Conv
  Block3_up: ConvTranspose → Concat(Block3) → Conv
  Block2_up: ConvTranspose → Concat(Block2) → Conv
  Block1_up: ConvTranspose → Concat(Block1) → Conv
    ↓
Output: Conv(64→1) + Sigmoid → (1, H, W)
```

**Implementation**: Standard PyTorch nn.Module

**Parameters**: ~7.5M

**Training**: Fast (~10 min on single GPU), but ANN baseline (not the focus of this work)

---

### 3. Spiking U-Net Baseline

**Key Difference**: Replace standard activations (ReLU) with LIF neurons

**LIF Neuron** (Leaky Integrate and Fire):
```python
V[t+1] = τ * V[t] * (1 - spike[t]) + input[t]
spike[t] = 1 if V[t] ≥ θ else 0
output_spike[t] = spike[t]
```

where $\tau \in [0, 1]$ is leak factor, $\theta$ is threshold.

**Architecture**:
```
Same as CNN but:
  ReLU → LIFNode(tau=0.25, threshold=1.0)
  Applied T=4 time steps (sequential)

For T time steps:
  spike_tensor = []
  V = init(0)
  for t in range(T):
    input = conv_layer(x_tcn[t])
    V = tau*V + input
    spike = (V ≥ θ).float()
    spike_tensor.append(spike)
  return stack(spike_tensor)  # (T, B, C, H, W)
```

**Key Properties**:
- Spikes ∈ {0, 1} → sparse computation
- No gradients through discrete spikes → use **surrogate gradient**
  - Typically: sigmoid derivative σ(x)(1-σ(x))
- Memory potential V carries temporal information

**Parameters**: ~Same as CNN (~7.5M)

**Training**: Slower (~1-2 hours), SNN unrolled T=4 times

---

### 4. Dual Temporal-Channel Attention (DTA) Module

**Motivation**: 
- SNNs operate on spike tensors (T, B, C, H, W)
- Standard attention ignores time OR channel dimension
- DTA uniquely couples both for transparent object edges

**Components**:

#### 4.1 Temporal Gate

```python
class TemporalGate(nn.Module):
    def forward(self, spike_tensor: Tensor) -> Tensor:
        """
        Input: (T, B, C, H, W)
        Output: (T, B, 1, 1, 1) — which timesteps matter?
        """
        # Reduce spatial: (T, B, C, H, W) → (T, B, C)
        reduced = adaptive_avg_pool_2d(spike_tensor, (1, 1))
        
        # Multi-head attention over T dimension
        # Reshape & compute attention
        att_weights = self.attention(reduced)  # (T, B)
        
        return att_weights.view(T, B, 1, 1, 1)
```

**Interpretation**: For transparent objects, which time-steps have edge activity?

#### 4.2 Channel Gate

```python
class ChannelGate(nn.Module):
    def forward(self, spike_tensor: Tensor) -> Tensor:
        """
        Input: (T, B, C, H, W)
        Output: (B, C, 1, 1) — which channels are relevant?
        """
        # Temporal reduction: (T, B, C, H, W) → (B, C, H, W)
        reduced = spike_tensor.mean(dim=0)
        
        # Spatial squeeze: (B, C, H, W) → (B, C)
        squeezed = adaptive_avg_pool_2d(reduced, (1, 1)).view(B, C)
        
        # Bottleneck FC: (B, C) → (B, C/r) → (B, C)
        att = self.fc_up(self.fc_down(squeezed))  # (B, C)
        
        return torch.sigmoid(att).view(B, C, 1, 1)
```

**Interpretation**: Which feature maps encode transparent object boundaries?

#### 4.3 Cross-Coupling

```python
class DynamicTemporalAttention(nn.Module):
    def forward(self, spike_tensor: Tensor) -> Tensor:
        """
        Integration of temporal + channel gates with learned coupling.
        Output: Weighted spike tensor same shape as input
        """
        temporal_gate = self.temporal_gate(spike_tensor)  # (T, B, 1, 1, 1)
        channel_gate = self.channel_gate(spike_tensor)    # (B, C, 1, 1)
        
        # Coupling: Learn interaction matrix (T, C)
        temporal_vec = temporal_gate.squeeze(-1).squeeze(-1)  # (T, B)
        channel_vec = channel_gate.squeeze(-1).squeeze(-1)    # (B, C)
        
        # Broadcast & apply coupling
        coupling = self.coupling_matrix(temporal_vec, channel_vec)
        
        # Apply to spike tensor
        weighted = spike_tensor * coupling.view(T, B, C, 1, 1)
        
        return weighted
```

---

### 5. DTA-SNN Full Architecture

```
Input TCN: (T, H, W)
    ↓
Encoder Block 1: Spiking8Conv(channels 1→64) + LIF × T steps
    ↓ MaxPool
Encoder Block 2: Spiking Conv(64→128) + LIF × T steps
    ↓ MaxPool
Encoder Block 3: Spiking Conv(128→256) + LIF × T steps
    ↓ MaxPool
Encoder Block 4: Spiking Conv(256→512) + LIF × T steps
    ↓ MaxPool
Bottleneck: Spiking Conv(512→512) + LIF × T steps
    ↓
┌──────────────────────────────────────┐
│   DTA Module (PATENT-PENDING)       │
│  - Temporal Gate (which timesteps)   │
│  - Channel Gate (which channels)     │
│  - Cross-Coupling (learned matrix)   │
└──────────────────────────────────────┘
    ↓
Decoder Block 4_up: ConvTranspose + Concat(skip) + LIF
    ↓
Decoder Block 3_up: ConvTranspose + Concat(skip) + LIF
    ↓
Decoder Block 2_up: ConvTranspose + Concat(skip) + LIF
    ↓
Decoder Block 1_up: ConvTranspose + Concat(skip) + LIF
    ↓
Output Layer: Conv(64→1) + Sigmoid
    ↓
Segmentation Mask: (1, H, W)
```

**Training Loss**:
```python
loss = dice_loss(pred_mask, gt_mask) + 0.1 * focal_loss(pred_mask, gt_mask)
```

**Optimizer**: AdamW (weight decay 1e-4)

**Scheduler**: Cosine annealing with 5-epoch warmup

---

### 6. Event Loader & Data Format

**Input Formats**:
- `.aedat2` binary format (standard DVS)
- `.npy` numpy arrays (sparse tuple format)
- Plain text event CSV

**Event Structure**:
```
Each event: (x, y, timestamp, polarity)
  - x ∈ [0, 639]  (spatial column)
  - y ∈ [0, 479]  (spatial row)
  - timestamp ∈ ℝ  (microseconds)
  - polarity ∈ {-1, +1} (brightness change direction)
```

**Loader Process**:
```python
events = load_events(file)  # (N_events, 4)
tcn = events_to_tcn(events, H=480, W=640, T=4)  # (T, 2, H, W)
return tcn.float()
```

---

### 7. Training Pipeline

**Data Split**:
- Synthetic train: 40,000 images
- Synthetic val: 5,000 images
- Real test: 286 images

**Augmentations**:
- Horizontal flip, vertical flip
- Random rotation (-5°, +5°)
- Brightness jitter
- Gaussian blur (σ = 0.5-1.5)
- Mixup (α = 0.2)

**Mixed Precision Training**:
```python
for epoch in range(epochs):
    for batch in dataloader:
        with autocast():
            output = model(batch)
            loss = criterion(output, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Checkpoint Strategy**:
- Save best model by validation IoU
- Keep last 3 checkpoints
- Full state_dict + optimizer state

---

### 8. Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **IoU** | TP/(TP+FP+FN) | Primary metric, segmentation quality |
| **Dice** | 2·TP/(2·TP+FP+FN) | Robust to class imbalance |
| **F1** | 2·(P·R)/(P+R) | Harmonic mean of precision/recall |
| **Precision** | TP/(TP+FP) | False positive rate |
| **Recall** | TP/(TP+FN) | False negative rate |

**Energy**:
- Spikes/sample: sum(spike_tensor) for SNNs
- vs. ANN: dense (H·W·C_max) operations

---

### 9. Robotics Integration

**Grasp Planning Pipeline**:

```
Segmentation Mask (1, H, W)
       ↓
1. Extract binary mask: mask > 0.5
       ↓
2. Find connected components
       ↓
3. For each component:
   - Compute centroid (x, y)
   - Compute bounding box
   - Estimate object dimensions from mask
       ↓
4. Grasp Hypothesis Generation:
   - Centroid ± εx, εy for multiple grasps
   - Orientation: principle axes of mask
       ↓
5. Collision Check (PyBullet):
   - Check gripper-object-scene collisions
   - Filter invalid grasps
       ↓
6. Rank & Select:
   - Score by: distance to centroid, collisions avoided
   - Pick highest-scoring grasp
       ↓
7. Robot IK Solver:
   - Map 2D grasp to 3D using camera calibration
   - Solve for joint angles (FK/IK)
       ↓
8. Trajectory Execution:
   - Plan collision-free path
   - Execute via robot controller
```

**PyBullet Simulation**:
```python
import pybullet as p

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load robot & scene
franka = p.loadURDF("franka_panda.urdf", [0, 0, 0])
table = p.loadURDF("plane.urdf")
bottle = p.loadURDF("transparent_bottle.urdf", [0.5, 0, 1.0])

# Simulate event camera
camera_matrix = get_calibration()
event_stream = simulate_events(bottle, camera_matrix)

# DTA-SNN inference
mask = dta_model(event_stream)

# Grasp planning
grasp_pose = plan_grasp(mask, bottle)
execute_grasp(franka, grasp_pose)
```

---

## Configuration Files

Each model variant has its own YAML config:

**configs/base.yaml** - Shared defaults
```yaml
model:
  input_channels: 1
  output_channels: 1
  height: 480
  width: 640
training:
  epochs: 50
data:
  train_split: 0.8
  val_split: 0.1
```

**configs/cnn.yaml** - CNN-specific
```yaml
model:
  type: cnn_unet
  depth: 4
  dropout: 0.2
training:
  optimizer: adam
  lr: 1e-3
```

**configs/snn.yaml** - SNN-specific
```yaml
model:
  type: spiking_unet
  timesteps: 4
  threshold: 1.0
training:
  lr: 5e-4
  surrogate: sigmoid
```

**configs/dta.yaml** - DTA-SNN-specific
```yaml
model:
  type: dta_snn
  attention_type: temporal_channel
  coupling: true
training:
  lr: 5e-5
  loss: focal_dice
```

---

## Performance Targets

| Component | Target |
|-----------|--------|
| Synthetic IoU (DTA-SNN) | **0.24-0.28** |
| Real-test IoU (DTA-SNN) | **0.16-0.22** |
| Inference latency | **18-30ms/image** |
| Memory usage | **~4-6GB GPU** |
| Spikes/sample | **75k-110k** |
| Training time (50 epochs) | **2-3 hours** |

---

*See README.md for usage examples and evaluation details.*
