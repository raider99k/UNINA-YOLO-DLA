# UNINA-YOLO-DLA: Deterministic Perception for Formula Student Driverless

This project implements a high-performance, DLA-optimized object detection pipeline designed for the NVIDIA Jetson Orin SoC. It prioritizes deterministic latency and hardware-aware design to ensure reliable cone detection in racing environments.

## 1. Strategic Vision: Silicon-Aware Architecture

Traditional object detection implementations (like stock YOLOv8/v11) often underutilize the specialized hardware on Jetson devices, leading to resource contention and non-deterministic latency. UNINA-YOLO-DLA adopts a **"Split-Compute" paradigm**:
- **DLA (Deep Learning Accelerator)**: Handles the entire 2D detection backbone and heads.
- **GPU**: Reserved for computationally intensive tasks like LiDAR clustering, SLAM, and Path Planning.

This isolation ensures that vision processing doesn't interfere with control-critical algorithms, maintaining an inference time of <15ms with minimal jitter.

## 2. Technical Blueprint

### Modified YOLOv11 Architecture
The model is customized to meet DLA hardware constraints and the specific needs of Formula Student:
- **Zero-Fallback Design**: Every layer is validated against DLA support.
- **Activations**: Systematic replacement of SiLU with **ReLU** for maximum DLA throughput and memory efficiency.
- **P2 Detection Head**: Added a high-resolution head (stride 4) to detect small cones at 20m+ distances, where they occupy as little as 10-15 pixels.
- **P5 Removal**: The stride 32 head is removed to save parameters and memory, as extremely close-range detection is handled by other sensors or redundant logic.
- **DLA-Friendly SPPF**: Re-engineered to avoid `torch.chunk` and slicing operations that trigger GPU fallback.

### Precision Engineering
- **INT8 Quantization**: Optimized execution on DLA using INT8 precision with **Entropy Calibration (KL Divergence)** to preserve sensitivity for small, distant objects.
- **Quantization Aware Training (QAT)**: Mandatory use of the `pytorch-quantization` toolkit to recover accuracy lost during quantization.

## 3. Implementation Modules

### Module 1: Data-Centric Training Pipeline
The project adopts a **Data-Centric AI strategy**, abandoning synthetic data in favor of a "Platinum-Standard" real-world dataset:
- **Auto-Labeling**: High-quality offline annotation using **GroundingDINO** (box proposals), **SAM** (segmentation masks), and **SAHI** (tiled high-res inference).
- **Active Learning**: Informative sample selection using **Entropy** and **Localization Variance** query functions to prioritize "hard" real examples.
- **Augmentation**: Real-to-real **Copy-Paste** augmentation using cone assets extracted from real frames.
- **Small Object Metrics**: Validation includes a custom `mAP_small` metric specifically for cones under 15x15 pixels.

### Module 2: Exporting for Deployment
Using the TensorRT Python API, models are compiled into engines with static shapes (640x640) for the target DLA core.
- **Script**: `unina_yolo_dla/export_trt.py`
- **Constraint**: Static batch size = 1.

### Module 3: Runtime & ROS 2 Integration
The C++ ROS 2 node (`perception_node`) implements a **Zero-Copy** data flow:
1. **Ingestion**: Direct mapping from ROS message to CUDA memory (`NvBufSurface`).
2. **Preprocessing**: GPU-based normalization.
3. **Inference**: Enqueued to the DLA context.
4. **Post-processing**: NMS and Conformal Prediction performed on GPU.

## 4. Requirements & Setup

- **Hardware**: NVIDIA Jetson Orin (AGX or NX).
- **Software**:
  - JetPack 5.x / 6.x
  - ROS 2 Humble/Jazzy
  - TensorRT 8.x / 10.x
  - `pytorch-quantization` toolkit
  - `ultralytics`

## 5. Project Structure

- `unina_yolo_dla/train.py`: Main training script with QAT logic.
- `unina_yolo_dla/auto_labeler.py`: Offline auto-labeling pipeline.
- `unina_yolo_dla/active_learning.py`: Dataset curation and Active Learning logic.
- `unina_yolo_dla/data_loader.py`: Reality-driven Dataloader with weight-based sampling.
- `unina_yolo_dla/model.py`: DLA-optimized architecture definition.
- `unina_yolo_dla/eval.py`: Evaluation script for local metrics.
- `ros2_ws/src/perception/`: ROS 2 package for C++ inference node.


