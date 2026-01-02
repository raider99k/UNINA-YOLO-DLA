# UNINA-YOLO-DLA

**UNINA-YOLO-DLA** is a high-performance perception pipeline developed by the UNINA Formula Student Team. It focuses on object detection for autonomous racing, specifically optimized for NVIDIA Jetson Orin DLA (Deep Learning Accelerator).

## 🚀 Key Features

- **Hardware-Aware Architecture**: YOLO-based model with ReLU activations and DLA-optimized blocks.
- **Advanced Training**: Support for Quantization Aware Training (QAT) to preserve INT8 accuracy on DLA.
- **Active Learning**: Intelligent dataset curation using Entropy sampling and Coreset selection.
- **Robust Detection**: Optimized P2/P3/P4 heads for small object detection (cones at distance).

## 🛠️ Quick Start

### 1. Requirements

- Python 3.10+
- (Optional) CUDA-enabled GPU and NVIDIA TensorRT for full performance.

### 2. Environment Setup

We provide scripts to automate the creation of a virtual environment and dependency installation.

**Windows (PowerShell):**
```powershell
.\setup_env.ps1
```

**Linux (Bash):**
```bash
./setup_env.sh
```

### 3. Local Dry-Run (No GPU required)

To verify the training pipeline on your local machine without CUDA/DLA:
```powershell
.\run_local_debug.ps1
```

## 📂 Project Structure

- `train.py`: Main training script for FP32 and QAT phases.
- `model.py`: DLA-optimized model definitions.
- `active_learning.py`: Active learning strategies and orchestrator.
- `data_loader.py`: Specialized dataset loaders for hybrid training.
- `ros2_ws/`: Production perception nodes for ROS 2.

## 📈 Training Pipeline

The training follows a two-phase approach:
1. **Phase 1 (FP32)**: Standard training with hardware-aware constraints.
2. **Phase 2 (QAT)**: Fine-tuning with fake quantization to prepare for INT8 deployment.

## 🏎️ Deployment

Models are exported to ONNX and then compiled to TensorRT engines with DLA core selection. Ensure zero-fallback on GPU for maximum efficiency.
