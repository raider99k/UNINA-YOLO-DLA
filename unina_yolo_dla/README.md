# UNINA-YOLO-DLA Project

This project implements a high-performance cone detection pipeline for Formula Student Driverless, optimized for the NVIDIA Jetson Orin's Deep Learning Accelerator (DLA).

## Architecture
- **Backbone**: CSP-Darknet (ReLU-only, no P5 head).
- **Optimization**: QAT (Quantization Aware Training) for INT8 precision.
- **Inference**: TensorRT 8.x/10.x on DLA Core 1.
- **Zero-Copy**: GPU-to-GPU data transfer via `NvBufSurface` and `GpuBufferHandle`.
- **ROS 2**: Lifecycle node for robust state management.

## Project Structure
- `model.py`: Standard PyTorch model definition.
- `qat.py`: QAT-enabled model using `pytorch-quantization`.
- `export_trt.py`: Python script for ONNX export and TensorRT engine building with DLA checks.
- `data_loader.py`: Hybrid data loader mixing real and synthetic datasets.
- `ros2_ws/src/perception/`: ROS 2 package containing the C++ node and CUDA kernels.

## Setup & Deployment
Refer to [walkthrough.md](.gemini/antigravity/brain/f6245b09-992a-477b-9565-6d7137946bb6/walkthrough.md) for detailed instructions on training, exporting, and running the node.

## Requirements
- NVIDIA Jetson Orin (AGX or NX).
- JetPack 5.x / 6.x.
- ROS 2 Humble/Jazzy.
- `pytorch-quantization` toolkit.
