"""
UNINA-YOLO-DLA: Quantization Aware Training (QAT) Module.

This module provides QAT-enabled versions of the model building blocks
for INT8 quantization on NVIDIA DLA with maximum accuracy preservation.

Requirements:
    - pytorch-quantization (NVIDIA's toolkit)
    - Install: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

Key Concepts:
    - QAT inserts "fake quantization" nodes during training
    - The model learns to compensate for quantization noise
    - Results in significantly better INT8 accuracy vs PTQ (Post-Training Quantization)
    - Critical for small object detection (cones at 15m+)

Usage:
    1. Replace standard model with QAT version
    2. Calibrate quantization ranges
    3. Fine-tune for a few epochs
    4. Export to ONNX -> TensorRT

Reference:
    - NVIDIA Blog: "Deploying YOLOv5 on NVIDIA Jetson Orin with cuDLA"
    - RESEARCH.md Section 4.3
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

# Try to import NVIDIA's pytorch-quantization toolkit
try:
    from pytorch_quantization import quant_modules
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import calib
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False
    print("WARNING: pytorch-quantization not found. QAT features disabled.")
    print("Install with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")


# --- Quantization Configuration ---

def initialize_quantization(calibrator: str = "histogram") -> None:
    """
    Initialize the global quantization configuration.
    
    Must be called BEFORE creating the model if using QAT.
    
    Args:
        calibrator: Calibration method - "histogram" (entropy) or "max" (minmax).
                   Use "histogram" for better small object accuracy.
    """
    if not QUANT_AVAILABLE:
        print("Quantization not available. Skipping initialization.")
        return
    
    # Configure the quantization descriptor
    # Using per-tensor quantization for DLA compatibility
    if calibrator == "histogram":
        # Histogram (Entropy) calibration - better for small objects
        quant_desc = QuantDescriptor(
            num_bits=8,
            calib_method="histogram",
            axis=None,  # Per-tensor quantization
        )
    else:
        # Max (MinMax) calibration
        quant_desc = QuantDescriptor(
            num_bits=8,
            calib_method="max",
            axis=None,
        )
    
    # Apply globally
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc)
    
    print(f"[QAT] Quantization initialized with {calibrator} calibration.")


def enable_calibration(model: nn.Module) -> None:
    """
    Enable calibration mode on all quantization nodes.
    
    In this mode, the model collects statistics to determine
    optimal quantization ranges.
    """
    if not QUANT_AVAILABLE:
        return
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    
    print("[QAT] Calibration mode enabled.")


def disable_calibration(model: nn.Module) -> None:
    """
    Disable calibration and enable quantization.
    
    Call this after calibration is complete to enable
    fake-quantization for QAT fine-tuning.
    """
    if not QUANT_AVAILABLE:
        return
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
    
    print("[QAT] Calibration disabled, quantization enabled.")


def collect_calibration_stats(
    model: nn.Module,
    calibration_dataloader,
    num_batches: int = 100,
    device: str = "cuda",
) -> None:
    """
    Collect calibration statistics from a representative dataset.
    
    Args:
        model: The QAT-enabled model.
        calibration_dataloader: DataLoader with calibration images.
        num_batches: Number of batches to use for calibration.
        device: Device to run calibration on.
    """
    if not QUANT_AVAILABLE:
        print("Skipping calibration - pytorch-quantization not available.")
        return
    
    # Robust device resolution: '0' -> 'cuda:0', 'cpu' -> 'cpu'
    if isinstance(device, str) and device.isdigit():
        device = f"cuda:{device}"
    
    model.eval()
    model.to(device)
    enable_calibration(model)
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_dataloader):
            if i >= num_batches:
                break
            
            # Calibration dataloader can return (images, labels) tuple or dict
            if isinstance(batch, dict):
                images = batch.get("images") or batch.get("image")
            else:
                images = batch[0]
                
            if images is None: continue
            
            images = images.to(device)
            model(images)
            if (i + 1) % 10 == 0:
                print(f"[QAT] Calibration batch {i + 1}/{num_batches}")
    
    disable_calibration(model)
    print("[QAT] Calibration complete.")


# --- QAT Building Blocks ---

class QuantConvBlock(nn.Module):
    """
    Quantized Convolution + BatchNorm + ReLU block.
    
    Uses QuantConv2d for fake-quantization during training.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        if QUANT_AVAILABLE:
            self.conv = quant_nn.QuantConv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False
            )
        else:
            # Fallback to standard Conv2d if quant toolkit not available
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class QuantBottleneck(nn.Module):
    """
    Quantized Bottleneck block with residual connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = QuantConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = QuantConvBlock(hidden_channels, out_channels, kernel_size=3)
        self.add = shortcut and in_channels == out_channels
        
        # Quantized addition for residual
        if QUANT_AVAILABLE and self.add:
            self.residual_quantizer = quant_nn.TensorQuantizer(
                QuantDescriptor(num_bits=8, calib_method="histogram")
            )
        else:
            self.residual_quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        if self.add:
            if self.residual_quantizer is not None:
                out = out + self.residual_quantizer(x)
            else:
                out = out + x
        return out


class QuantC3k2(nn.Module):
    """
    Quantized C3k2 block for CSP-style feature extraction.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.cv1 = QuantConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = QuantConvBlock(in_channels, hidden_channels, kernel_size=1)
        
        self.bottlenecks = nn.Sequential(
            *[QuantBottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) 
              for _ in range(n)]
        )
        
        self.cv3 = QuantConvBlock(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path1 = self.bottlenecks(self.cv1(x))
        path2 = self.cv2(x)
        return self.cv3(torch.cat([path1, path2], dim=1))


class QuantSPPF_DLA(nn.Module):
    """
    Quantized SPPF block (DLA-compatible).
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 5) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = QuantConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = QuantConvBlock(hidden_channels * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# --- Full QAT Model ---

class UNINA_YOLO_DLA_QAT(nn.Module):
    """
    Quantization Aware Training version of UNINA-YOLO-DLA.
    
    This model uses QuantConv2d layers that insert fake-quantization
    nodes during training, allowing the network to learn to compensate
    for INT8 quantization noise.
    
    Usage:
        1. initialize_quantization()  # Before model creation
        2. model = UNINA_YOLO_DLA_QAT()
        3. collect_calibration_stats(model, calib_loader)
        4. Train/fine-tune the model
        5. Export to ONNX -> TensorRT
    """
    def __init__(
        self,
        num_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        
        # --- Backbone ---
        self.stem = QuantConvBlock(3, c1, kernel_size=3, stride=2)
        
        self.stage1_conv = QuantConvBlock(c1, c2, kernel_size=3, stride=2)
        self.stage1_c3k2 = QuantC3k2(c2, c2, n=1)
        
        self.stage2_conv = QuantConvBlock(c2, c3, kernel_size=3, stride=2)
        self.stage2_c3k2 = QuantC3k2(c3, c3, n=2)
        
        self.stage3_conv = QuantConvBlock(c3, c4, kernel_size=3, stride=2)
        self.stage3_c3k2 = QuantC3k2(c4, c4, n=2)
        
        self.stage4_conv = QuantConvBlock(c4, c5, kernel_size=3, stride=2)
        self.stage4_sppf = QuantSPPF_DLA(c5, c5)
        
        # --- Neck (FPN + PAN) ---
        # Top-Down
        self.lateral_p4 = QuantConvBlock(c5, c4, kernel_size=1)
        self.fpn_c3k2_1 = QuantC3k2(c4 * 2, c4, n=1)
        
        self.lateral_p3 = QuantConvBlock(c4, c3, kernel_size=1)
        self.fpn_c3k2_2 = QuantC3k2(c3 * 2, c3, n=1)
        
        self.lateral_p2 = QuantConvBlock(c3, c2, kernel_size=1)
        self.fpn_c3k2_3 = QuantC3k2(c2 * 2, c2, n=1)
        
        # Bottom-Up
        self.down1 = QuantConvBlock(c2, c2, kernel_size=3, stride=2)
        self.pan_c3k2_1 = QuantC3k2(c2 + c3, c3, n=1)
        
        self.down2 = QuantConvBlock(c3, c3, kernel_size=3, stride=2)
        self.pan_c3k2_2 = QuantC3k2(c3 + c4, c4, n=1)
        
        # --- Detection Heads ---
        self.head_p2_cls = nn.Sequential(
            QuantConvBlock(c2, c2, kernel_size=3),
            QuantConvBlock(c2, c2, kernel_size=3),
            nn.Conv2d(c2, num_classes, kernel_size=1),
        )
        self.head_p2_reg = nn.Sequential(
            QuantConvBlock(c2, c2, kernel_size=3),
            QuantConvBlock(c2, c2, kernel_size=3),
            nn.Conv2d(c2, 4, kernel_size=1),
        )
        
        self.head_p3_cls = nn.Sequential(
            QuantConvBlock(c3, c3, kernel_size=3),
            QuantConvBlock(c3, c3, kernel_size=3),
            nn.Conv2d(c3, num_classes, kernel_size=1),
        )
        self.head_p3_reg = nn.Sequential(
            QuantConvBlock(c3, c3, kernel_size=3),
            QuantConvBlock(c3, c3, kernel_size=3),
            nn.Conv2d(c3, 4, kernel_size=1),
        )
        
        self.head_p4_cls = nn.Sequential(
            QuantConvBlock(c4, c4, kernel_size=3),
            QuantConvBlock(c4, c4, kernel_size=3),
            nn.Conv2d(c4, num_classes, kernel_size=1),
        )
        self.head_p4_reg = nn.Sequential(
            QuantConvBlock(c4, c4, kernel_size=3),
            QuantConvBlock(c4, c4, kernel_size=3),
            nn.Conv2d(c4, 4, kernel_size=1),
        )
        
        self.out_channels = [c2, c3, c4]

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # --- Backbone ---
        x = self.stem(x)
        
        x = self.stage1_conv(x)
        p2 = self.stage1_c3k2(x)
        
        x = self.stage2_conv(p2)
        p3 = self.stage2_c3k2(x)
        
        x = self.stage3_conv(p3)
        p4 = self.stage3_c3k2(x)
        
        x = self.stage4_conv(p4)
        p5_sppf = self.stage4_sppf(x)
        
        # --- Neck (Top-Down) ---
        p5_up = nn.functional.interpolate(self.lateral_p4(p5_sppf), scale_factor=2, mode='nearest')
        p4_fused = self.fpn_c3k2_1(torch.cat([p5_up, p4], dim=1))
        
        p4_up = nn.functional.interpolate(self.lateral_p3(p4_fused), scale_factor=2, mode='nearest')
        p3_fused = self.fpn_c3k2_2(torch.cat([p4_up, p3], dim=1))
        
        p3_up = nn.functional.interpolate(self.lateral_p2(p3_fused), scale_factor=2, mode='nearest')
        p2_fused = self.fpn_c3k2_3(torch.cat([p3_up, p2], dim=1))
        
        # --- Neck (Bottom-Up) ---
        p2_down = self.down1(p2_fused)
        p3_out = self.pan_c3k2_1(torch.cat([p2_down, p3_fused], dim=1))
        
        p3_down = self.down2(p3_out)
        p4_out = self.pan_c3k2_2(torch.cat([p3_down, p4_fused], dim=1))
        
        # --- Heads ---
        p2_cls = self.head_p2_cls(p2_fused)
        p2_reg = self.head_p2_reg(p2_fused)
        
        p3_cls = self.head_p3_cls(p3_out)
        p3_reg = self.head_p3_reg(p3_out)
        
        p4_cls = self.head_p4_cls(p4_out)
        p4_reg = self.head_p4_reg(p4_out)
        
        return [(p2_cls, p2_reg), (p3_cls, p3_reg), (p4_cls, p4_reg)]


# --- Training Utilities ---

def create_qat_model(num_classes: int = 4, base_channels: int = 32) -> UNINA_YOLO_DLA_QAT:
    """
    Factory function to create a properly initialized QAT model.
    
    This handles the correct initialization order.
    """
    initialize_quantization(calibrator="histogram")
    model = UNINA_YOLO_DLA_QAT(num_classes=num_classes, base_channels=base_channels)
    return model


def export_qat_onnx(
    model: UNINA_YOLO_DLA_QAT,
    filepath: str,
    input_size: int = 640,
) -> None:
    """
    Export a QAT model to ONNX format.
    
    The exported ONNX will contain QuantizeLinear/DequantizeLinear nodes
    that TensorRT can interpret for INT8 inference.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    if QUANT_AVAILABLE:
        # Enable INT8 mode for export
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        opset_version=13,
        input_names=['images'],
        output_names=['p2_cls', 'p2_reg', 'p3_cls', 'p3_reg', 'p4_cls', 'p4_reg'],
        dynamic_axes=None,
    )
    
    print(f"[QAT] Model exported to {filepath}")


# --- Example Usage ---

if __name__ == '__main__':
    print("UNINA-YOLO-DLA QAT Module")
    print("=" * 40)
    
    if QUANT_AVAILABLE:
        print("pytorch-quantization is available.")
        
        # Create QAT model
        model = create_qat_model(num_classes=4, base_channels=32)
        model.eval()
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        outputs = model(dummy_input)
        
        print(f"Input Shape: {dummy_input.shape}")
        for i, (cls_out, reg_out) in enumerate(outputs):
            print(f"P{i+2} Output: cls={cls_out.shape}, reg={reg_out.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")
    else:
        print("pytorch-quantization not available. Install with:")
        print("  pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")
