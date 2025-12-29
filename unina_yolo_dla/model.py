"""
UNINA-YOLO-DLA: A DLA-Optimized Object Detection Model for NVIDIA Jetson Orin.

This module defines the architecture for the UNINA-YOLO-DLA model, designed
for the Formula Student Driverless perception stack. All operations are
selected for native DLA support, avoiding GPU fallback.

Key Design Decisions:
    - ReLU activation only (SiLU is NOT DLA-native).
    - P2 Head (stride 4) for small object detection (distant cones).
    - P5 Head removed (stride 32 is too coarse for this use case).
    - No dynamic shapes (input is fixed at 640x640).
    - All layers verified against NVIDIA DLA Supported Layers documentation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --- Core Building Blocks (DLA-Compatible) ---

class ConvBlock(nn.Module):
    """
    Standard Convolution + BatchNorm + ReLU block.
    This is the fundamental building block, fully DLA-compatible.
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
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # NOTE: Using ReLU instead of SiLU for DLA compatibility.
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Standard Bottleneck block with two convolutional layers.
    Residual connection is added if in_channels == out_channels.
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
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBlock(hidden_channels, out_channels, kernel_size=3)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3k2(nn.Module):
    """
    C3k2: Cross-Stage Partial Bottleneck block with 2 convolutions.
    This is a DLA-friendly version of the CSP block, avoiding complex slicing.
    
    Uses a simple split-process-concat strategy that DLA handles well.
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
        
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        
        # Sequential bottleneck stack
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(n)]
        )
        
        # Output convolution to combine features
        self.cv3 = ConvBlock(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two-path processing (DLA-compatible split)
        path1 = self.bottlenecks(self.cv1(x))
        path2 = self.cv2(x)
        # Concatenation is DLA-native
        return self.cv3(torch.cat([path1, path2], dim=1))


class SPPF_DLA(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (DLA Compatible Version).
    
    Avoids dynamic slicing. Uses sequential MaxPool2d with fixed kernel sizes.
    MaxPool is natively supported on DLA.
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 5) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBlock(hidden_channels * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Upsample(nn.Module):
    """
    Upsample wrapper using nearest-neighbor interpolation.
    Transposed Convolutions are NOT recommended on DLA.
    """
    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode='nearest'
        )


# --- Backbone Definition ---

class Backbone(nn.Module):
    """
    CSP-Darknet style backbone optimized for DLA.
    Outputs feature maps at P2 (stride 4), P3 (stride 8), P4 (stride 16).
    P5 is intentionally omitted for small-object focus.
    """
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels      # 32
        c2 = base_channels * 2  # 64
        c3 = base_channels * 4  # 128
        c4 = base_channels * 8  # 256
        c5 = base_channels * 16 # 512

        # Stem: Input (3, 640, 640) -> (c1, 320, 320)
        self.stem = ConvBlock(3, c1, kernel_size=3, stride=2)

        # Stage 1: (c1, 320, 320) -> (c2, 160, 160) = P2 Output Level
        self.stage1_conv = ConvBlock(c1, c2, kernel_size=3, stride=2)
        self.stage1_c3k2 = C3k2(c2, c2, n=1)

        # Stage 2: (c2, 160, 160) -> (c3, 80, 80) = P3 Output Level
        self.stage2_conv = ConvBlock(c2, c3, kernel_size=3, stride=2)
        self.stage2_c3k2 = C3k2(c3, c3, n=2)

        # Stage 3: (c3, 80, 80) -> (c4, 40, 40) = P4 Output Level
        self.stage3_conv = ConvBlock(c3, c4, kernel_size=3, stride=2)
        self.stage3_c3k2 = C3k2(c4, c4, n=2)
        
        # Stage 4: (c4, 40, 40) -> (c5, 20, 20) = For SPPF context (NOT P5 Head)
        self.stage4_conv = ConvBlock(c4, c5, kernel_size=3, stride=2)
        self.stage4_sppf = SPPF_DLA(c5, c5)
        
        self.out_channels = [c2, c3, c4, c5]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)

        x = self.stage1_conv(x)
        p2 = self.stage1_c3k2(x)  # Output: stride 4, 160x160

        x = self.stage2_conv(p2)
        p3 = self.stage2_c3k2(x)  # Output: stride 8, 80x80

        x = self.stage3_conv(p3)
        p4 = self.stage3_c3k2(x)  # Output: stride 16, 40x40
        
        x = self.stage4_conv(p4)
        p5_sppf = self.stage4_sppf(x) # Contextual features, not for P5 head

        return p2, p3, p4, p5_sppf


# --- Neck (PANet-style) ---

class Neck(nn.Module):
    """
    Feature Pyramid Network (FPN) + Path Aggregation Network (PAN) style Neck.
    Combines multi-scale features for detection at P2, P3, P4.
    """
    def __init__(self, in_channels: list[int]) -> None:
        super().__init__()
        c2, c3, c4, c5 = in_channels

        # Top-Down (FPN-like) Pathway
        self.up1 = Upsample(scale_factor=2) # c5 -> 40x40
        self.lateral_p4 = ConvBlock(c5, c4, kernel_size=1)
        self.fpn_c3k2_1 = C3k2(c4 * 2, c4, n=1)

        self.up2 = Upsample(scale_factor=2) # c4 -> 80x80
        self.lateral_p3 = ConvBlock(c4, c3, kernel_size=1)
        self.fpn_c3k2_2 = C3k2(c3 * 2, c3, n=1)
        
        self.up3 = Upsample(scale_factor=2) # c3 -> 160x160
        self.lateral_p2 = ConvBlock(c3, c2, kernel_size=1)
        self.fpn_c3k2_3 = C3k2(c2 * 2, c2, n=1)

        # Bottom-Up (PAN-like) Pathway
        self.down1 = ConvBlock(c2, c2, kernel_size=3, stride=2) # 160->80
        self.pan_c3k2_1 = C3k2(c2 + c3, c3, n=1)

        self.down2 = ConvBlock(c3, c3, kernel_size=3, stride=2) # 80->40
        self.pan_c3k2_2 = C3k2(c3 + c4, c4, n=1)
        
        self.out_channels = [c2, c3, c4]

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        p2_in, p3_in, p4_in, p5_sppf = features

        # --- Top-Down (FPN) ---
        p5_up = self.up1(self.lateral_p4(p5_sppf))
        p4_fused = self.fpn_c3k2_1(torch.cat([p5_up, p4_in], dim=1))

        p4_up = self.up2(self.lateral_p3(p4_fused))
        p3_fused = self.fpn_c3k2_2(torch.cat([p4_up, p3_in], dim=1))
        
        p3_up = self.up3(self.lateral_p2(p3_fused))
        p2_fused = self.fpn_c3k2_3(torch.cat([p3_up, p2_in], dim=1))

        # --- Bottom-Up (PAN) ---
        p2_down = self.down1(p2_fused)
        p3_out = self.pan_c3k2_1(torch.cat([p2_down, p3_fused], dim=1))

        p3_down = self.down2(p3_out)
        p4_out = self.pan_c3k2_2(torch.cat([p3_down, p4_fused], dim=1))

        return p2_fused, p3_out, p4_out


# --- Detection Head (Decoupled) ---

class DetectionHead(nn.Module):
    """
    Decoupled Detection Head for a single feature level.
    Predicts bounding boxes and class probabilities.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels

        # Classification branch
        self.cls_branch = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=3),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, num_classes * num_anchors, kernel_size=1),
        )

        # Regression branch (x, y, w, h)
        self.reg_branch = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=3),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 4 * num_anchors, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_branch(x), self.reg_branch(x)


# --- Full Model ---

class UNINA_YOLO_DLA(nn.Module):
    """
    UNINA-YOLO-DLA Main Model Class.

    A YOLOv11-inspired architecture heavily modified for DLA execution on Jetson Orin.
    
    Architecture:
        - Backbone: CSP-Darknet with ReLU, outputs P2, P3, P4.
        - Neck: FPN + PAN for multi-scale feature fusion.
        - Head: Decoupled heads for P2 (stride 4), P3 (stride 8), P4 (stride 16).
        - NO P5 Head (stride 32 is too coarse for small cone detection).
    
    Args:
        num_classes: Number of object classes (e.g., 4 for yellow, blue, orange, large orange).
        base_channels: Base channel width for the backbone.
    """
    def __init__(
        self,
        num_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = Backbone(base_channels=base_channels)
        self.neck = Neck(self.backbone.out_channels)

        # Detection heads for P2, P3, P4
        self.head_p2 = DetectionHead(self.neck.out_channels[0], num_classes)
        self.head_p3 = DetectionHead(self.neck.out_channels[1], num_classes)
        self.head_p4 = DetectionHead(self.neck.out_channels[2], num_classes)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 640, 640).

        Returns:
            A list of tuples, one for each scale (P2, P3, P4).
            Each tuple contains (cls_output, reg_output).
        """
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)

        p2_out = self.head_p2(neck_features[0])  # Stride 4
        p3_out = self.head_p3(neck_features[1])  # Stride 8
        p4_out = self.head_p4(neck_features[2])  # Stride 16

        return [p2_out, p3_out, p4_out]

    def export_onnx(self, filepath: str, input_size: int = 640) -> None:
        """
        Exports the model to ONNX format for TensorRT.
        
        Args:
            filepath: Path to save the .onnx file.
            input_size: The fixed input resolution.
        """
        self.eval()
        dummy_input = torch.randn(1, 3, input_size, input_size)
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            opset_version=13,
            input_names=['images'],
            output_names=['p2_cls', 'p2_reg', 'p3_cls', 'p3_reg', 'p4_cls', 'p4_reg'],
            dynamic_axes=None,  # Static shape is mandatory for DLA.
        )
        print(f"Model exported to {filepath}")


if __name__ == '__main__':
    # Quick test
    model = UNINA_YOLO_DLA(num_classes=4, base_channels=32)
    model.eval()
    
    # Test input
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    
    print("UNINA-YOLO-DLA Model Instantiated Successfully.")
    print(f"  Input Shape: {dummy_input.shape}")
    for i, (cls_out, reg_out) in enumerate(outputs):
        print(f"  P{i+2} Output: cls={cls_out.shape}, reg={reg_out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")
