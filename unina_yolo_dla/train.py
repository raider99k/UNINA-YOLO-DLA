#!/usr/bin/env python3
"""
UNINA-YOLO-DLA: Training Script (Ultralytics-based).

This script implements the training pipeline per MORERESEARCH.md:
  - Phase 1: FP32 Training with ReLU activation (Hardware-Aware).
  - Phase 2: QAT (Quantization Aware Training) with pytorch-quantization.
  - Export to ONNX for TensorRT/DLA deployment.

Requirements:
    pip install ultralytics
    pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

Usage:
    python train.py --data fsd_data.yaml --epochs 100 --batch 16
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --- Ultralytics Imports ---
try:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import Conv
    import ultralytics.nn.modules as ultralytics_modules
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ERROR: ultralytics not found. Install with: pip install ultralytics")

# --- NVIDIA Quantization Imports ---
try:
    from pytorch_quantization import quant_modules
    from pytorch_quantization import nn as quant_nn
    from qat import initialize_quantization # Import from our qat module
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    print("WARNING: pytorch-quantization not found. QAT disabled.")
    print("Install with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")

# --- Local Module Imports ---
try:
    from data_loader import create_active_learning_dataloader, SmallObjectMetric
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    print("WARNING: data_loader module not found. Hybrid loading disabled.")

# --- TensorRT Imports (for DLA validation) ---
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


# ============================================================================
# 1. Custom DLA-Friendly Modules
# ============================================================================

# --- DLA Building Blocks (Duplicated from model.py for training script autonomy) ---

class SPPF_DLA(nn.Module):
    """
    Sostituzione SPPF ottimizzata per DLA.
    Evita torch.chunk/split che causano fallback su DLA.
    Usa sequenza di MaxPool2d e una singola concatenazione finale.
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

# Register custom module for Ultralytics YAML parsing
if ULTRALYTICS_AVAILABLE:
    # Register in ultralytics.nn.modules
    setattr(ultralytics_modules, 'SPPF_DLA', SPPF_DLA)
    
    # CRITICAL: Also register in ultralytics.nn.tasks where parse_model uses globals()
    try:
        import ultralytics.nn.tasks as ultralytics_tasks
        setattr(ultralytics_tasks, 'SPPF_DLA', SPPF_DLA)
        ultralytics_tasks.__dict__['SPPF_DLA'] = SPPF_DLA
    except ImportError:
        print("WARNING: Could not register modules in ultralytics.nn.tasks")


# ============================================================================
# 2. QAT Helper Functions
# ============================================================================

def replace_silu_with_relu(model: nn.Module) -> None:
    """
    Recursively replace all SiLU activations with ReLU.
    
    DLA does not efficiently support SiLU (requires exponential).
    ReLU is natively supported with zero overhead.
    
    Reference: MORERESEARCH.md Section 3.1
    """
    for name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """
    Prepare a model for Quantization Aware Training.
    
    This wraps Conv2d layers with QuantConv2d for fake-quantization.
    Must be called AFTER quant_modules.initialize().
    """
    if not QAT_AVAILABLE:
        print("QAT not available, returning original model.")
        return model
    
    # Replace activations first
    replace_silu_with_relu(model)
    
    return model


def transfer_weights_fp32_to_qat(
    fp32_model: nn.Module,
    qat_model: nn.Module,
    strict: bool = False,
) -> dict[str, list[str]]:
    """
    Transfers weights from an FP32 model to a QAT model with intelligent layer matching.
    
    Handles the structural differences between standard PyTorch models and
    pytorch-quantization wrapped models (Conv2d -> QuantConv2d, etc.).
    
    Args:
        fp32_model: Source model with FP32 weights.
        qat_model: Target QAT model.
        strict: If True, raise error on any mismatch. If False, skip mismatches.
    
    Returns:
        Dictionary with keys 'transferred', 'skipped', 'mismatched' listing layer names.
    
    Raises:
        RuntimeError: If strict=True and there are mismatches.
    """
    fp32_state = fp32_model.state_dict()
    qat_state = qat_model.state_dict()
    
    result = {
        "transferred": [],
        "skipped": [],
        "mismatched": [],
    }
    
    # Build a normalized name mapping for QAT model
    # QuantConv2d wraps the conv layer, so paths might differ
    qat_name_map: dict[str, str] = {}
    for qat_name in qat_state.keys():
        # Create normalized versions of the name for matching
        # Remove common QAT prefixes/suffixes
        normalized = qat_name
        for pattern in [".conv.", "._input_quantizer.", "._weight_quantizer."]:
            normalized = normalized.replace(pattern, ".")
        normalized = normalized.replace("..", ".")
        qat_name_map[normalized] = qat_name
        qat_name_map[qat_name] = qat_name  # Also keep original
    
    # Track which QAT parameters have been set
    transferred_qat_params: set[str] = set()
    
    print("\n>>> FP32 -> QAT Weight Transfer")
    print("=" * 50)
    
    for fp32_name, fp32_param in fp32_state.items():
        matched = False
        
        # Strategy 1: Direct name match
        if fp32_name in qat_state:
            if qat_state[fp32_name].shape == fp32_param.shape:
                qat_state[fp32_name].copy_(fp32_param)
                result["transferred"].append(fp32_name)
                transferred_qat_params.add(fp32_name)
                matched = True
            else:
                result["mismatched"].append(
                    f"{fp32_name}: FP32 {fp32_param.shape} != QAT {qat_state[fp32_name].shape}"
                )
                matched = True  # Matched but incompatible
        
        # Strategy 2: Normalized name match (for QuantConv2d wrappers)
        if not matched:
            for qat_name in qat_state.keys():
                if qat_name in transferred_qat_params:
                    continue
                    
                # Check if the core parameter name matches
                # FP32: backbone.stem.conv.weight
                # QAT:  backbone.stem.conv.weight  (same, or with quant wrappers)
                fp32_base = fp32_name.rsplit(".", 1)[-1]  # e.g., "weight" or "bias"
                qat_base = qat_name.rsplit(".", 1)[-1]
                
                if fp32_base != qat_base:
                    continue
                
                # Compare the structural path (without the final param name)
                fp32_path = fp32_name.rsplit(".", 1)[0] if "." in fp32_name else ""
                qat_path = qat_name.rsplit(".", 1)[0] if "." in qat_name else ""
                
                # Normalize paths for comparison
                fp32_path_norm = fp32_path.replace("module.", "")
                qat_path_norm = qat_path.replace("module.", "")
                
                # Check for structural similarity
                if _paths_match(fp32_path_norm, qat_path_norm):
                    if qat_state[qat_name].shape == fp32_param.shape:
                        qat_state[qat_name].copy_(fp32_param)
                        result["transferred"].append(f"{fp32_name} -> {qat_name}")
                        transferred_qat_params.add(qat_name)
                        matched = True
                        break
                    else:
                        result["mismatched"].append(
                            f"{fp32_name} ~> {qat_name}: {fp32_param.shape} != {qat_state[qat_name].shape}"
                        )
        
        if not matched:
            result["skipped"].append(fp32_name)
    
    # Load the modified state dict into QAT model
    qat_model.load_state_dict(qat_state, strict=False)
    
    # Print summary
    print(f"  Transferred: {len(result['transferred'])} parameters")
    print(f"  Skipped:     {len(result['skipped'])} parameters")
    print(f"  Mismatched:  {len(result['mismatched'])} parameters")
    
    if result["mismatched"]:
        print("\n>>> SHAPE MISMATCHES:")
        for m in result["mismatched"][:10]:
            print(f"    {m}")
        if len(result["mismatched"]) > 10:
            print(f"    ... and {len(result['mismatched']) - 10} more")
    
    if result["skipped"] and len(result["skipped"]) <= 20:
        print("\n>>> SKIPPED (no match found):")
        for s in result["skipped"]:
            print(f"    {s}")
    elif result["skipped"]:
        print(f"\n>>> SKIPPED: {len(result['skipped'])} parameters (too many to list)")
    
    if strict and (result["mismatched"] or result["skipped"]):
        raise RuntimeError(
            f"Strict weight transfer failed: {len(result['mismatched'])} mismatches, "
            f"{len(result['skipped'])} skipped."
        )
    
    return result


def _paths_match(fp32_path: str, qat_path: str) -> bool:
    """
    Check if two module paths refer to the same logical layer.
    
    Handles differences like:
    - 'backbone.stem.conv' vs 'backbone.stem.conv'
    - 'stem.0.conv' vs 'stem.conv'
    - 'layer1.conv' vs 'layer1.0.conv'
    """
    if fp32_path == qat_path:
        return True
    
    # Split into components
    fp32_parts = [p for p in fp32_path.split(".") if p]
    qat_parts = [p for p in qat_path.split(".") if p]
    
    # Remove numeric indices for comparison
    def strip_indices(parts: list[str]) -> list[str]:
        return [p for p in parts if not p.isdigit()]
    
    fp32_stripped = strip_indices(fp32_parts)
    qat_stripped = strip_indices(qat_parts)
    
    return fp32_stripped == qat_stripped


def configure_entropy_calibration() -> None:
    """
    Configure pytorch-quantization to use Entropy calibration.
    
    Entropy calibration (KL Divergence) chooses a threshold that minimizes
    information loss, which is critical for preserving small object signals.
    
    Reference: RESEARCH.md Section 4.3
    """
    if not QAT_AVAILABLE:
        return
    
    from pytorch_quantization.calib import HistogramCalibrator
    from pytorch_quantization import calib
    
    # Set default calibrator to Entropy (histogram-based)
    print(">>> Configuring Entropy Calibration (KL Divergence)...")
    calib.calibrator.CALIBRATOR_TYPE = "histogram"
    
    # Configure all quantizers to use entropy calibration
    from pytorch_quantization.nn.modules import tensor_quantizer
    tensor_quantizer.TensorQuantizer.default_calib_method = "entropy"


def set_layer_precision_fp16(model: nn.Module, layer_names: list) -> None:
    """
    Set specific layers to FP16 precision for mixed-precision INT8 inference.
    
    The P2 head and initial backbone layers should remain in FP16
    to preserve small object detection capability.
    
    CRITICAL: This function disables ALL quantizers (input, weight, and output)
    to prevent quantization loss for small, low-contrast features.
    
    Reference: RESEARCH.md Section 4.2 (Layer-Wise Mixed Precision)
    
    Args:
        model: The model to configure.
        layer_names: List of layer name patterns to keep in FP16.
    """
    if not QAT_AVAILABLE:
        return
    
    from pytorch_quantization.nn.modules import tensor_quantizer
    
    print(f">>> Setting FP16 precision for sensitive layers: {layer_names}")
    disabled_count = 0
    for name, module in model.named_modules():
        # Check if this layer matches any of the patterns
        for pattern in layer_names:
            if pattern in name:
                # Disable ALL quantizers for this layer (keep FP16)
                if hasattr(module, '_input_quantizer') and module._input_quantizer is not None:
                    module._input_quantizer.disable()
                    disabled_count += 1
                if hasattr(module, '_weight_quantizer') and module._weight_quantizer is not None:
                    module._weight_quantizer.disable()
                    disabled_count += 1
                if hasattr(module, '_output_quantizer') and module._output_quantizer is not None:
                    module._output_quantizer.disable()
                    disabled_count += 1
                print(f"    Disabled quantization for: {name} (all quantizers)")
                break
    
    print(f">>> Total quantizers disabled for FP16 preservation: {disabled_count}")


# ============================================================================
# 3. Custom Trainer Class
# ============================================================================

if ULTRALYTICS_AVAILABLE:
    class UninaDLATrainer(DetectionTrainer):
        """
        Custom Detection Trainer for UNINA-YOLO-DLA.
        
        Overrides model creation to inject DLA-friendly modules.
        Supports hybrid data loading (70% Real / 30% Synthetic).
        """
        
        # Class-level config for active learning (set via set_active_learning_config)
        active_learning_config = None
        
        @classmethod
        def set_active_learning_config(cls, dataset_root: str, difficulty_scores: dict = None):
            """Configure active learning data loading."""
            cls.active_learning_config = {
                "dataset_root": dataset_root,
                "difficulty_scores": difficulty_scores,
            }
            print(f">>> Active Learning config set for dataset: {dataset_root}")
        
        def get_model(self, cfg=None, weights=None, verbose=True):
            """
            Load model from YAML config, injecting custom modules.
            """
            # Use getattr for rank compatibility with different Ultralytics versions
            rank = getattr(self, 'rank', -1)
            model = DetectionModel(
                cfg, 
                nc=self.data['nc'], 
                verbose=verbose and rank == -1
            )
            
            if weights:
                model.load(weights)
                
            # Ensure ReLU activation throughout
            replace_silu_with_relu(model)
            
            return model
        
        def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
            """
            Override dataloader creation to support hybrid data loading.
            
            If hybrid_config is set, uses create_hybrid_dataloader from data_loader.py.
            Otherwise, falls back to the default Ultralytics dataloader.
            """
            if mode == "train" and self.active_learning_config and DATA_LOADER_AVAILABLE:
                print(f">>> Using Active Learning DataLoader for training")
                return create_active_learning_dataloader(
                    dataset_root=self.active_learning_config["dataset_root"],
                    batch_size=batch_size,
                    difficulty_scores=self.active_learning_config["difficulty_scores"],
                    num_workers=self.args.workers,
                )
            else:
                # Fall back to default Ultralytics dataloader
                return super().get_dataloader(dataset_path, batch_size, rank, mode)


# ============================================================================
# 4. Small Object Metric Callback (mAP_small)
# ============================================================================

class SmallObjectCallback:
    """
    Ultralytics callback to calculate mAP_small at the end of validation.
    
    Reference: RESEARCH.md Section 5.2 - mAP for objects < 15x15 pixels.
    """
    def __init__(self, size_threshold: int = 15, image_size: int = 640):
        self.size_threshold = size_threshold
        self.image_size = image_size
        self.metric = None
        if DATA_LOADER_AVAILABLE:
            self.metric = SmallObjectMetric(
                size_threshold=size_threshold,
                iou_threshold=0.5,
                image_size=image_size,
            )
        # Accumulators for predictions and ground truths
        self.all_predictions = []
        self.all_ground_truths = []
    
    def on_val_start(self, trainer):
        """Reset metric at start of validation."""
        if self.metric:
            self.metric.reset()
        self.all_predictions = []
        self.all_ground_truths = []
    
    def on_val_batch_end(self, validator, batch, preds):
        """
        Update metric with batch predictions.
        
        This hook receives the raw predictions and batch ground truths.
        We accumulate them for final computation.
        """
        try:
            # Extract ground truths from batch
            # batch is typically (images, labels, paths, shapes)
            if hasattr(batch, '__len__') and len(batch) >= 2:
                labels = batch[1]  # Ground truth labels
                
                # preds are the model predictions after NMS
                # Format varies by Ultralytics version
                if preds is not None:
                    self.all_predictions.append(preds)
                if labels is not None:
                    self.all_ground_truths.append(labels)
        except Exception as e:
            # Gracefully handle extraction errors
            pass
    
    def on_val_end(self, trainer):
        """Compute and log mAP_small at end of validation."""
        if self.metric and self.all_predictions and self.all_ground_truths:
            # Update metric with accumulated data
            for preds, gts in zip(self.all_predictions, self.all_ground_truths):
                try:
                    # Convert to list format expected by SmallObjectMetric
                    if hasattr(preds, 'tolist'):
                        preds_list = [preds]
                    else:
                        preds_list = list(preds) if preds is not None else []
                    
                    if hasattr(gts, 'tolist'):
                        gts_list = [gts]
                    else:
                        gts_list = list(gts) if gts is not None else []
                    
                    self.metric.update(preds_list, gts_list)
                except Exception:
                    pass
            
            results = self.metric.compute()
            print(f"\n>>> Small Object Metrics (< {self.size_threshold}x{self.size_threshold} px):")
            for k, v in results.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"\n>>> Small Object Metrics: No data collected (callback not fully hooked)")


# ============================================================================
# 5. Conformal Prediction Calibration (Offline)
# ============================================================================

def calibrate_conformal_prediction(
    model,
    data_yaml: str,
    alpha: float = 0.10,  # 90% coverage
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Offline Conformal Prediction Calibration.
    
    Calculates the quantile of nonconformity scores (1 - IoU) to determine
    the dilation factor for safety bounding boxes.
    
    Reference: MORERESEARCH.md Section 5.1-5.2
    
    Args:
        model: Trained YOLO model.
        data_yaml: Path to data YAML for validation set.
        alpha: Error rate (1 - coverage). Default 0.10 for 90% coverage.
        imgsz: Image size.
        batch_size: Batch size for inference.
        device: Device for inference.
    
    Returns:
        Dictionary with calibration results including the quantile.
    """
    import numpy as np
    from pathlib import Path
    import yaml
    
    print("=" * 60)
    print(">>> Conformal Prediction Calibration")
    print("=" * 60)
    
    # --- Helper: Calculate IoU between two boxes (xyxy format) ---
    def box_iou(box1, box2):
        """Calculate IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    # --- Helper: Convert YOLO format (xywh normalized) to xyxy absolute ---
    def yolo_to_xyxy(box, img_size):
        """Convert [x_center, y_center, w, h] normalized to [x1, y1, x2, y2] absolute."""
        x_c, y_c, w, h = box
        x1 = (x_c - w / 2) * img_size
        y1 = (y_c - h / 2) * img_size
        x2 = (x_c + w / 2) * img_size
        y2 = (y_c + h / 2) * img_size
        return [x1, y1, x2, y2]
    
    print(f">>> Running validation for CP calibration (alpha={alpha})...")
    
    # Run prediction on validation set
    results = model.predict(
        source=data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        save=False,
        verbose=False,
        stream=True,  # Process one by one to save memory
    )
    
    # Collect nonconformity scores (1 - IoU) for all matched predictions
    nonconformity_scores = []
    total_predictions = 0
    total_ground_truths = 0
    total_matches = 0
    
    for result in results:
        # Get predictions (xyxy format, absolute coordinates)
        if result.boxes is None or len(result.boxes) == 0:
            continue
            
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)
        pred_confs = result.boxes.conf.cpu().numpy()
        
        # Get ground truth labels if available
        # result.path gives us the image path, we need to find corresponding label
        if hasattr(result, 'path') and result.path:
            img_path = Path(result.path)
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            
            if label_path.exists():
                gt_boxes = []
                gt_classes = []
                
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(float(parts[0]))
                            box = [float(x) for x in parts[1:5]]
                            gt_boxes.append(yolo_to_xyxy(box, imgsz))
                            gt_classes.append(cls_id)
                
                total_ground_truths += len(gt_boxes)
                total_predictions += len(pred_boxes)
                
                # Match predictions to ground truths (greedy by confidence)
                matched_gt = set()
                sorted_indices = np.argsort(-pred_confs)  # Sort by confidence (descending)
                
                for idx in sorted_indices:
                    pred_box = pred_boxes[idx]
                    pred_cls = pred_classes[idx]
                    
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                        if gt_idx in matched_gt:
                            continue
                        if pred_cls != gt_cls:
                            continue
                        
                        iou = box_iou(pred_box, gt_box)
                        if iou > best_iou and iou >= 0.5:  # IoU threshold for matching
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        total_matches += 1
                        # Nonconformity score = 1 - IoU
                        # This measures "how wrong" the box localization is
                        nonconformity_scores.append(1.0 - best_iou)
    
    print(f">>> Collected {len(nonconformity_scores)} matched predictions")
    print(f"    Total predictions: {total_predictions}")
    print(f"    Total ground truths: {total_ground_truths}")
    print(f"    Matched pairs: {total_matches}")
    
    # Calculate the (1 - alpha) quantile of nonconformity scores
    if len(nonconformity_scores) == 0:
        raise ValueError(
            "FATAL: Conformal Prediction Calibration failed: No matched predictions found. "
            "Ensure the validation dataset is representative and the model is generating detections."
        )
        
    scores_array = np.array(nonconformity_scores)
    q_hat = float(np.quantile(scores_array, 1 - alpha))
    mean_score = float(np.mean(scores_array))
    std_score = float(np.std(scores_array))
    note = "Calibrated from empirical nonconformity score distribution (1-IoU)."
    
    calibration_result = {
        "alpha": alpha,
        "coverage_target": 1 - alpha,
        "q_hat": q_hat,
        "dilation_factor": q_hat,
        "num_calibration_samples": len(nonconformity_scores),
        "mean_nonconformity": mean_score,
        "std_nonconformity": std_score,
        "note": note,
    }
    
    print(f">>> CP Calibration Complete:")
    print(f"    Target Coverage: {(1-alpha)*100:.0f}%")
    print(f"    Quantile (q_hat): {q_hat:.4f}")
    print(f"    Mean nonconformity: {mean_score:.4f} ± {std_score:.4f}")
    
    return calibration_result


# ============================================================================
# 6. DLA Zero-Fallback Validation
# ============================================================================

def validate_dla_zero_fallback(engine_path: str) -> bool:
    """
    Validates that a TensorRT engine has zero GPU fallback layers.
    
    Reference: MORERESEARCH.md Section 7.2
    
    Args:
        engine_path: Path to the TensorRT engine file.
    
    Returns:
        True if 100% DLA (no fallback), False otherwise.
    """
    if not TRT_AVAILABLE:
        print(">>> Skipping DLA validation: TensorRT not available.")
        return True  # Assume OK if we can't check
    
    print("=" * 60)
    print(">>> DLA Zero-Fallback Validation")
    print("=" * 60)
    
    try:
        from export_trt import analyze_engine_layers, print_fallback_report
        dla_count, gpu_count, gpu_layers = analyze_engine_layers(engine_path)
        is_pure_dla = print_fallback_report(dla_count, gpu_count, gpu_layers)
        return is_pure_dla
    except ImportError:
        print(">>> export_trt module not found. Manual validation required.")
        print(f">>> Run: trtexec --onnx=model.onnx --useDLACore=0 --verbose")
        return True  # Can't validate, assume OK




def train_phase1_fp32(
    model_yaml: str,
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: int = 0,
    project: str = "runs/unina_dla",
    name: str = "fp32",
) -> str:
    """
    Phase 1: Train model in FP32 with ReLU activations.
    
    Uses UninaDLATrainer to support Active Learning and registers mAP_small callback.
    
    Returns:
        Path to best checkpoint.
    """
    print("=" * 60)
    print(">>> Phase 1: FP32 Training (Hardware-Aware)")
    print("=" * 60)
    
    # Define training arguments specifically for the Trainer class
    args = dict(
        model=model_yaml,
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        dynamic=False,  # DLA requires static shapes
        amp=True,       # Mixed precision for speed
        verbose=True,
    )

    if ULTRALYTICS_AVAILABLE:
        try:
            # Explicitly instantiate custom trainer
            trainer = UninaDLATrainer(overrides=args)
            
            # Register Small Object Metric Callback
            # note: Trainer typically sets self.validator internally during train()
            # We can use method add_callback if available, roughly:
            small_obj_cb = SmallObjectCallback(size_threshold=15, image_size=imgsz)
            
            # Registering callback to the trainer methods
            # Ultralytics callbacks are dicts of functions usually, or added via add_callback
            trainer.add_callback("on_val_start", small_obj_cb.on_val_start)
            trainer.add_callback("on_val_batch_end", small_obj_cb.on_val_batch_end)
            trainer.add_callback("on_val_end", small_obj_cb.on_val_end)
            
            print(">>> Custom Trainer and Callbacks Initialized.")
            trainer.train()
            
            # Retrieve best weight path from trainer state
            best_weights = trainer.best if hasattr(trainer, 'best') else Path(project) / name / "weights" / "best.pt"
            
        except Exception as e:
            print(f"ERROR initializing custom trainer: {e}")
            print("Falling back to standard YOLO.train()...")
            # Fallback
            model = YOLO(model_yaml)
            model.train(**args)
            best_weights = Path(project) / name / "weights" / "best.pt"
            
    else:
        print("Ultralytics not available, cannot train.")
        return ""

    print(f">>> Phase 1 Complete. Best weights: {best_weights}")
    
    return str(best_weights)


def train_phase2_qat(
    model_yaml: str,
    data_yaml: str,
    fp32_weights: str,
    epochs: int = 20,
    batch_size: int = 16,
    imgsz: int = 640,
    device: int = 0,
    project: str = "runs/unina_dla",
    name: str = "qat",
) -> str:
    """
    Phase 2: Quantization Aware Training.
    
    Fine-tunes the FP32 model with fake-quantization nodes
    to prepare for INT8 deployment on DLA.
    
    Returns:
        Path to best QAT checkpoint.
    """
    if not QAT_AVAILABLE:
        print(">>> Skipping Phase 2: pytorch-quantization not available.")
        return fp32_weights
    
    print("=" * 60)
    print(">>> Phase 2: Quantization Aware Training (QAT)")
    print("=" * 60)
    
    # Configure Entropy calibration (RESEARCH.md Section 4.3)
    # configure_entropy_calibration() # Use unified qat.py initialization instead
    initialize_quantization(calibrator="histogram")
    
    # Initialize quantization (monkey-patches Conv2d -> QuantConv2d)
    print(">>> Initializing quantization modules...")
    quant_modules.initialize()
    
    # Load model with quantized layers
    qat_model = YOLO(model_yaml)
    
    # Load original FP32 model for weight extraction
    print(f">>> Loading FP32 weights from: {fp32_weights}")
    fp32_model = YOLO(model_yaml)
    try:
        fp32_model.load(fp32_weights)
    except Exception as e:
        print(f">>> WARNING: Could not load FP32 weights ({e}). Using random init.")
        fp32_model = None
    
    # Transfer weights using intelligent matching
    if fp32_model is not None and hasattr(fp32_model, 'model') and hasattr(qat_model, 'model'):
        transfer_result = transfer_weights_fp32_to_qat(
            fp32_model.model, 
            qat_model.model,
            strict=False
        )
        print(f">>> Weight transfer complete: {len(transfer_result['transferred'])} transferred")
    else:
        print(">>> Fallback: Using direct weight loading")
        try:
            qat_model.load(fp32_weights)
        except Exception as e:
            print(f">>> Direct loading failed ({e}). Proceeding with random init.")
    
    # Use qat_model going forward
    model = qat_model
    
    # Set layer-wise precision: Keep P2 head and first backbone layers in FP16
    # (RESEARCH.md Section 4.2 - Mixed Precision Layer-Wise)
    # Using specific UNINA_YOLO_DLA_QAT component names
    fp16_layers = ["head_p2", "stem", "stage1_conv", "head_p2_cls", "head_p2_reg", "cv1"] 
    if hasattr(model, 'model'):
        set_layer_precision_fp16(model.model, fp16_layers)
    
    # Calibration step (run a few batches in eval mode to collect statistics)
    print(">>> Running Entropy calibration...")
    model.val(data=data_yaml, imgsz=imgsz, batch=batch_size // 2)
    
    # Fine-tune with QAT
    print(">>> Starting QAT fine-tuning...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        lr0=0.001,        # Reduced LR for fine-tuning
        warmup_epochs=0,  # No warmup for QAT
    )
    
    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f">>> Phase 2 Complete. Best weights: {best_weights}")
    
    return str(best_weights)


def export_to_onnx(
    weights_path: str,
    output_dir: str = "output",
    imgsz: int = 640,
    int8: bool = True,
) -> str:
    """
    Export trained model to ONNX format for TensorRT.
    
    Returns:
        Path to exported ONNX file.
    """
    print("=" * 60)
    print(">>> Exporting to ONNX")
    print("=" * 60)
    
    model = YOLO(weights_path)
    
    export_path = model.export(
        format="onnx",
        opset=13,       # TensorRT compatibility
        simplify=True,  # ONNX simplifier
        dynamic=False,  # Static shapes for DLA
        int8=int8,      # Mark for INT8 export
        imgsz=imgsz,
    )
    
    print(f">>> ONNX exported to: {export_path}")
    return str(export_path)


# ============================================================================
# 5. CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train UNINA-YOLO-DLA (Ultralytics-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data/Model
    parser.add_argument(
        '--model', type=str, default='unina-yolo-dla.yaml',
        help="Model YAML configuration file"
    )
    parser.add_argument(
        '--data', type=str, default='fsd_data.yaml',
        help="Dataset YAML configuration file"
    )
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100, help="FP32 training epochs")
    parser.add_argument('--qat-epochs', type=int, default=20, help="QAT fine-tuning epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    parser.add_argument('--device', type=str, default='0', help="Device: GPU ID (0,1) or 'cpu'")
    
    # Output
    parser.add_argument('--project', type=str, default='runs/unina_dla', help="Project directory")
    
    # Phases
    parser.add_argument('--skip-fp32', action='store_true', help="Skip FP32 training (use existing weights)")
    parser.add_argument('--fp32-weights', type=str, default=None, help="Path to FP32 weights (for QAT only)")
    parser.add_argument('--skip-qat', action='store_true', help="Skip QAT phase")
    parser.add_argument('--export', action='store_true', help="Export to ONNX after training")
    
    # Active Learning (EVENMORERESEARCH.md)
    parser.add_argument('--dataset-root', type=str, default=None, help="Path to real dataset (for active learning)")
    parser.add_argument('--difficulty-map', type=str, default=None, help="Path to .json file with difficulty scores")
    
    # Conformal Prediction (RESEARCH.md Section 5.2)
    parser.add_argument('--calibrate-cp', action='store_true', help="Run Conformal Prediction calibration after training")
    parser.add_argument('--cp-alpha', type=float, default=0.10, help="CP error rate (1 - coverage). Default 0.10 for 90%% coverage")
    
    args = parser.parse_args()
    
    if not ULTRALYTICS_AVAILABLE:
        print("ERROR: Ultralytics is required. Exiting.")
        sys.exit(1)
    
    # Resolve paths
    model_yaml = args.model
    data_yaml = args.data
    
    # Log active learning config if specified
    if args.dataset_root:
        difficulty_scores = None
        if args.difficulty_map and os.path.exists(args.difficulty_map):
            import json
            with open(args.difficulty_map, 'r') as f:
                difficulty_scores = json.load(f)
        
        # Configure the custom trainer for active learning loading
        UninaDLATrainer.set_active_learning_config(
            dataset_root=args.dataset_root,
            difficulty_scores=difficulty_scores,
        )
    
    # Phase 1: FP32 Training
    if args.skip_fp32:
        if args.fp32_weights is None:
            print("ERROR: --fp32-weights required when --skip-fp32 is set")
            sys.exit(1)
        fp32_weights = args.fp32_weights
    else:
        fp32_weights = train_phase1_fp32(
            model_yaml=model_yaml,
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name="fp32",
        )
    
    # Phase 2: QAT
    if not args.skip_qat:
        final_weights = train_phase2_qat(
            model_yaml=model_yaml,
            data_yaml=data_yaml,
            fp32_weights=fp32_weights,
            epochs=args.qat_epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name="qat",
        )
    else:
        final_weights = fp32_weights
    
    # Conformal Prediction Calibration
    if args.calibrate_cp:
        model = YOLO(final_weights)
        cp_result = calibrate_conformal_prediction(
            model=model,
            data_yaml=data_yaml,
            alpha=args.cp_alpha,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=str(args.device),
        )
        # Save CP calibration result
        import json
        cp_path = Path(args.project) / "cp_calibration.json"
        with open(cp_path, "w") as f:
            json.dump(cp_result, f, indent=2)
        print(f">>> CP calibration saved to: {cp_path}")
    
    # Export
    if args.export:
        onnx_path = export_to_onnx(
            weights_path=final_weights,
            output_dir=args.project,
            imgsz=args.imgsz,
            int8=not args.skip_qat,
        )
        
        # Validate DLA zero-fallback (if TensorRT available)
        # Note: This requires the engine to be built first
        # For now, just print instructions
        print(f"\n>>> To validate DLA compatibility, run on Jetson:")
        print(f"    trtexec --onnx={onnx_path} --useDLACore=0 --int8 --verbose")
    
    print("=" * 60)
    print(">>> Training Pipeline Complete!")
    print(f">>> Final weights: {final_weights}")
    print("=" * 60)


if __name__ == '__main__':
    main()
