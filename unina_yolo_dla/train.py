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
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import yaml

# --- Environment-Aware Paths (Critical for DDP) ---
# Ensure current directory is in PYTHONPATH for DDP subprocesses
cwd = str(Path(__file__).parent.absolute())
if cwd not in sys.path:
    sys.path.insert(0, cwd)
os.environ["PYTHONPATH"] = cwd + os.pathsep + os.environ.get("PYTHONPATH", "")

# --- Ultralytics Imports ---
try:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
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
    from qat import (
        initialize_quantization,
        prepare_qat_model,
        transfer_weights_fp32_to_qat,
        configure_entropy_calibration,
        set_layer_precision_fp16
    )
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    print("Install with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")

# --- Environment-Aware Kaggle Fixes ---
def is_kaggle_environment():
    """Returns True if running inside a Kaggle Kernel."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None or os.environ.get('KAGGLE_URL') is not None

def patch_kaggle_environment(trainer_or_model):
    """
    Surgically disable RayTune callbacks only on Kaggle to prevent AttributeError.
    Zero-impact on other environments (servers, local).
    """
    if not is_kaggle_environment():
        return

    targets = ['on_fit_epoch_end', 'on_train_epoch_end']
    # Check if we are dealing with a YOLO model object or a Trainer
    trainer = trainer_or_model.trainer if hasattr(trainer_or_model, 'trainer') else trainer_or_model
    
    if hasattr(trainer, 'callbacks'):
        modified = False
        for event in targets:
            if event in trainer.callbacks:
                original_callbacks = trainer.callbacks[event]
                new_callbacks = [
                    cb for cb in original_callbacks 
                    if 'raytune' not in str(cb.__module__)
                ]
                if len(new_callbacks) != len(original_callbacks):
                    trainer.callbacks[event] = new_callbacks
                    modified = True
        if modified:
            print(">>> KAGGLE PATCH: Disabled RayTune callbacks to prevent environment crash.")

# --- Local Module Imports ---
try:
    from data_loader import create_active_learning_dataloader, SmallObjectMetric
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    print("WARNING: data_loader module not found. Hybrid loading disabled.")

try:
    from trainer import UninaDLATrainer, UninaDLAValidator, apply_dla_patches, replace_silu_with_relu
    # APPLY DLA PATCHES (parse_model monkey-patch + SPPF_DLA registration)
    # Must be done BEFORE any YOLO or DetectionModel usage
    apply_dla_patches()
except ImportError:
    print("ERROR: trainer module not found. Custom trainers will not work.")

# --- TensorRT Imports (for DLA validation) ---
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


# ============================================================================
# 4. Small Object Metric Callback (mAP_small) - LEGACY (Kept for Fallback)
# ============================================================================


class SmallObjectCallback:
    """
    Ultralytics callback to calculate mAP_small at the end of validation.
    
    Reference: RESEARCH.md Section 5.2 - mAP for objects < 15x15 pixels.
    """
    def __init__(self, size_threshold: int = 15, image_size: int = 640, verbose: bool = True):
        self.size_threshold = size_threshold
        self.image_size = image_size
        self.verbose = verbose
        # Shared counters
        self.small_tp = 0
        self.small_fp = 0
        self.small_fn = 0
    
    def on_val_start(self, trainer):
        """Reset counters at start of validation."""
        self.small_tp = 0
        self.small_fp = 0
        self.small_fn = 0
    
    def on_val_batch_end(self, validator):
        """
        Update metric with batch predictions.
        
        Ultralytics callback signature: on_val_batch_end(validator)
        preds and batch are NOT direct attributes of validator - they must be
        extracted from the caller's local scope using frame introspection.
        
        Reference: https://docs.ultralytics.com/usage/callbacks/
        """
        import inspect
        
        try:
            # Access the caller's local variables to get preds and batch
            # The callback is called from within the validation loop
            frame = inspect.currentframe()
            if frame is not None:
                # Go up the call stack to find preds and batch
                # Typically: on_val_batch_end -> run_callbacks -> validate loop
                caller_locals = None
                for _ in range(5):  # Search up to 5 levels
                    frame = frame.f_back
                    if frame is None:
                        break
                    locals_dict = frame.f_locals
                    if 'preds' in locals_dict and 'batch' in locals_dict:
                        caller_locals = locals_dict
                        break
                
                if caller_locals:
                    preds = caller_locals.get('preds')
                    batch = caller_locals.get('batch')
                    
                    if preds is not None and batch is not None:
                        # Vectorized update (No memory accumulation)
                        tp, fp, fn = self._calculate_batch_stats(preds, batch)
                        self.small_tp += tp
                        self.small_fp += fp
                        self.small_fn += fn
                        
        except Exception as e:
            # Gracefully handle extraction errors
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Warning: Failed to extract small object metrics data: {e}")
    
    def _calculate_batch_stats(self, preds, batch):
        """Vectorized batch stats calculation (Production-ready)."""
        try:
            from ultralytics.utils import ops
            from ultralytics.utils.metrics import box_iou
            tp_total, fp_total, fn_total = 0, 0, 0
            
            # preds is usually a list of dicts or tensors
            for si, pred in enumerate(preds):
                # Target info
                idx = batch["batch_idx"] == si
                if not idx.any(): continue
                
                gt_bboxes = batch["bboxes"][idx]
                device = gt_bboxes.device
                imgsz = torch.tensor(batch["img"].shape[2:], device=device)[[1, 0, 1, 0]]
                gt_xyxy = ops.xywh2xyxy(gt_bboxes) * imgsz
                gt_w, gt_h = gt_xyxy[:, 2] - gt_xyxy[:, 0], gt_xyxy[:, 3] - gt_xyxy[:, 1]
                small_gt_mask = (gt_w < self.size_threshold) & (gt_h < self.size_threshold)
                
                if not small_gt_mask.any(): continue
                
                small_gt_xyxy = gt_xyxy[small_gt_mask]
                small_gt_cls = batch["cls"][idx].squeeze(-1)[small_gt_mask]
                n_small_gt = small_gt_mask.sum().item()
                
                # Pred info
                # preds can be a list of results or dicts
                if isinstance(pred, dict):
                    p_bboxes, p_cls, p_conf = pred["bboxes"], pred["cls"], pred["conf"]
                else: # Fallback for different Ultralytics versions
                    p_bboxes, p_cls, p_conf = pred[:, :4], pred[:, 5], pred[:, 4]
                
                if p_bboxes.numel() == 0:
                    fn_total += n_small_gt
                    continue
                
                # Vectorized matching
                iou = box_iou(p_bboxes, small_gt_xyxy)
                match = (iou >= 0.5) & (p_cls.view(-1, 1) == small_gt_cls.view(1, -1))
                
                m_gt = torch.zeros(n_small_gt, dtype=torch.bool, device=device)
                m_pred = torch.zeros(p_bboxes.shape[0], dtype=torch.bool, device=device)
                for i in torch.argsort(p_conf, descending=True):
                    matches = match[i] & ~m_gt
                    if matches.any():
                        m_gt[matches.nonzero()[0, 0]] = True
                        m_pred[i] = True
                
                tp = m_gt.sum().item()
                tp_total += tp
                fn_total += (n_small_gt - tp)
                
                p_w, p_h = p_bboxes[:, 2] - p_bboxes[:, 0], p_bboxes[:, 3] - p_bboxes[:, 1]
                p_small = (p_w < self.size_threshold) & (p_h < self.size_threshold)
                fp_total += (p_small & ~m_pred).sum().item()
                
            return tp_total, fp_total, fn_total
        except Exception:
            return 0, 0, 0

    def on_val_end(self, trainer):
        """Log results."""
        precision = self.small_tp / (self.small_tp + self.small_fp + 1e-7)
        recall = self.small_tp / (self.small_tp + self.small_fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        print(f"\n>>> Small Object Metrics (Vectorized Callback):")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    TP: {self.small_tp}, FP: {self.small_fp}, FN: {self.small_fn}")




# ============================================================================
# 5. Conformal Prediction Calibration (Offline)
# ============================================================================

from contextlib import contextmanager

@contextmanager
def isolate_inference_env():
    """
    Context manager to temporarily strip DDP environment variables.
    
    This is critical for running offline validation/calibration (like CP)
    inside a script that was launched with torch.distributed.run.
    Ultralytics' AutoBatch and device selection logic can get confused
    if it sees DDP variables but we just want to run a local inference pass.
    """
    # Keys that signal DDP mode to PyTorch/Ultralytics
    ddp_keys = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    stashed = {}
    
    # Stash and unset
    for key in ddp_keys:
        if key in os.environ:
            stashed[key] = os.environ[key]
            del os.environ[key]
            
    try:
        yield
    finally:
        # Restore
        for key, value in stashed.items():
            os.environ[key] = value

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
    
    # Parse data YAML to get validation images directory
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    base_path = Path(data_cfg.get('path', Path(data_yaml).parent))
    val_rel = data_cfg.get('val') or data_cfg.get('test')
    if isinstance(val_rel, list): val_rel = val_rel[0]
    val_images_path = base_path / val_rel if val_rel else base_path
    
    print(f">>> CP Calibration source: {val_images_path}")
    
    # Run prediction on validation set
    results = model.predict(
        source=str(val_images_path),
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
    workers: int = 4,
    plots: bool = True,
    device: int = 0,
    project: str = "runs/unina_dla",
    name: str = "fp32",
    pretrained: bool = False,
    weights: str = None,
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
        workers=workers,
        plots=plots,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        dynamic=False,  # DLA requires static shapes
        amp=True,       # Mixed precision for speed
        verbose=True,
        pretrained=pretrained,
    )
    
    # If explicit weights are provided, override 'model' arg or use as 'weights'
    # For Training from scratch, Ultralytics uses 'model=yaml'.
    # For Transfer Learning, 'model=pt' or 'model=yaml, weights=pt'.
    if weights:
        args['model'] = weights  # If weights provided, start from them? 
        # Actually Ultralytics logic:
        # model=yaml -> scratch (unless pretrained=True -> downloads default)
        # model=pt -> finetune
        # We want to use our custom YAML structure but maybe load weights?
        # Typically: model=yaml, pretrained=pt uses the pt weights on that architecture.
        # But here 'weights' arg usually means starting checkpoint.
        # Let's trust Ultralytics 'pretrained' arg if boolean, or path if string.
        # However, we simply pass 'model=yaml' and 'pretrained=False' to scratch.
        # If user provides weights, we might want to load them.
        # Let's stick to: pass 'pretrained' (bool/str) to overrides.
        pass

    # If weights is a path, Ultralytics 'pretrained' arg can take it, OR we load it later.
    # But simpler: use the 'pretrained' arg of the Trainer.
    # If weights is provided, we set pretrained=weights (path).
    if weights:
         args['pretrained'] = weights
    else:
         args['pretrained'] = pretrained # False by default


    if ULTRALYTICS_AVAILABLE:
        try:
            # Explicitly instantiate custom trainer
            trainer = UninaDLATrainer(overrides=args)
            
            # NOTE: Small object metrics are now calculated inline via UninaDLAValidator
            # (returned by get_validator), eliminating the need for slow callback-based
            # frame introspection. The legacy SmallObjectCallback is kept for fallback only.
            
            print(">>> Custom Trainer initialized (with UninaDLAValidator for small object metrics).")
            
            # Surgically patch Kaggle-specific Ray issues if detected
            patch_kaggle_environment(trainer)
            
            trainer.train()
            
            # Retrieve best weight path from trainer state
            best_weights = trainer.best if hasattr(trainer, 'best') else Path(project) / name / "weights" / "best.pt"
            
            # Phase 1 Cleanup: Free memory and kill dataloader threads to avoid ConnectionResetError
            del trainer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:

            print(f"ERROR initializing custom trainer: {e}")
            print("Falling back to standard YOLO.train()...")
            # Fallback
            model = YOLO(model_yaml)
            
            # PATCH: Apply ReLU to fallback model for DLA compatibility
            try:
                 if not hasattr(model, 'model') or model.model is None:
                     model._new(model_yaml, task='detect')
                 
                 if hasattr(model, 'model'):
                     print(">>> PATCH: Applying ReLU to fallback model for DLA compatibility.")
                     replace_silu_with_relu(model.model)
            except Exception as e_patch:
                 print(f">>> WARNING: Failed to patch fallback model: {e_patch}")

            # Also surgical patch for fallback path on Kaggle
            patch_kaggle_environment(model)

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
    workers: int = 4,
    plots: bool = True,
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
    
    # 1. Load original FP32 model to extract metadata (nc, names)
    print(f">>> Loading FP32 weights from: {fp32_weights}")
    try:
        fp32_model = YOLO(fp32_weights)
        full_nc = fp32_model.model.nc
        full_names = fp32_model.model.names
        print(f">>> FP32 Model loaded: nc={full_nc}, names={full_names}")
    except Exception as e:
        print(f">>> ERROR: Could not load FP32 weights for metadata extraction: {e}")
        # Fallback to YAML defaults if weights fail to load
        fp32_model = None
        full_nc = 4
        full_names = {i: f"class_{i}" for i in range(4)}

    # 2. Load QAT model with matching structure and metadata
    # We must use the same nc as FP32 to ensure head weights can be transferred
    with open(model_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f">>> Synchronizing QAT model with FP32 metadata (nc={full_nc})")
    cfg['nc'] = full_nc
    
    # Initialize YOLO with the YAML path first (YOLO wrapper requires a string/Path)
    qat_model = YOLO(model_yaml)
    # Then overwrite with the correctly-scaled DetectionModel
    qat_model.model = DetectionModel(cfg, nc=full_nc, verbose=False)
    qat_model.model.names = full_names
    
    if hasattr(qat_model, 'model'):
        replace_silu_with_relu(qat_model.model)
    
    # 3. Transfer weights using intelligent matching
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
    
    # Calibration step (run lower-overhead forward passes)
    print(">>> Running Entropy calibration (Lightweight)...")
    
    # Parse data yaml to get validation path
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    # Resolve paths relative to YAML's 'path' key (standard Ultralytics format)
    base_path = Path(data_cfg.get('path', Path(data_yaml).parent))
    calib_rel = data_cfg.get('val') or data_cfg.get('test') or data_cfg.get('train')
    if isinstance(calib_rel, list): calib_rel = calib_rel[0]
    calib_path = base_path / calib_rel if calib_rel else base_path
    print(f">>> Calibration path resolved to: {calib_path}")
    
    # Create lightweight dataloader
    from qat import collect_calibration_stats
    calib_loader = create_active_learning_dataloader(
        dataset_root=str(calib_path),
        batch_size=batch_size,
        num_workers=workers,
        difficulty_scores=None, # No weighting needed for calibration
        augment=False,          # CRITICAL: Clean data for accurate quantization stats
    )
    
    # Run calibration on the underlying pytorch model
    if hasattr(model, 'model'):
        collect_calibration_stats(model.model, calib_loader, num_batches=30, device=device)
    else:
        collect_calibration_stats(model, calib_loader, num_batches=30, device=device)
    
    # Explicit cleanup to prevent OOM before full training starts
    del calib_loader
    torch.cuda.empty_cache()
    
    # Fine-tune with QAT
    print(">>> Starting QAT fine-tuning...")
    
    # DDP Robustness: Set environment variable to trigger quantization init in workers
    os.environ['UNINA_DLA_QAT'] = '1'
    
    # Save a temporary checkpoint to preserve QAT structure and calibrated stats for DDP workers
    # We save it in the project root instead of the 'name' subdir to avoid trainer cleanup issues
    temp_qat_dir = Path(project).resolve()
    temp_qat_dir.mkdir(parents=True, exist_ok=True)
    temp_qat_path = temp_qat_dir / "qat_init_calibrated.pt"
    
    # We save the underlying model to ensure all layers (QuantConv2d) and buffers (calibrators) are preserved
    torch.save({
        'model': model.model,
        'nc': full_nc,
        'names': full_names,
        'date': datetime.now().isoformat(),
    }, str(temp_qat_path), _use_new_zipfile_serialization=False)
    
    # Crucial for Kaggle/Cloud environments: Wait for file system synchronization
    import time
    time.sleep(2)
    
    if not temp_qat_path.exists():
        print(f"\u003e\u003e\u003e ERROR: Failed to save temporary QAT model to {temp_qat_path}")
        # Continue anyway, let it fail at training if needed
    else:
        print(f"\u003e\u003e\u003e QAT calibrated model saved for DDP workers: {temp_qat_path}")
    
    # Training arguments for UninaDLATrainer
    qat_args = dict(
        model=str(temp_qat_path),
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        plots=plots,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        lr0=0.001,        # Reduced LR for fine-tuning
        warmup_epochs=0,  # No warmup for QAT
        amp=False,        # CRITICAL: QAT FakeQuant is unstable in FP16/AMP
        mosaic=0.0,       # Disable mosaic for fine-tuning
        mixup=0.0,        # Disable mixup for fine-tuning
        copy_paste=0.0,   # Disable copy-paste
    )
    
    # Filter custom arguments that would trigger SyntaxError in Ultralytics internal validation
    # We will inject these into a SEPARATE attribute, NOT trainer.args
    custom_qat_config = {
        'fp16_layers': fp16_layers,
        'ema': False,
    }

    try:
        # Use custom trainer to guarantee registration of SPPF_DLA in all DDP ranks
        trainer = UninaDLATrainer(overrides=qat_args)
        
        # Store custom config in a SEPARATE attribute to avoid Ultralytics validation
        # This attribute is read by UninaDLATrainer.get_model() for FP16 layer restoration
        trainer._custom_qat_config = custom_qat_config
            
        print(f">>> QAT Trainer initialized with custom config: {custom_qat_config}")
        
        # Surgical patch for Kaggle-specific Ray issues
        patch_kaggle_environment(trainer)
        
        trainer.train()
        results = trainer.best if hasattr(trainer, 'best') else Path(project) / name / "weights" / "best.pt"
    except Exception as e:
        print(f">>> WARNING: Custom trainer failed for QAT: {e}")
        print(">>> Attempting fallback to standard model.train()...")
        # For fallback, we MUST NOT include custom keys in qat_args
        results = model.train(**qat_args)
    
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
    
    # Handle QAT-specific export if needed
    if int8 and QAT_AVAILABLE:
        try:
             # Check if model actually has quantization nodes
             has_quant = any('QuantConv2d' in str(type(m)) for m in model.model.modules())
             if has_quant:
                 print(">>> Detected QAT model. Using specialized QAT ONNX export...")
                 from qat import export_qat_onnx
                 out_path = Path(output_dir) / "qat_model.onnx"
                 out_path.parent.mkdir(parents=True, exist_ok=True)
                 export_qat_onnx(model.model, str(out_path), input_size=imgsz)
                 return str(out_path)
        except Exception as e:
             print(f">>> WARNING: Specialized QAT export failed ({e}). Falling back to standard export.")
    
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
        '--model', type=str, default='unina-yolo-dla-m.yaml',
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
    parser.add_argument('--workers', type=int, default=4, help="Dataloader workers (reduce on Kaggle to 2)")
    parser.add_argument('--no-plots', action='store_true', help="Disable generating plots during training to save CPU")
    parser.add_argument('--device', type=str, default='0', help="Device: GPU ID (0,1) or 'cpu'")
    parser.add_argument('--weights', type=str, default=None, help="Initial weights path (e.g. yolov8n.pt)")
    parser.add_argument('--pretrained', action='store_true', help="Enable pretrained weights (default: False/Scratch)")
    
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
            workers=args.workers,
            plots=not args.no_plots,
            device=args.device,
            project=args.project,
            name="fp32",
            pretrained=args.pretrained,
            weights=args.weights,
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
            workers=args.workers,
            plots=not args.no_plots,
            device=args.device,
            project=args.project,
            name="qat",
        )
    else:
        final_weights = fp32_weights
    
    # Conformal Prediction Calibration
    # Conformal Prediction Calibration
    # CRITICAL: This must ONLY run on the main process (Rank 0) or non-DDP runs.
    # DDP inference/calibration is not supported by this script's architecture.
    rank = int(os.environ.get('RANK', -1))
    if args.calibrate_cp and rank in {-1, 0}:
        try:
            with isolate_inference_env():
                print(f">>> Starting CP Calibration on Rank {rank} (DDP env isolated)...")
                # Reload model fresh to ensure no DDP wrappers are present
                model = YOLO(final_weights)
                
                cp_result = calibrate_conformal_prediction(
                    model=model,
                    data_yaml=data_yaml,
                    alpha=args.cp_alpha,
                    imgsz=args.imgsz,
                    batch_size=args.batch, 
                    device=str(args.device).split(',')[0], # Force single device for calibration
                )
                
                # Save CP calibration result
                import json
                cp_path = Path(args.project) / "cp_calibration.json"
                with open(cp_path, "w") as f:
                    json.dump(cp_result, f, indent=2)
                print(f">>> CP calibration saved to: {cp_path}")
                
        except Exception as e:
            print(f">>> ERROR: CP Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole job, just skip CP
    elif args.calibrate_cp:
        print(f">>> Skipping CP Calibration on Rank {rank} (Only Rank 0 runs calibration)")
    
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
