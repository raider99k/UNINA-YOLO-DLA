import os
import sys
import torch
import torch.nn as nn
import yaml
import inspect
from pathlib import Path

# --- Ultralytics Imports ---
try:
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import Conv
    import ultralytics.nn.modules as ultralytics_modules
    import ultralytics.nn.tasks as ultralytics_tasks
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# --- Local Module Imports ---
try:
    from data_loader import create_active_learning_dataloader
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False

# ============================================================================
# 1. Monkey-Patching logic & DDP Compatibility
# ============================================================================

def apply_dla_patches():
    """
    Apply global patches to Ultralytics and register custom modules.
    Must be called at the start of any process (including DDP nodes).
    
    This implementation uses robust function wrapping instead of source injection
    to ensure 100% compatibility with DDP and environments where source code
    may not be available.
    """
    if not ULTRALYTICS_AVAILABLE:
        return

    # A. Surgically fix Ray Tube Session (Kaggle/Server DDP robustness)
    # Ref: Ray 2.x renamed _get_session to get_session
    try:
        import ray.train._internal.session as ray_session
        if hasattr(ray_session, 'get_session') and not hasattr(ray_session, '_get_session'):
            ray_session._get_session = ray_session.get_session
            print(">>> Ray Patch: Mapped _get_session to get_session.")
    except (ImportError, AttributeError):
        pass

    # B. Register SPPF_DLA
    if not hasattr(ultralytics_modules, 'SPPF_DLA'):
        setattr(ultralytics_modules, 'SPPF_DLA', SPPF_DLA)
        setattr(ultralytics_tasks, 'SPPF_DLA', SPPF_DLA)
        ultralytics_tasks.__dict__['SPPF_DLA'] = SPPF_DLA

    # C. Robust parse_model Patch (Wrapper-based)
    # Using a wrapper is production-ready as it doesn't modify source files
    # and handles local variable initialization by augmenting the input dict.
    try:
        import ultralytics.nn.tasks as tasks
        
        # Check if already wrapped to avoid infinite recursion
        if not hasattr(tasks.parse_model, '_dla_wrapped'):
            original_parse_model = tasks.parse_model
            
            def parse_model_wrapper(d, ch, verbose=True):
                """Wrapper to ensure 'scale' is present in the config dict."""
                # Ensure 'scale' exists in the config dictionary to satisfy the parser
                if isinstance(d, dict) and 'scale' not in d:
                    d['scale'] = 'm'  # Default to medium if missing
                return original_parse_model(d, ch, verbose)
            
            # Mark as wrapped and replace
            parse_model_wrapper._dla_wrapped = True
            tasks.parse_model = parse_model_wrapper
            print(">>> Monkey-Patch applied: Wrapper-based parse_model patch (DDP Ready).")
            
    except Exception as e:
        print(f">>> WARNING: Failed to apply robust monkey-patch: {e}")

# ============================================================================
# 2. Custom DLA Modules
# ============================================================================

class SPPF_DLA(nn.Module):
    """Sostituzione SPPF ottimizzata per DLA."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        if c2 == k and c2 < 16:
            c2 = c1
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

# ============================================================================
# 3. Custom Trainer & Validator
# ============================================================================

def replace_silu_with_relu(model: nn.Module) -> None:
    """Recursively replace SiLU with ReLU."""
    for name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)

class UninaDLATrainer(DetectionTrainer):
    """Custom Detection Trainer with Active Learning support."""
    active_learning_config = None
    
    @classmethod
    def set_active_learning_config(cls, dataset_root: str, difficulty_scores: dict = None):
        cls.active_learning_config = {"dataset_root": dataset_root, "difficulty_scores": difficulty_scores}

    def get_model(self, cfg=None, weights=None, verbose=True):
        rank = getattr(self, 'rank', -1)
        if isinstance(cfg, str):
            with open(cfg, 'r') as f:
                cfg = yaml.safe_load(f)
        
        # Ensure scale is present in the dictionary (second layer of defense)
        if isinstance(cfg, dict) and 'scale' not in cfg:
             cfg['scale'] = 'm'
        
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and rank == -1)
        if weights: model.load(weights)
        replace_silu_with_relu(model)
        return model

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        if mode == "train" and self.active_learning_config and DATA_LOADER_AVAILABLE:
            return create_active_learning_dataloader(
                dataset_root=self.active_learning_config["dataset_root"],
                batch_size=batch_size,
                difficulty_scores=self.active_learning_config["difficulty_scores"],
                num_workers=self.args.workers,
            )
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_validator(self):
        from copy import copy
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return UninaDLAValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

class UninaDLAValidator(DetectionValidator):
    """Custom Validator for small object metrics."""
    def __init__(self, *args, size_threshold: int = 15, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_threshold = size_threshold
        self.small_tp = 0
        self.small_fp = 0
        self.small_fn = 0

    def init_metrics(self, model):
        super().init_metrics(model)
        self.small_tp = self.small_fp = self.small_fn = 0

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        try:
            from ultralytics.utils import ops
            from ultralytics.utils.metrics import box_iou
            if "bboxes" not in batch or batch["bboxes"].numel() == 0: return
            
            for si, pred in enumerate(preds):
                idx = batch["batch_idx"] == si
                if not idx.any(): continue
                
                gt_cls = batch["cls"][idx].squeeze(-1)
                gt_bboxes = batch["bboxes"][idx]
                device = gt_bboxes.device
                h, w = batch["ori_shape"][si]
                scale_tensor = torch.tensor([w, h, w, h], device=device)
                gt_xyxy = ops.xywh2xyxy(gt_bboxes) * scale_tensor
                
                gw, gh = gt_xyxy[:, 2] - gt_xyxy[:, 0], gt_xyxy[:, 3] - gt_xyxy[:, 1]
                small_gt_mask = (gw < self.size_threshold) & (gh < self.size_threshold)
                
                if not small_gt_mask.any(): continue
                
                small_gt_xyxy = gt_xyxy[small_gt_mask]
                small_gt_cls = gt_cls[small_gt_mask]
                num_small_gt = small_gt_mask.sum().item()
                
                if isinstance(pred, dict):
                    p_bboxes, p_cls, p_conf = pred["bboxes"], pred["cls"], pred["conf"]
                else: 
                    p_bboxes, p_conf, p_cls = pred[:, :4], pred[:, 4], pred[:, 5]
                
                if p_bboxes.numel() == 0:
                    self.small_fn += num_small_gt
                    continue
                    
                iou_matrix = box_iou(p_bboxes, small_gt_xyxy)
                cls_match = p_cls.view(-1, 1) == small_gt_cls.view(1, -1)
                match_matrix = (iou_matrix > 0.45) & cls_match
                
                matched_gt = match_matrix.any(dim=0)
                tp = matched_gt.sum().item()
                self.small_tp += tp
                self.small_fn += (num_small_gt - tp)
                self.small_fp += (match_matrix.any(dim=1).sum().item() - tp)
        except Exception as e:
            print(f"WARNING: Error in small object metrics: {e}")

    def finalize_metrics(self, *args, **kwargs):
        stats = super().finalize_metrics(*args, **kwargs)
        precision = self.small_tp / (self.small_tp + self.small_fp + 1e-7)
        recall = self.small_tp / (self.small_tp + self.small_fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        print(f"\n[UNINA DLA] Small Object Metrics (<{self.size_threshold}px):")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    TP: {self.small_tp}, FP: {self.small_fp}, FN: {self.small_fn}")
        
        if stats and isinstance(stats, dict):
            stats["metrics/small_precision"] = precision
            stats["metrics/small_recall"] = recall
            stats["metrics/small_f1"] = f1
        return stats

# --- Global Initialization ---
# Apply patches automatically on import to support DDP worker processes
apply_dla_patches()
