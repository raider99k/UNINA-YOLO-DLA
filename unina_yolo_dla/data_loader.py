"""
UNINA-YOLO-DLA: Data Ingestion Module.

This module provides:
1. A hybrid DataLoader that mixes Real and Synthetic datasets.
2. A specialized validation metric for Small Objects (< 15x15 pixels).

Key Requirements:
    - The DataLoader must be compatible with standard PyTorch training loops.
    - Synthetic data (e.g., from Unreal Engine 5) is assumed to have the
      same annotation format as real data (YOLO format).
    - The Small Object metric is critical for Formula Student Driverless,
      where cones at 15m+ distance appear very small in the frame.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np

import cv2  # OpenCV for image loading


# --- Dataset Definitions ---

class YOLODataset(Dataset):
    """
    A dataset class for YOLO-format annotations.

    Expects a directory structure like:
        dataset_root/
            images/
                img_001.jpg
                img_002.png
                ...
            labels/
                img_001.txt
                img_002.txt
                ...

    Each label file contains lines in YOLO format:
        <class_id> <x_center> <y_center> <width> <height>
    All values are normalized (0-1 relative to image dimensions).

    Args:
        root: Path to the dataset root directory.
        transform: Optional callable to apply transformations to images.
    """
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.transform = transform

        # Gather all image paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        self.image_paths: list[Path] = sorted([
            p for p in self.images_dir.glob("*")
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary with 'image', 'labels', and 'path'.
        'image' is a (C, H, W) float tensor normalized to [0, 1].
        'labels' is a (N, 5) tensor: [class_id, x_c, y_c, w, h].
        """
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        # --- Image Loading (Real Implementation) ---
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # --- Label Loading ---
        labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_c, y_c, w, h = map(float, parts)
                        labels.append([class_id, x_c, y_c, w, h])
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

        # --- Transforms ---
        if self.transform:
            # Transforms should handle both image and labels (e.g., Albumentations).
            image, labels_tensor = self.transform(image, labels_tensor)

        return {
            "image": image,
            "labels": labels_tensor,
            "path": str(img_path),
        }


def create_active_learning_dataloader(
    dataset_root: str | Path,
    batch_size: int = 16,
    hard_example_ratio: float = 0.3,
    num_workers: int = 4,
    transform: Callable | None = None,
    difficulty_scores: dict[str, float] | None = None,
) -> DataLoader:
    """
    Creates a DataLoader optimized for Active Learning and Hard Example Mining.

    Uses WeightedRandomSampler to prioritize samples with high uncertainty
    or difficulty scores (e.g., from Entropy or Localization Variance).

    Args:
        dataset_root: Path to the real dataset.
        batch_size: The batch size for the DataLoader.
        hard_example_ratio: Target proportion of high-difficulty samples.
        num_workers: Number of worker processes for data loading.
        transform: Transforms to apply.
        difficulty_scores: Dictionary mapping image paths to difficulty scores.

    Returns:
        A configured PyTorch DataLoader.
    """
    dataset = YOLODataset(dataset_root, transform=transform)
    total_samples = len(dataset)

    # Default weights: uniform
    sample_weights = [1.0] * total_samples

    if difficulty_scores:
        # Assign weights based on difficulty scores
        # We normalize scores to use as sampling probabilities
        sample_weights = []
        for img_path in dataset.image_paths:
            # Score could be Entropy, Loss, or custom uncertainty metric
            score = difficulty_scores.get(str(img_path), 1.0)
            sample_weights.append(score)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for batching samples with variable-size labels.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    
    # Labels need to be kept as a list of tensors since N varies.
    labels = [item["labels"] for item in batch]
    
    return {
        "images": images,
        "labels": labels,
        "paths": paths,
    }


# --- Small Object Validation Metric ---

class SmallObjectMetric:
    """
    Calculates and tracks detection metrics specifically for Small Objects.

    A "small object" is defined as an annotated bounding box with
    dimensions < `size_threshold` pixels in the original image space.

    This metric is critical for Formula Student Driverless, where distant
    cones (15m+) appear as objects smaller than ~15x15 pixels.

    Args:
        size_threshold: Maximum pixel dimension (width or height) for an
                        object to be considered "small". Default: 15.
        iou_threshold: IoU threshold for a prediction to be a True Positive.
    """
    def __init__(
        self,
        size_threshold: int = 15,
        iou_threshold: float = 0.5,
        image_size: int = 640,
    ) -> None:
        self.size_threshold = size_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size

        # Accumulators
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0  # Ground truths not matched

    def reset(self) -> None:
        """Resets all accumulators."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def _is_small(self, box_wh: tuple[float, float]) -> bool:
        """Checks if a box is 'small' based on its pixel dimensions."""
        # box_wh is (width_norm, height_norm)
        w_px = box_wh[0] * self.image_size
        h_px = box_wh[1] * self.image_size
        return w_px < self.size_threshold and h_px < self.size_threshold

    def _iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Calculates IoU between two boxes in [x_c, y_c, w, h] format (normalized).
        """
        # Convert to corner format [x1, y1, x2, y2]
        b1_x1 = box1[0] - box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y2 = box1[1] + box1[3] / 2

        b2_x1 = box2[0] - box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y2 = box2[1] + box2[3] / 2

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union_area = b1_area + b2_area - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def update(
        self,
        predictions: list[torch.Tensor],  # List of (N_pred, 6) tensors [x,y,w,h,conf,cls]
        ground_truths: list[torch.Tensor], # List of (N_gt, 5) tensors [cls,x,y,w,h]
    ) -> None:
        """
        Updates the metric accumulators with a batch of predictions and ground truths.
        Only evaluates on 'small' ground truth objects.
        """
        for preds, gts in zip(predictions, ground_truths):
            # Filter ground truths to only small objects
            small_gt_indices = [
                i for i, gt in enumerate(gts)
                if self._is_small((gt[3].item(), gt[4].item()))  # w, h are at indices 3, 4
            ]
            
            if len(small_gt_indices) == 0:
                continue  # No small objects in this image

            small_gts = gts[small_gt_indices]
            matched_gt = set()

            # Sort predictions by confidence (descending)
            if preds.numel() == 0:
                self.false_negatives += len(small_gts)
                continue
            
            sorted_indices = torch.argsort(preds[:, 4], descending=True)
            sorted_preds = preds[sorted_indices]

            for pred in sorted_preds:
                # pred format: [x, y, w, h, conf, cls]
                pred_box = pred[:4]
                pred_cls = int(pred[5].item())
                
                best_iou = 0.0
                best_gt_idx = -1

                for i, gt in enumerate(small_gts):
                    # gt format: [cls, x, y, w, h]
                    gt_cls = int(gt[0].item())
                    gt_box = gt[1:5]

                    if i in matched_gt:
                        continue
                    if pred_cls != gt_cls:
                        continue
                    
                    iou = self._iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= self.iou_threshold:
                    # MATCH FOUND: Prediction correctly identified a small object.
                    self.true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    # This prediction did not match any small GT
                    # We only count FP if the prediction itself is also "small"
                    # to keep the metric focused.
                    if self._is_small((pred[2].item(), pred[3].item())):
                        self.false_positives += 1

            # Unmatched small GTs are false negatives
            self.false_negatives += len(small_gts) - len(matched_gt)

    def compute(self) -> dict[str, float]:
        """
        Computes Precision, Recall, and F1 for small objects.
        """
        precision = 0.0
        recall = 0.0
        f1 = 0.0

        if (self.true_positives + self.false_positives) > 0:
            precision = self.true_positives / (self.true_positives + self.false_positives)
        
        if (self.true_positives + self.false_negatives) > 0:
            recall = self.true_positives / (self.true_positives + self.false_negatives)

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "small_object_precision": precision,
            "small_object_recall": recall,
            "small_object_f1": f1,
            "small_object_tp": self.true_positives,
            "small_object_fp": self.false_positives,
            "small_object_fn": self.false_negatives,
        }


# --- Example Usage ---
if __name__ == '__main__':
    print("Data Ingestion Module Loaded.")
    
    # Example: SmallObjectMetric
    metric = SmallObjectMetric(size_threshold=15, iou_threshold=0.5, image_size=640)
    
    # Dummy ground truths: [cls, x, y, w, h] normalized
    # A small cone: w=0.02, h=0.04 -> 12.8px x 25.6px. h > 15, so not small.
    # A very small cone: w=0.01, h=0.02 -> 6.4px x 12.8px. Both < 15, small!
    dummy_gts = [torch.tensor([
        [0, 0.5, 0.5, 0.01, 0.02],  # Small (6.4x12.8 px)
        [1, 0.2, 0.3, 0.05, 0.08],  # Not small (32x51 px)
    ])]

    # Dummy predictions: [x, y, w, h, conf, cls]
    # A true positive for the small cone:
    dummy_preds = [torch.tensor([
        [0.51, 0.51, 0.012, 0.022, 0.95, 0],  # High IoU with small cone
    ])]

    metric.update(dummy_preds, dummy_gts)
    results = metric.compute()
    print("SmallObjectMetric Test Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
