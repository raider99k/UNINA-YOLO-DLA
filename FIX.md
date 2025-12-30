# Backlog: Codebase Fixes & Technical Debt

This document tracks incomplete implementations ("stubs"), broken logic connections, and placeholders identified during the code review (**2025-12-30**).

## 🚨 Critical: Broken Training Logic
**File**: [`unina_yolo_dla/train.py`](file:///c:/Users/pasqu/OneDrive/Vision/unina_yolo_dla/train.py)

The current training script defines advanced Data-Centric features but fails to use them during execution.

- [x] **Custom Trainer Unused**: The class `UninaDLATrainer` is defined but never instantiated. The function `train_phase1_fp32` calls `model.train()` which uses the default Ultralytics trainer. This means **Active Learning sampling is effectively disabled**.
- [x] **Callbacks Not Registered**: `SmallObjectCallback` is defined but never passed to the trainer. The `mAP_small` metric is currently not being calculated or logged.
- [x] **CLI Disconnect**: Arguments like `--dataset-root` and `--difficulty-map` are parsed but their effects are lost because the custom trainer is not invoked.

**Fix Strategy**:
1. Instantiate `UninaDLATrainer` manually in `train_phase1_fp32`.
2. Explicitly register `SmallObjectCallback` via `trainer.add_callback()`.

---

## 🚧 Incomplete Modules (Stubs)

### 1. Auto-Labeling Pipeline
**File**: [`unina_yolo_dla/auto_labeler.py`](file:///c:/Users/pasqu/OneDrive/Vision/unina_yolo_dla/auto_labeler.py)

The structure exists, but the core logic is missing.
- [x] **GroundingDINO Wrapper**: `__init__` and `predict` are empty (`pass`).
- [x] **SAM Wrapper**: `__init__` and `predict_mask` are empty (`pass`).
- [x] **Main Loop**: The processing loop in `main()` is a placeholder (`pass`).

### 2. Active Learning
**File**: [`unina_yolo_dla/active_learning.py`](file:///c:/Users/pasqu/OneDrive/Vision/unina_yolo_dla/active_learning.py)

- [x] **Coreset Selection**: Method `coreset_selection` is empty (`pass`). Needs implementation of embedding extraction and K-Means/K-Center.
- [x] **Copy-Paste Augmentation**: `CopyPasteAugmentor.apply` returns the input image unchanged.

### 3. Evaluation Script
**File**: [`unina_yolo_dla/eval.py`](file:///c:/Users/pasqu/OneDrive/Vision/unina_yolo_dla/eval.py)

- [x] **Metric Calculation**: Logic to parse validation JSON and compute `mAP_small` is marked as `TODO`. Currently relies only on standard metrics.
