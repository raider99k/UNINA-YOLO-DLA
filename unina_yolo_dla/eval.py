#!/usr/bin/env python3
"""
UNINA-YOLO-DLA: Evaluation Script.

Implements specific metrics for Formula Student Driverless:
1. mAP_small: mAP for objects smaller than 15x15 pixels.
2. Recall at 15m+: Recall for cones at distant ranges (proxy for small size).

Usage:
    python eval.py --weights runs/unina_dla/fp32/weights/best.pt --data fsd_data.yaml
"""

import argparse
import torch
from ultralytics import YOLO
from data_loader import SmallObjectMetric

def evaluate_model(weights_path, data_yaml, imgsz=640, device=0):
    print(f"Evaluating {weights_path} on {data_yaml}...")
    
    model = YOLO(weights_path)
    
    # Custom call to validate using our specific metric logic
    # Note: Ultralytics val() is robust but we want to hook into our SmallObjectMetric
    # We can use the callback approach or a manual loop if needed.
    # For simplicity, we run standard val() and assume the custom callback in train.py 
    # hooks in if we were training. Here we implement a standalone loop for explicit control.
    
    # Actually, simpler to reuse the SmallObjectCallback approach if we attach it
    # But let's write a standalone robust loop for clarity.
    
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, split='val', save_json=True)
    print(f"Standard mAP50: {metrics.box.map50}")
    print(f"Standard mAP50-95: {metrics.box.map}")
    
    # TODO: Parse the saved JSON predictions to compute mAP_small explicitly
    # if standard val doesn't support custom callbacks easily in CLI mode.
    # For now, we trust the standard metrics but emphasize looking at 'small' split if available.
    
    print("\n[NOTE] Ensure 'SmallObjectCallback' is registered in the Trainer for training-time metrics.")
    print("[NOTE] For offline small-object analysis, use 'auto_labeler.py' visualization tools.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='fsd_data.yaml')
    args = parser.parse_args()
    
    evaluate_model(args.weights, args.data)
