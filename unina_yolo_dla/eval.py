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
    
    # Parse outputs for mAP_small
    # Ultralytics saves predictions to runs/detect/val/predictions.json typically, or similar.
    # We will need the save directory from metrics.
    save_dir = metrics.save_dir
    pred_json = save_dir / "predictions.json"
    
    if pred_json.exists():
        import json
        import cv2
        from pathlib import Path
        
        print(f"\n[SmallObjectMetric] Computing mAP for objects < 15px on {pred_json}...")
        
        with open(pred_json, 'r') as f:
            preds_data = json.load(f)
            
        # Group predictions by image_id
        preds_by_img = {}
        for p in preds_data:
            img_id = p['image_id']
            if img_id not in preds_by_img:
                preds_by_img[img_id] = []
            # format per prediction dict: {'image_id': '...', 'category_id': 0, 'bbox': [x, y, w, h], 'score': 0.9}
            # metric expects: [x, y, w, h, conf, cls] (normalized)
            # JSON bbox is usually [x_min, y_min, w, h] in absolute pixels
            preds_by_img[img_id].append(p)

        # We need ground truths. Ultralytics dataloader has them, or we parse from YAML path.
        # For this standalone script, we assume standard YOLO dataset structure.
        # Parsing data_yaml to find validation path
        import yaml
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        val_path = Path(data_cfg.get('val', ''))
        
        # Initialize metric
        so_metric = SmallObjectMetric(size_threshold=15, image_size=imgsz)
        
        # Iterate over validation images to match GT and Preds
        # Note: robust matching requires aligned paths. 
        # Here we perform a simplified iteration assuming file names match image_ids in JSON.
        
        # Fallback if image_id is simply the stem
        processed_count = 0
        
        # If val_path is a directory, glob it
        img_files = sorted(list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))) if val_path.is_dir() else []
        
        for img_file in img_files:
            img_stem = img_file.stem
            
            # 1. Get Predictions
            img_preds_raw = preds_by_img.get(img_stem, [])
            
            # Convert to [x, y, w, h, conf, cls] normalized tensor
            # Need image dimensions
            img = cv2.imread(str(img_file))
            if img is None: continue
            h, w = img.shape[:2]
            
            pred_tensor_list = []
            for p in img_preds_raw:
                bbox = p['bbox'] # x_min, y_min, w_p, h_p
                score = p['score']
                cat = p['category_id']
                
                # Convert to normalized xywh (center)
                x_center = (bbox[0] + bbox[2]/2) / w
                y_center = (bbox[1] + bbox[3]/2) / h
                w_norm = bbox[2] / w
                h_norm = bbox[3] / h
                
                pred_tensor_list.append([x_center, y_center, w_norm, h_norm, score, cat])
                
            pred_tensor = torch.tensor(pred_tensor_list) if pred_tensor_list else torch.empty((0, 6))

            # 2. Get Ground Truths from label file
            # Assume labels are in sibling 'labels' dir
            label_file = img_file.parent.parent / "labels" / (img_stem + ".txt")
            gt_tensor_list = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        cls_id = float(parts[0])
                        gx, gy, gw, gh = map(float, parts[1:5])
                        gt_tensor_list.append([cls_id, gx, gy, gw, gh])
            
            gt_tensor = torch.tensor(gt_tensor_list) if gt_tensor_list else torch.empty((0, 5))
            
            # 3. Update Metric
            so_metric.update([pred_tensor], [gt_tensor])
            processed_count += 1
            
        if processed_count > 0:
            res = so_metric.compute()
            print("\n>>> Small Object Metrics Calculation Complete:")
            for k, v in res.items():
                print(f"    {k}: {v}")
        else:
            print("[SmallObjectMetric] Could not match JSON image_ids to local files. Skipping.")
    else:
        print(f"[SmallObjectMetric] Predictions JSON not found at {pred_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='fsd_data.yaml')
    args = parser.parse_args()
    
    evaluate_model(args.weights, args.data)
