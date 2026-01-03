"""
UNINA-YOLO-DLA: Data Mining Script

This script implements the 'Mining' phase of the Active Learning workflow.
It uses a trained FP32 model to estimate the uncertainty (entropy) of images
in an unlabeled dataset and generates a 'difficulty_map.json'.

Usage:
    python mine_data.py --model runs/unina_dla/fp32/weights/best.pt \
                        --data /path/to/unlabeled_images \
                        --output difficulty_map.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Import ActiveLearner
try:
    from active_learning import ActiveLearner
    from train import replace_silu_with_relu
except ImportError:
    print("ERROR: Could not import active_learning or train modules.")
    print("Run this script from the project root.")
    sys.exit(1)

# Import Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.modules import Detect
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ERROR: ultralytics not found.")
    sys.exit(1)


class UnlabeledDataset(Dataset):
    """
    Simple dataset for loading unlabeled images with YOLO preprocessing (LetterBox).
    """
    def __init__(self, root: str, imgsz: int = 640):
        self.root = Path(root)
        self.imgsz = imgsz
        self.extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.files = sorted([
            f for f in self.root.rglob('*') 
            if f.suffix.lower() in self.extensions
        ])
        
        if len(self.files) == 0:
            print(f"WARNING: No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = str(self.files[idx])
        
        # Load image (BGR)
        img0 = cv2.imread(path)
        if img0 is None:
            # Return dummy if failed
            print(f"WARNING: Could not load {path}")
            return {'images': torch.zeros(3, self.imgsz, self.imgsz), 'paths': path}

        # Preprocess: LetterBox -> CHW -> RGB -> Norm
        img = LetterBox(self.imgsz, auto=False, stride=32)(image=img0)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        
        return {
            'images': torch.from_numpy(img),
            'paths': path
        }


def collate_mining_batch(batch):
    """
    Custom collate function to properly batch dicts from UnlabeledDataset.
    """
    images = torch.stack([item['images'] for item in batch])
    paths = [item['paths'] for item in batch]
    return {'images': images, 'paths': paths}


class MiningModelWrapper(nn.Module):
    """
    Wraps an Ultralytics DetectionModel to expose raw head outputs.
    
    ActiveLearner expects a list of (cls_head, reg_head) tuples,
    but standard YOLO returns a concatenated output or detailed list depending on execution mode.
    We intercept the forward pass to extract exactly what we need.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        # Identify the Detect layer
        self.detect_layer = None
        # We assume model is a DetectionModel which is a Sequential-like
        # The last layer is usually Detect
        if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
            m = model.model[-1]
            if isinstance(m, Detect):
                self.detect_layer = m
                # Hook into the layer *before* Detect to capture its inputs
                # Or simply run the model up to Detect
                self.backbone_and_neck = model.model[:-1]
                
        if self.detect_layer is None:
            raise ValueError("Could not find Detect layer in the model.")

    def forward(self, x):
        # 1. Run Backbone + Neck
        # DetectionModel calls each layer; some layers depend on previous outputs (save list)
        # We can't just run model.model[:-1](x) because of the save/concat logic.
        # BEST APPROACH: Re-implement the loop from DetectionModel.forward
        # But that's fragile.
        
        # ALTERNATIVE: Hook the Detect layer.
        self.captured_inputs = None
        
        def hook_fn(module, input, output):
            self.captured_inputs = input
            
        handle = self.detect_layer.register_forward_hook(hook_fn)
        
        # Run full model
        # We don't care about the final output, just what went into Detect
        _ = self.model(x)
        
        handle.remove()
        
        if self.captured_inputs is None:
            return [] # Should not happen
            
        # The input to Detect is a list of feature maps [P2, P3, P4] (usually tuple)
        features = self.captured_inputs[0] if isinstance(self.captured_inputs, tuple) else self.captured_inputs
        
        outputs = []
        # Simulate Detect layer forward (without the final concat/view)
        # cv2 is box, cv3 is cls
        for i in range(self.detect_layer.nl):
            x_i = features[i]
            cls_out = self.detect_layer.cv3[i](x_i)
            reg_out = self.detect_layer.cv2[i](x_i)
            outputs.append((cls_out, reg_out))
            
        return outputs


def main():
    parser = argparse.ArgumentParser(description="Active Learning Data Mining")
    parser.add_argument('--model', type=str, required=True, help="Path to trained .pt model")
    parser.add_argument('--data', type=str, required=True, help="Path to unlabeled dataset folder")
    parser.add_argument('--output', type=str, default="difficulty_map.json", help="Output JSON path")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--device', type=str, default='0', help="Device (0, cpu)")
    parser.add_argument('--limit', type=int, default=0, help="Limit number of images (0=all)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(">>> UNINA-YOLO-DLA: Active Learning Data Mining")
    print("=" * 60)
    
    # 1. Load Model
    print(f">>> Loading model from {args.model}...")
    try:
        yolo = YOLO(args.model)
        # Force reload to ensure weights are applied
        # Assuming args.model is a pt file
        model = yolo.model
        
        # Apply DLA Patch (ReLU)
        replace_silu_with_relu(model)
        
        # Wrap for Mining
        mining_model = MiningModelWrapper(model)
        
        # Handle device properly
        if args.device.isdigit():
            device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        mining_model.to(device)
        mining_model.eval()
        print(f">>> Model loaded and wrapped for mining on {device}.")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
        
    # 2. Setup Dataset
    print(f">>> Scanning dataset: {args.data}")
    dataset = UnlabeledDataset(args.data, imgsz=args.imgsz)
    print(f"    Found {len(dataset)} images.")
    
    if len(dataset) == 0:
        print("Exiting.")
        sys.exit(0)
        
    # Apply limit if specified
    if args.limit > 0 and args.limit < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(args.limit))
        print(f"    Limited to {args.limit} images.")
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_mining_batch
    )
    
    # 3. Active Learning Mining
    print(">>> Starting mining (Entropy calculation)...")
    learner = ActiveLearner(mining_model, gold_set_path=Path(".")) # gold path unused here
    
    # Compute scores
    scores = learner.compute_difficulty_scores(dataloader, mode="entropy")
    
    print(f">>> Mining complete. Computed scores for {len(scores)} images.")
    
    # 4. Save Results
    output_path = Path(args.output)
    print(f">>> Saving difficulty map to {output_path}...")
    
    # Sort for logging/inspection (optional, but nice)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Export dict
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)
        
    print(">>> Done!")
    print("\nTop 5 Most Uncertain Images:")
    for path, score in sorted_items[:5]:
        print(f"  {Path(path).name}: {score:.4f}")

if __name__ == "__main__":
    main()
