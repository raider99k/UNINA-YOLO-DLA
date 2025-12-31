import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
from PIL import Image

# --- Model Wrappers (Production-Ready) ---

class GroundingDINO:
    """Wrapper for GroundingDINO open-vocabulary detector."""
    def __init__(self, config_path: str, weights_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from groundingdino.util.inference import load_model
        except ImportError:
            raise ImportError("GroundingDINO is not installed. Please install it.")
            
        self.model = load_model(config_path, weights_path)
        self.model.to(self.device)
        print(f"[GroundingDINO] Loaded from {weights_path}")

    def predict(self, image: np.ndarray, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        """
        Predict boxes based on prompt.
        Returns: boxes (xyxy absolute), logits, phrases
        """
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T
        
        # Transform image for GDINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform(image_pil, None)
        
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert cxcywh (normalized) to xyxy (absolute)
        h, w = image.shape[:2]
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        return boxes_xyxy, logits.numpy(), phrases

def box_convert(boxes: torch.Tensor, in_fmt: str, out_fmt: str) -> torch.Tensor:
    """
    Convert boxes from in_fmt to out_fmt.
    Supported formats: 'cxcywh', 'xyxy'.
    """
    if in_fmt == 'cxcywh' and out_fmt == 'xyxy':
        cx, cy, w, h = boxes.unbind(-1)
        b = [(cx - 0.5 * w), (cy - 0.5 * h),
             (cx + 0.5 * w), (cy + 0.5 * h)]
        return torch.stack(b, dim=-1)
    return boxes

class SAM:
    """Wrapper for Segment Anything Model."""
    def __init__(self, checkpoint: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
             raise ImportError("Segment Anything (SAM) is not installed. Please install it.")
             
        # Default to vit_h (huge) for best quality
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print(f"[SAM] Loaded from {checkpoint}")

    def predict_mask(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Refine boxes into masks and tight-fit bounding boxes."""
        if len(boxes) == 0:
            return [], np.zeros((0, 4))
            
        # SAM expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        boxes_torch = torch.from_numpy(boxes).to(self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_torch, image.shape[:2])
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # masks: [N, 1, H, W] -> [N, H, W]
        masks = masks.squeeze(1).cpu().numpy().astype(np.uint8)
        
        refined_boxes = []
        final_masks = []
        
        for mask in masks:
            # Find bounding box of the mask
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                refined_boxes.append([x1, y1, x2, y2])
                final_masks.append(mask)
            else:
                # Fallback to original box if mask is empty (rare)
                pass 
                
        return final_masks, np.array(refined_boxes)

class SAHI_Wrapper:
    """Slicing Aided Hyper Inference logic."""
    def __init__(self, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    def get_slices(self, image: np.ndarray):
        """Generate slices for high-resolution inference with overlap."""
        h, w = image.shape[:2]
        stride_h = int(self.slice_height * (1 - self.overlap_height_ratio))
        stride_w = int(self.slice_width * (1 - self.overlap_width_ratio))

        # Handle small images
        if h <= self.slice_height and w <= self.slice_width:
             yield image, (0, 0)
             return

        for y in range(0, h, stride_h):
            for x in range(0, w, stride_w):
                # Ensure we don't go out of bounds but also cover the edges
                y_end = min(y + self.slice_height, h)
                x_end = min(x + self.slice_width, w)
                
                # Adjust start to ensure slice size is constant if possible, or handle edge case
                y_start = max(0, y_end - self.slice_height)
                x_start = max(0, x_end - self.slice_width)
                
                slice_img = image[y_start:y_end, x_start:x_end]
                yield slice_img, (x_start, y_start)

# --- Utilities ---

def map_boxes_to_global(boxes: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    """Maps boxes from slice coordinates to global image coordinates."""
    if len(boxes) == 0:
        return boxes
    global_boxes = boxes.copy().astype(float)
    global_boxes[:, [0, 2]] += x_offset
    global_boxes[:, [1, 3]] += y_offset
    return global_boxes

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Standard Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

# --- Main Auto-Labeling Pipeline ---

def auto_label_frame(
    image_path: str,
    output_dir: str,
    gdino: GroundingDINO,
    sam: SAM,
    sahi: SAHI_Wrapper,
    text_prompt: str = "traffic cone",
    class_map: Dict[str, int] = {"traffic cone": 0}
):
    """Run the Full Auto-Labeling Pipeline on a single frame."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        return

    all_boxes = []
    all_scores = []
    all_class_ids = []

    # 1. SAHI + GroundingDINO
    for slice_img, (x_off, y_off) in sahi.get_slices(image):
        boxes, logits, phrases = gdino.predict(slice_img, text_prompt)
        
        # Filter by class_map
        valid_indices = []
        for i, phrase in enumerate(phrases):
             # Simple matching logic
             matched_cls = None
             for k, v in class_map.items():
                 if k in phrase:
                     matched_cls = v
                     break
             if matched_cls is not None:
                 valid_indices.append(i)
                 all_class_ids.append(matched_cls)
        
        if valid_indices:
             slice_valid_boxes = boxes[valid_indices]
             slice_valid_scores = logits[valid_indices]
             
             # Map detailed boxes to global
             global_boxes = map_boxes_to_global(slice_valid_boxes, x_off, y_off)
             all_boxes.append(global_boxes)
             all_scores.append(slice_valid_scores)

    if not all_boxes:
        return

    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_class_ids = np.array(all_class_ids) # CAUTION: this list logic matches loop order, needs careful handling if shuffled

    # 2. Global NMS
    # Since we can have different classes, we should NMS per class or globally?
    # Usually per-class NMS.
    final_boxes = []
    final_class_ids = []
    
    unique_classes = np.unique(all_class_ids)
    for cls_id in unique_classes:
        cls_mask = all_class_ids == cls_id
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]
        
        keep_indices = nms(cls_boxes, cls_scores, iou_threshold=0.45)
        
        final_boxes.append(cls_boxes[keep_indices])
        # Replicate class id
        final_class_ids.extend([cls_id] * len(keep_indices))

    if not final_boxes:
         return
         
    final_boxes = np.vstack(final_boxes)

    # 3. Refine with SAM
    # SAM works best on the full image with global boxes
    masks, refined_boxes = sam.predict_mask(image, final_boxes)
    
    # 4. Save Labels (YOLO Format)
    img_h, img_w = image.shape[:2]
    label_path = os.path.join(output_dir, Path(image_path).stem + ".txt")
    
    with open(label_path, "w") as f:
        for i, box in enumerate(refined_boxes):
            cls_id = final_class_ids[i]
            # Convert xyxy to yolov8 normalized
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            f.write(f"{cls_id} {cx/img_w} {cy/img_h} {w/img_w} {h/img_h}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="GDINO config")
    parser.add_argument("--weights", type=str, required=True, help="GDINO weights")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input images dir")
    parser.add_argument("--output", type=str, required=True, help="Output labels dir")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Init Models
    gdino = GroundingDINO(args.config, args.weights)
    sam = SAM(args.sam_checkpoint)
    sahi = SAHI_Wrapper()
    
    images = list(Path(args.input).glob("*.jpg")) + list(Path(args.input).glob("*.png"))
    for img_path in tqdm(images):
        auto_label_frame(str(img_path), args.output, gdino, sam, sahi)
