#!/usr/bin/env python3
"""
UNINA-YOLO-DLA: Synthetic FSD Dataset Generator.

Generates a structurally valid YOLO-format dataset for local dry-run testing
on systems without real FSD data.

Output Structure:
    datasets/fsd_synth/
    ├── images/
    │   ├── train/  (50+ images)
    │   └── val/    (10+ images)
    ├── labels/
    │   ├── train/  (50+ .txt files)
    │   └── val/    (10+ .txt files)
    └── fsd_synth.yaml

Classes:
    0: blue_cone     (Blue triangle)
    1: yellow_cone   (Yellow triangle)
    2: orange_cone   (Orange triangle)
    3: large_orange_cone (Large orange rectangle)

Usage:
    python tools/generate_synthetic_fsd.py [--num-train 50] [--num-val 10]
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# --- Constants ---
IMG_SIZE = 640
CLASSES = {
    0: ("blue_cone", (255, 100, 50)),       # Blue (BGR -> RGB for display)
    1: ("yellow_cone", (50, 230, 230)),     # Yellow
    2: ("orange_cone", (50, 140, 255)),     # Orange
    3: ("large_orange_cone", (30, 120, 255)),  # Large Orange
}

# Cone sizes in pixels (min, max)
CONE_SIZE_SMALL = (20, 50)
CONE_SIZE_LARGE = (60, 100)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def draw_triangle(img: np.ndarray, cx: int, cy: int, size: int, color: tuple) -> tuple:
    """
    Draw a filled triangle (cone shape) centered at (cx, cy).
    
    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) in pixels.
    """
    # Triangle pointing up
    half_base = size // 2
    height = int(size * 0.87)  # Equilateral-ish triangle
    
    pts = np.array([
        [cx, cy - height // 2],           # Top
        [cx - half_base, cy + height // 2],  # Bottom-left
        [cx + half_base, cy + height // 2],  # Bottom-right
    ], dtype=np.int32)
    
    cv2.fillPoly(img, [pts], color)
    
    x_min = cx - half_base
    y_min = cy - height // 2
    x_max = cx + half_base
    y_max = cy + height // 2
    
    return (x_min, y_min, x_max, y_max)


def draw_rectangle(img: np.ndarray, cx: int, cy: int, width: int, height: int, color: tuple) -> tuple:
    """
    Draw a filled rectangle centered at (cx, cy).
    
    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) in pixels.
    """
    x_min = cx - width // 2
    y_min = cy - height // 2
    x_max = cx + width // 2
    y_max = cy + height // 2
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, -1)
    
    return (x_min, y_min, x_max, y_max)


def generate_random_cones(img_size: int = IMG_SIZE, min_cones: int = 2, max_cones: int = 8) -> tuple:
    """
    Generate a synthetic image with random cones.
    
    Returns:
        Tuple of (image, labels) where labels is a list of [class_id, x_c, y_c, w, h] (normalized).
    """
    # Black background
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Optional: Add slight noise for realism
    noise = np.random.randint(0, 15, (img_size, img_size, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    labels = []
    num_cones = random.randint(min_cones, max_cones)
    
    # Track occupied regions to avoid overlap
    occupied = []
    
    for _ in range(num_cones):
        class_id = random.choice([0, 1, 2, 3])
        _, color = CLASSES[class_id]
        
        # Large orange cone uses rectangle
        is_large = (class_id == 3)
        
        # Random position (avoid edges)
        margin = 80
        cx = random.randint(margin, img_size - margin)
        cy = random.randint(margin, img_size - margin)
        
        # Check for overlap with existing cones
        overlap = False
        for (ox_min, oy_min, ox_max, oy_max) in occupied:
            if (ox_min - 30 < cx < ox_max + 30) and (oy_min - 30 < cy < oy_max + 30):
                overlap = True
                break
        
        if overlap:
            continue
        
        # Draw the cone
        if is_large:
            width = random.randint(*CONE_SIZE_LARGE)
            height = int(width * 1.5)  # Taller rectangle
            bbox = draw_rectangle(img, cx, cy, width, height, color)
        else:
            size = random.randint(*CONE_SIZE_SMALL)
            bbox = draw_triangle(img, cx, cy, size, color)
        
        x_min, y_min, x_max, y_max = bbox
        occupied.append(bbox)
        
        # Convert to YOLO format (normalized)
        x_center = clamp((x_min + x_max) / 2.0 / img_size)
        y_center = clamp((y_min + y_max) / 2.0 / img_size)
        width_norm = clamp((x_max - x_min) / img_size)
        height_norm = clamp((y_max - y_min) / img_size)
        
        # Validate bounding box is within image
        if width_norm > 0 and height_norm > 0:
            labels.append([class_id, x_center, y_center, width_norm, height_norm])
    
    return img, labels


def validate_labels(labels: list, tolerance: float = 1e-6) -> bool:
    """
    Validate that all labels have normalized coordinates in [0, 1].
    
    Returns:
        True if all labels are valid.
    """
    for label in labels:
        class_id, x_c, y_c, w, h = label
        
        if class_id < 0 or class_id > 3:
            print(f"ERROR: Invalid class_id {class_id}")
            return False
        
        for val, name in [(x_c, "x_center"), (y_c, "y_center"), (w, "width"), (h, "height")]:
            if val < -tolerance or val > 1.0 + tolerance:
                print(f"ERROR: {name}={val} out of bounds [0, 1]")
                return False
    
    return True


def generate_dataset(output_dir: Path, num_train: int = 50, num_val: int = 10) -> None:
    """Generate the complete synthetic dataset."""
    
    # Create directory structure
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"
    
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f">>> Generating {num_train} training images...")
    for i in range(num_train):
        img, labels = generate_random_cones()
        
        if not validate_labels(labels):
            print(f"WARNING: Skipping invalid image {i}")
            continue
        
        # Save image
        img_path = images_train / f"synth_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Save labels
        label_path = labels_train / f"synth_{i:04d}.txt"
        with open(label_path, "w") as f:
            for label in labels:
                class_id, x_c, y_c, w, h = label
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    print(f">>> Generating {num_val} validation images...")
    for i in range(num_val):
        img, labels = generate_random_cones()
        
        if not validate_labels(labels):
            print(f"WARNING: Skipping invalid image {i}")
            continue
        
        # Save image
        img_path = images_val / f"synth_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Save labels
        label_path = labels_val / f"synth_{i:04d}.txt"
        with open(label_path, "w") as f:
            for label in labels:
                class_id, x_c, y_c, w, h = label
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    # Generate YAML config
    yaml_path = output_dir / "fsd_synth.yaml"
    yaml_content = f"""# Synthetic FSD Dataset for Local Dry-Run Testing
# Auto-generated by generate_synthetic_fsd.py

path: {output_dir.resolve()}
train: images/train
val: images/val

# Class definitions (matches real FSD cones)
names:
  0: blue_cone
  1: yellow_cone
  2: orange_cone
  3: large_orange_cone

nc: 4
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"\n>>> Dataset generated successfully!")
    print(f"    Output directory: {output_dir.resolve()}")
    print(f"    Training images:  {num_train}")
    print(f"    Validation images: {num_val}")
    print(f"    Config file: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic FSD dataset for local testing."
    )
    parser.add_argument(
        "--num-train", type=int, default=50,
        help="Number of training images to generate (default: 50)"
    )
    parser.add_argument(
        "--num-val", type=int, default=10,
        help="Number of validation images to generate (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: datasets/fsd_synth relative to script)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: datasets/fsd_synth relative to project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        output_dir = project_root / "datasets" / "fsd_synth"
    
    generate_dataset(output_dir, args.num_train, args.num_val)


if __name__ == "__main__":
    main()
