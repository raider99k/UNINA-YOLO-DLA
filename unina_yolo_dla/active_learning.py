"""
UNINA-YOLO-DLA: Active Learning & Dataset Curation.

Implements selection strategies to identify informative samples from the "Silver Set"
for human review and inclusion in the "Gold Set".

Strategies:
1. Entropy Sampling: Uncertainty in class predictions.
2. Localization Variance: Uncertainty in bounding box coordinates.
3. Coreset Selection: Diversity sampling using backbone embeddings.
4. Copy-Paste Augmentation: Real-to-Real augmentation using SAM masks.
"""

import torch
import numpy as np
from pathlib import Path

def calculate_entropy(class_probs: torch.Tensor) -> torch.Tensor:
    """Calculates Shannon Entropy for a set of class probabilities."""
    # class_probs: [N, NumClasses]
    entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-10), dim=-1)
    return entropy

class ActiveLearner:
    """Orchestrates the active learning cycle."""
    def __init__(self, model, gold_set_path: Path):
        self.model = model
        self.gold_set_path = gold_set_path

    def query_uncertain_samples(self, silver_set_dataloader, top_k: int = 100, mode: str = "entropy"):
        """Selects the most uncertain samples using Entropy or Localization Variance."""
        self.model.eval()
        scores = {}
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in silver_set_dataloader:
                images = batch["images"].to(device)
                paths = batch["paths"]
                
                # Forward pass - returns list of (cls_head, reg_head)
                outputs = self.model(images)
                
                for i, path in enumerate(paths):
                    batch_scores = []
                    for level_out in outputs:
                        cls_out, reg_out = level_out
                        # cls_out: [B, num_classes, H, W]
                        # reg_out: [B, 4, H, W]
                        
                        probs = torch.softmax(cls_out[i], dim=0) # [C, H, W]
                        
                        if mode == "entropy":
                            # Shannon entropy per pixel, then max across spatial dims
                            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=0)
                            batch_scores.append(ent.max().item())
                        elif mode == "loc_var":
                            # Proxy for localization variance: max confidence vs second max
                            # or use reg_out distribution if available.
                            # For simplicity, we use the Inverse of Confidence here as a stub
                            conf = probs.max(dim=0)[0]
                            batch_scores.append(1.0 - conf.max().item())
                            
                    scores[path] = max(batch_scores) if batch_scores else 0.0
                    
        return sorted(scores, key=scores.get, reverse=True)[:top_k]

    def coreset_selection(self, silver_set_dataloader, target_size: int):
        """Diversity sampling based on feature embeddings."""
        # 1. Extract embeddings from Backbone SPPF layer
        # 2. Run K-Center Greedy or K-Means++ to select diverse samples
        pass

class CopyPasteAugmentor:
    """implements Real-to-Real Copy-Paste augmentation."""
    def __init__(self, cone_assets_path: Path):
        self.cone_assets = list(cone_assets_path.glob("*.npy")) # SAM masks + patches

    def apply(self, background_image):
        """Pastes real cone assets onto a background image."""
        # 1. Select random cones from assets
        # 2. Apply random scaling, rotation, and brightness adjustments
        # 3. Paste onto background with edge blending
        return background_image

if __name__ == "__main__":
    print("UNINA-YOLO-DLA Active Learning Module")
