/**
 * UNINA-YOLO-DLA: Post-processing Utilities
 *
 * Provides CPU-based decoding and NMS for the YOLO heads.
 * Can be replaced by CUDA kernels for further optimization.
 */

#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <algorithm>
#include <cmath>
#include <vector>

struct Detection {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;

  float width() const { return x2 - x1; }
  float height() const { return y2 - y1; }
  float area() const { return width() * height(); }
};

/**
 * @brief Computes Intersection over Union (IoU) between two boxes.
 */
inline float compute_iou(const Detection &a, const Detection &b) {
  float inter_x1 = std::max(a.x1, b.x1);
  float inter_y1 = std::max(a.y1, b.y1);
  float inter_x2 = std::min(a.x2, b.x2);
  float inter_y2 = std::min(a.y2, b.y2);

  if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2)
    return 0.0f;

  float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  return inter_area / (a.area() + b.area() - inter_area);
}

/**
 * @brief Performs Non-Maximum Suppression (NMS).
 */
inline std::vector<Detection> nms(std::vector<Detection> &detections,
                                  float iou_threshold) {
  std::sort(detections.begin(), detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  std::vector<Detection> result;
  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i])
      continue;
    result.push_back(detections[i]);
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (!suppressed[j] && detections[i].class_id == detections[j].class_id) {
        if (compute_iou(detections[i], detections[j]) > iou_threshold) {
          suppressed[j] = true;
        }
      }
    }
  }
  return result;
}

/**
 * @brief Applies Conformal Prediction dilation for formal safety.
 *
 * Expands the bounding box by a factor 'q' to guarantee ground truth
 * coverage with 1-alpha probability.
 *
 * Formula (from RESEARCH.md 5.2): B_safe = Dilate(B, f(conf, q))
 */
inline void apply_conformal_prediction(Detection &det, float q_factor) {
  float dw = det.width() * q_factor;
  float dh = det.height() * q_factor;

  det.x1 -= dw;
  det.y1 -= dh;
  det.x2 += dw;
  det.y2 += dh;
}

/**
 * @brief Decodes raw tensors from an anchor-free task-aligned head.
 *
 * Assumption:
 * - cls_data: [num_classes, H, W]
 * - reg_data: [4, H, W] (TLBR offsets or raw coords)
 */
inline void decode_head(const float *cls_data, const float *reg_data, int width,
                        int height, int stride, int num_classes,
                        float conf_threshold,
                        float q_factor, // Conformal Quantile factor
                        std::vector<Detection> &detections) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Find best class
      int best_class = -1;
      float max_conf = 0.0f;
      for (int c = 0; c < num_classes; ++c) {
        float conf =
            1.0f /
            (1.0f +
             std::exp(
                 -cls_data[c * height * width + y * width + x])); // Sigmoid
        if (conf > max_conf) {
          max_conf = conf;
          best_class = c;
        }
      }

      if (max_conf > conf_threshold) {
        // Decode TLBR (relative to cell center * stride)
        int grid_idx = y * width + x;
        float x_center = (x + 0.5f) * stride;
        float y_center = (y + 0.5f) * stride;

        // Assuming raw reg_data is [l, t, r, b] multipliers of stride
        float l = reg_data[0 * height * width + grid_idx] * stride;
        float t = reg_data[1 * height * width + grid_idx] * stride;
        float r = reg_data[2 * height * width + grid_idx] * stride;
        float b = reg_data[3 * height * width + grid_idx] * stride;

        Detection det;
        det.x1 = x_center - l;
        det.y1 = y_center - t;
        det.x2 = x_center + r;
        det.y2 = y_center + b;
        det.confidence = max_conf;
        det.class_id = best_class;

        // RESEARCH.md Section 5.2: Apply Conformal Prediction
        if (q_factor > 0.0f) {
          apply_conformal_prediction(det, q_factor);
        }

        detections.push_back(det);
      }
    }
  }
}

#endif
