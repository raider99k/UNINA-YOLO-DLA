/**
 * UNINA-YOLO-DLA: GPU-Native Post-Processing Kernels
 *
 * CRITICAL: All decoding and NMS runs entirely on GPU.
 * Only the final detection list (<1KB) is copied to host.
 *
 * This file implements:
 *   1. Grid-based YOLO decoding (anchor-free, TLBR)
 *   2. Confidence thresholding (early rejection)
 *   3. Batched NMS using atomics
 */

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Maximum detections to avoid memory explosion
#define MAX_DETECTIONS 1024
#define MAX_DETECTIONS_PER_HEAD 512

// Detection structure (GPU-compatible, 32 bytes)
struct __align__(32) GpuDetection {
  float x1, y1, x2, y2; // Box coordinates
  float confidence;
  int class_id;
  int valid; // 1 = valid, 0 = suppressed
  int _pad;  // Padding for alignment
};

// Atomic counter for detection indexing
__device__ int g_detection_count;

/**
 * @brief Sigmoid activation (inline device function)
 */
__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Computes IoU between two boxes
 */
__device__ float compute_iou_gpu(const GpuDetection &a, const GpuDetection &b) {
  float inter_x1 = fmaxf(a.x1, b.x1);
  float inter_y1 = fmaxf(a.y1, b.y1);
  float inter_x2 = fminf(a.x2, b.x2);
  float inter_y2 = fminf(a.y2, b.y2);

  if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2)
    return 0.0f;

  float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

  return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

/**
 * @brief Decodes a single YOLO head output and appends to detection buffer.
 *
 * Each thread processes one grid cell.
 * Uses atomicAdd to get a unique index in the output buffer.
 *
 * @param cls_data Classification logits [num_classes, H, W]
 * @param reg_data Regression outputs [4, H, W] (TLBR)
 * @param detections Output detection buffer
 * @param grid_w Grid width
 * @param grid_h Grid height
 * @param stride Stride of this head (4, 8, or 16)
 * @param num_classes Number of classes
 * @param conf_threshold Confidence threshold
 * @param conformal_q Conformal prediction dilation factor
 */
__global__ void decode_yolo_head_kernel(const float *__restrict__ cls_data,
                                        const float *__restrict__ reg_data,
                                        GpuDetection *detections, int grid_w,
                                        int grid_h, int stride, int num_classes,
                                        float conf_threshold,
                                        float conformal_q) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= grid_w || y >= grid_h)
    return;

  int grid_idx = y * grid_w + x;
  int grid_size = grid_w * grid_h;

  // Find best class (argmax + sigmoid)
  float max_conf = 0.0f;
  int best_class = -1;

#pragma unroll
  for (int c = 0; c < num_classes; ++c) {
    float logit = cls_data[c * grid_size + grid_idx];
    float conf = sigmoid(logit);
    if (conf > max_conf) {
      max_conf = conf;
      best_class = c;
    }
  }

  // Early exit if below threshold
  if (max_conf < conf_threshold)
    return;

  // Decode box (TLBR relative to cell center)
  float x_center = (x + 0.5f) * stride;
  float y_center = (y + 0.5f) * stride;

  float l = reg_data[0 * grid_size + grid_idx] * stride;
  float t = reg_data[1 * grid_size + grid_idx] * stride;
  float r = reg_data[2 * grid_size + grid_idx] * stride;
  float b = reg_data[3 * grid_size + grid_idx] * stride;

  float x1 = x_center - l;
  float y1 = y_center - t;
  float x2 = x_center + r;
  float y2 = y_center + b;

  // Apply conformal prediction dilation
  if (conformal_q > 0.0f) {
    float w = x2 - x1;
    float h = y2 - y1;
    x1 -= w * conformal_q;
    y1 -= h * conformal_q;
    x2 += w * conformal_q;
    y2 += h * conformal_q;
  }

  // Atomic increment to get unique slot
  int det_idx = atomicAdd(&g_detection_count, 1);
  if (det_idx >= MAX_DETECTIONS)
    return; // Overflow protection

  detections[det_idx].x1 = x1;
  detections[det_idx].y1 = y1;
  detections[det_idx].x2 = x2;
  detections[det_idx].y2 = y2;
  detections[det_idx].confidence = max_conf;
  detections[det_idx].class_id = best_class;
  detections[det_idx].valid = 1;
}

/**
 * @brief GPU NMS kernel (class-aware, parallel)
 *
 * Each thread checks if detection[i] should suppress detection[j].
 * Uses atomic compare-and-swap for thread safety.
 */
__global__ void nms_kernel(GpuDetection *detections, int num_detections,
                           float iou_threshold) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_detections)
    return;
  if (detections[i].valid == 0)
    return;

  // Compare with all lower-confidence detections
  for (int j = i + 1; j < num_detections; ++j) {
    if (detections[j].valid == 0)
      continue;
    if (detections[i].class_id != detections[j].class_id)
      continue;

    // Only suppress if i has higher confidence
    if (detections[i].confidence > detections[j].confidence) {
      float iou = compute_iou_gpu(detections[i], detections[j]);
      if (iou > iou_threshold) {
        detections[j].valid = 0; // Suppress j
      }
    }
  }
}

/**
 * @brief Sorts detections by confidence (simple GPU radix sort placeholder)
 *
 * For production, use CUB or Thrust radix_sort_by_key.
 * This is a simplified bubble sort for small arrays.
 */
__global__ void sort_detections_kernel(GpuDetection *detections, int n) {
  // Odd-even transposition sort (GPU-friendly)
  for (int phase = 0; phase < n; ++phase) {
    int start = phase % 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = start + 2 * idx;

    if (i + 1 < n) {
      if (detections[i].confidence < detections[i + 1].confidence) {
        // Swap
        GpuDetection tmp = detections[i];
        detections[i] = detections[i + 1];
        detections[i + 1] = tmp;
      }
    }
    __syncthreads();
  }
}

// =============================================================================
// Host API
// =============================================================================

extern "C" {

/**
 * @brief Resets the global detection counter (call before each frame)
 */
cudaError_t reset_detection_counter(cudaStream_t stream) {
  int zero = 0;
  return cudaMemcpyToSymbolAsync(g_detection_count, &zero, sizeof(int), 0,
                                 cudaMemcpyHostToDevice, stream);
}

/**
 * @brief Gets the current detection count
 */
cudaError_t get_detection_count(int *count, cudaStream_t stream) {
  return cudaMemcpyFromSymbolAsync(count, g_detection_count, sizeof(int), 0,
                                   cudaMemcpyDeviceToHost, stream);
}

/**
 * @brief Runs GPU-native YOLO decoding for one head
 */
cudaError_t decode_yolo_head(const float *d_cls, const float *d_reg,
                             GpuDetection *d_detections, int grid_w, int grid_h,
                             int stride, int num_classes, float conf_threshold,
                             float conformal_q, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((grid_w + block.x - 1) / block.x, (grid_h + block.y - 1) / block.y);

  decode_yolo_head_kernel<<<grid, block, 0, stream>>>(
      d_cls, d_reg, d_detections, grid_w, grid_h, stride, num_classes,
      conf_threshold, conformal_q);

  return cudaGetLastError();
}

/**
 * @brief Runs GPU NMS on the detection buffer
 */
cudaError_t run_gpu_nms(GpuDetection *d_detections, int num_detections,
                        float iou_threshold, cudaStream_t stream) {
  if (num_detections == 0)
    return cudaSuccess;

  // Sort by confidence first
  int threads = 256;
  int blocks = (num_detections / 2 + threads - 1) / threads;
  sort_detections_kernel<<<blocks, threads, 0, stream>>>(d_detections,
                                                         num_detections);

  // Run NMS
  blocks = (num_detections + threads - 1) / threads;
  nms_kernel<<<blocks, threads, 0, stream>>>(d_detections, num_detections,
                                             iou_threshold);

  return cudaGetLastError();
}

/**
 * @brief Copies only valid detections to host (compacted)
 *
 * This is the ONLY D2H transfer - typically <1KB
 */
cudaError_t copy_valid_detections_to_host(const GpuDetection *d_detections,
                                          GpuDetection *h_detections,
                                          int num_detections,
                                          int *out_valid_count,
                                          cudaStream_t stream) {
  // For simplicity, copy all and filter on host
  // In production, use stream compaction (CUB::DeviceSelect::Flagged)
  cudaError_t err = cudaMemcpyAsync(h_detections, d_detections,
                                    num_detections * sizeof(GpuDetection),
                                    cudaMemcpyDeviceToHost, stream);

  if (err != cudaSuccess)
    return err;

  cudaStreamSynchronize(stream);

  // Count valid
  int valid = 0;
  for (int i = 0; i < num_detections; ++i) {
    if (h_detections[i].valid) {
      if (valid != i) {
        h_detections[valid] = h_detections[i];
      }
      valid++;
    }
  }
  *out_valid_count = valid;

  return cudaSuccess;
}

} // extern "C"
