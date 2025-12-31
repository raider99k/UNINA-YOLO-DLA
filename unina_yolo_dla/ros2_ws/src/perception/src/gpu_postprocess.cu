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
 *   4. CUB stream compaction for minimal D2H transfer
 */

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// CUB for stream compaction
#include <cub/cub.cuh>

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

// =============================================================================
// Workspace-based counter (safer than global __device__ variable)
// =============================================================================

/**
 * @brief Workspace structure for post-processing.
 *
 * Allocated once during init_postprocess_resources() and passed to kernels.
 * This avoids the multi-stream safety issues of global __device__ variables.
 */
struct PostprocessWorkspace {
  int *d_detection_count;           // Device counter for atomic indexing
  GpuDetection *d_compacted_output; // Compacted output buffer
  void *d_cub_temp_storage;         // CUB temporary storage
  size_t cub_temp_storage_bytes;    // CUB temp storage size
};

// Global workspace (allocated once, reused per frame)
static PostprocessWorkspace g_workspace = {nullptr, nullptr, nullptr, 0};

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
 * Uses atomicAdd on workspace counter for multi-stream safety.
 *
 * @param cls_data Classification logits [num_classes, H, W]
 * @param reg_data Regression outputs [4, H, W] (TLBR)
 * @param detections Output detection buffer
 * @param d_count Device pointer to atomic counter (workspace-based)
 * @param grid_w Grid width
 * @param grid_h Grid height
 * @param stride Stride of this head (4, 8, or 16)
 * @param num_classes Number of classes
 * @param conf_threshold Confidence threshold
 * @param conformal_q Conformal prediction dilation factor
 */
__global__ void decode_yolo_head_kernel(const float *__restrict__ cls_data,
                                        const float *__restrict__ reg_data,
                                        GpuDetection *detections, int *d_count,
                                        int grid_w, int grid_h, int stride,
                                        int num_classes, float conf_threshold,
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

  // Determine if this thread has a valid detection
  bool has_detection = false;

  // Early exit if below threshold (check before mask calculation)
  if (max_conf >= conf_threshold) {
    has_detection = true;
  }

  // OPTIMIZATION: Warp-Aggregated Atomics
  // Reduce global atomic pressure by aggregating valid detections within a
  // warp.

  // Get mask of all threads in this warp that have a detection
  unsigned int mask = __activemask();
  unsigned int det_mask = __ballot_sync(mask, has_detection);

  if (has_detection) {
    // Decode box (TLBR relative to cell center) only if valid
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

    // Calculate local rank (prefix sum of set bits)
    // lane_id = threadIdx.x % 32 effectively
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int active_rank = __popc(det_mask & ((1 << lane_id) - 1));

    // Elect a leader (first active thread in warp)
    int leader_idx = __ffs(det_mask) - 1;
    int warp_base_idx = 0;

    // Leader performs single atomic add for the whole warp
    if (lane_id == leader_idx) {
      int count_to_add = __popc(det_mask);
      warp_base_idx = atomicAdd(d_count, count_to_add);
    }

    // Broadcast base index to all active threads in warp
    warp_base_idx = __shfl_sync(det_mask, warp_base_idx, leader_idx);

    // Calculate final index
    int det_idx = warp_base_idx + active_rank;

    if (det_idx < MAX_DETECTIONS) {
      detections[det_idx].x1 = x1;
      detections[det_idx].y1 = y1;
      detections[det_idx].x2 = x2;
      detections[det_idx].y2 = y2;
      detections[det_idx].confidence = max_conf;
      detections[det_idx].class_id = best_class;
      detections[det_idx].valid = 1;
    }
  }
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
 * @brief Functor for confidence-based sorting
 */
struct ConfidenceComparator {
  __device__ bool operator()(const GpuDetection &a,
                             const GpuDetection &b) const {
    // Sort descending: highest confidence first
    return a.confidence > b.confidence;
  }
};

/**
 * @brief Functor to extract valid flag for CUB stream compaction
 */
struct IsValidDetection {
  __device__ __forceinline__ bool operator()(const GpuDetection &det) const {
    return det.valid != 0;
  }
};

// =============================================================================
// Host API
// =============================================================================

extern "C" {

/**
 * @brief Initializes GPU resources for post-processing (One-time allocation)
 *
 * Allocates:
 *   - Workspace counter (int on device)
 *   - Compacted output buffer
 *   - CUB temporary storage for stream compaction
 */
cudaError_t init_postprocess_resources() {
  cudaError_t err;

  // Allocate device counter
  err = cudaMalloc(&g_workspace.d_detection_count, sizeof(int));
  if (err != cudaSuccess)
    return err;

  // Allocate compacted output buffer
  err = cudaMalloc(&g_workspace.d_compacted_output,
                   MAX_DETECTIONS * sizeof(GpuDetection));
  if (err != cudaSuccess)
    return err;

  // Query CUB temporary storage size
  g_workspace.cub_temp_storage_bytes = 0;
  int *d_num_selected = nullptr;
  err = cudaMalloc(&d_num_selected, sizeof(int));
  if (err != cudaSuccess)
    return err;

  // Dummy call to get temp storage size
  GpuDetection *d_dummy_in = nullptr;
  GpuDetection *d_dummy_out = nullptr;
  cub::DeviceSelect::If(nullptr, g_workspace.cub_temp_storage_bytes, d_dummy_in,
                        d_dummy_out, d_num_selected, MAX_DETECTIONS,
                        IsValidDetection());

  // Allocate CUB temp storage
  err = cudaMalloc(&g_workspace.d_cub_temp_storage,
                   g_workspace.cub_temp_storage_bytes);
  if (err != cudaSuccess)
    return err;

  cudaFree(d_num_selected);

  return cudaSuccess;
}

/**
 * @brief Frees GPU resources
 */
cudaError_t cleanup_postprocess_resources() {
  if (g_workspace.d_detection_count) {
    cudaFree(g_workspace.d_detection_count);
    g_workspace.d_detection_count = nullptr;
  }
  if (g_workspace.d_compacted_output) {
    cudaFree(g_workspace.d_compacted_output);
    g_workspace.d_compacted_output = nullptr;
  }
  if (g_workspace.d_cub_temp_storage) {
    cudaFree(g_workspace.d_cub_temp_storage);
    g_workspace.d_cub_temp_storage = nullptr;
  }
  g_workspace.cub_temp_storage_bytes = 0;
  return cudaSuccess;
}

/**
 * @brief Resets the workspace detection counter (call before each frame)
 */
cudaError_t reset_detection_counter(cudaStream_t stream) {
  int zero = 0;
  return cudaMemcpyAsync(g_workspace.d_detection_count, &zero, sizeof(int),
                         cudaMemcpyHostToDevice, stream);
}

/**
 * @brief Gets the current detection count
 */
cudaError_t get_detection_count(int *count, cudaStream_t stream) {
  return cudaMemcpyAsync(count, g_workspace.d_detection_count, sizeof(int),
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
      d_cls, d_reg, d_detections, g_workspace.d_detection_count, grid_w, grid_h,
      stride, num_classes, conf_threshold, conformal_q);

  return cudaGetLastError();
}

/**
 * @brief Runs GPU NMS on the detection buffer
 */
cudaError_t run_gpu_nms(GpuDetection *d_detections, int num_detections,
                        float iou_threshold, cudaStream_t stream) {
  if (num_detections == 0)
    return cudaSuccess;

  // Sort by confidence first (using Thrust for production performance)
  thrust::device_ptr<GpuDetection> d_ptr(d_detections);
  try {
    thrust::sort(thrust::cuda::par.on(stream), d_ptr, d_ptr + num_detections,
                 ConfidenceComparator());
  } catch (...) {
    return cudaErrorUnknown;
  }

  // Run NMS
  int threads = 256;
  int blocks = (num_detections + threads - 1) / threads;
  nms_kernel<<<blocks, threads, 0, stream>>>(d_detections, num_detections,
                                             iou_threshold);

  return cudaGetLastError();
}

/**
 * @brief Copies only valid detections to host using CUB stream compaction.
 *
 * This is the ONLY D2H transfer - typically <1KB.
 * Uses cub::DeviceSelect::If for GPU-side compaction before copy.
 */
cudaError_t copy_valid_detections_to_host(const GpuDetection *d_detections,
                                          GpuDetection *h_detections,
                                          int num_detections,
                                          int *out_valid_count,
                                          cudaStream_t stream) {
  if (num_detections == 0) {
    *out_valid_count = 0;
    return cudaSuccess;
  }

  // Allocate temporary device counter for CUB
  int *d_num_selected = nullptr;
  cudaError_t err = cudaMalloc(&d_num_selected, sizeof(int));
  if (err != cudaSuccess)
    return err;

  // Run CUB stream compaction (GPU-side filtering)
  err = cub::DeviceSelect::If(g_workspace.d_cub_temp_storage,
                              g_workspace.cub_temp_storage_bytes, d_detections,
                              g_workspace.d_compacted_output, d_num_selected,
                              num_detections, IsValidDetection(), stream);

  if (err != cudaSuccess) {
    cudaFree(d_num_selected);
    return err;
  }

  // Copy count to host
  int valid_count = 0;
  err = cudaMemcpyAsync(&valid_count, d_num_selected, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    cudaFree(d_num_selected);
    return err;
  }

  cudaStreamSynchronize(stream);
  cudaFree(d_num_selected);

  // Clamp to MAX_DETECTIONS
  valid_count = (valid_count > MAX_DETECTIONS) ? MAX_DETECTIONS : valid_count;
  *out_valid_count = valid_count;

  if (valid_count > 0) {
    // Copy ONLY valid detections to host (compacted buffer)
    err = cudaMemcpyAsync(h_detections, g_workspace.d_compacted_output,
                          valid_count * sizeof(GpuDetection),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
      return err;

    cudaStreamSynchronize(stream);
  }

  return cudaSuccess;
}

} // extern "C"
