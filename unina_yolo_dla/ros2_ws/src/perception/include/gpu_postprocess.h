/**
 * UNINA-YOLO-DLA: GPU Post-Processing Header
 *
 * Provides C++ interface for GPU-native YOLO decoding and NMS.
 */

#ifndef GPU_POSTPROCESS_H
#define GPU_POSTPROCESS_H

// =============================================================================
// MOCK_CUDA: Stub types for compiling without CUDA toolkit
// =============================================================================
#ifdef MOCK_CUDA
typedef int cudaError_t;
typedef void *cudaStream_t;
#define cudaSuccess 0
#define __align__(x)
#else
#include <cuda_runtime.h>
#endif

#define MAX_DETECTIONS 1024

// GPU-compatible detection structure
struct __align__(32) GpuDetection {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;
  int valid;
  int _pad;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Resets the atomic detection counter (call before each frame)
 */
cudaError_t reset_detection_counter(cudaStream_t stream);

/**
 * @brief Gets the current detection count after decoding
 */
cudaError_t get_detection_count(int *count, cudaStream_t stream);

/**
 * @brief Decodes a single YOLO head on GPU
 */
cudaError_t decode_yolo_head(const float *d_cls, const float *d_reg,
                             GpuDetection *d_detections, int grid_w, int grid_h,
                             int stride, int num_classes, float conf_threshold,
                             float conformal_q, cudaStream_t stream);

/**
 * @brief Runs GPU NMS (class-aware)
 */
cudaError_t run_gpu_nms(GpuDetection *d_detections, int num_detections,
                        float iou_threshold, cudaStream_t stream);

/**
 * @brief Copies valid detections to host (~1KB, not 5MB)
 */
cudaError_t copy_valid_detections_to_host(const GpuDetection *d_detections,
                                          GpuDetection *h_detections,
                                          int num_detections,
                                          int *out_valid_count,
                                          cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPU_POSTPROCESS_H
