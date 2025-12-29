/**
 * UNINA-YOLO-DLA: CUDA Preprocessing Header (Production)
 *
 * C++ interface for CUDA preprocessing kernels.
 * Designed for JetPack 5.x / 6.x environment.
 */

#ifndef CUDA_PREPROCESS_H
#define CUDA_PREPROCESS_H

#include <cstdint>
#include <cuda_runtime.h>


#ifdef JETPACK_AVAILABLE
#include <nvbufsurface.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// --- Normalization Parameters ---
/**
 * @brief Configurable normalization parameters.
 *
 * Allows tuning mean/std without recompiling kernels.
 */
typedef struct {
  float mean_r;
  float mean_g;
  float mean_b;
  float std_r;
  float std_g;
  float std_b;
} NormParams;

/**
 * @brief Create ImageNet normalization parameters.
 */
NormParams create_norm_params_imagenet(void);

/**
 * @brief Create custom normalization parameters.
 */
NormParams create_norm_params(float mean_r, float mean_g, float mean_b,
                              float std_r, float std_g, float std_b);

// --- Preprocessing Functions ---

/**
 * @brief Preprocess BGRA image: resize + convert + normalize.
 */
cudaError_t preprocess_bgra_resize(const uint8_t *d_input, float *d_output,
                                   int src_width, int src_height, int src_pitch,
                                   int dst_width, int dst_height,
                                   NormParams params, cudaStream_t stream);

/**
 * @brief Preprocess BGRA image without resize.
 */
cudaError_t preprocess_bgra(const uint8_t *d_input, float *d_output, int width,
                            int height, int pitch, NormParams params,
                            cudaStream_t stream);

/**
 * @brief Preprocess NV12 (YUV420sp) image.
 */
cudaError_t preprocess_nv12(const uint8_t *d_y_plane, const uint8_t *d_uv_plane,
                            float *d_output, int width, int height, int y_pitch,
                            int uv_pitch, NormParams params,
                            cudaStream_t stream);

#ifdef JETPACK_AVAILABLE
/**
 * @brief Preprocess from NvBufSurface (JetPack Multimedia API).
 */
cudaError_t preprocess_nvbufsurface(NvBufSurface *surface, float *d_output,
                                    int dst_width, int dst_height,
                                    NormParams params, cudaStream_t stream);
#endif

// --- Memory Management ---

/**
 * @brief Allocate output buffer for preprocessed tensor.
 */
float *allocate_preprocess_buffer(int width, int height);

/**
 * @brief Free preprocessing buffer.
 */
void free_preprocess_buffer(float *d_buffer);

/**
 * @brief Create a CUDA stream for async preprocessing.
 */
cudaStream_t create_preprocess_stream(void);

/**
 * @brief Destroy CUDA stream.
 */
void destroy_preprocess_stream(cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PREPROCESS_H
