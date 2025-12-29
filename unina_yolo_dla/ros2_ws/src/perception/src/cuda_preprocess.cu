/**
 * UNINA-YOLO-DLA: CUDA Preprocessing Kernels (Production)
 *
 * GPU-accelerated image preprocessing for zero-copy inference pipeline.
 * Uses real NVIDIA headers (cuda_runtime.h, nvbufsurface.h).
 *
 * Operations:
 *   1. Color space conversion (BGRA -> RGB)
 *   2. Resize (bilinear interpolation to target size)
 *   3. Normalization (configurable mean/std)
 *   4. HWC -> CHW layout transformation
 *
 * Target Hardware:
 *   - NVIDIA Jetson Orin AGX/NX (JetPack 5.x / 6.x)
 *   - Input: NvBufSurface (from camera/VIC/ZED)
 *   - Output: float32 tensor in GPU memory for TensorRT/DLA
 *
 * Zero-Copy Design:
 *   - Input buffer is directly accessed from NvBufSurface
 *   - Output tensor is pre-allocated and reused
 *   - No CPU involvement in the preprocessing pipeline
 *
 * Build Requirements:
 *   - CUDA Toolkit (nvcc)
 *   - JetPack Multimedia API (for nvbufsurface.h)
 *   - Compile with: nvcc -arch=sm_87 cuda_preprocess.cu
 */

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// JetPack Multimedia API for NvBufSurface
// Available in: /usr/src/jetson_multimedia_api/include/
#ifdef JETPACK_AVAILABLE
#include <nvbufsurface.h>
#endif

// --- Error Checking Macro ---
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      return err;                                                              \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_VOID(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

// --- Normalization Parameters Structure ---
/**
 * @brief Configurable normalization parameters.
 *
 * Passed as a uniform buffer to avoid recompilation when tuning.
 * Default values are ImageNet statistics.
 */
struct NormParams {
  float mean_r;
  float mean_g;
  float mean_b;
  float std_r;
  float std_g;
  float std_b;

  // Default constructor with ImageNet values
  __host__ __device__ NormParams()
      : mean_r(0.485f), mean_g(0.456f), mean_b(0.406f), std_r(0.229f),
        std_g(0.224f), std_b(0.225f) {}

  // Custom constructor
  __host__ __device__ NormParams(float mr, float mg, float mb, float sr,
                                 float sg, float sb)
      : mean_r(mr), mean_g(mg), mean_b(mb), std_r(sr), std_g(sg), std_b(sb) {}
};

// --- Kernel: BGRA to RGB + Normalize (Configurable) ---
/**
 * @brief Converts BGRA uint8 input to RGB float32 output with configurable
 * normalization.
 *
 * @param input     Pointer to BGRA input image in device memory.
 * @param output    Pointer to output tensor in device memory (CHW layout).
 * @param width     Image width.
 * @param height    Image height.
 * @param pitch     Row pitch (stride) of input buffer in bytes.
 * @param params    Normalization parameters (mean/std per channel).
 */
__global__ void bgra_to_rgb_normalize_kernel(const uint8_t *__restrict__ input,
                                             float *__restrict__ output,
                                             int width, int height, int pitch,
                                             NormParams params) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Input offset (HWC layout, BGRA order)
  int input_idx = y * pitch + x * 4;

  // Read BGRA values
  uint8_t b = input[input_idx + 0];
  uint8_t g = input[input_idx + 1];
  uint8_t r = input[input_idx + 2];

  // Output offsets (CHW layout, RGB order)
  int plane_size = width * height;
  int output_idx = y * width + x;

  // Normalize and write to CHW format
  output[0 * plane_size + output_idx] =
      ((r / 255.0f) - params.mean_r) / params.std_r;
  output[1 * plane_size + output_idx] =
      ((g / 255.0f) - params.mean_g) / params.std_g;
  output[2 * plane_size + output_idx] =
      ((b / 255.0f) - params.mean_b) / params.std_b;
}

// --- Kernel: Bilinear Resize + BGRA to RGB + Normalize ---
/**
 * @brief Resizes BGRA input with bilinear interpolation, converts to RGB,
 * normalizes.
 *
 * @param input       Pointer to BGRA input buffer.
 * @param output      Pointer to output tensor buffer (CHW, float32).
 * @param src_width   Source image width.
 * @param src_height  Source image height.
 * @param src_pitch   Source row pitch in bytes.
 * @param dst_width   Target width.
 * @param dst_height  Target height.
 * @param params      Normalization parameters.
 */
__global__ void resize_bgra_to_rgb_normalize_kernel(
    const uint8_t *__restrict__ input, float *__restrict__ output,
    int src_width, int src_height, int src_pitch, int dst_width, int dst_height,
    NormParams params) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_width || dst_y >= dst_height)
    return;

  // Source coordinates (bilinear interpolation)
  float scale_x = (float)src_width / dst_width;
  float scale_y = (float)src_height / dst_height;

  float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
  float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

  // Clamp to valid range
  src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
  src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

  // Integer and fractional parts
  int x0 = (int)src_x;
  int y0 = (int)src_y;
  int x1 = min(x0 + 1, src_width - 1);
  int y1 = min(y0 + 1, src_height - 1);

  float fx = src_x - x0;
  float fy = src_y - y0;

  // Bilinear interpolation weights
  float w00 = (1.0f - fx) * (1.0f - fy);
  float w01 = fx * (1.0f - fy);
  float w10 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  // Sample 4 corner pixels (BGRA format)
  int idx00 = y0 * src_pitch + x0 * 4;
  int idx01 = y0 * src_pitch + x1 * 4;
  int idx10 = y1 * src_pitch + x0 * 4;
  int idx11 = y1 * src_pitch + x1 * 4;

  // Interpolate each channel (BGR -> RGB reordering)
  float r = w00 * input[idx00 + 2] + w01 * input[idx01 + 2] +
            w10 * input[idx10 + 2] + w11 * input[idx11 + 2];
  float g = w00 * input[idx00 + 1] + w01 * input[idx01 + 1] +
            w10 * input[idx10 + 1] + w11 * input[idx11 + 1];
  float b = w00 * input[idx00 + 0] + w01 * input[idx01 + 0] +
            w10 * input[idx10 + 0] + w11 * input[idx11 + 0];

  // Write to CHW output with normalization
  int plane_size = dst_width * dst_height;
  int output_idx = dst_y * dst_width + dst_x;

  output[0 * plane_size + output_idx] =
      ((r / 255.0f) - params.mean_r) / params.std_r;
  output[1 * plane_size + output_idx] =
      ((g / 255.0f) - params.mean_g) / params.std_g;
  output[2 * plane_size + output_idx] =
      ((b / 255.0f) - params.mean_b) / params.std_b;
}

// --- Kernel: NV12 to RGB + Normalize (Common camera format) ---
/**
 * @brief Converts NV12 (YUV420sp) to RGB float32 with normalization.
 *
 * NV12 is common in camera pipelines (ZED, CSI cameras via VIC).
 */
__global__ void
nv12_to_rgb_normalize_kernel(const uint8_t *__restrict__ y_plane,
                             const uint8_t *__restrict__ uv_plane,
                             float *__restrict__ output, int width, int height,
                             int y_pitch, int uv_pitch, NormParams params) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y_coord = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y_coord >= height)
    return;

  // Read Y value
  float Y = y_plane[y_coord * y_pitch + x];

  // Read UV values (subsampled 2x2)
  int uv_x = x / 2;
  int uv_y = y_coord / 2;
  int uv_idx = uv_y * uv_pitch + uv_x * 2;
  float U = uv_plane[uv_idx + 0] - 128.0f;
  float V = uv_plane[uv_idx + 1] - 128.0f;

  // YUV to RGB conversion (BT.601)
  float r = Y + 1.402f * V;
  float g = Y - 0.344136f * U - 0.714136f * V;
  float b = Y + 1.772f * U;

  // Clamp to [0, 255]
  r = fmaxf(0.0f, fminf(255.0f, r));
  g = fmaxf(0.0f, fminf(255.0f, g));
  b = fmaxf(0.0f, fminf(255.0f, b));

  // Output (CHW layout)
  int plane_size = width * height;
  int output_idx = y_coord * width + x;

  output[0 * plane_size + output_idx] =
      ((r / 255.0f) - params.mean_r) / params.std_r;
  output[1 * plane_size + output_idx] =
      ((g / 255.0f) - params.mean_g) / params.std_g;
  output[2 * plane_size + output_idx] =
      ((b / 255.0f) - params.mean_b) / params.std_b;
}

// =============================================================================
// HOST WRAPPER FUNCTIONS
// =============================================================================

/**
 * @brief Create default normalization parameters (ImageNet).
 */
extern "C" NormParams create_norm_params_imagenet() { return NormParams(); }

/**
 * @brief Create custom normalization parameters.
 */
extern "C" NormParams create_norm_params(float mean_r, float mean_g,
                                         float mean_b, float std_r, float std_g,
                                         float std_b) {
  return NormParams(mean_r, mean_g, mean_b, std_r, std_g, std_b);
}

/**
 * @brief Preprocess BGRA image: resize + convert + normalize.
 *
 * Main entry point for preprocessing pipeline.
 */
extern "C" cudaError_t
preprocess_bgra_resize(const uint8_t *d_input, float *d_output, int src_width,
                       int src_height, int src_pitch, int dst_width,
                       int dst_height, NormParams params, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((dst_width + block.x - 1) / block.x,
            (dst_height + block.y - 1) / block.y);

  resize_bgra_to_rgb_normalize_kernel<<<grid, block, 0, stream>>>(
      d_input, d_output, src_width, src_height, src_pitch, dst_width,
      dst_height, params);

  return cudaGetLastError();
}

/**
 * @brief Preprocess BGRA image without resize.
 */
extern "C" cudaError_t preprocess_bgra(const uint8_t *d_input, float *d_output,
                                       int width, int height, int pitch,
                                       NormParams params, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  bgra_to_rgb_normalize_kernel<<<grid, block, 0, stream>>>(
      d_input, d_output, width, height, pitch, params);

  return cudaGetLastError();
}

/**
 * @brief Preprocess NV12 (YUV420sp) image.
 */
extern "C" cudaError_t preprocess_nv12(const uint8_t *d_y_plane,
                                       const uint8_t *d_uv_plane,
                                       float *d_output, int width, int height,
                                       int y_pitch, int uv_pitch,
                                       NormParams params, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  nv12_to_rgb_normalize_kernel<<<grid, block, 0, stream>>>(
      d_y_plane, d_uv_plane, d_output, width, height, y_pitch, uv_pitch,
      params);

  return cudaGetLastError();
}

#ifdef JETPACK_AVAILABLE
/**
 * @brief Preprocess from NvBufSurface (JetPack Multimedia API).
 *
 * Extracts device pointer from NvBufSurface and calls appropriate kernel.
 * Supports BGRA and NV12 formats.
 */
extern "C" cudaError_t preprocess_nvbufsurface(NvBufSurface *surface,
                                               float *d_output, int dst_width,
                                               int dst_height,
                                               NormParams params,
                                               cudaStream_t stream) {
  if (surface == nullptr || surface->numFilled < 1) {
    fprintf(stderr, "Invalid NvBufSurface\n");
    return cudaErrorInvalidValue;
  }

  NvBufSurfaceParams *surf_params = &surface->surfaceList[0];

  // Map to CUDA if needed
  if (NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_READ) != 0) {
    fprintf(stderr, "Failed to map NvBufSurface\n");
    return cudaErrorMapBufferObjectFailed;
  }

  cudaError_t result;

  switch (surf_params->colorFormat) {
  case NVBUF_COLOR_FORMAT_BGRA:
  case NVBUF_COLOR_FORMAT_BGRx: {
    const uint8_t *d_input = static_cast<const uint8_t *>(surf_params->dataPtr);
    result = preprocess_bgra_resize(d_input, d_output, surf_params->width,
                                    surf_params->height, surf_params->pitch,
                                    dst_width, dst_height, params, stream);
    break;
  }

  case NVBUF_COLOR_FORMAT_NV12:
  case NVBUF_COLOR_FORMAT_NV12_10LE: {
    const uint8_t *d_y = static_cast<const uint8_t *>(surf_params->dataPtr);
    const uint8_t *d_uv = d_y + surf_params->pitch * surf_params->height;

    // For NV12 with resize, we need a two-step process
    // Here we do direct conversion (assuming same size or handle resize
    // separately)
    result = preprocess_nv12(d_y, d_uv, d_output, surf_params->width,
                             surf_params->height, surf_params->pitch,
                             surf_params->pitch, params, stream);
    break;
  }

  default:
    fprintf(stderr, "Unsupported color format: %d\n", surf_params->colorFormat);
    result = cudaErrorNotSupported;
    break;
  }

  NvBufSurfaceUnMap(surface, 0, 0);
  return result;
}
#endif // JETPACK_AVAILABLE

// =============================================================================
// MEMORY MANAGEMENT
// =============================================================================

/**
 * @brief Allocate output buffer for preprocessed tensor.
 */
extern "C" float *allocate_preprocess_buffer(int width, int height) {
  float *d_buffer = nullptr;
  size_t size = 3 * width * height * sizeof(float);
  cudaError_t err = cudaMalloc(&d_buffer, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate preprocess buffer: %s\n",
            cudaGetErrorString(err));
    return nullptr;
  }
  return d_buffer;
}

/**
 * @brief Free preprocessing buffer.
 */
extern "C" void free_preprocess_buffer(float *d_buffer) {
  if (d_buffer != nullptr) {
    CUDA_CHECK_VOID(cudaFree(d_buffer));
  }
}

/**
 * @brief Create a CUDA stream for async preprocessing.
 */
extern "C" cudaStream_t create_preprocess_stream() {
  cudaStream_t stream;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to create CUDA stream: %s\n",
            cudaGetErrorString(err));
    return nullptr;
  }
  return stream;
}

/**
 * @brief Destroy CUDA stream.
 */
extern "C" void destroy_preprocess_stream(cudaStream_t stream) {
  if (stream != nullptr) {
    CUDA_CHECK_VOID(cudaStreamDestroy(stream));
  }
}
