/**
 * UNINA-YOLO-DLA: ROS2 Lifecycle Perception Node (Production - Zero-Copy)
 *
 * Real-time object detection for Formula Student Driverless.
 * Uses TensorRT on NVIDIA DLA Core 1 for deterministic latency.
 *
 * CRITICAL DESIGN:
 *   - ALL post-processing runs on GPU (no 5MB D2H transfer)
 *   - Only final detections (~1KB) are copied to host
 *   - Zero-copy input via GpuBufferHandle (ZED SDK / NvBufSurface)
 *   - sensor_msgs::Image path is DEPRECATED (CPU copy fallback only)
 *
 * Build Requirements:
 *   - JetPack 5.x / 6.x
 *   - ROS 2 Humble/Jazzy
 *   - TensorRT 8.x+
 *   - ZED SDK (for zero-copy camera input)
 */

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// ROS 2 Lifecycle
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

// Custom zero-copy message
#include <perception/msg/gpu_buffer_ptr.hpp>

// =============================================================================
// MOCK_CUDA: Compile without CUDA/TensorRT for ROS 2 infrastructure testing
// =============================================================================
#ifdef MOCK_CUDA

#include <cstdint>

// Fake CUDA types
typedef int cudaError_t;
typedef void *cudaStream_t;
#define cudaSuccess 0

inline const char *cudaGetErrorString(cudaError_t) { return "MOCK_CUDA"; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  return cudaSuccess;
}
inline cudaError_t cudaFree(void *ptr) {
  free(ptr);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void *, const void *, size_t, int,
                                   cudaStream_t) {
  return cudaSuccess;
}
#define cudaMemcpyHostToDevice 0

// Fake TensorRT namespace
namespace nvinfer1 {
enum class Severity { kINTERNAL_ERROR, kERROR, kWARNING, kINFO, kVERBOSE };
enum class TensorIOMode { kINPUT, kOUTPUT };
struct Dims {
  int nbDims;
  int d[8];
};

class ILogger {
public:
  virtual void log(Severity, const char *) noexcept = 0;
  virtual ~ILogger() = default;
};
class IRuntime {
public:
  void setDLACore(int) {}
  void destroy() {}
};
class ICudaEngine {
public:
  int getNbIOTensors() { return 0; }
  const char *getIOTensorName(int) { return ""; }
  TensorIOMode getTensorIOMode(const char *) { return TensorIOMode::kOUTPUT; }
  int getBindingIndex(const char *) { return -1; }
  Dims getTensorShape(const char *) { return Dims{4, {1, 3, 640, 640}}; }
  void destroy() {}
};
class IExecutionContext {
public:
  void setTensorAddress(const char *, void *) {}
  bool enqueueV3(cudaStream_t) { return true; }
  void destroy() {}
};
inline IRuntime *createInferRuntime(ILogger &) { return nullptr; }
} // namespace nvinfer1

// Fake preprocess functions
struct NormParams {
  float mean_r, mean_g, mean_b, std_r, std_g, std_b;
};
inline NormParams create_norm_params(float mr, float mg, float mb, float sr,
                                     float sg, float sb) {
  return NormParams{mr, mg, mb, sr, sg, sb};
}
inline cudaStream_t create_preprocess_stream() { return nullptr; }
inline void destroy_preprocess_stream(cudaStream_t) {}
inline float *allocate_preprocess_buffer(int, int) { return nullptr; }
inline void free_preprocess_buffer(float *) {}
inline cudaError_t preprocess_bgra_resize(const uint8_t *, float *, int, int,
                                          int, int, int, NormParams,
                                          cudaStream_t) {
  return cudaSuccess;
}

// Fake postprocess functions
struct GpuDetection {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;
  int valid;
  int _pad;
};
#define MAX_DETECTIONS 1024
inline cudaError_t init_postprocess_resources() { return cudaSuccess; }
inline cudaError_t cleanup_postprocess_resources() { return cudaSuccess; }
inline cudaError_t reset_detection_counter(cudaStream_t) { return cudaSuccess; }
inline cudaError_t get_detection_count(int *count, cudaStream_t) {
  *count = 0;
  return cudaSuccess;
}
inline cudaError_t decode_yolo_head(const float *, const float *,
                                    GpuDetection *, int, int, int, int, float,
                                    float, cudaStream_t) {
  return cudaSuccess;
}
inline cudaError_t run_gpu_nms(GpuDetection *, int, float, cudaStream_t) {
  return cudaSuccess;
}
inline cudaError_t copy_valid_detections_to_host(const GpuDetection *,
                                                 GpuDetection *, int,
                                                 int *count, cudaStream_t) {
  *count = 0;
  return cudaSuccess;
}

#else // !MOCK_CUDA

// CUDA Runtime
#include <cuda_runtime.h>

// TensorRT
#include <NvInfer.h>
#include <NvInferRuntime.h>

// JetPack Multimedia API (for NvBufSurface)
#ifdef JETPACK_AVAILABLE
#include <nvbufsurface.h>
#endif

// ZED SDK (optional)
#ifdef ZED_SDK_AVAILABLE
#include <sl/Camera.hpp>
#endif

// Local CUDA kernels
#include "cuda_preprocess.h"
#include "gpu_postprocess.h" // GPU-native decode + NMS

#endif // MOCK_CUDA

// =============================================================================
// CUDA Error Checking
// =============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      RCLCPP_FATAL(this->get_logger(), "CUDA Error at %s:%d - %s", __FILE__,   \
                   __LINE__, cudaGetErrorString(err));                         \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// =============================================================================
// TensorRT Logger
// =============================================================================

class TRTLogger : public nvinfer1::ILogger {
public:
  explicit TRTLogger(rclcpp::Logger ros_logger) : ros_logger_(ros_logger) {}

  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      RCLCPP_ERROR(ros_logger_, "[TensorRT] %s", msg);
      break;
    case Severity::kWARNING:
      RCLCPP_WARN(ros_logger_, "[TensorRT] %s", msg);
      break;
    case Severity::kINFO:
      RCLCPP_INFO(ros_logger_, "[TensorRT] %s", msg);
      break;
    default:
      RCLCPP_DEBUG(ros_logger_, "[TensorRT] %s", msg);
      break;
    }
  }

private:
  rclcpp::Logger ros_logger_;
};

// =============================================================================
// TensorRT Engine Wrapper
// =============================================================================

class TensorRTEngine {
public:
  TensorRTEngine() = default;
  ~TensorRTEngine() { unload(); }

  bool load(const std::string &engine_path, TRTLogger &logger,
            int dla_core = -1) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
      return false;

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size))
      return false;
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_)
      return false;

    if (dla_core >= 0) {
      runtime_->setDLACore(dla_core);
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_)
      return false;

    context_.reset(engine_->createExecutionContext());
    if (!context_)
      return false;

    loaded_ = true;
    return true;
  }

  void unload() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    loaded_ = false;
  }

  void setInputTensorAddress(const char *name, void *ptr) {
    if (context_)
      context_->setTensorAddress(name, ptr);
  }

  void setOutputTensorAddress(const char *name, void *ptr) {
    if (context_)
      context_->setTensorAddress(name, ptr);
  }

  bool enqueueV3(cudaStream_t stream) {
    if (!context_)
      return false;
    return context_->enqueueV3(stream);
  }

  bool isLoaded() const { return loaded_; }
  nvinfer1::ICudaEngine *getEngine() { return engine_.get(); }

  /**
   * @brief Gets the expected input dimensions from the compiled engine.
   *
   * CRITICAL: DLA engines are statically compiled. Dimension mismatch = memory
   * corruption.
   *
   * @param[out] width Expected input width
   * @param[out] height Expected input height
   * @return true if dimensions were retrieved successfully
   */
  bool getInputDimensions(int &width, int &height) const {
    if (!engine_)
      return false;

    // Find the "images" input tensor
    int tensor_idx = engine_->getBindingIndex("images");
    if (tensor_idx < 0) {
      // Try first input if "images" not found
      for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        if (engine_->getTensorIOMode(engine_->getIOTensorName(i)) ==
            nvinfer1::TensorIOMode::kINPUT) {
          tensor_idx = i;
          break;
        }
      }
    }

    if (tensor_idx < 0)
      return false;

    auto dims = engine_->getTensorShape(engine_->getIOTensorName(tensor_idx));
    if (dims.nbDims < 4)
      return false;

    // Assuming NCHW format: dims = [N, C, H, W]
    height = dims.d[2];
    width = dims.d[3];
    return true;
  }

private:
  struct RuntimeDeleter {
    void operator()(nvinfer1::IRuntime *p) {
      if (p)
        p->destroy();
    }
  };
  struct EngineDeleter {
    void operator()(nvinfer1::ICudaEngine *p) {
      if (p)
        p->destroy();
    }
  };
  struct ContextDeleter {
    void operator()(nvinfer1::IExecutionContext *p) {
      if (p)
        p->destroy();
    }
  };

  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
  bool loaded_ = false;
};

// =============================================================================
// GPU Buffer Handle (Zero-Copy Transport)
// =============================================================================

struct GpuBufferHandle {
  void *device_ptr = nullptr;
  int width = 0;
  int height = 0;
  int pitch = 0;
  int format = 0; // 0=BGRA, 1=NV12, 2=RGB
  uint64_t timestamp_ns = 0;

  bool isValid() const {
    return device_ptr != nullptr && width > 0 && height > 0;
  }
};

// =============================================================================
// Lifecycle Perception Node (Zero-Copy Compliant)
// =============================================================================

class PerceptionNodeLifecycle : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit PerceptionNodeLifecycle(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : rclcpp_lifecycle::LifecycleNode("perception_node", options),
        trt_logger_(this->get_logger()),
        engine_(std::make_unique<TensorRTEngine>()) {

    // --- Parameters ---
    this->declare_parameter<std::string>("engine_path",
                                         "unina_yolo_dla.engine");
    this->declare_parameter<std::string>("detections_topic",
                                         "/perception/detections");
    this->declare_parameter<float>("confidence_threshold", 0.5f);
    this->declare_parameter<float>("iou_threshold", 0.45f);
    this->declare_parameter<float>("conformal_quantile", 0.1f);
    this->declare_parameter<int>("input_width", 640);
    this->declare_parameter<int>("input_height", 640);
    this->declare_parameter<std::vector<double>>("norm_mean",
                                                 {0.485, 0.456, 0.406});
    this->declare_parameter<std::vector<double>>("norm_std",
                                                 {0.229, 0.224, 0.225});
    this->declare_parameter<int>("dla_core", 1);
    this->declare_parameter<std::string>("gpu_buffer_topic",
                                         "/camera/gpu_buffer");

    RCLCPP_INFO(this->get_logger(), "PerceptionNodeLifecycle constructed.");
  }

  ~PerceptionNodeLifecycle() { cleanup_resources(); }

  // =========================================================================
  // Lifecycle Callbacks
  // =========================================================================

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(this->get_logger(), "Configuring...");

    std::string engine_path = this->get_parameter("engine_path").as_string();
    std::string detections_topic =
        this->get_parameter("detections_topic").as_string();
    confidence_threshold_ =
        this->get_parameter("confidence_threshold").as_double();
    iou_threshold_ = this->get_parameter("iou_threshold").as_double();
    conformal_q_ = this->get_parameter("conformal_quantile").as_double();
    input_width_ = this->get_parameter("input_width").as_int();
    input_height_ = this->get_parameter("input_height").as_int();

    auto norm_mean = this->get_parameter("norm_mean").as_double_array();
    auto norm_std = this->get_parameter("norm_std").as_double_array();
    norm_params_ = create_norm_params(
        static_cast<float>(norm_mean[0]), static_cast<float>(norm_mean[1]),
        static_cast<float>(norm_mean[2]), static_cast<float>(norm_std[0]),
        static_cast<float>(norm_std[1]), static_cast<float>(norm_std[2]));

    // --- Load TensorRT Engine ---
    int dla_core = this->get_parameter("dla_core").as_int();
    if (!engine_->load(engine_path, trt_logger_, dla_core)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load TensorRT engine.");
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
          CallbackReturn::FAILURE;
    }
    RCLCPP_INFO(this->get_logger(), "Loaded engine on DLA Core %d", dla_core);

    // --- CRITICAL: Validate engine dimensions match ROS parameters ---
    int engine_width = 0, engine_height = 0;
    if (engine_->getInputDimensions(engine_width, engine_height)) {
      if (engine_width != input_width_ || engine_height != input_height_) {
        RCLCPP_FATAL(this->get_logger(),
                     "FATAL: Engine dimension mismatch! "
                     "Engine expects %dx%d, but ROS params specify %dx%d. "
                     "This WILL cause memory corruption on DLA.",
                     engine_width, engine_height, input_width_, input_height_);
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
            CallbackReturn::FAILURE;
      }
      RCLCPP_INFO(this->get_logger(), "Engine dimensions validated: %dx%d",
                  engine_width, engine_height);
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "WARNING: Could not validate engine dimensions. "
                  "Ensure input_width/height match the compiled engine!");
    }

    // --- Calculate Head Dimensions ---
    p2_w_ = input_width_ / 4;
    p2_h_ = input_height_ / 4;
    p3_w_ = input_width_ / 8;
    p3_h_ = input_height_ / 8;
    p4_w_ = input_width_ / 16;
    p4_h_ = input_height_ / 16;

    RCLCPP_INFO(this->get_logger(),
                "Head dimensions: P2=%dx%d, P3=%dx%d, P4=%dx%d", p2_w_, p2_h_,
                p3_w_, p3_h_, p4_w_, p4_h_);

    // --- Allocate CUDA Resources ---
    cuda_stream_ = create_preprocess_stream();
    d_preprocess_output_ =
        allocate_preprocess_buffer(input_width_, input_height_);
    allocate_detection_buffers();

    // --- Initialize Post-Process Resources (Zero-Allocation) ---
    CUDA_CHECK(init_postprocess_resources());

    // --- GPU Detection Buffer (for GPU-native postprocess) ---
    CUDA_CHECK(
        cudaMalloc(&d_detections_, MAX_DETECTIONS * sizeof(GpuDetection)));
    h_detections_.resize(MAX_DETECTIONS);

    // --- Publisher ---
    detections_pub_ =
        this->create_publisher<vision_msgs::msg::Detection2DArray>(
            detections_topic, rclcpp::QoS(1).best_effort());

    // --- Zero-Copy Subscriber (PRIMARY PATH) ---
    std::string gpu_buffer_topic =
        this->get_parameter("gpu_buffer_topic").as_string();
    rclcpp::QoS qos(1);
    qos.best_effort().durability_volatile();

    gpu_buffer_sub_ = this->create_subscription<perception::msg::GpuBufferPtr>(
        gpu_buffer_topic, qos,
        std::bind(&PerceptionNodeLifecycle::gpuBufferCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Zero-copy subscriber listening on: %s",
                gpu_buffer_topic.c_str());

    RCLCPP_INFO(this->get_logger(), "Configured. Ready to activate.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(this->get_logger(), "Activating...");
    detections_pub_->on_activate();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(this->get_logger(), "Deactivating...");
    detections_pub_->on_deactivate();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(this->get_logger(), "Cleaning up...");
    cleanup_resources();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(this->get_logger(), "Shutting down...");
    cleanup_resources();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  // =========================================================================
  // Zero-Copy ROS 2 Callback (PRIMARY PATH)
  // =========================================================================

  /**
   * @brief Callback for GpuBufferPtr messages (TRUE ZERO-COPY).
   *
   * This is the production path. Receives GPU pointer from ZED SDK wrapper
   * or NvBufSurface publisher, then calls processGpuBuffer().
   */
  void gpuBufferCallback(const perception::msg::GpuBufferPtr::SharedPtr msg) {
    // Convert ROS message to internal handle
    GpuBufferHandle buffer;
    buffer.device_ptr = reinterpret_cast<void *>(msg->device_ptr);
    buffer.width = msg->width;
    buffer.height = msg->height;
    buffer.pitch = msg->pitch;
    buffer.format = msg->format;
    buffer.timestamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();

    if (!buffer.isValid()) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "Received invalid GpuBufferPtr (null device_ptr or zero dimensions)");
      return;
    }

    // Process via zero-copy pipeline
    processGpuBuffer(buffer, buffer.timestamp_ns);
  }

  // =========================================================================
  // Primary Inference Pipeline (Zero-Copy GPU Buffer)
  // =========================================================================

  /**
   * @brief Process a GPU buffer from ZED SDK or NvBufSurface.
   *
   * This is the PRODUCTION path. No CPU copies for input or postprocess.
   */
  void processGpuBuffer(const GpuBufferHandle &buffer, uint64_t timestamp_ns) {
    if (this->get_current_state().id() !=
        lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {

      return;
    }

    // MEMORY ALIGNMENT GUARD (HPC Phase 2)
    // DLA and efficient CUDA vector loads require aligned pitch.
    if (buffer.pitch % 256 != 0) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "FATAL: Invalid pitch alignment (%d)! Must be "
                            "256-byte aligned for DLA/CUDA efficiency.",
                            buffer.pitch);
      return;
    }

    auto start_time = std::chrono::steady_clock::now();

    // --- Step 1: Preprocess (GPU-to-GPU) ---
    cudaError_t err = preprocess_bgra_resize(
        static_cast<const uint8_t *>(buffer.device_ptr), d_preprocess_output_,
        buffer.width, buffer.height, buffer.pitch, input_width_, input_height_,
        norm_params_, cuda_stream_);

    if (err != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "GPU preprocess failed.");
      return;
    }

    // --- Step 2: Bind TensorRT I/O ---
    engine_->setInputTensorAddress("images", d_preprocess_output_);
    engine_->setOutputTensorAddress("p2_cls", d_output_p2_cls_);
    engine_->setOutputTensorAddress("p2_reg", d_output_p2_reg_);
    engine_->setOutputTensorAddress("p3_cls", d_output_p3_cls_);
    engine_->setOutputTensorAddress("p3_reg", d_output_p3_reg_);
    engine_->setOutputTensorAddress("p4_cls", d_output_p4_cls_);
    engine_->setOutputTensorAddress("p4_reg", d_output_p4_reg_);

    // --- Step 3: DLA Inference ---
    if (!engine_->enqueueV3(cuda_stream_)) {
      RCLCPP_ERROR(this->get_logger(), "TensorRT inference failed.");
      return;
    }

    // --- Step 4: GPU Post-Processing (ZERO D2H for feature maps) ---
    reset_detection_counter(cuda_stream_);

    // P2 (stride 4)
    decode_yolo_head(d_output_p2_cls_, d_output_p2_reg_, d_detections_, p2_w_,
                     p2_h_, 4, 4, confidence_threshold_, conformal_q_,
                     cuda_stream_);
    // P3 (stride 8)
    decode_yolo_head(d_output_p3_cls_, d_output_p3_reg_, d_detections_, p3_w_,
                     p3_h_, 8, 4, confidence_threshold_, conformal_q_,
                     cuda_stream_);
    // P4 (stride 16)
    decode_yolo_head(d_output_p4_cls_, d_output_p4_reg_, d_detections_, p4_w_,
                     p4_h_, 16, 4, confidence_threshold_, conformal_q_,
                     cuda_stream_);

    // Get detection count
    int num_detections = 0;
    get_detection_count(&num_detections, cuda_stream_);
    cudaStreamSynchronize(cuda_stream_);

    if (num_detections > 0) {
      num_detections = std::min(num_detections, (int)MAX_DETECTIONS);

      // GPU NMS
      run_gpu_nms(d_detections_, num_detections, iou_threshold_, cuda_stream_);

      // --- Step 5: Copy ONLY final detections (~1KB, not 5MB!) ---
      int valid_count = 0;
      copy_valid_detections_to_host(d_detections_, h_detections_.data(),
                                    num_detections, &valid_count, cuda_stream_);

      // --- Step 6: Publish ---
      auto det_msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
      det_msg->header.stamp = rclcpp::Time(timestamp_ns);

      for (int i = 0; i < valid_count; ++i) {
        const auto &det = h_detections_[i];
        vision_msgs::msg::Detection2D d;
        d.bbox.center.position.x = (det.x1 + det.x2) / 2.0;
        d.bbox.center.position.y = (det.y1 + det.y2) / 2.0;
        d.bbox.size_x = det.x2 - det.x1;
        d.bbox.size_y = det.y2 - det.y1;

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = std::to_string(det.class_id);
        hyp.hypothesis.score = det.confidence;
        d.results.push_back(hyp);

        det_msg->detections.push_back(d);
      }

      detections_pub_->publish(std::move(det_msg));

      RCLCPP_DEBUG(this->get_logger(), "Published %d detections", valid_count);
    }

    // Latency
    auto end_time = std::chrono::steady_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time - start_time)
                          .count();
    RCLCPP_DEBUG(this->get_logger(), "Latency: %.2f ms", latency_us / 1000.0);
  }

private:
  // =========================================================================
  // Resource Management
  // =========================================================================

  void allocate_detection_buffers() {
    size_t p2_size = 4 * p2_w_ * p2_h_ * sizeof(float);
    size_t p3_size = 4 * p3_w_ * p3_h_ * sizeof(float);
    size_t p4_size = 4 * p4_w_ * p4_h_ * sizeof(float);

    cudaMalloc(&d_output_p2_cls_, p2_size);
    cudaMalloc(&d_output_p2_reg_, p2_size);
    cudaMalloc(&d_output_p3_cls_, p3_size);
    cudaMalloc(&d_output_p3_reg_, p3_size);
    cudaMalloc(&d_output_p4_cls_, p4_size);
    cudaMalloc(&d_output_p4_reg_, p4_size);
  }

  void cleanup_resources() {
    engine_->unload();
    cleanup_postprocess_resources();

    if (cuda_stream_) {
      destroy_preprocess_stream(cuda_stream_);
      cuda_stream_ = nullptr;
    }

    free_preprocess_buffer(d_preprocess_output_);
    d_preprocess_output_ = nullptr;

    if (d_detections_) {
      cudaFree(d_detections_);
      d_detections_ = nullptr;
    }
    if (d_output_p2_cls_) {
      cudaFree(d_output_p2_cls_);
      d_output_p2_cls_ = nullptr;
    }
    if (d_output_p2_reg_) {
      cudaFree(d_output_p2_reg_);
      d_output_p2_reg_ = nullptr;
    }
    if (d_output_p3_cls_) {
      cudaFree(d_output_p3_cls_);
      d_output_p3_cls_ = nullptr;
    }
    if (d_output_p3_reg_) {
      cudaFree(d_output_p3_reg_);
      d_output_p3_reg_ = nullptr;
    }
    if (d_output_p4_cls_) {
      cudaFree(d_output_p4_cls_);
      d_output_p4_cls_ = nullptr;
    }
    if (d_output_p4_reg_) {
      cudaFree(d_output_p4_reg_);
      d_output_p4_reg_ = nullptr;
    }

    detections_pub_.reset();
  }

  // =========================================================================
  // Members
  // =========================================================================

  TRTLogger trt_logger_;
  std::unique_ptr<TensorRTEngine> engine_;

  cudaStream_t cuda_stream_ = nullptr;
  float *d_preprocess_output_ = nullptr;

  // DLA output buffers
  float *d_output_p2_cls_ = nullptr;
  float *d_output_p2_reg_ = nullptr;
  float *d_output_p3_cls_ = nullptr;
  float *d_output_p3_reg_ = nullptr;
  float *d_output_p4_cls_ = nullptr;
  float *d_output_p4_reg_ = nullptr;

  // GPU detection buffer (for GPU-native postprocess)
  GpuDetection *d_detections_ = nullptr;
  std::vector<GpuDetection> h_detections_;

  NormParams norm_params_;
  float confidence_threshold_ = 0.5f;
  float iou_threshold_ = 0.45f;
  float conformal_q_ = 0.1f;
  int input_width_ = 640;
  int input_height_ = 640;

  // Head dimensions
  int p2_w_, p2_h_, p3_w_, p3_h_, p4_w_, p4_h_;

  // Zero-copy subscriber
  rclcpp::Subscription<perception::msg::GpuBufferPtr>::SharedPtr
      gpu_buffer_sub_;

  std::shared_ptr<
      rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>>
      detections_pub_;
};

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PerceptionNodeLifecycle>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  RCLCPP_INFO(node->get_logger(), "UNINA-YOLO-DLA Zero-Copy Node ready. Use "
                                  "lifecycle commands to activate.");
  RCLCPP_INFO(node->get_logger(),
              "  ros2 lifecycle set /perception_node configure");
  RCLCPP_INFO(node->get_logger(),
              "  ros2 lifecycle set /perception_node activate");

  executor.spin();
  rclcpp::shutdown();
  return 0;
}
