/**
 * UNINA-YOLO-DLA: ROS2 Lifecycle Perception Node (Production)
 *
 * Real-time object detection for Formula Student Driverless.
 * Uses TensorRT on NVIDIA DLA Core 1 for deterministic latency.
 *
 * Key Features:
 *   - Lifecycle-managed node (rclcpp_lifecycle)
 *   - Zero-copy GPU buffer integration (NvBufSurface / ZED SDK)
 *   - TensorRT DLA execution with explicit core selection
 *   - CUDA preprocessing kernels for BGRA->RGB+Normalize
 *
 * Build Requirements:
 *   - JetPack 5.x / 6.x
 *   - ROS 2 Humble/Jazzy
 *   - TensorRT 8.x+
 *   - ZED SDK (optional, for ZED camera zero-copy)
 *
 * Target Hardware:
 *   - NVIDIA Jetson Orin AGX/NX
 *   - DLA Core 1 (Core 0 can be used for secondary tasks)
 */

#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ROS 2 Lifecycle
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

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

// Local CUDA preprocessing
#include "cuda_preprocess.h"
#include "postprocess.hpp"

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

  /**
   * @brief Loads a serialized TensorRT engine from file.
   *
   * The engine should have been built with DLA Core 1 targeting.
   */
  bool load(const std::string &engine_path, TRTLogger &logger,
            int dla_core = -1) {
    // Read engine file
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size)) {
      return false;
    }
    file.close();

    // Create runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
      return false;
    }

    // --- Explicit DLA Core Selection at Runtime ---
    if (dla_core >= 0) {
      runtime_->setDLACore(dla_core);
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
      return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      return false;
    }

    loaded_ = true;
    return true;
  }

  void unload() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    loaded_ = false;
  }

  /**
   * @brief Sets the address of an input tensor (Zero-Copy).
   */
  void setInputTensorAddress(const char *name, void *ptr) {
    if (context_) {
      context_->setTensorAddress(name, ptr);
    }
  }

  /**
   * @brief Sets the address of an output tensor.
   */
  void setOutputTensorAddress(const char *name, void *ptr) {
    if (context_) {
      context_->setTensorAddress(name, ptr);
    }
  }

  /**
   * @brief Enqueues inference on DLA (async).
   */
  bool enqueueV3(cudaStream_t stream) {
    if (!context_)
      return false;
    return context_->enqueueV3(stream);
  }

  bool isLoaded() const { return loaded_; }

  nvinfer1::ICudaEngine *getEngine() { return engine_.get(); }

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
// GPU Buffer Handle (for Zero-Copy Transport)
// =============================================================================

/**
 * @brief Message type for GPU buffer pointer transport.
 *
 * This can be used with ROS 2 Type Adaptation (REP-2007) to avoid
 * serializing image data through the middleware.
 *
 * In production, integrate with ZED SDK's sl::Mat::getPtr<sl::GPU>()
 * or NvBufSurface from DeepStream/VIC.
 */
struct GpuBufferHandle {
  void *device_ptr = nullptr; // CUdeviceptr or void* to GPU memory
  int width = 0;
  int height = 0;
  int pitch = 0;  // Row stride in bytes
  int format = 0; // 0=BGRA, 1=NV12, 2=RGB
  uint64_t timestamp_ns = 0;

  bool isValid() const {
    return device_ptr != nullptr && width > 0 && height > 0;
  }
};

// =============================================================================
// Lifecycle Perception Node
// =============================================================================

class PerceptionNodeLifecycle : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit PerceptionNodeLifecycle(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : rclcpp_lifecycle::LifecycleNode("perception_node", options),
        trt_logger_(this->get_logger()),
        engine_(std::make_unique<TensorRTEngine>()) {
    // --- Declare Parameters ---
    this->declare_parameter<std::string>("engine_path",
                                         "unina_yolo_dla.engine");
    this->declare_parameter<std::string>("image_topic", "/zed/image_raw");
    this->declare_parameter<std::string>("detections_topic",
                                         "/perception/detections");
    this->declare_parameter<float>("confidence_threshold", 0.5f);
    this->declare_parameter<float>("iou_threshold", 0.45f);
    this->declare_parameter<float>("conformal_quantile",
                                   0.1f); // Dilation factor for safety
    this->declare_parameter<int>("input_width", 640);
    this->declare_parameter<int>("input_height", 640);

    // Normalization parameters (tunable without recompile)
    this->declare_parameter<std::vector<double>>("norm_mean",
                                                 {0.485, 0.456, 0.406});
    this->declare_parameter<std::vector<double>>("norm_std",
                                                 {0.229, 0.224, 0.225});

    // DLA configuration
    this->declare_parameter<int>("dla_core", 1); // Default to DLA Core 1

    RCLCPP_INFO(this->get_logger(),
                "PerceptionNodeLifecycle constructed (Unconfigured).");
  }

  ~PerceptionNodeLifecycle() { cleanup_resources(); }

  // =========================================================================
  // Lifecycle Callbacks
  // =========================================================================

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State & /*state*/) override {
    RCLCPP_INFO(this->get_logger(), "Configuring...");

    // Get parameters
    std::string engine_path = this->get_parameter("engine_path").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string detections_topic =
        this->get_parameter("detections_topic").as_string();
    confidence_threshold_ =
        this->get_parameter("confidence_threshold").as_double();
    input_width_ = this->get_parameter("input_width").as_int();
    input_height_ = this->get_parameter("input_height").as_int();

    // Get normalization parameters
    auto norm_mean = this->get_parameter("norm_mean").as_double_array();
    auto norm_std = this->get_parameter("norm_std").as_double_array();
    norm_params_ = create_norm_params(
        static_cast<float>(norm_mean[0]), static_cast<float>(norm_mean[1]),
        static_cast<float>(norm_mean[2]), static_cast<float>(norm_std[0]),
        static_cast<float>(norm_std[1]), static_cast<float>(norm_std[2]));

    RCLCPP_INFO(
        this->get_logger(),
        "Normalization: mean=[%.3f, %.3f, %.3f], std=[%.3f, %.3f, %.3f]",
        norm_params_.mean_r, norm_params_.mean_g, norm_params_.mean_b,
        norm_params_.std_r, norm_params_.std_g, norm_params_.std_b);

    // --- Load TensorRT Engine ---
    int dla_core = this->get_parameter("dla_core").as_int();
    if (!engine_->load(engine_path, trt_logger_, dla_core)) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to load TensorRT engine: %s (DLA Core %d)",
                   engine_path.c_str(), dla_core);
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
          CallbackReturn::FAILURE;
    }
    RCLCPP_INFO(this->get_logger(), "TensorRT engine loaded: %s on DLA Core %d",
                engine_path.c_str(), dla_core);

    // --- Create CUDA Resources ---
    cuda_stream_ = create_preprocess_stream();
    if (!cuda_stream_) {
      RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA stream.");
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
          CallbackReturn::FAILURE;
    }

    // Allocate preprocessing output buffer
    d_preprocess_output_ =
        allocate_preprocess_buffer(input_width_, input_height_);
    if (!d_preprocess_output_) {
      RCLCPP_ERROR(this->get_logger(), "Failed to allocate preprocess buffer.");
      return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
          CallbackReturn::FAILURE;
    }

    // Allocate detection output buffers
    // P2: 160x160, P3: 80x80, P4: 40x40 (for 640x640 input)
    allocate_detection_buffers();

    // --- Create Publisher ---
    detections_pub_ =
        this->create_publisher<vision_msgs::msg::Detection2DArray>(
            detections_topic, rclcpp::QoS(1).best_effort());

    // --- Create Subscriber ---
    // Using sensor_msgs::Image for now; for true zero-copy, use custom
    // transport
    rclcpp::QoS qos(1);
    qos.best_effort().durability_volatile();

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, qos,
        std::bind(&PerceptionNodeLifecycle::imageCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Configured. Ready to activate.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State & /*state*/) override {
    RCLCPP_INFO(this->get_logger(),
                "Activating... Starting inference on DLA Core %d.",
                this->get_parameter("dla_core").as_int());
    detections_pub_->on_activate();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State & /*state*/) override {
    RCLCPP_INFO(this->get_logger(), "Deactivating...");
    detections_pub_->on_deactivate();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State & /*state*/) override {
    RCLCPP_INFO(this->get_logger(), "Cleaning up...");
    cleanup_resources();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State & /*state*/) override {
    RCLCPP_INFO(this->get_logger(), "Shutting down...");
    cleanup_resources();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

private:
  // =========================================================================
  // Image Callback (Main Inference Pipeline)
  // =========================================================================

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // Only process if active
    if (this->get_current_state().id() !=
        lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
      return;
    }

    auto start_time = std::chrono::steady_clock::now();

    // --- Step 1: Get Input Data ---
    // For true zero-copy with ZED SDK, replace this with:
    //   sl::Mat zed_image;
    //   zed.retrieveImage(zed_image, sl::VIEW::LEFT, sl::MEM::GPU);
    //   void* d_input = zed_image.getPtr<sl::GPU>();
    //
    // For NvBufSurface from DeepStream/VIC:
    //   NvBufSurface* surface = ...;
    //   preprocess_nvbufsurface(surface, d_preprocess_output_, ...);

    // Current: Copy from CPU (sensor_msgs::Image) to GPU
    // This is NOT zero-copy! Replace with proper GPU pointer transport.
    uint8_t *d_input = nullptr;
    bool needs_copy = true;

    if (needs_copy) {
      // Allocate temp buffer if needed
      size_t input_size = msg->step * msg->height;
      CUDA_CHECK(cudaMalloc(&d_input, input_size));
      CUDA_CHECK(cudaMemcpyAsync(d_input, msg->data.data(), input_size,
                                 cudaMemcpyHostToDevice, cuda_stream_));
    }

    // --- Step 2: Preprocess (GPU) ---
    int src_pitch = msg->step;
    cudaError_t preprocess_err = preprocess_bgra_resize(
        d_input, d_preprocess_output_, msg->width, msg->height, src_pitch,
        input_width_, input_height_, norm_params_, cuda_stream_);

    if (preprocess_err != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "Preprocess failed: %s",
                   cudaGetErrorString(preprocess_err));
      if (needs_copy)
        cudaFree(d_input);
      return;
    }

    // --- Step 3: Set TensorRT Input/Output Addresses ---
    engine_->setInputTensorAddress("images", d_preprocess_output_);
    engine_->setOutputTensorAddress("p2_cls", d_output_p2_cls_);
    engine_->setOutputTensorAddress("p2_reg", d_output_p2_reg_);
    engine_->setOutputTensorAddress("p3_cls", d_output_p3_cls_);
    engine_->setOutputTensorAddress("p3_reg", d_output_p3_reg_);
    engine_->setOutputTensorAddress("p4_cls", d_output_p4_cls_);
    engine_->setOutputTensorAddress("p4_reg", d_output_p4_reg_);

    // --- Step 4: Enqueue Inference (DLA) ---
    if (!engine_->enqueueV3(cuda_stream_)) {
      RCLCPP_ERROR(this->get_logger(), "TensorRT inference failed.");
      if (needs_copy)
        cudaFree(d_input);
      return;
    }

    // --- Step 5: Synchronize ---
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    // --- Step 6: Post-process (Decode + NMS) ---
    std::vector<Detection> detections = postprocess();

    // --- Step 7: Publish Detections ---
    auto det_msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
    det_msg->header = msg->header;

    for (const auto &det : detections) {
      if (det.confidence < confidence_threshold_)
        continue;

      vision_msgs::msg::Detection2D d;
      d.bbox.center.position.x = det.x;
      d.bbox.center.position.y = det.y;
      d.bbox.size_x = det.w;
      d.bbox.size_y = det.h;

      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(det.class_id);
      hyp.hypothesis.score = det.confidence;
      d.results.push_back(hyp);

      det_msg->detections.push_back(d);
    }

    detections_pub_->publish(std::move(det_msg));

    // Cleanup temp buffer
    if (needs_copy) {
      cudaFree(d_input);
    }

    // --- Latency Logging ---
    auto end_time = std::chrono::steady_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time - start_time)
                          .count();

    RCLCPP_DEBUG(this->get_logger(),
                 "Inference latency: %.2f ms, detections: %zu",
                 latency_us / 1000.0, det_msg->detections.size());
  }

  // =========================================================================
  // Zero-Copy GPU Buffer Callback (for ZED SDK / NvBufSurface)
  // =========================================================================

  /**
   * @brief Alternative callback for true zero-copy GPU buffer input.
   *
   * Use this when integrating with ZED SDK or custom GPU buffer transport.
   */
  void gpuBufferCallback(const GpuBufferHandle &buffer) {
    if (!buffer.isValid()) {
      RCLCPP_WARN(this->get_logger(), "Invalid GPU buffer received.");
      return;
    }

    // Preprocess directly from GPU buffer (TRUE ZERO-COPY)
    cudaError_t err = preprocess_bgra_resize(
        static_cast<const uint8_t *>(buffer.device_ptr), d_preprocess_output_,
        buffer.width, buffer.height, buffer.pitch, input_width_, input_height_,
        norm_params_, cuda_stream_);

    if (err != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "Zero-copy preprocess failed.");
      return;
    }

    // Continue with inference...
    // (Same as imageCallback from Step 3 onwards)
  }

  // =========================================================================
  // Post-processing
  // =========================================================================

  struct Detection {
    float x, y, w, h;
    float confidence;
    int class_id;
  };

  std::vector<Detection> postprocess() {
    std::vector<Detection> all_detections;
    int num_classes = 4; // Yellow, Blue, Orange, Large Orange
    float iou_threshold = this->get_parameter("iou_threshold").as_double();
    size_t b_size = 1; // Batch size fixed at 1

    // Host pointers for copying results
    // In a full GPU pipeline, we would avoid these transfers
    std::vector<float> h_p2_cls(4 * 160 * 160);
    std::vector<float> h_p2_reg(4 * 160 * 160);
    std::vector<float> h_p3_cls(4 * 80 * 80);
    std::vector<float> h_p3_reg(4 * 80 * 80);
    std::vector<float> h_p4_cls(4 * 40 * 40);
    std::vector<float> h_p4_reg(4 * 40 * 40);

    // DLA -> GPU -> CPU (or DLA -> CPU directly if pinned memory)
    // Using async copy to maximize parallelism
    cudaMemcpyAsync(h_p2_cls.data(), d_output_p2_cls_,
                    h_p2_cls.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    cuda_stream_);
    cudaMemcpyAsync(h_p2_reg.data(), d_output_p2_reg_,
                    h_p2_reg.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    cuda_stream_);
    cudaMemcpyAsync(h_p3_cls.data(), d_output_p3_cls_,
                    h_p3_cls.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    cuda_stream_);
    cudaMemcpyAsync(h_p3_reg.data(), d_output_p3_reg_,
                    h_p3_reg.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    cuda_stream_);
    cudaMemcpyAsync(h_p4_cls.data(), d_output_p4_cls_,
                    h_p4_cls.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    cuda_stream_);
    cudaMemcpyAsync(h_p4_reg.data(), d_output_p4_reg_,
                    h_p4_reg.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    cuda_stream_);

    cudaStreamSynchronize(cuda_stream_);

    float conformal_q = this->get_parameter("conformal_quantile").as_double();

    // --- Decode Outputs ---
    // P2 (Stride 4)
    decode_head(h_p2_cls.data(), h_p2_reg.data(), 160, 160, 4, num_classes,
                confidence_threshold_, conformal_q, all_detections);
    // P3 (Stride 8)
    decode_head(h_p3_cls.data(), h_p3_reg.data(), 80, 80, 8, num_classes,
                confidence_threshold_, conformal_q, all_detections);
    // P4 (Stride 16)
    decode_head(h_p4_cls.data(), h_p4_reg.data(), 40, 40, 16, num_classes,
                confidence_threshold_, conformal_q, all_detections);

    // --- NMS ---
    return nms(all_detections, iou_threshold);
  }

  // =========================================================================
  // Resource Management
  // =========================================================================

  void allocate_detection_buffers() {
    // P2: 160x160, 4 classes + 4 box coords
    size_t p2_cls_size = 4 * 160 * 160 * sizeof(float);
    size_t p2_reg_size = 4 * 160 * 160 * sizeof(float);

    // P3: 80x80
    size_t p3_cls_size = 4 * 80 * 80 * sizeof(float);
    size_t p3_reg_size = 4 * 80 * 80 * sizeof(float);

    // P4: 40x40
    size_t p4_cls_size = 4 * 40 * 40 * sizeof(float);
    size_t p4_reg_size = 4 * 40 * 40 * sizeof(float);

    cudaMalloc(&d_output_p2_cls_, p2_cls_size);
    cudaMalloc(&d_output_p2_reg_, p2_reg_size);
    cudaMalloc(&d_output_p3_cls_, p3_cls_size);
    cudaMalloc(&d_output_p3_reg_, p3_reg_size);
    cudaMalloc(&d_output_p4_cls_, p4_cls_size);
    cudaMalloc(&d_output_p4_reg_, p4_reg_size);
  }

  void cleanup_resources() {
    engine_->unload();

    if (cuda_stream_) {
      destroy_preprocess_stream(cuda_stream_);
      cuda_stream_ = nullptr;
    }

    free_preprocess_buffer(d_preprocess_output_);
    d_preprocess_output_ = nullptr;

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

    image_sub_.reset();
    detections_pub_.reset();
  }

  // =========================================================================
  // Members
  // =========================================================================

  TRTLogger trt_logger_;
  std::unique_ptr<TensorRTEngine> engine_;

  cudaStream_t cuda_stream_ = nullptr;
  float *d_preprocess_output_ = nullptr;

  // Detection output buffers
  float *d_output_p2_cls_ = nullptr;
  float *d_output_p2_reg_ = nullptr;
  float *d_output_p3_cls_ = nullptr;
  float *d_output_p3_reg_ = nullptr;
  float *d_output_p4_cls_ = nullptr;
  float *d_output_p4_reg_ = nullptr;

  NormParams norm_params_;
  float confidence_threshold_ = 0.5f;
  int input_width_ = 640;
  int input_height_ = 640;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
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

  // Use SingleThreadedExecutor for deterministic timing
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  RCLCPP_INFO(node->get_logger(), "Lifecycle node started. Use 'ros2 lifecycle "
                                  "set' to configure/activate.");
  RCLCPP_INFO(node->get_logger(),
              "  ros2 lifecycle set /perception_node configure");
  RCLCPP_INFO(node->get_logger(),
              "  ros2 lifecycle set /perception_node activate");

  executor.spin();
  rclcpp::shutdown();
  return 0;
}
