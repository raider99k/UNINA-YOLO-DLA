cmake_minimum_required(VERSION 3.8)
project(perception)

# Default to C++17 for modern features
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Find Dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- Custom Message Generation (Zero-Copy GPU Buffer Pointer) ---
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GpuBufferPtr.msg"
  DEPENDENCIES std_msgs
)

# Allow perception_node to use generated messages
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# CUDA (for preprocessing kernels)
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION}")
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
else()
  message(WARNING "CUDA not found. Building with MOCK_CUDA stubs for CI/testing.")
  # Define MOCK_CUDA to enable stub implementations in perception_node.cpp
  add_definitions(-DMOCK_CUDA)
endif()

# TensorRT (for inference)
# On Jetson, TensorRT is typically in /usr/lib/aarch64-linux-gnu/
# Adjust paths if needed.
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu/ /usr/lib/x86_64-linux-gnu/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu/ /usr/include/x86_64-linux-gnu/)

if(TENSORRT_LIBRARY AND TENSORRT_INCLUDE_DIR)
  message(STATUS "TensorRT found: ${TENSORRT_LIBRARY}")
  include_directories(${TENSORRT_INCLUDE_DIR})
else()
  message(WARNING "TensorRT not found. perception_node will not link against TensorRT.")
endif()

# --- Build the Perception Node ---
add_executable(perception_node
  src/perception_node.cpp
)

target_include_directories(perception_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(perception_node
  rclcpp
  rclcpp_lifecycle
  lifecycle_msgs
  sensor_msgs
  vision_msgs
  std_msgs
)

# --- CUDA Preprocessing and Post-Processing Library ---
if(CUDA_FOUND)
  # Build CUDA kernels as a static library
  add_library(cuda_kernels STATIC
    src/cuda_preprocess.cu
    src/gpu_postprocess.cu
  )
  
  # Set CUDA architecture for Jetson Orin (SM 8.7)
  set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "87"  # Jetson Orin AGX/NX
  )
  
  target_include_directories(cuda_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
  )
  
  # Link CUDA kernels to perception node
  target_link_libraries(perception_node cuda_kernels ${CUDA_LIBRARIES})
  
  message(STATUS "CUDA kernels enabled (preprocess + postprocess, SM 8.7)")

else()
  message(WARNING "CUDA not found. Preprocessing will be CPU-only (slow).")
endif()

# Link TensorRT if found
if(TENSORRT_LIBRARY)
  target_link_libraries(perception_node ${TENSORRT_LIBRARY})
endif()

# Link generated message typesupport
target_link_libraries(perception_node ${cpp_typesupport_target})


# --- Install ---
install(TARGETS
  perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL  # Optional: won't fail if directories don't exist yet
)

ament_package()
