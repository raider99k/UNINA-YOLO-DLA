# GpuBufferPtr.msg
# Custom message for zero-copy GPU buffer pointer transport.
#
# CRITICAL: This message transports a raw GPU pointer as uint64.
# The receiving node MUST be in the same process or use shared memory
# with proper synchronization. Do NOT use across network!
#
# Usage:
#   - ZED SDK: Publish sl::Mat::getPtr<sl::GPU>() as device_ptr
#   - NvBufSurface: Publish NvBufSurface->surfaceList[0].dataPtr
#   - DeepStream: Publish NvDsFrameMeta->batch_meta->frame_meta_list

std_msgs/Header header

# GPU device pointer (void* cast to uint64)
uint64 device_ptr

# Image dimensions
uint32 width
uint32 height
uint32 pitch  # Row stride in bytes

# Pixel format
# 0 = BGRA (ZED default)
# 1 = NV12 (NVIDIA capture)
# 2 = RGB
# 3 = RGBA
uint8 format

# Frame metadata
uint64 frame_id
uint64 exposure_time_ns
