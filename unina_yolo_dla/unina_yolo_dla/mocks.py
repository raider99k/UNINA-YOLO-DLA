"""
UNINA-YOLO-DLA: NVIDIA Library Mocks for CPU-Only Development.

This module provides mock implementations of TensorRT and pytorch-quantization
APIs to enable local testing on systems without NVIDIA hardware.

Usage:
    # Add this at the TOP of train.py or any script that imports NVIDIA libs:
    
    try:
        import tensorrt
    except ImportError:
        from unina_yolo_dla.mocks import install_mocks
        install_mocks()
        import tensorrt  # Now uses mock

All mock classes log their calls with [MOCK] prefix for transparency.
"""

from __future__ import annotations

import sys
import types
from typing import Any


# =============================================================================
# TensorRT Mock Module
# =============================================================================

def _create_tensorrt_mock() -> types.ModuleType:
    """Create a complete mock of the tensorrt module."""
    
    trt = types.ModuleType("tensorrt")
    
    # --- Logger ---
    class MockLogger:
        """Mock TensorRT Logger."""
        
        # Severity levels
        VERBOSE = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
        INTERNAL_ERROR = 4
        
        def __init__(self, severity: int = 2):
            print(f"[MOCK] TensorRT call: Logger.__init__(severity={severity})")
            self.severity = severity
        
        def log(self, severity: int, msg: str) -> None:
            print(f"[MOCK] TensorRT Logger: [{severity}] {msg}")
    
    trt.Logger = MockLogger
    
    # --- NetworkDefinitionCreationFlag ---
    class MockNetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 1
        EXPLICIT_PRECISION = 2
    
    trt.NetworkDefinitionCreationFlag = MockNetworkDefinitionCreationFlag
    
    # --- BuilderFlag ---
    class MockBuilderFlag:
        FP16 = 1
        INT8 = 2
        DEBUG = 4
        GPU_FALLBACK = 8
        STRICT_TYPES = 16
        REFIT = 32
    
    trt.BuilderFlag = MockBuilderFlag
    
    # --- DeviceType ---
    class MockDeviceType:
        GPU = 0
        DLA = 1
    
    trt.DeviceType = MockDeviceType
    
    # --- MemoryPoolType ---
    class MockMemoryPoolType:
        WORKSPACE = 0
        DLA_MANAGED_SRAM = 1
        DLA_LOCAL_DRAM = 2
        DLA_GLOBAL_DRAM = 3
    
    trt.MemoryPoolType = MockMemoryPoolType
    
    # --- LayerInformationFormat ---
    class MockLayerInformationFormat:
        ONELINE = 0
        JSON = 1
    
    trt.LayerInformationFormat = MockLayerInformationFormat
    
    # --- INetworkDefinition ---
    class MockNetworkDefinition:
        """Mock TensorRT Network Definition."""
        
        def __init__(self):
            print("[MOCK] TensorRT call: NetworkDefinition.__init__()")
            self.num_layers = 0
            self.num_inputs = 1
            self.num_outputs = 6
        
        def get_input(self, index: int) -> Any:
            return MockTensor(f"input_{index}")
        
        def get_output(self, index: int) -> Any:
            return MockTensor(f"output_{index}")
    
    trt.INetworkDefinition = MockNetworkDefinition
    
    # --- Mock Tensor ---
    class MockTensor:
        def __init__(self, name: str = "tensor"):
            self.name = name
            self.shape = (1, 3, 640, 640)
            self.dtype = "float32"
    
    # --- BuilderConfig ---
    class MockBuilderConfig:
        """Mock TensorRT Builder Config."""
        
        def __init__(self):
            print("[MOCK] TensorRT call: BuilderConfig.__init__()")
            self.default_device_type = MockDeviceType.GPU
            self.DLA_core = 0
            self.int8_calibrator = None
            self._flags = 0
        
        def set_memory_pool_limit(self, pool_type: int, size: int) -> None:
            print(f"[MOCK] TensorRT call: set_memory_pool_limit(pool={pool_type}, size={size})")
        
        def set_flag(self, flag: int) -> None:
            print(f"[MOCK] TensorRT call: set_flag({flag})")
            self._flags |= flag
        
        def clear_flag(self, flag: int) -> None:
            self._flags &= ~flag
    
    trt.IBuilderConfig = MockBuilderConfig
    
    # --- Builder ---
    class MockBuilder:
        """Mock TensorRT Builder."""
        
        def __init__(self, logger: MockLogger):
            print("[MOCK] TensorRT call: Builder.__init__()")
            self.logger = logger
            self.platform_has_fast_fp16 = True
            self.platform_has_fast_int8 = True
            self.max_DLA_batch_size = 16
        
        def create_network(self, flags: int = 0) -> MockNetworkDefinition:
            print(f"[MOCK] TensorRT call: create_network(flags={flags})")
            return MockNetworkDefinition()
        
        def create_builder_config(self) -> MockBuilderConfig:
            print("[MOCK] TensorRT call: create_builder_config()")
            return MockBuilderConfig()
        
        def build_serialized_network(
            self, network: MockNetworkDefinition, config: MockBuilderConfig
        ) -> bytes:
            print("[MOCK] TensorRT call: build_serialized_network()")
            # Return dummy serialized engine
            return b"MOCK_TENSORRT_ENGINE_V1"
    
    trt.Builder = MockBuilder
    
    # --- OnnxParser ---
    class MockOnnxParser:
        """Mock TensorRT ONNX Parser."""
        
        def __init__(self, network: MockNetworkDefinition, logger: MockLogger):
            print("[MOCK] TensorRT call: OnnxParser.__init__()")
            self.network = network
            self.logger = logger
            self.num_errors = 0
        
        def parse(self, data: bytes) -> bool:
            print(f"[MOCK] TensorRT call: OnnxParser.parse(data_len={len(data)})")
            return True
        
        def get_error(self, index: int) -> str:
            return f"Mock error {index}"
    
    trt.OnnxParser = MockOnnxParser
    
    # --- Runtime ---
    class MockRuntime:
        """Mock TensorRT Runtime."""
        
        def __init__(self, logger: MockLogger):
            print("[MOCK] TensorRT call: Runtime.__init__()")
            self.logger = logger
        
        def deserialize_cuda_engine(self, data: bytes) -> Any:
            print(f"[MOCK] TensorRT call: deserialize_cuda_engine(data_len={len(data)})")
            return MockCudaEngine()
    
    trt.Runtime = MockRuntime
    
    # --- CudaEngine ---
    class MockCudaEngine:
        """Mock TensorRT CUDA Engine."""
        
        def __init__(self):
            self.num_io_tensors = 7
            self.num_optimization_profiles = 1
            self.hardware_compatibility_level = None
        
        def create_engine_inspector(self) -> Any:
            return MockEngineInspector()
    
    # --- EngineInspector ---
    class MockEngineInspector:
        """Mock TensorRT Engine Inspector."""
        
        def get_layer_information(self, index: int, format: int) -> str:
            import json
            return json.dumps({
                "Name": f"layer_{index}",
                "Device": "DLA",
                "Precision": "INT8"
            })
    
    # --- IInt8Calibrator base classes ---
    class MockIInt8Calibrator:
        """Base mock calibrator."""
        pass
    
    class MockIInt8EntropyCalibrator2(MockIInt8Calibrator):
        """Mock IInt8EntropyCalibrator2 for INT8 calibration."""
        
        def __init__(self):
            print("[MOCK] TensorRT call: IInt8EntropyCalibrator2.__init__()")
        
        def get_batch_size(self) -> int:
            return 1
        
        def get_batch(self, names: list) -> list | None:
            return None
        
        def read_calibration_cache(self) -> bytes | None:
            return None
        
        def write_calibration_cache(self, cache: bytes) -> None:
            pass
    
    trt.IInt8Calibrator = MockIInt8Calibrator
    trt.IInt8EntropyCalibrator2 = MockIInt8EntropyCalibrator2
    trt.IInt8MinMaxCalibrator = MockIInt8Calibrator  # Alias
    
    return trt


# =============================================================================
# pytorch-quantization Mock Module
# =============================================================================

def _create_pytorch_quantization_mock() -> types.ModuleType:
    """Create a complete mock of the pytorch_quantization module."""
    import torch
    import torch.nn as nn
    
    # Main module
    pq = types.ModuleType("pytorch_quantization")
    
    # --- quant_modules submodule ---
    quant_modules = types.ModuleType("pytorch_quantization.quant_modules")
    
    def mock_initialize(**kwargs):
        print(f"[MOCK] QAT call: quant_modules.initialize({kwargs})")
    
    quant_modules.initialize = mock_initialize
    quant_modules.deactivate = lambda: print("[MOCK] QAT call: quant_modules.deactivate()")
    
    pq.quant_modules = quant_modules
    
    # --- nn submodule (quant_nn) ---
    quant_nn = types.ModuleType("pytorch_quantization.nn")
    
    # QuantConv2d - pass-through wrapper around Conv2d
    class MockQuantConv2d(nn.Conv2d):
        """Mock QuantConv2d that behaves exactly like Conv2d."""
        
        def __init__(self, *args, **kwargs):
            # Remove quantization-specific kwargs
            for key in ["quant_desc_input", "quant_desc_weight"]:
                kwargs.pop(key, None)
            super().__init__(*args, **kwargs)
            print(f"[MOCK] QAT call: QuantConv2d.__init__(in={args[0] if args else '?'}, out={args[1] if len(args) > 1 else '?'})")
            
            # Add mock quantizers
            self._input_quantizer = MockTensorQuantizer()
            self._weight_quantizer = MockTensorQuantizer()
    
    quant_nn.QuantConv2d = MockQuantConv2d
    
    # QuantLinear - pass-through wrapper around Linear
    class MockQuantLinear(nn.Linear):
        """Mock QuantLinear that behaves exactly like Linear."""
        
        def __init__(self, *args, **kwargs):
            for key in ["quant_desc_input", "quant_desc_weight"]:
                kwargs.pop(key, None)
            super().__init__(*args, **kwargs)
            self._input_quantizer = MockTensorQuantizer()
            self._weight_quantizer = MockTensorQuantizer()
    
    quant_nn.QuantLinear = MockQuantLinear
    
    # TensorQuantizer
    class MockTensorQuantizer(nn.Module):
        """Mock TensorQuantizer that does nothing."""
        
        use_fb_fake_quant = False
        default_calib_method = "histogram"
        
        def __init__(self, quant_desc=None, **kwargs):
            super().__init__()
            self._disabled = False
            self._calibrating = False
        
        def forward(self, x):
            return x  # Pass-through
        
        def disable(self):
            self._disabled = True
        
        def enable(self):
            self._disabled = False
        
        def enable_calib(self):
            self._calibrating = True
        
        def disable_calib(self):
            self._calibrating = False
    
    quant_nn.TensorQuantizer = MockTensorQuantizer
    
    pq.nn = quant_nn
    
    # --- tensor_quant submodule ---
    tensor_quant = types.ModuleType("pytorch_quantization.tensor_quant")
    
    class MockQuantDescriptor:
        """Mock QuantDescriptor for quantization configuration."""
        
        def __init__(self, num_bits: int = 8, calib_method: str = "histogram", **kwargs):
            self.num_bits = num_bits
            self.calib_method = calib_method
    
    tensor_quant.QuantDescriptor = MockQuantDescriptor
    pq.tensor_quant = tensor_quant
    
    # Also expose at top level
    pq.QuantDescriptor = MockQuantDescriptor
    
    # --- calib submodule ---
    calib = types.ModuleType("pytorch_quantization.calib")
    
    class MockHistogramCalibrator:
        """Mock histogram calibrator."""
        pass
    
    class MockMaxCalibrator:
        """Mock max calibrator."""
        pass
    
    calib.HistogramCalibrator = MockHistogramCalibrator
    calib.MaxCalibrator = MockMaxCalibrator
    calib.calibrator = types.SimpleNamespace(CALIBRATOR_TYPE="histogram")
    
    pq.calib = calib
    
    # --- nn.modules submodule ---
    nn_modules = types.ModuleType("pytorch_quantization.nn.modules")
    tensor_quantizer_module = types.ModuleType("pytorch_quantization.nn.modules.tensor_quantizer")
    tensor_quantizer_module.TensorQuantizer = MockTensorQuantizer
    nn_modules.tensor_quantizer = tensor_quantizer_module
    quant_nn.modules = nn_modules
    
    return pq


# =============================================================================
# Mock Installation
# =============================================================================

def install_mocks() -> None:
    """
    Install all NVIDIA library mocks into sys.modules.
    
    Call this BEFORE importing tensorrt or pytorch_quantization.
    """
    print("=" * 60)
    print("[MOCK] Installing NVIDIA library mocks for CPU-only development")
    print("=" * 60)
    
    # Install TensorRT mock
    if "tensorrt" not in sys.modules:
        trt_mock = _create_tensorrt_mock()
        sys.modules["tensorrt"] = trt_mock
        print("[MOCK] Installed: tensorrt")
    
    # Install pytorch_quantization mock
    if "pytorch_quantization" not in sys.modules:
        pq_mock = _create_pytorch_quantization_mock()
        sys.modules["pytorch_quantization"] = pq_mock
        sys.modules["pytorch_quantization.quant_modules"] = pq_mock.quant_modules
        sys.modules["pytorch_quantization.nn"] = pq_mock.nn
        sys.modules["pytorch_quantization.tensor_quant"] = pq_mock.tensor_quant
        sys.modules["pytorch_quantization.calib"] = pq_mock.calib
        sys.modules["pytorch_quantization.nn.modules"] = pq_mock.nn.modules
        sys.modules["pytorch_quantization.nn.modules.tensor_quantizer"] = pq_mock.nn.modules.tensor_quantizer
        print("[MOCK] Installed: pytorch_quantization")
    
    print("=" * 60)


def is_mock_installed() -> bool:
    """Check if mocks are currently installed."""
    if "tensorrt" in sys.modules:
        trt = sys.modules["tensorrt"]
        return hasattr(trt, "Logger") and "Mock" in str(type(trt.Logger))
    return False


# =============================================================================
# Injection Snippet (for train.py)
# =============================================================================

INJECTION_SNIPPET = '''
# --- MOCK INJECTION FOR LOCAL DEVELOPMENT ---
# Add this block at the TOP of train.py (before any tensorrt/pytorch_quantization imports)

import sys
_MOCK_NVIDIA = True  # Set to False to use real libraries

if _MOCK_NVIDIA:
    try:
        import tensorrt
    except ImportError:
        try:
            from unina_yolo_dla.mocks import install_mocks
            install_mocks()
        except ImportError:
            print("WARNING: mocks.py not found. NVIDIA imports may fail.")

# --- END MOCK INJECTION ---
'''


if __name__ == "__main__":
    print("UNINA-YOLO-DLA NVIDIA Library Mocks")
    print("=" * 50)
    print("\nTo use these mocks, add the following to your script:")
    print(INJECTION_SNIPPET)
    print("\nTesting mock installation...")
    install_mocks()
    
    # Test imports
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    from pytorch_quantization import quant_modules
    quant_modules.initialize()
    
    print("\n[MOCK] All mocks working correctly!")
