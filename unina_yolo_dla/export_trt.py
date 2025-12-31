"""
UNINA-YOLO-DLA: TensorRT Export Module (Enhanced).

This module provides the complete workflow for exporting the PyTorch model
to a TensorRT engine optimized for execution on the NVIDIA DLA.

Workflow:
    1. Load a trained UNINA_YOLO_DLA PyTorch model.
    2. Export to ONNX (Static shape, Opset 13).
    3. Build a TensorRT engine using the Python API.
       - INT8 precision with Entropy Calibration.
       - Explicit DLA Core mapping.
       - GPU Fallback Detection: Prints RED ERROR if any layer is on GPU.

Requirements:
    - tensorrt (pycuda is NOT required for DLA-only inference, but useful for testing).
    - A calibration dataset for INT8 quantization.

Target Hardware:
    - NVIDIA Jetson Orin AGX (DLA Core 0 or 1).
"""
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# TensorRT import will only succeed on systems with TensorRT installed (e.g., Jetson).
# We provide a try-except to allow the script to be imported on dev machines for linting.
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not found. Export functions will not work.")


# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


# --- ONNX Export ---

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: int = 640,
    opset_version: int = 13,
) -> Path:
    """
    Exports a PyTorch model (Standard or QAT) to ONNX format.

    Args:
        model: The PyTorch model instance.
        output_path: Path to save the .onnx file.
        input_size: The fixed input resolution (e.g., 640 for 640x640).
        opset_version: ONNX opset version (13+ recommended for TensorRT).

    Returns:
        The Path to the saved ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Check for QAT model and enable fake quantization export
    is_qat = False
    try:
        from pytorch_quantization import nn as quant_nn
        # Heuristic: Check if model has any QuantConv2d layers
        for m in model.modules():
            if isinstance(m, quant_nn.QuantConv2d):
                is_qat = True
                break
        
        if is_qat:
            print("[ONNX Export] Detected QAT model. Enabling use_fb_fake_quant.")
            quant_nn.TensorQuantizer.use_fb_fake_quant = True
    except ImportError:
        pass

    # Output names should match model's forward() return structure.
    # P2, P3, P4 each have cls and reg outputs.
    output_names = ['p2_cls', 'p2_reg', 'p3_cls', 'p3_reg', 'p4_cls', 'p4_reg']

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=None,  # CRITICAL: Static shapes for DLA.
    )
    print(f"[ONNX Export] Model exported to: {output_path}")
    return output_path


# --- Cone Calibration Stream ---

class ConeCalibrationStream:
    """
    Calibration data stream that reads images from a folder.
    
    CRITICAL: Preprocessing MUST match cuda_preprocess.cu exactly!
    - Resize to input_size x input_size
    - Normalize with ImageNet mean/std
    - Output range: approximately [-2.1, +2.6]
    
    Args:
        calib_folder: Path to folder containing calibration images.
        input_size: Target input size (images will be resized).
        batch_size: Number of images per batch.
        mean: ImageNet mean (RGB).
        std: ImageNet std (RGB).
    """
    
    # ImageNet normalization (MUST match cuda_preprocess.cu NormParams!)
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        calib_folder: str | Path,
        input_size: int = 640,
        batch_size: int = 1,
        mean: tuple[float, float, float] = DEFAULT_MEAN,
        std: tuple[float, float, float] = DEFAULT_STD,
    ) -> None:
        self.calib_folder = Path(calib_folder)
        self.input_size = input_size
        self.batch_size = batch_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.current_index = 0
        
        # Find all images in the folder
        self.image_paths = sorted(
            glob.glob(str(self.calib_folder / "*.jpg")) +
            glob.glob(str(self.calib_folder / "*.jpeg")) +
            glob.glob(str(self.calib_folder / "*.png"))
        )
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"FATAL: No calibration images found in {self.calib_folder}. "
                                    "Production-ready engines require real calibration data.")
        
        print(f"[ConeCalibrationStream] Found {len(self.image_paths)} images")
        print(f"[ConeCalibrationStream] Using mean={mean}, std={std}")
    
    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Loads an image and preprocesses it EXACTLY like cuda_preprocess.cu.
        
        CRITICAL: This must match the CUDA kernel normalization!
        Formula: (pixel/255.0 - mean) / std
        Output range: approximately [-2.1, +2.6] for ImageNet stats
        """
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.input_size, self.input_size))
            img_np = np.array(img, dtype=np.float32)
        except (ImportError, Exception) as e:
            raise RuntimeError(f"FATAL: Failed to load calibration image {image_path}: {e}")
        
        # HWC -> CHW
        img_np = img_np.transpose(2, 0, 1)
        
        # CRITICAL: Apply ImageNet normalization (matching CUDA kernel!)
        # Formula: (pixel/255.0 - mean) / std
        img_np = img_np / 255.0
        img_np = (img_np - self.mean) / self.std
        
        return img_np

    
    def get_batch(self) -> np.ndarray | None:
        """Returns the next batch of images, or None if exhausted."""
        if self.current_index >= len(self.image_paths):
            return None
        
        batch_end = min(self.current_index + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[self.current_index:batch_end]
        self.current_index = batch_end
        
        batch_images = [self._load_and_preprocess(p) for p in batch_paths]
        return np.stack(batch_images, axis=0)
    
    def reset(self) -> None:
        """Resets the stream to the beginning."""
        self.current_index = 0


# --- INT8 Calibrator ---

class EntropyCalibrator(trt.IInt8EntropyCalibrator2 if TRT_AVAILABLE else object):
    """
    INT8 Entropy Calibrator for TensorRT.
    
    USES EXCLUSIVELY trt.IInt8EntropyCalibrator2 (Crucial for small objects).
    DO NOT use MinMax calibrator.

    Args:
        calibration_stream: A ConeCalibrationStream or similar data source.
        cache_file: Path to cache the calibration data for faster subsequent builds.
    """
    def __init__(
        self,
        calibration_stream: ConeCalibrationStream,
        cache_file: str | Path = "calibration.cache",
    ) -> None:
        if TRT_AVAILABLE:
            super().__init__()
        
        self.stream = calibration_stream
        self.cache_file = Path(cache_file)
        
        # Allocate device memory placeholder
        # In a real implementation with pycuda:
        # self.d_input = cuda.mem_alloc(batch_size * 3 * input_size * input_size * 4)
        self._device_input_placeholder: np.ndarray | None = None

    def get_batch_size(self) -> int:
        return self.stream.batch_size

    def get_batch(self, names: list[str]) -> list | None:
        """
        Returns the next batch of calibration data.
        Returns None when all data has been processed.
        """
        batch = self.stream.get_batch()
        if batch is None:
            return None
        
        # Store for device pointer access
        self._device_input_placeholder = np.ascontiguousarray(batch)
        
        # In a real implementation, copy to GPU:
        #   cuda.memcpy_htod(self.d_input, batch)
        #   return [int(self.d_input)]
        
        return [self._device_input_placeholder.ctypes.data]

    def read_calibration_cache(self) -> bytes | None:
        """Reads cached calibration data if available."""
        if self.cache_file.exists():
            print(f"[Calibrator] Reading cache from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Writes calibration data to cache."""
        print(f"[Calibrator] Writing cache to: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# --- GPU Fallback Analyzer ---

def analyze_engine_layers(engine_path: str | Path) -> tuple[int, int, list[str]]:
    """
    Analyzes a TensorRT engine to detect GPU fallback layers.
    
    Returns:
        Tuple of (dla_layer_count, gpu_layer_count, gpu_layer_names)
    """
    if not TRT_AVAILABLE:
        return (0, 0, [])
    
    engine_path = Path(engine_path)
    if not engine_path.exists():
        print(f"ERROR: Engine file not found: {engine_path}")
        return (0, 0, [])
    
    # Load the engine
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("ERROR: Failed to load engine for analysis.")
        return (0, 0, [])
    
    dla_count = 0
    gpu_count = 0
    gpu_layers: list[str] = []
    
    # Analyze each layer
    inspector = engine.create_engine_inspector()
    
    # Get layer info (TensorRT 8.5+)
    try:
        for i in range(engine.num_io_tensors + engine.num_optimization_profiles):
            layer_info = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
            if layer_info:
                import json
                info = json.loads(layer_info)
                layer_name = info.get("Name", f"layer_{i}")
                device = info.get("Device", "GPU")
                precision = info.get("Precision", "FP32")
                
                if "DLA" in device.upper():
                    dla_count += 1
                else:
                    gpu_count += 1
                    gpu_layers.append(layer_name)
                
                # Check for P2 layers ending up in INT8 (CRITICAL FOR SMALL OBJECTS)
                if "p2" in layer_name.lower() and "int8" in precision.lower():
                    print(f"{Colors.YELLOW}WARNING: P2 layer '{layer_name}' is quantized to INT8!{Colors.RESET}")
                    print(f"{Colors.YELLOW}This may degrade small object detection accuracy. "
                          f"Consider keeping P2 layers in FP16.{Colors.RESET}")
                    
    except Exception as e:
        print(f"Note: Could not analyze individual layers ({e})")
        # Fallback: check engine-level DLA status
        if engine.hardware_compatibility_level:
            print("Engine has hardware compatibility enabled.")
    
    return (dla_count, gpu_count, gpu_layers)


def print_fallback_report(dla_count: int, gpu_count: int, gpu_layers: list[str]) -> bool:
    """
    Prints a colored report of DLA/GPU layer distribution.
    
    Returns:
        True if 100% DLA (no GPU fallback), False otherwise.
    """
    total = dla_count + gpu_count
    if total == 0:
        print("WARNING: No layers analyzed.")
        return False
    
    dla_pct = 100.0 * dla_count / total
    gpu_pct = 100.0 * gpu_count / total
    
    print("\n" + "=" * 60)
    print("TensorRT Engine Layer Analysis")
    print("=" * 60)
    print(f"  DLA Layers: {dla_count} ({dla_pct:.1f}%)")
    print(f"  GPU Layers: {gpu_count} ({gpu_pct:.1f}%)")
    
    if gpu_count == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SUCCESS: 100% DLA - Zero GPU Fallback!{Colors.RESET}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ ERROR: GPU FALLBACK DETECTED!{Colors.RESET}")
        print(f"{Colors.RED}The following {gpu_count} layers are running on GPU:{Colors.RESET}")
        for layer in gpu_layers[:10]:  # Show first 10
            print(f"  {Colors.YELLOW}- {layer}{Colors.RESET}")
        if len(gpu_layers) > 10:
            print(f"  ... and {len(gpu_layers) - 10} more")
        print(f"\n{Colors.RED}This engine is NOT production-ready for DLA deployment.{Colors.RESET}")
        print(f"{Colors.RED}Review the model architecture for unsupported operations.{Colors.RESET}")
        return False


# --- TensorRT Engine Builder ---

def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    calibrator: "EntropyCalibrator | None" = None,
    precision: str = "int8",  # "fp16" or "int8"
    dla_core: int = 0,
    allow_gpu_fallback: bool = True,  # Enable for build, check after
    strict_dla: bool = True,          # Enforce 0% GPU fallback
    workspace_size_gb: float = 2.0,
) -> Path | None:
    """
    Builds a TensorRT engine from an ONNX model, targeting the DLA.

    Args:
        onnx_path: Path to the input ONNX file.
        engine_path: Path to save the output .engine (or .trt) file.
        calibrator: The INT8 calibrator instance (required for int8 precision).
        precision: Target precision ("fp16" or "int8").
        dla_core: Which DLA core to use (0 or 1 on Orin).
        allow_gpu_fallback: If True, allows layers to fall back to GPU.
                            We enable this for the build, then CHECK for fallback.
        workspace_size_gb: Maximum GPU memory for TensorRT to use during build.

    Returns:
        Path to the saved engine file, or None if build failed.
    """
    if not TRT_AVAILABLE:
        print("ERROR: TensorRT is not available. Cannot build engine.")
        return None

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # --- Create Builder ---
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # --- Parse ONNX ---
    print(f"[TensorRT] Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model.")
            for i in range(parser.num_errors):
                print(f"  Parser Error {i}: {parser.get_error(i)}")
            return None

    # --- Configure Builder ---
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_size_gb * (1 << 30))
    )

    # --- Set Precision ---
    if precision == "fp16":
        print("[TensorRT] Enabling FP16 mode.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        print("[TensorRT] Enabling INT8 mode (with Entropy Calibration).")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            print("ERROR: INT8 precision requires a calibrator.")
            return None
        config.int8_calibrator = calibrator
    else:
        print(f"[TensorRT] Using default FP32 precision.")

    # --- Configure DLA ---
    print(f"[TensorRT] Targeting DLA Core: {dla_core}")
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = dla_core

    # Enable GPU fallback for build (we'll check for actual fallback after)
    if allow_gpu_fallback:
        print("[TensorRT] GPU Fallback ENABLED for build (will verify after).")
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    else:
        print("[TensorRT] GPU Fallback DISABLED. Build will FAIL if DLA incompatible.")

    # --- Build Engine ---
    print("[TensorRT] Building serialized engine... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Engine build failed. Check logs for DLA layer compatibility issues.")
        return None

    # --- Save Engine ---
    print(f"[TensorRT] Saving engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print("[TensorRT] Engine build complete!")
    
    # --- Analyze for GPU Fallback ---
    print("\n[TensorRT] Analyzing engine for GPU fallback...")
    dla_count, gpu_count, gpu_layers = analyze_engine_layers(engine_path)
    is_pure_dla = print_fallback_report(dla_count, gpu_count, gpu_layers)
    
    if not is_pure_dla:
        if strict_dla:
            error_msg = (
                f"\n{Colors.RED}{Colors.BOLD}FATAL: Strict DLA enforcement failed!{Colors.RESET}\n"
                f"The following layers fell back to GPU:\n"
            )
            for layer in gpu_layers:
                error_msg += f"- {layer}\n"
            error_msg += "Modify the model architecture to use DLA-supported operations."
            raise RuntimeError(error_msg)
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}⛔ BLOCKING ERROR: Engine has GPU fallback.{Colors.RESET}")
            print(f"{Colors.RED}Objective was 100% DLA. Fix the model architecture.{Colors.RESET}\n")
            # Note: We still return the engine path if strict_dla is False, but print error
    
    return engine_path


# --- Full Export Pipeline ---

def export_pipeline(
    model: torch.nn.Module,
    output_dir: str | Path,
    calib_folder: str | Path | None = None,
    precision: str = "int8",
    dla_core: int = 1,  # Default to DLA Core 1 (Core 0 reserved for other tasks)
    input_size: int = 640,
    strict_dla: bool = True,
    num_calib_images: int = 50,
) -> Path | None:
    """
    Full export pipeline: PyTorch -> ONNX -> TensorRT Engine (DLA).

    Args:
        model: The trained UNINA_YOLO_DLA model.
        output_dir: Directory to save ONNX and engine files.
        calib_folder: Path to folder with calibration images (e.g., calib_imgs/).
        precision: "fp16" or "int8".
        dla_core: Target DLA core.
        input_size: Input resolution.
        num_calib_images: Expected number of calibration images (for validation).

    Returns:
        Path to the final TensorRT engine, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "unina_yolo_dla.onnx"
    engine_path = output_dir / f"unina_yolo_dla_{precision}_dla{dla_core}.engine"
    cache_path = output_dir / "calibration.cache"

    # Step 1: Export to ONNX
    export_to_onnx(model, onnx_path, input_size=input_size)

    # Step 2: Build TensorRT Engine
    calibrator = None
    if precision == "int8":
        if calib_folder is None:
            raise ValueError(
                "FATAL: No 'calib_folder' provided for INT8 export. "
                "Dummy data is prohibited for production deployment."
            )
        
        calib_folder = Path(calib_folder)
        if not calib_folder.exists():
            raise FileNotFoundError(f"FATAL: Calibration folder not found: {calib_folder}")
            
        stream = ConeCalibrationStream(calib_folder, input_size=input_size)
        
        if len(stream.image_paths) < num_calib_images:
            raise RuntimeError(
                f"FATAL: Insufficient calibration data. Found {len(stream.image_paths)} images, "
                f"but {num_calib_images} are required for production calibration."
            )
        
        calibrator = EntropyCalibrator(stream, cache_file=cache_path)

    engine_path = build_trt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        calibrator=calibrator,
        precision=precision,
        dla_core=dla_core,
        dla_core=dla_core,
        allow_gpu_fallback=True,  # Allow for build, verify after
        strict_dla=strict_dla,
    )

    return engine_path


# --- Example Usage ---
if __name__ == '__main__':
    print("TensorRT Export Module Loaded (Enhanced).")
    
    if not TRT_AVAILABLE:
        print("Skipping TensorRT tests as tensorrt is not installed.")
        print("To test ONNX export only, import the model and call export_to_onnx().")
    else:
        print("TensorRT is available. Ready to build engines.")
    
    # Example usage:
    # from unina_yolo_dla.model import UNINA_YOLO_DLA
    # model = UNINA_YOLO_DLA(num_classes=4, base_channels=32)
    # export_pipeline(
    #     model,
    #     output_dir="output",
    #     calib_folder="calib_imgs",
    #     precision="int8",
    #     dla_core=0,
    # )
