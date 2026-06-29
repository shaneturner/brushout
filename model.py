import os
from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# Full-precision model for GPU — float32 RGB, CUDA EP compatible
GPU_MODEL_REPO = "Carve/LaMa-ONNX"
GPU_MODEL_FILE = "lama_fp32.onnx"
GPU_MODEL_FILE_SIM = "lama_fp32_simplified.onnx"  # constant-folded, ~10% faster

# Quantized model for CPU — smaller, faster on CPU, expects BGR
CPU_MODEL_REPO = "opencv/inpainting_lama"
CPU_MODEL_FILE = "inpainting_lama_2025jan.onnx"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


@dataclass
class InpaintSession:
    session: ort.InferenceSession
    use_bgr: bool        # opencv model needs BGR; Carve model needs RGB
    output_scale: float  # opencv outputs [0,255]; Carve outputs [0,1]


def _download(repo: str, filename: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(local_path):
        size_hint = "~200MB" if "fp32" in filename else "~100MB"
        print(f"Downloading {filename} ({size_hint})...")
        downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=MODELS_DIR)
        if downloaded != local_path and not os.path.exists(local_path):
            import shutil
            shutil.copy2(downloaded, local_path)
        print("Download complete.")
    return local_path


def _simplify_gpu_model(src_path: str) -> str:
    """Constant-fold the GPU model on first use to reduce CPU/GPU Memcpy nodes."""
    dst_path = os.path.join(MODELS_DIR, GPU_MODEL_FILE_SIM)
    if os.path.exists(dst_path):
        return dst_path
    print("Optimising model (one-time, takes ~30s)...")
    import onnx
    from onnxsim import simplify
    model = onnx.load(src_path)
    simplified, ok = simplify(
        model,
        test_input_shapes={"image": [1, 3, 512, 512], "mask": [1, 1, 512, 512]},
    )
    if ok:
        onnx.save(simplified, dst_path)
        print("Optimisation complete.")
        return dst_path
    print("Optimisation check failed, using original model.")
    return src_path


def load_session() -> InpaintSession:
    cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()

    if cuda_available:
        base_path = _download(GPU_MODEL_REPO, GPU_MODEL_FILE)
        model_path = _simplify_gpu_model(base_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        use_bgr = False
        output_scale = 1.0
    else:
        model_path = _download(CPU_MODEL_REPO, CPU_MODEL_FILE)
        providers = ["CPUExecutionProvider"]
        use_bgr = True
        output_scale = 1.0

    session = ort.InferenceSession(model_path, providers=providers)
    active = session.get_providers()[0]
    model_label = "fp32/RGB" if cuda_available else "quantized/BGR"
    print(f"ONNX Runtime using: {active} ({model_label} model)")
    return InpaintSession(session=session, use_bgr=use_bgr, output_scale=output_scale)


def run_inference(info: InpaintSession, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    image: float32 [1, 3, H, W] RGB in [0, 1]
    mask:  float32 [1, 1, H, W], 1.0 = region to inpaint

    Returns float32 [3, H, W] RGB in [0, 255]
    """
    if info.use_bgr:
        image = image[:, ::-1, :, :]  # RGB → BGR

    outputs = info.session.run(None, {"image": image, "mask": mask})
    output = outputs[0]

    if output.ndim == 4:
        output = output[0]  # [1, 3, H, W] → [3, H, W]

    output = output * info.output_scale  # normalise to [0, 255]

    if info.use_bgr:
        output = output[::-1]  # BGR → RGB

    return output
