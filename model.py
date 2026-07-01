import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

ort.set_default_logger_severity(3)  # suppress W-level messages from the global C++ logger

if getattr(sys, 'frozen', False):
    MODELS_DIR = os.path.join(os.path.dirname(sys.executable), "models")
else:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class ModelType(Enum):
    LAMA  = "lama"
    MIGAN = "migan"


# LaMa — full-precision float32 RGB, fixed 512×512, GPU-friendly
_LAMA_REPO = "Carve/LaMa-ONNX"
_LAMA_FILE = "lama_fp32.onnx"
_LAMA_FILE_SIM = "lama_fp32_simplified.onnx"

# MI-GAN — uint8 RGB, dynamic size, all pre/post-processing built-in
_MIGAN_REPO = "andraniksargsyan/migan"
_MIGAN_FILE = "migan_pipeline_v2.onnx"


@dataclass
class InpaintSession:
    session: ort.InferenceSession
    model_type: ModelType
    # LaMa-specific: Carve model is RGB [0,255] out; opencv CPU model is BGR [0,255] out
    use_bgr: bool = False


def _download(repo: str, filename: str, size_hint: str = "") -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(local_path):
        hint = f" ({size_hint})" if size_hint else ""
        print(f"Downloading {filename}{hint}...")
        downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=MODELS_DIR)
        if downloaded != local_path and not os.path.exists(local_path):
            import shutil
            shutil.copy2(downloaded, local_path)
        print("Download complete.")
    return local_path


def _simplify_lama(src_path: str) -> str:
    dst_path = os.path.join(MODELS_DIR, _LAMA_FILE_SIM)
    if os.path.exists(dst_path):
        return dst_path
    print("Optimising LaMa model (one-time, ~30s)...")
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


def _cuda_available() -> bool:
    # Packaged builds never bundle the CUDA/cuDNN runtime DLLs onnxruntime-gpu needs
    # (~2GB — cuBLAS, cuDNN, cuFFT, cuRAND — not shipped by onnxruntime-gpu itself and
    # deliberately excluded from the installer for size). Always use CPU when frozen.
    if getattr(sys, 'frozen', False):
        return False

    # onnxruntime-gpu always lists CUDAExecutionProvider in get_available_providers(),
    # even on machines with no NVIDIA driver — it reports what was compiled in, not
    # what actually works. Probe for a real driver via nvidia-smi instead, since
    # letting onnxruntime try (and fail) to load the CUDA/cuDNN DLLs can hang for a
    # long time on Windows.
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=3,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _session_opts() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # ERROR only — suppress W-level Memcpy/graph warnings
    return opts


def load_session(model_type: ModelType = ModelType.LAMA) -> InpaintSession:
    return _load_session(model_type, cuda=_cuda_available())


def _load_session(model_type: ModelType, cuda: bool) -> InpaintSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    opts = _session_opts()

    if model_type == ModelType.MIGAN:
        path = _download(_MIGAN_REPO, _MIGAN_FILE, "~26MB")
        try:
            session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        except Exception:
            if not cuda:
                raise
            print("CUDA session creation failed, falling back to CPU.")
            return _load_session(model_type, cuda=False)
        active = session.get_providers()[0]
        print(f"ONNX Runtime using: {active} (MI-GAN pipeline)")
        return InpaintSession(session=session, model_type=ModelType.MIGAN)

    # LaMa
    if cuda:
        base = _download(_LAMA_REPO, _LAMA_FILE, "~200MB")
        path = _simplify_lama(base)
        use_bgr = False
    else:
        path = _download("opencv/inpainting_lama", "inpainting_lama_2025jan.onnx", "~100MB")
        use_bgr = True

    try:
        session = ort.InferenceSession(path, sess_options=opts, providers=providers)
    except Exception:
        if not cuda:
            raise
        print("CUDA session creation failed, falling back to CPU.")
        return _load_session(model_type, cuda=False)
    active = session.get_providers()[0]
    label = "fp32/RGB" if cuda else "quantized/BGR"
    print(f"ONNX Runtime using: {active} (LaMa {label})")
    return InpaintSession(session=session, model_type=ModelType.LAMA, use_bgr=use_bgr)


def run_inference(info: InpaintSession, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LaMa path:
      image: float32 [1, 3, H, W] RGB in [0, 1]
      mask:  float32 [1, 1, H, W], 1.0 = region to inpaint
      returns: float32 [3, H, W] RGB in [0, 255]

    MI-GAN path:
      image: uint8 [1, 3, H, W] RGB
      mask:  uint8 [1, 1, H, W], 0 = region to inpaint, 255 = keep
      returns: uint8 [3, H, W] RGB
    """
    if info.model_type == ModelType.MIGAN:
        output = info.session.run(None, {"image": image, "mask": mask})[0]
        return output[0] if output.ndim == 4 else output

    if info.use_bgr:
        image = image[:, ::-1, :, :]
    output = info.session.run(None, {"image": image, "mask": mask})[0]
    if output.ndim == 4:
        output = output[0]
    if info.use_bgr:
        output = output[::-1]
    return output
