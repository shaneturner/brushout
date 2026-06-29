import numpy as np
from PIL import Image

MODEL_INPUT_SIZE = 512
MIN_CROP_PADDING = 32


def _mask_bbox(mask_arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) bounding box of non-zero pixels, or None."""
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _to_rgb_blob(img_arr: np.ndarray, size: int) -> np.ndarray:
    """PIL RGB uint8 HWC → float32 [1, 3, size, size] RGB in [0, 1]."""
    pil = Image.fromarray(img_arr)
    pil = pil.resize((size, size), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32)   # HWC RGB
    arr = arr.transpose(2, 0, 1)            # HWC → CHW
    arr = arr / 255.0
    return arr[np.newaxis]                  # → [1, 3, H, W]


def _to_mask_blob(mask_arr: np.ndarray, size: int) -> np.ndarray:
    """Grayscale uint8 HW → float32 [1, 1, size, size], binary 0/1."""
    pil = Image.fromarray(mask_arr, "L")
    pil = pil.resize((size, size), Image.NEAREST)
    arr = np.array(pil, dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    return arr[np.newaxis, np.newaxis]      # → [1, 1, H, W]


def inpaint(
    session,
    image: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Remove masked region from image using localized context cropping.

    image: RGB PIL image (any size)
    mask:  L (grayscale) PIL image, same size; 255 = region to remove
    Returns: RGB PIL image with masked region filled in
    """
    img_arr = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))

    bbox = _mask_bbox(mask_arr)
    if bbox is None:
        return image.copy()

    W, H = image.size
    x0, y0, x1, y1 = bbox

    # Adaptive padding: at least half the mask size so the model sees as much
    # context outside the mask as inside — critical for plausible fill.
    mask_w, mask_h = x1 - x0, y1 - y0
    pad = max(MIN_CROP_PADDING, max(mask_w, mask_h) // 2)

    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(W, x1 + pad)
    y1p = min(H, y1 + pad)

    crop_img = img_arr[y0p:y1p, x0p:x1p]
    crop_mask = mask_arr[y0p:y1p, x0p:x1p]
    crop_h, crop_w = crop_img.shape[:2]

    img_blob = _to_rgb_blob(crop_img, MODEL_INPUT_SIZE)
    mask_blob = _to_mask_blob(crop_mask, MODEL_INPUT_SIZE)

    from model import run_inference
    output = run_inference(session, img_blob, mask_blob)

    # output: [3, H, W] float32 RGB in [0, 255]
    result_hwc = np.clip(output, 0, 255).astype(np.uint8).transpose(1, 2, 0)

    result_pil = Image.fromarray(result_hwc, "RGB")
    result_pil = result_pil.resize((crop_w, crop_h), Image.LANCZOS)

    output_arr = img_arr.copy()
    output_arr[y0p:y1p, x0p:x1p] = np.array(result_pil)

    return Image.fromarray(output_arr)
