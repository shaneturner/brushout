import numpy as np
from PIL import Image, ImageFilter

from model import InpaintSession, ModelType, run_inference

MODEL_INPUT_SIZE = 512
MIN_CROP_PADDING = 32


def _mask_bbox(mask_arr: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _to_rgb_blob(img_arr: np.ndarray, size: int) -> np.ndarray:
    """uint8 HWC RGB → float32 [1, 3, size, size] in [0, 1]."""
    pil = Image.fromarray(img_arr).resize((size, size), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr[np.newaxis]


def _to_mask_blob(mask_arr: np.ndarray, size: int) -> np.ndarray:
    """uint8 HW → float32 [1, 1, size, size] binary 0/1."""
    pil = Image.fromarray(mask_arr, "L").resize((size, size), Image.NEAREST)
    arr = (np.array(pil) > 127).astype(np.float32)
    return arr[np.newaxis, np.newaxis]


def inpaint(session: InpaintSession, image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    image: RGB PIL image (any size)
    mask:  L (grayscale) PIL image, same size; 255 = region to remove
    Returns: RGB PIL image with masked region filled in
    """
    if session.model_type == ModelType.MIGAN:
        return _inpaint_migan(session, image, mask)
    return _inpaint_lama(session, image, mask)


def _inpaint_migan(session: InpaintSession, image: Image.Image, mask: Image.Image) -> Image.Image:
    img_arr  = np.array(image.convert("RGB"))           # HWC uint8
    mask_arr = np.array(mask.convert("L"))              # HW  uint8, 255=masked

    # MI-GAN convention: 0=fill, 255=keep  →  invert our mask
    migan_mask = np.where(mask_arr > 127, 0, 255).astype(np.uint8)

    img_blob  = img_arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W] uint8
    mask_blob = migan_mask[np.newaxis, np.newaxis]       # [1, 1, H, W] uint8

    result = run_inference(session, img_blob, mask_blob)  # [3, H, W] uint8
    return Image.fromarray(result.transpose(1, 2, 0), "RGB")


def _inpaint_lama(session: InpaintSession, image: Image.Image, mask: Image.Image) -> Image.Image:
    img_arr  = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))

    bbox = _mask_bbox(mask_arr)
    if bbox is None:
        return image.copy()

    W, H = image.size
    x0, y0, x1, y1 = bbox
    mask_w, mask_h = x1 - x0, y1 - y0
    pad = max(MIN_CROP_PADDING, max(mask_w, mask_h) // 2)

    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(W, x1 + pad)
    y1p = min(H, y1 + pad)

    crop_img  = img_arr[y0p:y1p, x0p:x1p]
    crop_mask = mask_arr[y0p:y1p, x0p:x1p]
    crop_h, crop_w = crop_img.shape[:2]

    output = run_inference(
        session,
        _to_rgb_blob(crop_img, MODEL_INPUT_SIZE),
        _to_mask_blob(crop_mask, MODEL_INPUT_SIZE),
    )  # [3, H, W] float32 RGB in [0, 255]

    result_hwc = np.clip(output, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    result_pil = Image.fromarray(result_hwc, "RGB").resize((crop_w, crop_h), Image.LANCZOS)

    # Composite: only replace masked pixels with feathered edges
    comp_mask = Image.fromarray(crop_mask, "L").resize((crop_w, crop_h), Image.NEAREST)
    comp_mask = comp_mask.filter(ImageFilter.GaussianBlur(radius=3))
    alpha = np.array(comp_mask, dtype=np.float32) / 255.0

    blended = crop_img.astype(np.float32) + alpha[..., None] * (
        np.array(result_pil, dtype=np.float32) - crop_img.astype(np.float32)
    )
    output_arr = img_arr.copy()
    output_arr[y0p:y1p, x0p:x1p] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(output_arr)
