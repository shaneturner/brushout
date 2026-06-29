Context Summary: Building an Efficient Custom Object-Removal Tool (Image Inpainting)
Objective: Build a lightweight, fast, and local object-removal (image inpainting) tool that mimics Microsoft Photos' "Erase Object" feature.

How Microsoft Photos Works (The Architecture):
Backend: It uses native Windows AI Imaging APIs (ImageObjectErase), executing via the ONNX Runtime and DirectML to target NPUs/GPUs directly without heavy Python framework overhead.
Method: It optimizes performance using Localized Context Cropping — it doesn't process the whole canvas, only a cropped bounding box around the user's brush stroke mask.
Models: It leverages heavily quantized (likely int8) distilled local diffusion backbones for offline erasing, and hybrid cloud models for text-based generations.

Technical Milestones Discussed:
Traditional Computer Vision (CV): Built a basic script using OpenCV (cv2.inpaint with INPAINT_TELEA). Ultra-lightweight (<100MB) and fast, but lacks true generative AI intelligence for complex textures.
Heavy AI Diffusion (SDXL): Explored Latent Consistency Models (LCM) down to 4 inference steps, but the fp16 model size (~6GB) is too large for a lightweight tool.
Compact AI Diffusion (SD 1.5 + LCM): Discussed dropping to a Stable Diffusion 1.5 architecture optimized with an LCM Scheduler. Cuts VRAM footprint to ~2.5GB while keeping processing times to a few seconds.
Ultra-Lightweight AI (LaMa): LaMa (Large Mask Inpainting) is a non-diffusion, Fourier Convolution-based model. A quantized ONNX version was added to OpenCV in May 2025 — ~100MB, runs under 1 second on CPU.

Selected Architecture (Implemented):
Primary Model: LaMa quantized ONNX from opencv/inpainting_lama on HuggingFace (~100MB)
Runtime: ONNX Runtime — CUDAExecutionProvider (NVIDIA GPU) with automatic CPU fallback
GUI: PySide6 (cross-platform: Linux, Windows, Mac)
Key technique: Localized Context Cropping — crop bounding box of mask + padding, infer at 512×512, paste result back

NOTE — IOPaint / lama-cleaner: This project was archived August 2025 and is no longer maintained. It was not used as a dependency.

Current State:
Implementation is complete. See:
  app.py           — PySide6 GUI (open image, brush mask, remove object, undo, save)
  inpaint.py       — Localized Context Cropping pipeline
  model.py         — ONNX model loader + HuggingFace auto-download
  pyproject.toml   — uv project config (Python 3.12)
  .python-version  — pins Python 3.12 for uv

Setup and run (using uv):
  uv sync              # install dependencies into virtualenv
  uv run python app.py # launch the GUI

First launch auto-downloads the LaMa ONNX model (~100MB) from HuggingFace.

Potential quality upgrade paths:
  MAT (Mask-Aware Transformer): ~500MB, ~2-3s, better on complex textures
  TurboFill (CVPR 2025): GPU-required, best quality/speed in 2025
