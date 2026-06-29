# Testing the Object Remover

## Setup (first time only)

```bash
cd ~/code/remover
uv sync
```

This installs all dependencies including PySide6 and the ONNX runtime. Python 3.12 is pinned via `.python-version`.

> **Note:** The LaMa model (~100MB) downloads automatically on first run from HuggingFace. You'll see a progress message in the terminal.

## Run the app

```bash
uv run python app.py
```

A window will open with a dark canvas and a toolbar across the top.

## Manual test walkthrough

### 1. Open an image
Click **Open Image** (or Ctrl+O) and pick any photo. The image fits to the window.

### 2. Paint a mask over the object to remove
- **Left-drag** to paint a red mask over the object
- **Right-drag** to erase parts of the mask
- **Scroll wheel** to zoom in/out for precision
- Use the **Brush** slider in the toolbar to change brush size

### 3. Remove the object
Click **Remove Object** (or press **Enter**).

The status bar shows "Removing object…" while processing, then "Done." when finished. Expect **1–2 seconds** on CPU.

### 4. Save the result
Click **Save** (or Ctrl+S) and choose PNG or JPEG.

### 5. Undo
Click **Undo** (or Ctrl+Z) to step back. Up to 10 undo steps are kept.

### 6. Clear mask without undoing
Click **Clear Mask** to wipe the red overlay without changing the image.

## Things to verify

- [ ] App opens without errors in the terminal
- [ ] Image loads and fits the canvas
- [ ] Red mask appears while painting, disappears from erased areas
- [ ] "Remove Object" replaces the masked region with plausible background
- [ ] Result looks natural (no dark blobs or hard edges)
- [ ] Undo restores the previous image state
- [ ] Save writes a valid file

## Known limitations

- **GPU not active yet** — the NVIDIA GPU requires cuDNN 9 which isn't installed. The app runs on CPU (~1.5s per removal). To enable GPU later: `sudo apt install libcudnn9-cuda-12`.
- **Large masks** take slightly longer but still work — the adaptive crop padding ensures the model has enough context.
- **LaMa quality** is best for uniform backgrounds (sky, grass, walls). Complex textures near the mask edge may show slight blurring.
