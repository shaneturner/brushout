# Testing Brushout

## Setup (first time only)

```bash
cd ~/code/remover
uv sync
```

## Run the app

```bash
uv run python app.py
```

## Loading an image

Three ways to open an image:

- **Ctrl+O** — file dialog
- **Drag and drop** a file onto the canvas (not supported in WSL; use Ctrl+V instead)
- **Ctrl+V** — paste from clipboard (works with files copied in Explorer, screenshots from Win+Shift+S, or images copied in a browser)

## Manual test walkthrough

### 1. Open an image

Load any photo using one of the methods above. The image fits to the window and the hint overlay disappears.

### 2. Paint a mask

- **Left-drag** — paint a red mask over the object to remove
- **Right-drag** — erase parts of the mask
- **Scroll wheel** — zoom in/out (zooms toward cursor)
- **Space+drag** — pan around the image
- **Brush slider** in the toolbar — change brush size

### 3. Remove the object

Click **Remove Object** or press **Enter**.

The status bar shows "Removing object..." while processing, then "Done." when finished. Expect 1-2 seconds on CPU.

### 4. Switch models

Click the **Model: LaMa / Model: MI-GAN** button in the toolbar to toggle between models. Both are preloaded in the background on startup.

- **LaMa** — better for uniform backgrounds (sky, walls, grass)
- **MI-GAN** — sharper edges, better for complex textures

### 5. Save the result

Click **Save** or Ctrl+S and choose PNG or JPEG.

### 6. Undo

Ctrl+Z steps back through up to 10 undo states.

### 7. Clear mask

Click **Clear Mask** to wipe the red overlay without changing the image.

### 8. Close image

Click **Close Image** or Ctrl+W to return to the empty canvas with the hint overlay.

## Things to verify

- [ ] App opens without terminal errors
- [ ] Hint overlay shows on startup; disappears when an image is loaded
- [ ] All three load methods work (Ctrl+O, drag-and-drop, Ctrl+V)
- [ ] Red mask appears while painting, disappears when erased
- [ ] Zoom toward cursor works on scroll
- [ ] Space+drag pans the canvas
- [ ] Remove Object fills the masked region with plausible background
- [ ] Model toggle switches between LaMa and MI-GAN; status bar confirms
- [ ] Undo restores previous image state
- [ ] Clear Mask wipes mask without affecting the image
- [ ] Close Image returns to the empty canvas
- [ ] Save writes a valid file

## Known limitations

- **GPU not active by default** — runs on CPU (~1.5s per removal). To enable NVIDIA GPU: `sudo apt install libcudnn9-cuda-12`
- **Large masks** take slightly longer but still work
- **LaMa quality** is best for uniform backgrounds; complex textures near the mask edge may show slight blurring
- **WSL** — drag-and-drop from Windows Explorer is not supported; use Ctrl+V instead
