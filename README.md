# Brushout

Paint over an object in a photo and it disappears. Brushout uses AI inpainting to fill the selected region with plausible background.

![Before and after: person in chicken costume removed from wedding photo](beach-brushout.jpg)

## How it works

Paint a red mask over whatever you want to remove, then press Enter. The masked region is passed to an ONNX inpainting model (LaMa or MI-GAN) which reconstructs the background. Results are ready in 1-2 seconds on CPU.

## Download

Pre-built installers are attached to each [GitHub release](../../releases):

- **Windows** — `Brushout-Setup.exe` (installer, no Python required)
- **Linux** — `Brushout-x86_64.AppImage` (portable, no install required)

## Running from source

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd remover
uv sync
uv run python app.py
```

Models download automatically from HuggingFace on first run (~100-200 MB total).

## Usage

| Action | How |
|--------|-----|
| Open image | Ctrl+O or drag and drop |
| Paste image | Ctrl+V |
| Paint mask | Left-drag |
| Erase mask | Right-drag |
| Adjust brush size | Scroll wheel or toolbar slider |
| Zoom | Ctrl+scroll |
| Remove object | Enter or click Remove |
| Undo | Ctrl+Z (up to 10 steps) |
| Clear mask | Clear Mask button |
| Save result | Ctrl+S |

## Models

Two models are available via the toolbar toggle:

- **LaMa** — better for uniform backgrounds (sky, walls, grass)
- **MI-GAN** — sharper edges, handles complex textures

## GPU acceleration

The app runs on CPU by default. To enable NVIDIA GPU acceleration:

```bash
sudo apt install libcudnn9-cuda-12
```

## Building

### Windows

```bash
uv run pyinstaller brushout.spec
iscc installer.iss
# Output: Output/Brushout-Setup.exe
```

### Linux (AppImage)

```bash
uv run pyinstaller brushout.spec
wget -O appimagetool https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool
# ... (see .github/workflows/build.yml for full steps)
```

CI builds both targets automatically on tagged releases.

## License

MIT
