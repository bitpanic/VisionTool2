# VisionTool

VisionTool is a modular, plugin-based image processing application for scientific and industrial vision tasks. It allows you to load images, define regions of interest (ROI), and apply a customizable pipeline of filters and detectors. The software is designed for flexibility and extensibility, supporting both built-in and user-defined plugins.

## Features
- Load and view images
- Define and adjust ROI interactively (including enable/disable toggle)
- Build a processing pipeline with filters and detectors
- Per-step enable/disable checkboxes for the pipeline
- Parameter panel directly below the pipeline for real-time tuning
- Measurement tools with calibration (µm / mm per pixel)
- Histogram with adjustable LUT window for contrast emphasis
- Export of both the current image and the annotated view
- Save and load pipelines
- Cross-platform: Windows and Mac (Apple Silicon)

---

## Installation

### Windows
1. Make sure you have Python 3.9+ installed and added to your PATH.
2. Double-click or run `install_windows.bat` in a command prompt.
3. This will create a virtual environment and install all required dependencies.

### Mac (Apple Silicon: M1/M2/M3/M4)
1. Make sure you have Python 3 installed (the system Python at `/usr/bin/python3` is recommended).
2. Open a terminal and run:
   ```sh
   chmod +x install_mac.command
   ./install_mac.command
   ```
3. This will create a virtual environment and install all required dependencies.

**Apple Silicon Notes:**
- The script uses the system Python 3 at `/usr/bin/python3` for best compatibility.
- If you have issues with native wheels, you may need to use `arch -arm64` before python/pip commands.
- If pip is not available, the script will attempt to install it using `ensurepip`.

---

## Running VisionTool

After installation, activate the virtual environment:

### Windows
```bat
venv\Scripts\activate
python main.py
```

### Mac
```sh
source venv/bin/activate
python3 main.py
```

---

## Quick User Manual

### Layout
- **Center**: Image viewer with toolbar (zoom, measurement, export).
- **Right (top → bottom)**:
  - ROI controls & hints.
  - Histogram + LUT window sliders.
  - Plugin browser (filters / detectors).
  - Pipeline list (with enable/disable checkboxes).
  - Parameter panel for the selected pipeline step.

### ROI handling
- **Create ROI**: Hold **Ctrl** and drag in the image.
- **Move ROI**: Hold **Ctrl** and drag **inside** the existing ROI.
- **Resize ROI**: Hold **Ctrl** and drag on ROI edges/corners.
- **Edit numerically**: Change X/Y/Width/Height in the ROI panel and press **Apply ROI**.
- **Temporarily hide ROI**: Uncheck **Enable ROI** in the ROI panel (values are kept).
- **Clear ROI**: Click **Clear ROI**.

### Measurement and calibration
- Toggle **Measure** in the top toolbar, then left‑drag to draw measurement lines.
- Set **Units / pixel** and choose **px / µm / mm** to calibrate.
- Use **Clear Measurements** to remove all measurement lines.

### Histogram & LUT
- The histogram shows grayscale distribution of the current image.
- Drag the **left** (cyan) and **right** (yellow) vertical bars to define a grayscale window.
- The viewer remaps that window to full contrast, helping to reveal low‑contrast defects.

### Export
- **Export Image** (toolbar): saves the current processed image (with LUT, without ROI/measurements).
- **File → Export View**: saves the current view including ROI and measurement overlays.

---

## Built‑in Filters and Detectors

### Filters
- **Gaussian Blur**
  - Softens the image using a Gaussian kernel.
  - **Parameters**: `kernel_size` (odd integer; controls blur strength).
- **Threshold**
  - Converts the image to a binary black/white mask.
  - **Parameters**: `thresh` (threshold level), `maxval` (value for white).
- **Edge Detection**
  - Runs Canny edge detection.
  - **Parameters**: `threshold1`, `threshold2` (Canny thresholds).
- **Contrast Enhancement**
  - Enhances local contrast using CLAHE on the L channel in LAB space.
  - **Parameters**: `clip_limit`, `tile_size`, `apply_clahe` (on/off).

### Detectors
- **Blob Detector**
  - Uses OpenCV `SimpleBlobDetector` to find roughly blob‑shaped regions.
  - **Parameters**: `min_threshold`, `max_threshold`, `threshold_step`,
    `min_area`, `max_area`, `min_circularity`, `min_convexity`, `max_convexity`,
    `min_inertia_ratio`.
- **Edge Point Detector**
  - Uses Canny to find strong edges, then marks each edge pixel with a small dot.
  - **Parameters**: `threshold1`, `threshold2` (Canny thresholds).
- **Contour Detector**
  - Extracts and draws object contours based on a binary threshold.
  - **Parameters**: `threshold`, `min_area`, `max_area`.

---

## Project Structure
- `main.py` — Main application entry point
- `install_windows.bat` — Windows installation script
- `install_mac.command` — Mac (Apple Silicon) installation script
- `requirements.txt` — Python dependencies
- `plugins/` — Plugin directory (filters, detectors, base classes)
- `README.md` — This file

---

## Support
For issues or questions, please open an issue on the project repository or contact the maintainer. 