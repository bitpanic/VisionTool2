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

### Edge Measurement Tool
- Add the **Edge Measurement** filter from the Filters list into the pipeline and select it.
- Use the viewer toolbar **Measure** button to draw a line across the edge you want to analyse; the latest line becomes the evaluation path.
- In the **parameter panel (under the pipeline)** configure:
  - **Measurement mode** (drop‑down):
    - **Along line (single profile)**: uses only the drawn line and computes one 1D profile \(I(s)\) and its derivative; this is the classic edge‑spread measurement.
    - **Perpendicular cross-sections (multi profile)**: treats the drawn line as an evaluation path and takes many short cross‑sections orthogonal to it; each cross‑section gets the same metrics as in the single‑profile case and they are aggregated (mean/median/std, etc.).
  - **Sampling mode**:
    - **Step (px)**: choose a physical step size along the line (e.g. 0.25, 0.5, 1.0 px).
    - **Num samples**: choose an explicit number of samples along the line, independent of its length.
  - **Cross‑sections (multi‑profile mode only)**:
    - `Cross-section count M`: number of cross‑sections sampled along the evaluation path.
    - `Cross-section length W (px)`: physical length of each cross‑section, in pixels.
  - **Gradient alignment (optional)**:
    - Enable **Align cross-sections to local gradient** to rotate each cross‑section to follow the local edge normal instead of the simple perpendicular to the drawn line.
    - Select **Gradient method** (Sobel/Scharr), optionally increase **Gradient sigma** and **Gradient radius r** to smooth noisy gradients, and set **Min gradient magnitude** to ignore very weak edges.
  - **Robustness (Multi‑line average)**:
    - Enable `Multi-line average enabled`, then set `Multi-line count K` and `Multi-line offset d (px)` to sample several parallel profiles around the main one and average them; this reduces the influence of texture/noise.
  - **Pre‑processing**:
    - Choose `Smoothing type` (None / Gaussian / Savitzky‑Golay) and its parameters.
    - Toggle `Normalize 0-1` to normalize the profile before derivative/width computation.
- Press the pipeline **Run** button:
  - The image content itself is unchanged, but overlays appear on top:
    - In **Along line** mode: peak locations and (optionally) 10–90% edge‑width markers on the line.
    - In **Cross‑section** modes: a subset of cross‑section lines plus an optional color “heat” along the evaluation path indicating a chosen metric (e.g. peak_to_peak or edge_width).
  - The **Edge Measurement** panel (next to the image) shows:
    - Current mode and parameter summary.
    - Core metrics: `peak_height_pos`, `peak_height_neg`, `peak_to_peak`, `area_pos`, `area_neg`, `area_total`, and optional `edge_width`.
    - For cross‑sections: aggregated mean/median/std/p10/p90 across all valid cross‑sections, and a selector to inspect individual profiles.
  - The embedded plot displays the intensity profile \(I(s)\) and derivative \(dI/ds\), with peak positions and (if present) 10–90% width markers.
- Use the **Export CSV** button in the Edge Measurement panel to save results:
  - Metadata: image filename, timestamp, and the active parameter settings.
  - **Along line** mode: a single metrics row, plus optional raw profile/derivative samples (one row per sample).
  - **Cross‑section** modes: one row per cross‑section (center, endpoints, direction, metrics), followed by an aggregate statistics block.

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