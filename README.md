# VisionTool

VisionTool is a modular, plugin-based image processing application for scientific and industrial vision tasks. It allows you to load images, define regions of interest (ROI), and apply a customizable pipeline of filters and detectors. The software is designed for flexibility and extensibility, supporting both built-in and user-defined plugins.

## Features
- Load and view images
- Define and adjust ROI interactively
- Build a processing pipeline with filters and detectors
- Parameter panel for real-time plugin adjustments
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