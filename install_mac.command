#!/bin/bash
set -e

# VisionTool Mac Installation Script (Apple Silicon M3/M4)
# This script sets up a Python virtual environment and installs dependencies.
# NOTE: On Apple Silicon (M1/M2/M3/M4), use the system Python 3 at /usr/bin/python3.
# If you have issues with native wheels, you may need to use 'arch -arm64' before python/pip commands.
# If pip is not available, this script will attempt to install it using ensurepip.

PYTHON_PATH="/usr/bin/python3"

if ! $PYTHON_PATH --version >/dev/null 2>&1; then
  echo "Python 3 is not installed. Please install it from https://www.python.org/downloads/macos/"
  exit 1
fi

# Create virtual environment (Apple Silicon: arm64)
$PYTHON_PATH -m venv venv

# Activate virtual environment
source venv/bin/activate

# Ensure pip is available
if ! command -v pip >/dev/null 2>&1; then
  echo "pip not found, installing with ensurepip..."
  $PYTHON_PATH -m ensurepip
fi

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Installation complete! To activate the environment, run:"
echo "source venv/bin/activate" 