#!/bin/bash
# Setup script for HFT Market Simulator project

echo "Setting up HFT Market Simulator project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Note: TA-Lib may require additional system dependencies."
echo "On macOS: brew install ta-lib"
echo "On Ubuntu: sudo apt-get install ta-lib"
echo "On Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"

