#!/bin/bash

# Neural Network Steganography - Web Interface Launcher
# Quick start script for the web application

echo "========================================"
echo "Neural Network Steganography"
echo "Web Interface Launcher"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python version: $PYTHON_VERSION"

# Check if Flask is installed
if ! python -c "import flask" &> /dev/null; then
    echo "❌ Flask not found"
    echo "Installing Flask..."
    pip install flask werkzeug
fi

echo "✓ Flask installed"

# Check if PyTorch is installed
if ! python -c "import torch" &> /dev/null; then
    echo "❌ PyTorch not found"
    echo "Please install PyTorch first:"
    echo "  pip install torch torchvision"
    exit 1
fi

echo "✓ PyTorch installed"

# Check CUDA availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &> /dev/null; then
    echo "✓ CUDA available"
else
    echo "⚠️  CUDA not available (using CPU)"
fi

echo ""
echo "========================================"
echo "Starting web server..."
echo "========================================"
echo ""
echo "Access the web interface at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Flask application
python app.py
