#!/bin/bash

# Local MkDocs Build Test Script
# This script mimics the CI build process for testing documentation locally

set -e  # Exit on error

echo "=========================================="
echo "Testing MkDocs Build Locally"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip -q
echo "✓ pip upgraded"

echo ""
echo "Step 4: Installing documentation dependencies..."
pip install -q mkdocs-material mkdocs-autorefs mkdocstrings[python] mkdocs-glightbox pillow cairosvg
echo "✓ Dependencies installed"

echo ""
echo "Step 5: Building documentation with MkDocs..."
echo "Running: mkdocs build --strict --verbose"
echo "=========================================="
mkdocs build --strict --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS: Documentation built successfully!"
    echo "=========================================="
    echo ""
    echo "To view the built documentation:"
    echo "  1. Run: mkdocs serve"
    echo "  2. Open: http://127.0.0.1:8000"
    echo ""
    echo "Or open the built site directly:"
    echo "  file://$(pwd)/site/index.html"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ FAILED: Documentation build failed"
    echo "=========================================="
    exit 1
fi
