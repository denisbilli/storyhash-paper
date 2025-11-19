#!/bin/bash
# Setup script for StoryHash paper reproduction

set -e

echo "================================================"
echo "StoryHash Paper - Setup Script"
echo "================================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/davis
mkdir -p embeddings
mkdir -p indices
mkdir -p results

# Create placeholder files
touch embeddings/.gitkeep
touch indices/.gitkeep
touch results/.gitkeep

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Download DAVIS 2017:"
echo "   wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
echo "   unzip DAVIS-2017-trainval-480p.zip -d data/davis"
echo "3. Extract features: bash scripts/extract_all_features.sh"
echo "4. Run benchmark: python scripts/benchmark_robustness.py"
echo ""
