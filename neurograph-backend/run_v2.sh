#!/bin/bash

# NeuroGraph V2 - Installation & Launch Script
# This script installs all dependencies and starts the V2 backend

set -e

echo "=========================================="
echo "  NeuroGraph V2 - Advanced Knowledge Graph"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install system dependencies for OCR and graph libraries
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    tesseract-ocr \
    tesseract-ocr-dan \
    tesseract-ocr-eng \
    libpoppler-cpp-dev \
    libigraph0-dev \
    pkg-config \
    > /dev/null 2>&1 || true

# Install Python packages
echo "Installing Python packages (this may take 5-10 minutes)..."
pip install -q --no-cache-dir -r requirements.txt

# Download spaCy models
echo "Downloading spaCy language models..."
python -m spacy download da_core_news_lg -q || true
python -m spacy download en_core_web_lg -q || true

# Create data directories
echo "Setting up data directories..."
mkdir -p data/uploads
mkdir -p data/embedding_cache
mkdir -p data/entity_cache

# Show installation summary
echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Starting NeuroGraph V2 server..."
echo ""
echo "📡 API Documentation: http://localhost:8001/docs"
echo "🔍 Health Check:      http://localhost:8001/api/v2/health"
echo "ℹ️  System Info:       http://localhost:8001/api/v2/info"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server on port 8001 (V2)
uvicorn app.v2.api:app --host 0.0.0.0 --port 8001 --reload
