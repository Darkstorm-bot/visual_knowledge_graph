#!/bin/bash
# NeuroGraph V2 Backend - Installation and Run Script

set -e

echo "========================================"
echo "NeuroGraph V2 Backend Setup"
echo "========================================"

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm || true

# Create data directory
mkdir -p data

# Start server
echo ""
echo "========================================"
echo "Starting NeuroGraph V2 Server"
echo "API Docs: http://localhost:8001/docs"
echo "========================================"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
