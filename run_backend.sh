#!/bin/bash
# NeuroGraph Backend Startup Script

echo "🚀 Starting NeuroGraph Knowledge Base Backend..."
echo ""

# Check if in correct directory
if [ ! -d "neurograph-backend" ]; then
    echo "❌ Error: neurograph-backend directory not found"
    exit 1
fi

cd neurograph-backend

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Download spaCy model if not present
echo "⬇️  Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
    echo "Downloading en_core_web_sm model..."
    python -m spacy download en_core_web_sm
}

# Start the server
echo ""
echo "✅ Starting FastAPI server..."
echo "📍 API available at: http://localhost:8000"
echo "📚 Docs available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
