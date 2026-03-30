# NeuroGraph Knowledge Base - Real Implementation

A fully functional personal knowledge base with real ML/AI processing, replacing all fake/hardcoded logic from the original demo.

## What Was Fixed

### Original Issues (Fake/Hardcoded Logic)
1. ❌ **Random file hashing** - Used `Math.random()` instead of cryptographic hash
2. ❌ **Fake node/edge generation** - Random numbers instead of actual graph construction
3. ❌ **Random language detection** - No actual language analysis
4. ❌ **Fake OCR decisions** - Random chance instead of real OCR processing
5. ❌ **Hardcoded search results** - Always returned first 5 documents with random match %
6. ❌ **Simulated model downloads** - Just animated progress bars
7. ❌ **Fake VRAM/storage calculations** - Made-up formulas
8. ❌ **Pipeline demo was pure animation** - No actual processing

### New Real Implementation
✅ **SHA-256 file hashing** - Proper cryptographic deduplication
✅ **Real text extraction** - PDF (PyPDF2), DOCX (python-docx), Images (pytesseract OCR)
✅ **Actual language detection** - Heuristic-based detection (Danish/English)
✅ **Real entity extraction** - spaCy NLP for named entity recognition
✅ **True semantic search** - Sentence transformers + cosine similarity
✅ **Knowledge graph construction** - NetworkX with document similarity edges
✅ **Persistent storage** - JSON files + NumPy arrays for embeddings
✅ **Real embedding generation** - all-MiniLM-L6-v2 sentence transformers

## Project Structure

```
/workspace
├── index.html              # Frontend (needs to be updated to connect to backend)
├── index.html.backup       # Original demo version
└── neurograph-backend/
    ├── app/
    │   ├── __init__.py
    │   └── main.py         # FastAPI backend with all real logic
    ├── uploads/            # Uploaded files stored here
    ├── data/
    │   ├── documents.json  # Document metadata
    │   ├── graph.json      # Knowledge graph structure
    │   └── embeddings.npy  # Vector embeddings
    ├── requirements.txt    # Python dependencies
    └── main.py             # Entry point
```

## Installation & Setup

### 1. Install Dependencies

```bash
cd /workspace/neurograph-backend
pip install -r requirements.txt
```

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Start the Backend Server

```bash
cd /workspace/neurograph-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Get system status and stats |
| POST | `/api/upload` | Upload and process a document |
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Get specific document |
| DELETE | `/api/documents/{id}` | Delete a document |
| GET | `/api/search?q=query` | Semantic search |
| GET | `/api/graph` | Get knowledge graph data |
| GET | `/api/models` | List loaded ML models |
| POST | `/api/models/load` | Load ML models |
| GET | `/api/stats` | Get detailed statistics |
| GET | `/api/export` | Export all data |

## How It Works

### Document Upload Pipeline

1. **File Upload** → Receive file via HTTP POST
2. **Hash Generation** → SHA-256 hash for deduplication
3. **Text Extraction** → Parse PDF/DOCX/TXT or OCR images
4. **Language Detection** → Analyze text for Danish/English
5. **Entity Extraction** → spaCy NLP for named entities
6. **Embedding Generation** → Sentence transformer creates 384-dim vector
7. **Graph Update** → Add document node, calculate similarity edges
8. **Persistence** → Save to disk (JSON + NumPy)

### Semantic Search

1. User enters query
2. Query embedded using same model
3. Cosine similarity calculated against all document embeddings
4. Top-K results returned with scores
5. Matched entities highlighted

### Knowledge Graph

- **Document Nodes** (green) - Each uploaded document
- **Entity Nodes** (blue) - Extracted named entities (PERSON, ORG, GPE, etc.)
- **Similarity Edges** - Connect similar documents (cosine > 0.3)
- **Contains Edges** - Connect documents to their entities

## Testing the Backend

```python
import requests

# Check status
response = requests.get('http://localhost:8000/api/status')
print(response.json())

# Upload a file
with open('test.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload', files={'file': f})
    print(response.json())

# Search
response = requests.get('http://localhost:8000/api/search', params={'q': 'machine learning'})
print(response.json())

# Get graph
response = requests.get('http://localhost:8000/api/graph')
print(response.json())
```

## Frontend Integration

The frontend (`index.html`) needs to be updated to call the real backend APIs. Key changes:

1. Replace fake `AppState` with API calls
2. Use `fetch()` to call `/api/upload` instead of simulating
3. Call `/api/search` for real semantic search
4. Fetch `/api/graph` for actual graph data
5. Display real entity counts, language detection results, etc.

## Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **Sentence Transformers** - Text embeddings
- **spaCy** - NLP and entity extraction
- **NetworkX** - Graph data structures
- **PyPDF2** - PDF parsing
- **python-docx** - Word document parsing
- **pytesseract** - OCR for images
- **NumPy/Scikit-learn** - Vector operations and similarity

### Frontend (existing)
- Vanilla JavaScript
- Tailwind CSS
- vis-network for graph visualization

## Performance Notes

- First upload triggers model download (~300MB for sentence-transformers)
- Embedding generation: ~50ms per document on CPU
- Similarity search: O(n) but fast for <1000 documents
- Consider adding FAISS or Annoy for larger datasets

## Future Improvements

1. Add PostgreSQL for production persistence
2. Implement chunking for long documents
3. Add multi-language support (better than heuristic detection)
4. Integrate RAG pipeline for Q&A
5. Add user authentication
6. Deploy with Docker

## License

MIT License
