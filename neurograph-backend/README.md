# NeuroGraph V2 Backend

Real ML/AI-powered Knowledge Graph Backend that replaces all fake/hardcoded logic from the original demo.

## Features Implemented

### ✅ Real File Processing
- **SHA-256 Hashing**: Cryptographic file deduplication (not random strings)
- **PDF Extraction**: PyPDF2 for text extraction from PDFs
- **DOCX Extraction**: python-docx for Word documents
- **OCR**: pytesseract for image text extraction

### ✅ NLP & ML
- **Language Detection**: langdetect library (not random assignment)
- **Entity Extraction**: spaCy with 18+ entity types (PERSON, ORG, GPE, etc.)
- **Concept Extraction**: Noun phrases and key terms
- **Semantic Embeddings**: sentence-transformers (all-MiniLM-L6-v2)

### ✅ Knowledge Graph
- **NetworkX Graph**: Real graph structure with nodes and edges
- **Similarity Edges**: Cosine similarity between document embeddings
- **Community Detection**: Louvain algorithm for clustering
- **Graph Statistics**: Density, communities, average degree

### ✅ Search
- **Hybrid Search**: Keyword + Entity + Semantic similarity
- **Re-ranking**: Combined scoring algorithm
- **Snippets**: Context-aware result previews

### ✅ Persistence
- **JSON Storage**: Documents and graph saved to disk
- **State Recovery**: Automatic loading on startup

## Quick Start

```bash
cd neurograph-backend
./run.sh
```

Visit http://localhost:8001/docs for API documentation.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload and process a document |
| `/api/upload/batch` | POST | Upload multiple documents |
| `/api/search` | GET/POST | Search documents |
| `/api/graph` | GET | Get knowledge graph data |
| `/api/stats` | GET | Get system statistics |
| `/api/documents` | GET | List all documents |
| `/api/documents/{id}` | GET | Get document details |
| `/api/documents/{id}` | DELETE | Delete a document |
| `/api/analyze` | POST | Analyze text without saving |
| `/api/models/status` | GET | Check ML model status |
| `/api/reset` | POST | Reset entire system |

## Architecture

```
app/
├── main.py          # FastAPI server and routes
├── engine.py        # Core ML/AI processing logic
└── __init__.py

data/                # Persistent storage
└── state.json       # Documents and graph state
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- fastapi, uvicorn - Web framework
- spacy - NLP and entity extraction
- sentence-transformers - Semantic embeddings
- networkx - Graph processing
- PyPDF2, python-docx, pytesseract - File processing

## Comparison: V1 Fake vs V2 Real

| Feature | V1 (Fake) | V2 (Real) |
|---------|-----------|-----------|
| File Hash | `Math.random()` | SHA-256 |
| Language | Random choice | langdetect |
| Entities | None | spaCy NLP |
| Embeddings | None | sentence-transformers |
| Search | First 5 docs | Hybrid semantic search |
| Graph Edges | Random count | Cosine similarity |
| Storage | In-memory | JSON persistence |
| OCR | Random boolean | pytesseract |

## Testing

```bash
# Test upload
curl -X POST http://localhost:8001/api/upload \
  -F "file=@/path/to/document.pdf"

# Test search
curl "http://localhost:8001/api/search?q=machine+learning&top_k=5"

# Get graph
curl http://localhost:8001/api/graph

# Get stats
curl http://localhost:8001/api/stats
```
