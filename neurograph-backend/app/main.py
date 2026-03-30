"""
NeuroGraph Knowledge Base - Real Backend Implementation
Fully functional backend with actual ML/AI processing
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import hashlib
import json
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import aiofiles

# ML imports
from sentence_transformers import SentenceTransformer
import spacy
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from docx import Document
import io

app = FastAPI(title="NeuroGraph Knowledge Base API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Database file paths
DOCUMENTS_DB = DATA_DIR / "documents.json"
GRAPH_DB = DATA_DIR / "graph.json"
EMBEDDINGS_DB = DATA_DIR / "embeddings.npy"

# Global state (in production, use proper database)
class AppState:
    def __init__(self):
        self.documents = []
        self.graph_data = {"nodes": [], "edges": []}
        self.embeddings = None
        self.embedding_model = None
        self.nlp_model = None
        self.knowledge_graph = nx.Graph()
        self.loaded = False
    
    def save(self):
        """Persist state to disk"""
        with open(DOCUMENTS_DB, 'w') as f:
            json.dump(self.documents, f, indent=2, default=str)
        with open(GRAPH_DB, 'w') as f:
            graph_data = {
                "nodes": list(self.knowledge_graph.nodes(data=True)),
                "edges": list(self.knowledge_graph.edges(data=True))
            }
            json.dump(graph_data, f, indent=2, default=str)
        if self.embeddings is not None:
            np.save(EMBEDDINGS_DB, self.embeddings)
    
    def load(self):
        """Load state from disk"""
        if DOCUMENTS_DB.exists():
            with open(DOCUMENTS_DB, 'r') as f:
                self.documents = json.load(f)
        if GRAPH_DB.exists():
            with open(GRAPH_DB, 'r') as f:
                graph_data = json.load(f)
                self.knowledge_graph.clear()
                for node, data in graph_data.get("nodes", []):
                    self.knowledge_graph.add_node(node, **data)
                for u, v, data in graph_data.get("edges", []):
                    self.knowledge_graph.add_edge(u, v, **data)
        if EMBEDDINGS_DB.exists():
            self.embeddings = np.load(EMBEDDINGS_DB)
        self.loaded = True

state = AppState()

# Pydantic models
class DocumentInfo(BaseModel):
    id: str
    filename: str
    filetype: str
    size: int
    hash: str
    upload_date: str
    status: str
    language: str
    ocr_applied: bool = False
    content_preview: str = ""
    entities: List[Dict[str, Any]] = []
    embedding_id: Optional[int] = None

class SearchResult(BaseModel):
    document_id: str
    filename: str
    similarity_score: float
    content_preview: str
    matched_entities: List[str]

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    size: float
    color: str

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: float

class PipelineStatus(BaseModel):
    step: str
    status: str
    progress: float
    message: str

class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    size_mb: float
    loaded: bool

# Initialize ML models
def load_models():
    """Load ML models on startup"""
    print("Loading sentence-transformers model...")
    state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loading spaCy NLP model...")
    try:
        state.nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm...")
        from spacy.cli import download
        download("en_core_web_sm")
        state.nlp_model = spacy.load("en_core_web_sm")
    state.loaded = True
    print("Models loaded successfully!")

def generate_file_hash(file_content: bytes) -> str:
    """Generate real SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

def detect_language(text: str) -> str:
    """Detect language of text using simple heuristics"""
    # In production, use langdetect or similar
    danish_chars = set('æøåÆØÅ')
    has_danish = any(c in text for c in danish_chars)
    
    if has_danish:
        return 'da'
    
    # Simple English detection
    common_english_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with'}
    words = text.lower().split()[:100]
    english_count = sum(1 for w in words if w in common_english_words)
    
    if english_count > 3:
        return 'en'
    
    return 'mixed'

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {str(e)}")

def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {str(e)}")

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities using spaCy"""
    if not state.nlp_model:
        return []
    
    doc = state.nlp_model(text[:10000])  # Limit text length
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    
    return entities

def generate_embedding(text: str) -> np.ndarray:
    """Generate sentence embedding using transformers"""
    if not state.embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    embedding = state.embedding_model.encode(text, convert_to_numpy=True)
    return embedding

def build_knowledge_graph(documents: List[Dict], embeddings: np.ndarray) -> None:
    """Build knowledge graph from documents and their embeddings"""
    state.knowledge_graph.clear()
    
    # Add document nodes
    for doc in documents:
        state.knowledge_graph.add_node(
            doc['id'],
            type='document',
            label=doc['filename'],
            size=15,
            color='#4CAF50'
        )
    
    # Calculate similarity and add edges
    if len(embeddings) > 1:
        similarity_matrix = cosine_similarity(embeddings)
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.3:  # Threshold for connection
                    state.knowledge_graph.add_edge(
                        documents[i]['id'],
                        documents[j]['id'],
                        type='similarity',
                        weight=float(similarity)
                    )
    
    # Add entity nodes and connect to documents
    for doc in documents:
        for entity in doc.get('entities', []):
            entity_id = f"entity_{entity['text'].lower().replace(' ', '_')}_{entity['label']}"
            
            if not state.knowledge_graph.has_node(entity_id):
                state.knowledge_graph.add_node(
                    entity_id,
                    type='entity',
                    label=entity['text'],
                    entity_type=entity['label'],
                    size=10,
                    color='#2196F3'
                )
            
            state.knowledge_graph.add_edge(
                doc['id'],
                entity_id,
                type='contains_entity',
                weight=1.0
            )

def search_similar(query: str, top_k: int = 10) -> List[SearchResult]:
    """Search for similar documents using semantic similarity"""
    if not state.embedding_model or state.embeddings is None or len(state.documents) == 0:
        return []
    
    # Generate query embedding
    query_embedding = state.embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, state.embeddings)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum threshold
            doc = state.documents[idx]
            results.append(SearchResult(
                document_id=doc['id'],
                filename=doc['filename'],
                similarity_score=float(similarities[idx]),
                content_preview=doc.get('content_preview', '')[:200],
                matched_entities=[e['text'] for e in doc.get('entities', [])[:5]]
            ))
    
    return results

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    state.load()
    if state.documents:
        load_models()

@app.get("/")
async def root():
    return {"message": "NeuroGraph Knowledge Base API", "version": "1.0.0"}

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "documents_count": len(state.documents),
        "graph_nodes": state.knowledge_graph.number_of_nodes(),
        "graph_edges": state.knowledge_graph.number_of_edges(),
        "models_loaded": state.loaded,
        "storage_used_mb": sum(os.path.getsize(f) for f in UPLOAD_DIR.glob("*")) / (1024 * 1024) if UPLOAD_DIR.exists() else 0
    }

@app.post("/api/upload", response_model=DocumentInfo)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a document"""
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Generate real hash
    file_hash = generate_file_hash(content)
    
    # Check for duplicates
    for doc in state.documents:
        if doc['hash'] == file_hash:
            raise HTTPException(status_code=409, detail="Duplicate file already exists")
    
    # Determine file type
    filename = file.filename or "unknown"
    ext = filename.split('.')[-1].lower()
    filetype_map = {
        'pdf': 'pdf',
        'docx': 'document',
        'doc': 'document',
        'txt': 'text',
        'png': 'image',
        'jpg': 'image',
        'jpeg': 'image',
        'gif': 'image',
        'bmp': 'image'
    }
    filetype = filetype_map.get(ext, 'other')
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(content)
    
    # Extract text based on file type
    ocr_applied = False
    text_content = ""
    
    try:
        if filetype == 'pdf':
            text_content = extract_text_from_pdf(content)
        elif filetype == 'document':
            if ext == 'docx':
                text_content = extract_text_from_docx(content)
            else:
                text_content = content.decode('utf-8', errors='ignore')
        elif filetype == 'text':
            text_content = content.decode('utf-8', errors='ignore')
        elif filetype == 'image':
            text_content = extract_text_from_image(content)
            ocr_applied = True
    except Exception as e:
        # Continue without text extraction
        text_content = f"[Could not extract text: {str(e)}]"
    
    # Detect language
    language = detect_language(text_content) if text_content else 'unknown'
    
    # Extract entities
    entities = extract_entities(text_content) if text_content and state.nlp_model else []
    
    # Generate embedding
    embedding = None
    if text_content and state.embedding_model:
        embedding = generate_embedding(text_content[:5000])  # Limit length
    
    # Create document record
    doc_info = DocumentInfo(
        id=file_id,
        filename=filename,
        filetype=filetype,
        size=file_size,
        hash=file_hash,
        upload_date=datetime.now().isoformat(),
        status="processed",
        language=language,
        ocr_applied=ocr_applied,
        content_preview=text_content[:500] if text_content else "",
        entities=entities
    )
    
    # Update state
    state.documents.append(doc_info.dict())
    
    # Update embeddings
    if embedding is not None:
        if state.embeddings is None:
            state.embeddings = embedding.reshape(1, -1)
        else:
            state.embeddings = np.vstack([state.embeddings, embedding.reshape(1, -1)])
        doc_info.embedding_id = len(state.documents) - 1
    
    # Rebuild knowledge graph
    if state.embeddings is not None:
        build_knowledge_graph(state.documents, state.embeddings)
    
    # Save state
    background_tasks.add_task(state.save)
    
    return doc_info

@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents(limit: int = Query(100, le=1000)):
    """Get list of all documents"""
    return state.documents[:limit]

@app.get("/api/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """Get specific document details"""
    for doc in state.documents:
        if doc['id'] == doc_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, background_tasks: BackgroundTasks):
    """Delete a document"""
    # Find document index
    doc_index = None
    for i, doc in enumerate(state.documents):
        if doc['id'] == doc_id:
            doc_index = i
            break
    
    if doc_index is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from state
    state.documents.pop(doc_index)
    
    # Remove embedding
    if state.embeddings is not None and doc_index < len(state.embeddings):
        state.embeddings = np.delete(state.embeddings, doc_index, axis=0)
    
    # Delete file
    for file_path in UPLOAD_DIR.glob(f"{doc_id}_*"):
        file_path.unlink()
    
    # Rebuild graph
    if state.embeddings is not None and len(state.documents) > 0:
        build_knowledge_graph(state.documents, state.embeddings)
    else:
        state.knowledge_graph.clear()
    
    # Save state
    background_tasks.add_task(state.save)
    
    return {"message": "Document deleted successfully"}

@app.get("/api/search", response_model=List[SearchResult])
async def search_documents(q: str = Query(..., min_length=1), limit: int = Query(10, le=100)):
    """Search documents using semantic similarity"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    results = search_similar(q, top_k=limit)
    return results

@app.get("/api/graph")
async def get_graph():
    """Get knowledge graph data"""
    nodes = []
    edges = []
    
    for node, data in state.knowledge_graph.nodes(data=True):
        nodes.append({
            "id": node,
            "label": data.get('label', node),
            "type": data.get('type', 'unknown'),
            "size": data.get('size', 10),
            "color": data.get('color', '#999999')
        })
    
    for u, v, data in state.knowledge_graph.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "type": data.get('type', 'related'),
            "weight": data.get('weight', 1.0)
        })
    
    return {"nodes": nodes, "edges": edges}

@app.get("/api/models")
async def get_models():
    """Get information about loaded ML models"""
    models = []
    
    if state.embedding_model:
        models.append(ModelInfo(
            name="all-MiniLM-L6-v2",
            type="sentence-transformer",
            status="loaded",
            size_mb=90.0,
            loaded=True
        ))
    
    if state.nlp_model:
        models.append(ModelInfo(
            name="en_core_web_sm",
            type="spacy-nlp",
            status="loaded",
            size_mb=12.0,
            loaded=True
        ))
    
    return models

@app.post("/api/models/load")
async def load_models_endpoint(background_tasks: BackgroundTasks):
    """Load ML models"""
    if state.loaded:
        return {"message": "Models already loaded"}
    
    background_tasks.add_task(load_models)
    return {"message": "Models loading in background"}

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    total_size = sum(doc['size'] for doc in state.documents)
    
    # Count entities by type
    entity_counts = {}
    for doc in state.documents:
        for entity in doc.get('entities', []):
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    # Language distribution
    language_counts = {}
    for doc in state.documents:
        lang = doc['language']
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    return {
        "total_documents": len(state.documents),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "graph_nodes": state.knowledge_graph.number_of_nodes(),
        "graph_edges": state.knowledge_graph.number_of_edges(),
        "entity_types": entity_counts,
        "languages": language_counts,
        "avg_entities_per_doc": sum(len(doc.get('entities', [])) for doc in state.documents) / max(len(state.documents), 1)
    }

@app.get("/api/export")
async def export_data():
    """Export all data as JSON"""
    return {
        "documents": state.documents,
        "graph": {
            "nodes": [(n, d) for n, d in state.knowledge_graph.nodes(data=True)],
            "edges": [(u, v, d) for u, v, d in state.knowledge_graph.edges(data=True)]
        },
        "exported_at": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
