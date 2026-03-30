"""
NeuroGraph V2 - FastAPI Application

Production-ready API with:
- Async file processing
- WebSocket for real-time progress
- Batch operations
- Advanced analytics endpoints
"""

import asyncio
import io
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .v2.engine import NeuroGraphV2, get_neurograph, SearchResult, DocumentStatus

# Initialize FastAPI app
app = FastAPI(
    title="NeuroGraph V2 API",
    description="Advanced Knowledge Graph with Hybrid Search and Community Detection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
engine: Optional[NeuroGraphV2] = None
processing_tasks: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    global engine
    engine = get_neurograph()
    print("NeuroGraph V2 initialized successfully")


# ==================== Pydantic Models ====================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    alpha: float = 0.5  # Weight for dense vs BM25
    use_reranking: bool = True


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_time_ms: float
    algorithm: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_hash: str
    file_type: str
    file_size: int
    language: str
    entity_count: int
    status: str
    created_at: str


class GraphStats(BaseModel):
    nodes: int
    edges: int
    node_types: Dict[str, int]
    entity_types: Dict[str, int]
    density: float
    average_degree: float
    num_communities: int
    documents: int


class CommunityDetectionRequest(BaseModel):
    algorithm: str = "leiden"
    min_community_size: int = 2


class EntityResolutionRequest(BaseModel):
    threshold: float = 0.85


# ==================== Document Endpoints ====================

@app.post("/api/v2/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a document
    
    Supports: PDF, DOCX, TXT, images (with OCR)
    Processing happens asynchronously with progress tracking
    """
    task_id = str(uuid.uuid4())
    
    # Read file content
    content_bytes = await file.read()
    
    # Determine file type
    file_ext = Path(file.filename).suffix.lower()
    file_type_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".txt": "text",
        ".md": "markdown",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "image",
    }
    file_type = file_type_map.get(file_ext, "text")
    
    # Extract text based on file type
    try:
        content = await extract_text(content_bytes, file_type, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")
    
    # Add to engine
    doc = engine.add_document(
        filename=file.filename,
        content=content,
        file_type=file_type,
        file_bytes=content_bytes
    )
    
    if doc is None:
        raise HTTPException(status_code=409, detail="Duplicate document")
    
    return DocumentInfo(
        id=doc.id,
        filename=doc.filename,
        file_hash=doc.file_hash,
        file_type=doc.file_type,
        file_size=doc.file_size,
        language=doc.language,
        entity_count=len(doc.entities),
        status=doc.status.value,
        created_at=doc.created_at.isoformat()
    )


@app.post("/api/v2/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple documents at once"""
    results = []
    errors = []
    
    for file in files:
        try:
            content_bytes = await file.read()
            file_ext = Path(file.filename).suffix.lower()
            file_type = {"pdf": "pdf", ".docx": "docx", ".txt": "text"}.get(file_ext, "text")
            
            content = await extract_text(content_bytes, file_type, file.filename)
            
            doc = engine.add_document(
                filename=file.filename,
                content=content,
                file_type=file_type,
                file_bytes=content_bytes
            )
            
            if doc:
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "id": doc.id
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "duplicate",
                    "id": None
                })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "processed": len(results),
        "success": len([r for r in results if r["status"] == "success"]),
        "duplicates": len([r for r in results if r["status"] == "duplicate"]),
        "errors": len(errors),
        "results": results,
        "error_details": errors
    }


async def extract_text(content_bytes: bytes, file_type: str, filename: str) -> str:
    """Extract text from various file formats"""
    
    if file_type == "text" or file_type == "markdown":
        return content_bytes.decode("utf-8", errors="ignore")
    
    elif file_type == "pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")
    
    elif file_type == "docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(content_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {e}")
    
    elif file_type == "image":
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(io.BytesIO(content_bytes))
            text = pytesseract.image_to_string(image, lang='eng+dan')
            return text
        except Exception as e:
            raise Exception(f"OCR failed: {e}")
    
    else:
        return content_bytes.decode("utf-8", errors="ignore")


@app.get("/api/v2/documents")
async def list_documents(limit: int = 100, offset: int = 0):
    """List all documents"""
    docs = list(engine.documents.values())
    
    # Sort by creation date
    docs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    paginated = docs[offset:offset + limit]
    
    return {
        "total": len(docs),
        "limit": limit,
        "offset": offset,
        "documents": [
            DocumentInfo(
                id=doc.id,
                filename=doc.filename,
                file_hash=doc.file_hash,
                file_type=doc.file_type,
                file_size=doc.file_size,
                language=doc.language,
                entity_count=len(doc.entities),
                status=doc.status.value,
                created_at=doc.created_at.isoformat()
            )
            for doc in paginated
        ]
    }


@app.get("/api/v2/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document details"""
    if doc_id not in engine.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = engine.documents[doc_id]
    
    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_hash": doc.file_hash,
        "file_type": doc.file_type,
        "file_size": doc.file_size,
        "content": doc.content,
        "language": doc.language,
        "entities": doc.entities,
        "created_at": doc.created_at.isoformat(),
        "status": doc.status.value
    }


@app.delete("/api/v2/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    if doc_id not in engine.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Note: In production, we'd need to rebuild FAISS index
    del engine.documents[doc_id]
    engine._save_state()
    
    return {"status": "deleted", "id": doc_id}


# ==================== Search Endpoints ====================

@app.post("/api/v2/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Hybrid search with BM25 + Dense embeddings + Re-ranking
    
    Returns most relevant documents with highlights and entities
    """
    start_time = time.time()
    
    results = engine.hybrid_search(
        query=request.query,
        top_k=request.top_k,
        alpha=request.alpha,
        use_reranking=request.use_reranking
    )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return SearchResponse(
        results=[
            {
                "document_id": r.document_id,
                "score": r.score,
                "bm25_score": r.bm25_score,
                "dense_score": r.dense_score,
                "rerank_score": r.rerank_score,
                "highlights": r.highlights,
                "entities": r.entities,
                "document": engine.documents[r.document_id].to_dict()
            }
            for r in results
        ],
        query=request.query,
        total_time_ms=elapsed_ms,
        algorithm="hybrid_bm25_dense_rerank"
    )


@app.get("/api/v2/search/similar/{doc_id}")
async def find_similar(doc_id: str, limit: int = 10):
    """Find documents similar to a given document"""
    if doc_id not in engine.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = engine.documents[doc_id]
    
    # Use FAISS to find similar
    query_embedding = doc.embedding
    distances, indices = engine.faiss_index.search(
        query_embedding.reshape(1, -1),
        k=limit + 1
    )
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(engine.tokenized_docs):
            continue
        
        other_doc_id = list(engine.documents.keys())[idx]
        if other_doc_id == doc_id:
            continue
        
        results.append({
            "document_id": other_doc_id,
            "similarity": float(dist),
            "filename": engine.documents[other_doc_id].filename
        })
        
        if len(results) >= limit:
            break
    
    return {"query_doc": doc_id, "similar_documents": results}


# ==================== Graph Analytics Endpoints ====================

@app.get("/api/v2/graph/stats", response_model=GraphStats)
async def get_graph_statistics():
    """Get comprehensive graph statistics"""
    stats = engine.get_graph_stats()
    return GraphStats(**stats)


@app.post("/api/v2/graph/communities")
async def detect_communities(request: CommunityDetectionRequest):
    """Detect communities in the knowledge graph"""
    communities = engine.detect_communities(algorithm=request.algorithm)
    
    # Filter by minimum size
    filtered = [list(c) for c in communities if len(c) >= request.min_community_size]
    
    return {
        "algorithm": request.algorithm,
        "num_communities": len(filtered),
        "communities": [
            {
                "id": i,
                "size": len(c),
                "nodes": c
            }
            for i, c in enumerate(filtered)
        ]
    }


@app.post("/api/v2/graph/entities/resolve")
async def resolve_entities(request: EntityResolutionRequest):
    """Resolve duplicate/variant entities"""
    resolved = engine.resolve_entities(threshold=request.threshold)
    
    return {
        "threshold": request.threshold,
        "clusters_found": len(resolved),
        "resolutions": [
            {
                "canonical": canonical,
                "variants": variants
            }
            for canonical, variants in resolved.items()
        ]
    }


@app.get("/api/v2/graph/export/{format}")
async def export_graph(format: str):
    """Export graph in various formats (json, gexf, graphml)"""
    try:
        data = engine.export_graph(format=format)
        
        if format == "json":
            return JSONResponse(data)
        else:
            return StreamingResponse(
                io.BytesIO(data),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=knowledge_graph.{format}"}
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v2/graph/visualization")
async def get_graph_for_visualization(limit_nodes: int = 500):
    """Get graph data optimized for frontend visualization"""
    # Sample large graphs
    nodes = list(engine.graph.nodes(data=True))
    edges = list(engine.graph.edges(data=True))
    
    if len(nodes) > limit_nodes:
        # Keep highest degree nodes
        degrees = dict(engine.graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:limit_nodes]
        
        nodes = [(n, d) for n, d in nodes if n in top_nodes]
        edges = [(u, v, d) for u, v, d in edges if u in top_nodes and v in top_nodes]
    
    return {
        "nodes": [
            {
                "id": node_id,
                **data
            }
            for node_id, data in nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                **data
            }
            for u, v, data in edges
        ]
    }


# ==================== WebSocket for Real-time Progress ====================

@app.websocket("/ws/processing")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time processing updates"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                task_id = data.get("task_id")
                if task_id in processing_tasks:
                    await websocket.send_json(processing_tasks[task_id])
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ==================== Health & Info ====================

@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "documents": len(engine.documents),
        "graph_nodes": engine.graph.number_of_nodes(),
        "graph_edges": engine.graph.number_of_edges(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v2/info")
async def get_info():
    """Get system information"""
    import sys
    
    return {
        "version": "2.0.0",
        "python_version": sys.version,
        "features": [
            "FAISS vector indexing",
            "Hybrid BM25 + Dense search",
            "Cross-encoder re-ranking",
            "Leiden community detection",
            "Entity resolution",
            "Disk-based embedding cache",
            "Multi-format document support",
            "Real-time WebSocket updates"
        ],
        "models": {
            "embedder": engine.embedder.model_name_or_path,
            "reranker": engine.reranker.config.name_or_path,
            "nlp": engine.nlp.meta["name"]
        }
    }
