"""
NeuroGraph V2 - FastAPI Backend Server
Provides REST API endpoints for document processing, search, and knowledge graph
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import shutil
from pathlib import Path

from .engine import get_processor, DocumentProcessor

app = FastAPI(
    title="NeuroGraph V2 API",
    description="Real ML/AI-powered Knowledge Graph Backend",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_hash: str
    file_type: str
    file_size: int
    language: str
    entity_count: int
    concept_count: int
    ocr_applied: bool
    created_at: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_semantic: bool = True


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    total_documents: int
    total_concepts: int
    total_entities: int
    total_size_bytes: int
    languages: Dict[str, int]
    graph: Dict[str, Any]


class EntityResponse(BaseModel):
    entities: List[Dict[str, Any]]
    concepts: List[str]


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "message": "NeuroGraph V2 Backend - Real ML/AI Processing"
    }


@app.post("/api/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    - Extracts text (PDF, DOCX, images with OCR)
    - Detects language
    - Extracts entities and concepts
    - Generates semantic embeddings
    - Adds to knowledge graph
    """
    processor = get_processor()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process the file
        doc = processor.process_file(tmp_path)
        
        return DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_hash=doc.file_hash,
            file_type=doc.file_type,
            file_size=doc.file_size,
            language=doc.language,
            entity_count=len(doc.entities),
            concept_count=len(doc.concepts),
            ocr_applied=doc.ocr_applied,
            created_at=doc.created_at
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/upload/batch", response_model=List[DocumentResponse])
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple documents in batch
    """
    processor = get_processor()
    results = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            doc = processor.process_file(tmp_path)
            results.append(DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_hash=doc.file_hash,
                file_type=doc.file_type,
                file_size=doc.file_size,
                language=doc.language,
                entity_count=len(doc.entities),
                concept_count=len(doc.concepts),
                ocr_applied=doc.ocr_applied,
                created_at=doc.created_at
            ))
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return results


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search documents using hybrid approach:
    - Keyword matching
    - Entity matching
    - Semantic similarity (optional)
    """
    processor = get_processor()
    
    results = processor.search(
        query=request.query,
        top_k=request.top_k,
        use_semantic=request.use_semantic
    )
    
    return SearchResponse(
        results=results,
        total=len(results)
    )


@app.get("/api/search", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    use_semantic: bool = True
):
    """GET endpoint for search (for browser compatibility)"""
    processor = get_processor()
    
    results = processor.search(
        query=q,
        top_k=top_k,
        use_semantic=use_semantic
    )
    
    return SearchResponse(
        results=results,
        total=len(results)
    )


@app.get("/api/graph", response_model=GraphResponse)
async def get_graph():
    """
    Get knowledge graph data for visualization
    Returns nodes (documents + entities) and edges (similarity relationships)
    """
    processor = get_processor()
    graph_data = processor.get_graph_data()
    
    return GraphResponse(
        nodes=graph_data['nodes'],
        edges=graph_data['edges'],
        stats=graph_data['stats']
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    processor = get_processor()
    stats = processor.get_stats()
    
    return StatsResponse(**stats)


@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """List all processed documents with pagination"""
    processor = get_processor()
    
    docs = list(processor.documents.values())[offset:offset+limit]
    
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_hash=doc.file_hash,
            file_type=doc.file_type,
            file_size=doc.file_size,
            language=doc.language,
            entity_count=len(doc.entities),
            concept_count=len(doc.concepts),
            ocr_applied=doc.ocr_applied,
            created_at=doc.created_at
        )
        for doc in docs
    ]


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get detailed information about a specific document"""
    processor = get_processor()
    
    if doc_id not in processor.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = processor.documents[doc_id]
    return {
        **doc.to_dict(),
        "content_preview": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge graph"""
    processor = get_processor()
    
    if doc_id not in processor.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from documents
    del processor.documents[doc_id]
    
    # Remove from graph nodes
    if doc_id in processor.knowledge_graph.nodes_data:
        del processor.knowledge_graph.nodes_data[doc_id]
    
    # Remove edges connected to this document
    processor.knowledge_graph.edges_data = [
        e for e in processor.knowledge_graph.edges_data
        if e['source'] != doc_id and e['target'] != doc_id
    ]
    
    # Save state
    processor._save_state()
    
    return {"status": "deleted", "id": doc_id}


@app.post("/api/analyze", response_model=EntityResponse)
async def analyze_text(text: str = Query(..., max_length=10000)):
    """
    Analyze text without saving:
    - Extract entities
    - Extract concepts
    - Detect language
    """
    processor = get_processor()
    
    entities = processor.entity_extractor.extract(text)
    concepts = processor.entity_extractor.extract_concepts(text)
    
    return EntityResponse(
        entities=entities,
        concepts=concepts
    )


@app.get("/api/models/status")
async def get_models_status():
    """Check status of ML models"""
    from .engine import (
        SPACY_AVAILABLE, 
        SENTENCE_TRANSFORMERS_AVAILABLE,
        PDF_AVAILABLE,
        DOCX_AVAILABLE,
        OCR_AVAILABLE,
        NETWORKX_AVAILABLE,
        LANGDETECT_AVAILABLE
    )
    
    return {
        "spacy": {"available": SPACY_AVAILABLE, "loaded": SPACY_AVAILABLE},
        "sentence_transformers": {"available": SENTENCE_TRANSFORMERS_AVAILABLE},
        "pdf_reader": {"available": PDF_AVAILABLE},
        "docx_reader": {"available": DOCX_AVAILABLE},
        "ocr": {"available": OCR_AVAILABLE},
        "networkx": {"available": NETWORKX_AVAILABLE},
        "langdetect": {"available": LANGDETECT_AVAILABLE}
    }


@app.post("/api/reset")
async def reset_system():
    """
    Reset the entire system (delete all documents and graph)
    Use with caution!
    """
    processor = get_processor()
    
    # Clear all data
    processor.documents.clear()
    processor.concept_nodes.clear()
    processor.knowledge_graph = type(processor.knowledge_graph)()
    
    # Remove state file
    state_file = processor.data_dir / 'state.json'
    if state_file.exists():
        state_file.unlink()
    
    return {
        "status": "reset_complete",
        "message": "All documents and graph data have been deleted"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
