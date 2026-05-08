# -*- coding: utf-8 -*-
"""
Router for Vectorial RAG operations (Retrieval, PCA, Chunking info).
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from backend.services.vectorial import VectorService
from typing import Dict, Any
from pydantic import BaseModel
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

router = APIRouter(prefix="/vectorial", tags=["Vectorial"])

# Lazy-load service to avoid hanging imports
_vector_service = None


class SummarizeRequest(BaseModel):
    query: str
    chunks: list

def get_vector_service():
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service

@router.get("/pca")
async def get_pca():
    """Returns 2D PCA projection for visualization."""
    vector_service = get_vector_service()
    if not vector_service.load_index():
        raise HTTPException(status_code=404, detail="Index not loaded. Run ingestion first.")
    return vector_service.get_pca_projection()


@router.get("/pca_image")
async def get_pca_image():
    """Return a PNG image (matplotlib) of the PCA 2D projection."""
    vector_service = get_vector_service()
    if not vector_service.load_index():
        raise HTTPException(status_code=404, detail="Index not loaded. Run ingestion first.")

    proj = vector_service.get_pca_projection()
    # `get_pca_projection()` returns a list of point dicts
    if isinstance(proj, dict):
        points = proj.get('points', [])
    else:
        points = proj or []

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [p.get('x', 0.0) for p in points]
    ys = [p.get('y', 0.0) for p in points]
    labels = [p.get('label', '') for p in points]
    if points:
        ax.scatter(xs, ys, s=10, alpha=0.7)
        # annotate a small sample
        for i, txt in enumerate(labels[:20]):
            ax.annotate(txt, (xs[i], ys[i]), fontsize=6, alpha=0.8)
    else:
        ax.text(0.5, 0.5, 'No embeddings available', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('PCA 2D projection of embeddings')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type='image/png')


@router.post('/summarize')
async def summarize_chunks(payload: SummarizeRequest):
    """Return a lightweight summary of provided chunks for demo purposes.

    This is a placeholder summarizer (concatenation + truncation).
    Replace with an LLM call in production.
    """
    query = payload.query
    chunks = payload.chunks

    if not chunks:
        return {"summary": "No chunks provided."}

    # naive summarization: join tops of chunks and truncate
    texts = [c.get('text', '') for c in chunks]
    joined = ' '.join(texts)
    summary = joined[:800]
    if len(joined) > 800:
        summary += '...'
    return {"summary": summary}


@router.get('/cleaned_pdf')
async def get_cleaned_pdf():
    """Serve the cleaned thesis PDF for download/viewing."""
    # Exact cleaned thesis PDF path provided by the user
    pdf_paths = [
        'data/cleaned_corpus/16_cleaned.pdf',
    ]
    
    for path in pdf_paths:
        if os.path.exists(path):
            return FileResponse(path, media_type='application/pdf', filename='cleaned_thesis.pdf')
    
    # If no PDF found, return a placeholder message
    raise HTTPException(status_code=404, detail="Cleaned thesis PDF not found. Check backend data folder.")

@router.get("/search")
async def search_vectorial(query: str):
    """Semantic search with similarity scores."""
    vector_service = get_vector_service()
    if not vector_service.load_index():
        raise HTTPException(status_code=404, detail="Index not loaded.")
    results = vector_service.hybrid_search(query)
    return {
        "query": query,
        "results": results
    }

@router.get("/stats")
async def get_stats():
    """Returns info about the vector store, including chunking method and samples."""
    vector_service = get_vector_service()
    if not vector_service.load_index():
        return {"loaded": False}
    
    # Get a few sample chunks
    samples = []
    for c in vector_service.chunks[:5]:
        samples.append({
            "text": c.get("text", "")[:200] + "...",
            "page": c.get("metadata", {}).get("page_label"),
            "keywords": c.get("metadata", {}).get("keywords", [])
        })

    return {
        "loaded": True,
        "chunk_count": len(vector_service.chunks),
        "model": vector_service.model_name,
        "chunking_method": "RecursiveCharacterTextSplitter (LangChain)",
        "chunk_params": {
            "size": 1500,
            "overlap": 400
        },
        "samples": samples
    }

@router.get("/eval-metrics")
async def get_eval_metrics():
    """Returns retrieval evaluation metrics from the CSV report."""
    path = "data/eval/reports/final_retrieval_comparison.csv"
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # Handle NaN values to ensure valid JSON
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error loading eval metrics: {e}")
        return []
