# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:07:27 2026

@author: nisrine
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.config import get_settings
from backend.routers import query, health, vectorial, graph, agentic

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic Vectorial Graph RAG System",
    version=settings.APP_VERSION,
)

# ── CORS ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query.router)
app.include_router(vectorial.router)
app.include_router(graph.router)
app.include_router(agentic.router)


@app.get("/")
def root():
    return {"status": "ok", "message": f"{settings.APP_NAME} running"}


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
