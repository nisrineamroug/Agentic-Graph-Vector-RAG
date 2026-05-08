# -*- coding: utf-8 -*-
"""
Router for Graph RAG operations (Neo4j viz, Louvain, Metrics).
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.services.graph import GraphService

router = APIRouter(prefix="/graph", tags=["Graph"])

def get_graph_service():
    """Create a fresh GraphService instance per request."""
    return GraphService()

@router.get("/visualization")
async def get_viz(refresh: bool = Query(default=False)):
    """Returns nodes and links for the React-Force-Graph."""
    graph_service = get_graph_service()
    try:
        graph_service.connect()
        return graph_service.get_visualization_data(force_refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Graph visualization unavailable: {exc}") from exc
    finally:
        graph_service.close()

@router.get("/communities")
async def get_communities(refresh: bool = Query(default=False)):
    """Computes Louvain automatically and returns community sizes."""
    graph_service = get_graph_service()
    try:
        graph_service.connect()
        return graph_service.run_louvain_communities(force_refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Graph communities unavailable: {exc}") from exc
    finally:
        graph_service.close()


@router.get("/communities/{community_id}")
async def get_single_community(
    community_id: int,
    min_weight: float = Query(default=1.0, ge=0.0),
    limit: int = Query(default=500, ge=50, le=5000),
):
    """Returns one community as its own graph for isolated visualization."""
    graph_service = get_graph_service()
    try:
        graph_service.connect()
        return graph_service.get_community_subgraph(
            community_id=community_id,
            min_weight=min_weight,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Community subgraph unavailable: {exc}") from exc
    finally:
        graph_service.close()

@router.get("/metrics")
async def get_metrics(community_id: Optional[int] = Query(default=None)):
    """Returns centrality and connectivity metrics, optionally filtered by community."""
    graph_service = get_graph_service()
    try:
        graph_service.connect()
        return graph_service.get_graph_metrics(community_id=community_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Graph metrics unavailable: {exc}") from exc
    finally:
        graph_service.close()
