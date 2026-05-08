# -*- coding: utf-8 -*-
"""
/query endpoint – accepts a user question and returns an answer
using the agentic RAG pipeline.
"""

import os
import json
from fastapi import APIRouter, HTTPException
from backend.models.schemas import QueryRequest, QueryResponse
from backend.services.agentic import AgenticService
from backend.services.graph import GraphService
from backend.services.vectorial import VectorService
from backend.services.auto_reward import AutoRewardScorer
from backend.services.llm import LLMService

router = APIRouter(prefix="/query", tags=["Query"])

@router.get("/history")
async def get_query_history():
    """Returns the list of recent queries from logs."""
    reward_scorer = get_reward_scorer()
    history = []
    if os.path.exists(reward_scorer.log_path):
        with open(reward_scorer.log_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("query"):
                    history.append(entry["query"])
    return list(reversed(history))[:20]

# Lazy-load services to avoid hanging imports
_agent_service = None
_vector_service = None
_graph_service = None
_reward_scorer = None
_llm_service = None

def get_agent_service():
    global _agent_service
    if _agent_service is None:
        _agent_service = AgenticService()
    return _agent_service

def get_vector_service():
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service

def get_graph_service():
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service

def get_reward_scorer():
    global _reward_scorer
    if _reward_scorer is None:
        _reward_scorer = AutoRewardScorer()
    return _reward_scorer


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def _normalize_route(route_value: str) -> str:
    route_name = route_value.lower().strip()
    aliases = {
        "vector": "vectorial",
        "vectorial": "vectorial",
        "graph": "graph",
        "hybrid": "hybrid",
    }
    if route_name not in aliases:
        raise HTTPException(status_code=400, detail="route_override must be vectorial, graph, or hybrid")
    return aliases[route_name]


def _build_vector_sources(results):
    sources = []
    for result in results:
        metadata = result.get("metadata", {})
        sources.append({
            "page": metadata.get("page_label"),
            "score": result.get("score"),
            "text": result.get("text"),
            "keywords": metadata.get("keywords", []),
        })
    return sources


@router.post("/", response_model=QueryResponse)
async def ask_question(payload: QueryRequest):
    """
    Main agentic query endpoint.
    Step 1: Classify query features → Step 2: Q-Learning selects route →
    Step 3: Execute retrieval → Step 4: LLM generates answer →
    Step 5: Compute reward & update Q-table.
    """
    agent_service = get_agent_service()
    vector_service = get_vector_service()
    graph_service = get_graph_service()
    reward_scorer = get_reward_scorer()
    llm_service = get_llm_service()

    # ── Step 1 & 2: Route the query ──
    if payload.route_override:
        route_name = _normalize_route(payload.route_override)
        route_info = {"route": route_name, "confidence": 1.0}
        action_index = {"vectorial": 0, "graph": 1, "hybrid": 2}[route_name]
    else:
        route_info = agent_service.route_query(payload.query)
        route_name = route_info["route"]
        action_index = route_info["action_index"]

    # ── Step 3: Execute retrieval based on chosen route ──
    reward = 0.0
    reward_details = {"reward": 0.0}
    sources = []
    answer = ""
    llm_context = ""

    if route_name == "vectorial":
        if not vector_service.load_index():
            raise HTTPException(status_code=404, detail="Vector index not loaded. Run ingest.py first.")
        results = vector_service.hybrid_search(payload.query, top_k=payload.top_k)
        llm_context = "\n\n".join(
            [f"[Chunk {i+1}] {r.get('text', '')}" for i, r in enumerate(results)]
        )
        answer = llm_service.generate_answer(payload.query, llm_context, route_name)
        sources = _build_vector_sources(results)
        reward_res = reward_scorer.score_vector_results_detailed(payload.query, results)
        reward = reward_res["reward"]
        reward_details = reward_res["details"]
        
        # Add vector visualization data to thought_process
        route_info["vector_viz"] = {
            "query": payload.query,
            "similarities": [
                {"text": r.get("text", "")[:100], "score": float(r.get("score", 0)), "method": r.get("method", "vector")} 
                for r in results[:10]
            ]
        }

    elif route_name == "graph":
        try:
            graph_results = graph_service.graph_search(payload.query, top_k=payload.top_k)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Graph search failed: {exc}") from exc
        finally:
            graph_service.close()

        llm_context = graph_results.get("context", "")
        answer = llm_service.generate_answer(payload.query, llm_context, route_name)

        # Build sources from retrieved graph chunks
        for chunk in graph_results.get("sources", []):
            sources.append({
                "text": chunk.get("text", ""),
                "page": chunk.get("page_label", ""),
                "source": chunk.get("source", ""),
                "terms": chunk.get("related_terms", []),
            })

        # Add graph reasoning to thought_process
        route_info["cypher_queries"] = graph_results.get("cypher_queries", [])
        route_info["subgraph"] = graph_results.get("subgraph", {})
        route_info["matched_terms"] = graph_results.get("matched_terms", [])
        route_info["relationships"] = graph_results.get("relationships", [])

        # Reward based on retrieval quality
        n_matched = len(graph_results.get("matched_terms", []))
        n_chunks = graph_results.get("chunks_retrieved", 0)
        term_cov = min(n_matched / max(len(payload.query.split()), 1), 1.0)
        chunk_cov = min(n_chunks / max(payload.top_k, 1), 1.0)
        reward = 0.5 * term_cov + 0.5 * chunk_cov
        reward_details = {
            "reward": reward,
            "term_coverage": term_cov,
            "chunk_coverage": chunk_cov,
            "matched_terms": n_matched,
            "chunks_retrieved": n_chunks,
        }

    else:  # hybrid
        # Vector retrieval
        vector_context = ""
        vector_sources = []
        if vector_service.load_index():
            vector_results = vector_service.hybrid_search(payload.query, top_k=payload.top_k)
            vector_context = "\n\n".join(
                [f"[Vector Chunk {i+1}] {r.get('text', '')}" for i, r in enumerate(vector_results)]
            )
            vector_sources = _build_vector_sources(vector_results)
        else:
            vector_results = []

        # Graph retrieval
        graph_context = ""
        graph_sources = []
        try:
            graph_results = graph_service.graph_search(payload.query, top_k=3)
            graph_context = graph_results.get("context", "")
            for chunk in graph_results.get("sources", []):
                graph_sources.append({
                    "text": chunk.get("text", ""),
                    "page": chunk.get("page_label", ""),
                    "source": chunk.get("source", ""),
                    "terms": chunk.get("related_terms", []),
                })
            route_info["cypher_queries"] = graph_results.get("cypher_queries", [])
            route_info["matched_terms"] = graph_results.get("matched_terms", [])
            route_info["subgraph"] = graph_results.get("subgraph", {})
            route_info["relationships"] = graph_results.get("relationships", [])
        except Exception as exc:
            print(f"Graph search in hybrid mode failed: {exc}")
            graph_results = {"context": ""}
        finally:
            graph_service.close()

        llm_context = f"{vector_context}\n\n{graph_context}".strip()
        answer = llm_service.generate_answer(payload.query, llm_context, route_name)
        sources = vector_sources + graph_sources

        # Hybrid reward
        graph_data_for_reward = {"nodes": graph_results.get("subgraph", {}).get("nodes", []),
                                  "links": graph_results.get("subgraph", {}).get("links", [])}
        reward_res = reward_scorer.score_hybrid_results_detailed(payload.query, vector_results, graph_data_for_reward)
        reward = reward_res["reward"]
        reward_details = reward_res["details"]

    # ── Step 4: Update Q-table with reward ──
    update_info = agent_service.auto_update_from_reward(
        query=payload.query,
        action_index=action_index,
        reward=reward
    )

    # ── Step 5: Log for offline analysis ──
    reward_scorer.log_query(
        query=payload.query,
        route=route_name,
        reward=reward,
        query_state=route_info.get("state"),
        action_index=action_index
    )

    return QueryResponse(
        query=payload.query,
        route=route_name,
        answer=answer,
        confidence=float(route_info.get("confidence", 0.0)),
        sources=sources,
        thought_process=route_info,
        reward_details=reward_details,
        update_details=update_info
    )
