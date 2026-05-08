# -*- coding: utf-8 -*-
"""
Auto-Reward Scorer for implicit feedback without user interaction.
Computes retrieval quality from multiple signals.
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime
import json
import os


class AutoRewardScorer:
    """
    Automatically scores retrieval quality without explicit user feedback.
    
    Reward signals:
    - Semantic relevance: avg similarity score of top-k results
    - Result diversity: cosine distance variance (avoiding redundancy)
    - Coverage: number of unique terms matched in query
    - Confidence: max similarity score of top result
    """
    
    def __init__(self, 
                 log_path: str = "data/query_logs.jsonl",
                 relevance_weight: float = 0.5,
                 diversity_weight: float = 0.2,
                 coverage_weight: float = 0.2,
                 confidence_weight: float = 0.1):
        self.log_path = log_path
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        self.confidence_weight = confidence_weight
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def score_vector_results_detailed(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results or len(results) == 0:
            return {"reward": 0.0, "details": {"relevance": 0, "confidence": 0, "diversity": 0, "coverage": 0}}

        scores = [r.get("score", 0.0) for r in results]
        relevance = np.mean(scores[:min(3, len(scores))]) if scores else 0.0
        confidence = max(scores) if scores else 0.0
        diversity = float(np.std(scores)) if len(scores) > 1 else 0.5
        
        query_tokens = set(query.lower().split())
        coverage = 0.0
        if results:
            top_text = " ".join([r.get("text", "").lower() for r in results[:2]])
            matched_tokens = sum(1 for token in query_tokens if token in top_text)
            coverage = matched_tokens / max(len(query_tokens), 1)
        
        reward = (
            self.relevance_weight * relevance +
            self.confidence_weight * confidence +
            self.diversity_weight * min(diversity, 1.0) +
            self.coverage_weight * coverage
        )
        
        return {
            "reward": float(np.clip(reward, 0.0, 1.0)),
            "details": {
                "reward": float(np.clip(reward, 0.0, 1.0)),
                "relevance": float(relevance),
                "confidence": float(confidence),
                "diversity": float(diversity),
                "coverage": float(coverage)
            }
        }

    def score_vector_results(self, query: str, results: List[Dict[str, Any]]) -> float:
        return self.score_vector_results_detailed(query, results)["reward"]

    def score_graph_results_detailed(self, query: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        if not graph_data:
            return {"reward": 0.0, "details": {"density": 0, "diversity": 0, "coverage": 0}}
        
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])
        
        density = len(links) / max(len(nodes), 1) if nodes else 0.0
        
        rel_types = set()
        for link in links:
            rel_type = link.get("type", link.get("value", "CO_OCCURS_WITH"))
            if rel_type in ["CONTROLS", "INFLUENCES", "PART_OF", "CAUSES", "DRIVES"]:
                rel_types.add(rel_type)
        diversity = len(rel_types) / 5.0
        
        query_tokens = set(query.lower().split())
        node_labels = " ".join([n.get("text", n.get("label", "")).lower() for n in nodes])
        coverage = 0.0
        if node_labels:
            matched = sum(1 for token in query_tokens if token in node_labels)
            coverage = matched / max(len(query_tokens), 1)
        
        reward = (0.4 * min(density, 1.0) + 0.3 * diversity + 0.3 * coverage)
        
        return {
            "reward": float(np.clip(reward, 0.0, 1.0)),
            "details": {
                "reward": float(np.clip(reward, 0.0, 1.0)),
                "density": float(density),
                "diversity": float(diversity),
                "coverage": float(coverage)
            }
        }

    def score_graph_results(self, query: str, graph_data: Dict[str, Any]) -> float:
        return self.score_graph_results_detailed(query, graph_data)["reward"]

    def score_hybrid_results_detailed(self, 
                                     query: str, 
                                     vector_results: List[Dict[str, Any]], 
                                     graph_data: Dict[str, Any]) -> Dict[str, Any]:
        v_res = self.score_vector_results_detailed(query, vector_results)
        g_res = self.score_graph_results_detailed(query, graph_data)
        
        v_score = v_res["reward"]
        g_score = g_res["reward"]
        
        agreement = 1.0 - abs(v_score - g_score)
        reward = (v_score + g_score) / 2.0 * (0.7 + 0.3 * agreement)
        
        return {
            "reward": float(np.clip(reward, 0.0, 1.0)),
            "details": {
                "reward": float(np.clip(reward, 0.0, 1.0)),
                "vector_reward": v_score,
                "graph_reward": g_score,
                "agreement": float(agreement),
                "vector_breakdown": v_res["details"],
                "graph_breakdown": g_res["details"]
            }
        }

    def score_hybrid_results(self, 
                            query: str, 
                            vector_results: List[Dict[str, Any]], 
                            graph_data: Dict[str, Any]) -> float:
        return self.score_hybrid_results_detailed(query, vector_results, graph_data)["reward"]

    def log_query(self, 
                  query: str, 
                  route: str, 
                  reward: float,
                  query_state: str = None,
                  action_index: int = None):
        """
        Log query + route + reward for analysis.
        Useful for offline evaluation and debugging.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "route": route,
            "reward": reward,
            "state": query_state,
            "action_index": action_index,
        }
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Could not log query: {e}")

    def get_route_stats(self) -> Dict[str, Any]:
        """
        Analyze logged queries to see which routes work best.
        Returns: {route: {avg_reward, count, best_state}}
        """
        stats = {}
        
        if not os.path.exists(self.log_path):
            return stats
        
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    route = entry.get("route")
                    reward = entry.get("reward", 0.0)
                    
                    if route not in stats:
                        stats[route] = {
                            "count": 0,
                            "total_reward": 0.0,
                            "avg_reward": 0.0,
                            "max_reward": 0.0,
                            "best_state": None,
                        }
                    
                    stats[route]["count"] += 1
                    stats[route]["total_reward"] += reward
                    stats[route]["max_reward"] = max(stats[route]["max_reward"], reward)
                    
                    if reward > (stats[route].get("best_reward", 0.0) or 0):
                        stats[route]["best_state"] = entry.get("state")
                        stats[route]["best_reward"] = reward
            
            # Compute averages
            for route in stats:
                if stats[route]["count"] > 0:
                    stats[route]["avg_reward"] = stats[route]["total_reward"] / stats[route]["count"]
        
        except Exception as e:
            print(f"Warning: Could not read logs: {e}")
        
        return stats
